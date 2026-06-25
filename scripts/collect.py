"""Phase 2 collection CLI — continuous AdFlex collection with cursor checkpointing.

Commands:
    collect plan          — Dry run: show all queries and their pagination state
    collect run           — Start/resume collection (paginate until budget exhausted)
    collect assess        — Print assessment stats
    collect enrich        — Fetch detail for high-tier ads (post-scoring)
    collect status        — Overall collection progress
    collect reset         — Rebuild state from JSONL, reset cursors, archive legacy files
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from draper.collection.sweep import (
    CollectionCheckpoint,
    FilterConfig,
    SweepExecutor,
    SweepPlanner,
    load_enriched_ids,
    load_seen_ids,
    save_enrich_stats,
    save_enriched_ids,
    save_run_stats,
    select_queries_for_budget,
)
from draper.scraping.adflex import AdFlexClient
from draper.scraping.config import ScrapingConfig
from draper.scraping.rate_limiter import RateLimiter
from draper.scraping.schemas import RawAd
from draper.utils.io import read_jsonl, update_jsonl_records
from draper.utils.logging import setup_logging

load_dotenv()

app = typer.Typer(help="Phase 2 AdFlex collection (continuous, cursor-checkpointed)")
console = Console()
log = setup_logging()


def _init_planner(
    plan_path: str = "configs/sweep_plans.yaml",
    filters_dir: str = "configs/filters",
) -> SweepPlanner:
    return SweepPlanner(plan_path, FilterConfig(filters_dir))


@app.command()
def plan(
    plan_path: str = typer.Option("configs/sweep_plans.yaml"),
    filters_dir: str = typer.Option("configs/filters"),
    output_dir: str = typer.Option("data/raw"),
    max_pages: int = typer.Option(10, help="Max pages per query"),
) -> None:
    """Dry run: show all queries and their pagination state."""
    planner = _init_planner(plan_path, filters_dir)
    raw_dir = Path(output_dir)
    checkpoint = CollectionCheckpoint(raw_dir)

    # Generate all queries
    all_queries = []
    for platform in planner.get_platform_names():
        for sweep_name in planner.get_sweep_names(platform):
            queries = planner.generate_queries(platform, sweep_name)
            all_queries.extend(queries)

    total_budget = planner.get_total_budget()
    calls_made = checkpoint.total_calls_made()

    console.print("\n[bold]Collection Plan[/bold]")
    console.print(f"Total queries: {len(all_queries)}")
    console.print(f"Max pages per query: {max_pages}")
    remaining = total_budget - calls_made
    console.print(f"Budget: {total_budget} calls ({calls_made} used, {remaining} remaining)")
    console.print()

    # Summary table
    table = Table(title="Query Plan")
    table.add_column("Platform")
    table.add_column("Sweep")
    table.add_column("Queries", justify="right")
    table.add_column("Done", justify="right")
    table.add_column("In Progress", justify="right")
    table.add_column("Not Started", justify="right")

    by_sweep: dict[str, dict[str, int]] = {}
    for q in all_queries:
        key = f"{q.platform}:{q.sweep_name}"
        if key not in by_sweep:
            by_sweep[key] = {"total": 0, "done": 0, "in_progress": 0, "not_started": 0}
        by_sweep[key]["total"] += 1

        progress = checkpoint.get_progress(q.key)
        if progress.done:
            by_sweep[key]["done"] += 1
        elif progress.pages_fetched > 0:
            by_sweep[key]["in_progress"] += 1
        else:
            by_sweep[key]["not_started"] += 1

    totals = {"total": 0, "done": 0, "in_progress": 0, "not_started": 0}
    for sweep_key in sorted(by_sweep):
        platform, sweep_name = sweep_key.split(":", 1)
        d = by_sweep[sweep_key]
        totals["total"] += d["total"]
        totals["done"] += d["done"]
        totals["in_progress"] += d["in_progress"]
        totals["not_started"] += d["not_started"]
        table.add_row(
            platform,
            sweep_name,
            str(d["total"]),
            str(d["done"]),
            str(d["in_progress"]),
            str(d["not_started"]),
        )

    table.add_row(
        "[bold]Total[/bold]",
        "",
        f"[bold]{totals['total']}[/bold]",
        f"[bold]{totals['done']}[/bold]",
        f"[bold]{totals['in_progress']}[/bold]",
        f"[bold]{totals['not_started']}[/bold]",
    )
    console.print(table)

    max_calls = totals["total"] * max_pages
    console.print(f"\nMax possible calls: {max_calls:,} ({max_calls * 100:,} credits)")


@app.command()
def run(
    budget: int = typer.Option(750, help="Max API calls this session"),
    max_pages: int = typer.Option(10, help="Max pages per query"),
    plan_path: str = typer.Option("configs/sweep_plans.yaml"),
    filters_dir: str = typer.Option("configs/filters"),
    output_dir: str = typer.Option("data/raw"),
    auto: bool = typer.Option(False, help="Skip confirmation prompt"),
) -> None:
    """Start or resume collection. Paginates with cursor checkpointing."""
    api_key = os.environ.get("ADFLEX_API_KEY")
    if not api_key:
        console.print("[red]ADFLEX_API_KEY not set in .env[/red]")
        raise typer.Exit(1)

    planner = _init_planner(plan_path, filters_dir)
    raw_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = CollectionCheckpoint(raw_dir)

    # Load global dedup set
    seen_ids = load_seen_ids(raw_dir)
    console.print(f"Loaded {len(seen_ids)} existing ad IDs for dedup")

    # Generate all queries
    all_queries = []
    for platform in planner.get_platform_names():
        for sweep_name in planner.get_sweep_names(platform):
            queries = planner.generate_queries(platform, sweep_name)
            all_queries.extend(queries)

    # Select queries with remaining pages, distributed by platform and sweep
    selected = select_queries_for_budget(all_queries, budget, planner, checkpoint, max_pages)

    calls_used = checkpoint.total_calls_made()
    console.print(f"\n[bold]Collection run[/bold] — budget: {budget} calls")
    console.print(
        f"  {len(all_queries)} total queries, "
        f"{len(selected)} selected (distributed across platforms + sweeps)"
    )
    console.print(f"  {calls_used} calls already made across all sessions")

    if not selected:
        console.print("[yellow]All queries fully paginated. Nothing to do.[/yellow]")
        return

    est_credits = budget * planner.credits_per_call
    console.print(f"  Estimated credits this session: up to {est_credits:,}")

    if not auto:
        typer.confirm("Proceed?", abort=True)

    async def _run() -> None:
        _rl = ScrapingConfig.from_yaml().rate_limits.adflex
        limiter = RateLimiter(
            requests_per_minute=_rl.requests_per_minute, burst_size=_rl.burst_size
        )
        async with AdFlexClient(api_key=api_key, rate_limiter=limiter) as client:
            executor = SweepExecutor(client, raw_dir, checkpoint, seen_ids)
            calls_remaining = budget
            queries_processed = 0

            for query, page_limit in selected:
                if calls_remaining <= 0:
                    break

                try:
                    calls_used = await executor.execute_query(query, max_pages=page_limit)
                    calls_remaining -= calls_used
                    queries_processed += 1

                    # Save stats periodically
                    if queries_processed % 5 == 0:
                        checkpoint.update_stats(executor.stats)

                    # Progress report every 10 queries
                    if queries_processed % 10 == 0:
                        s = executor.stats
                        console.print(
                            f"  [{queries_processed}/{len(selected)}] "
                            f"{s.total_ads_unique} unique / {s.total_ads_raw} raw "
                            f"({s.total_credits:,} credits, "
                            f"{calls_remaining} calls left)"
                        )

                except Exception as e:
                    log.warning(f"Query failed: {query.key}: {e}")

            # Final save
            checkpoint.update_stats(executor.stats)
            stats_path = save_run_stats(raw_dir, executor.stats)

            s = executor.stats
            console.print("\n[green]Collection session complete.[/green]")
            console.print(f"  Stats saved to {stats_path}")
            console.print(
                f"  {s.total_calls} calls, "
                f"{s.total_ads_unique} unique / {s.total_ads_raw} raw, "
                f"{s.total_credits:,} credits"
            )

    asyncio.run(_run())
    console.print("\nRun [bold]collect assess[/bold] to review results.")


@app.command()
def assess(
    output_dir: str = typer.Option("data/raw"),
) -> None:
    """Print assessment stats."""
    raw_dir = Path(output_dir)
    checkpoint = CollectionCheckpoint(raw_dir)
    stats = checkpoint.get_stats()

    if stats is None:
        console.print("[red]No stats found. Run collection first.[/red]")
        raise typer.Exit(1)

    console.print("\n[bold]Collection Assessment[/bold]\n")

    console.print(f"Total calls: {stats.total_calls}")
    console.print(f"Credits used: {stats.total_credits:,}")
    console.print(f"Estimated remaining: ~{500000 - stats.total_credits:,}")

    if stats.total_ads_raw > 0:
        dedup_rate = 1 - (stats.total_ads_unique / stats.total_ads_raw)
        console.print(f"Dedup rate: {dedup_rate:.1%}")
    console.print()

    # Per-platform table
    table = Table(title="Results by Platform")
    table.add_column("Platform")
    table.add_column("Calls", justify="right")
    table.add_column("Raw", justify="right")
    table.add_column("Unique", justify="right")
    table.add_column("Unique/Call", justify="right")

    for platform in sorted(stats.by_platform):
        p = stats.by_platform[platform]
        upc = p["unique"] / p["calls"] if p["calls"] else 0
        table.add_row(
            platform,
            str(p["calls"]),
            str(p["raw"]),
            str(p["unique"]),
            f"{upc:.1f}",
        )
    console.print(table)
    console.print()

    # Per-sweep table
    table2 = Table(title="Results by Sweep Type")
    table2.add_column("Sweep")
    table2.add_column("Calls", justify="right")
    table2.add_column("Unique", justify="right")
    table2.add_column("Yield/Call", justify="right")

    sorted_sweeps = sorted(
        stats.by_sweep,
        key=lambda k: stats.by_sweep[k]["unique"],
        reverse=True,
    )
    for sweep_key in sorted_sweeps:
        s = stats.by_sweep[sweep_key]
        ypc = s["unique"] / s["calls"] if s["calls"] else 0
        table2.add_row(sweep_key, str(s["calls"]), str(s["unique"]), f"{ypc:.1f}")
    console.print(table2)

    # Pagination progress
    console.print()
    states = checkpoint.query_states
    done_count = sum(1 for p in states.values() if p.done)
    in_progress = sum(1 for p in states.values() if not p.done and p.pages_fetched > 0)
    console.print(
        f"Query progress: {done_count} done, {in_progress} in progress, {len(states)} tracked"
    )


@app.command()
def status(
    output_dir: str = typer.Option("data/raw"),
) -> None:
    """Show overall collection progress."""
    raw_dir = Path(output_dir)

    seen_ids = load_seen_ids(raw_dir)
    checkpoint = CollectionCheckpoint(raw_dir)

    console.print("\n[bold]Collection Status[/bold]\n")
    console.print(f"Total unique ads: {len(seen_ids)}")
    console.print(f"Total API calls: {checkpoint.total_calls_made()}")

    stats = checkpoint.get_stats()
    if stats:
        console.print(f"Credits used: {stats.total_credits:,} / 500,000")
        console.print(f"Budget remaining: ~{500000 - stats.total_credits:,}")
        if stats.total_ads_raw > 0:
            dedup_rate = 1 - (stats.total_ads_unique / stats.total_ads_raw)
            console.print(f"Overall dedup rate: {dedup_rate:.1%}")

    # Query progress summary
    states = checkpoint.query_states
    if states:
        done = sum(1 for p in states.values() if p.done)
        active = sum(1 for p in states.values() if not p.done and p.pages_fetched > 0)
        # Don't count queries not in checkpoint yet
        console.print(f"\nQuery pagination: {done} exhausted, {active} in progress")

    # Data files
    console.print("\nData files:")
    for f in sorted(raw_dir.glob("adflex_*.jsonl")):
        records = read_jsonl(f)
        console.print(f"  {f.name}: {len(records)} records")


# Platform name mapping for detail calls
_PLATFORM_TO_ADFLEX = {
    "facebook": "facebook",
    "tiktok": "tiktok",
    "twitter": "x",
    "reddit": "reddit",
    "pinterest": "pinterest",
}


@app.command()
def enrich(
    tier: str = typer.Option("high", help="Tier to enrich: high, medium, or all"),
    top_n: int | None = typer.Option(None, help="Enrich top N ads by score (overrides --tier)"),
    budget: int = typer.Option(500, help="Max detail API calls"),
    output_dir: str = typer.Option("data/raw"),
    scored_dir: str = typer.Option("data/scored"),
    auto: bool = typer.Option(False, help="Skip confirmation prompt"),
) -> None:
    """Fetch detail endpoint for high-value ads (run after scoring)."""
    api_key = os.environ.get("ADFLEX_API_KEY")
    if not api_key:
        console.print("[red]ADFLEX_API_KEY not set in .env[/red]")
        raise typer.Exit(1)

    raw_dir = Path(output_dir)
    scored_path = Path(scored_dir) / "scored_ads.jsonl"
    if not scored_path.exists():
        console.print(f"[red]No scored ads at {scored_path}. Run 'score score_all' first.[/red]")
        raise typer.Exit(1)

    scored_records = read_jsonl(scored_path)
    console.print(f"Loaded {len(scored_records)} scored ads")

    if top_n is not None:
        scored_records.sort(key=lambda r: r.get("composite_score", 0), reverse=True)
        candidates = scored_records[:top_n]
        console.print(f"Selected top {len(candidates)} by score")
    else:
        candidates = [r for r in scored_records if r.get("tier") == tier]
        console.print(f"Selected {len(candidates)} {tier}-tier ads")

    if not candidates:
        console.print("[yellow]No candidates to enrich.[/yellow]")
        return

    enriched_ids = load_enriched_ids(raw_dir)
    to_enrich = []
    for rec in candidates:
        ad = rec.get("ad", {})
        ad_id = ad.get("ad_id", "")
        if ad_id and ad_id not in enriched_ids:
            to_enrich.append(ad)

    if not to_enrich:
        console.print("[yellow]All candidates already enriched.[/yellow]")
        return

    to_enrich = to_enrich[:budget]
    console.print(
        f"  {len(enriched_ids)} already enriched, {len(to_enrich)} to fetch (budget: {budget})"
    )

    est_credits = len(to_enrich) * 100
    console.print(f"  Estimated credits: ~{est_credits:,}")

    if not auto:
        typer.confirm("Proceed?", abort=True)

    async def _run() -> None:
        _rl = ScrapingConfig.from_yaml().rate_limits.adflex
        limiter = RateLimiter(
            requests_per_minute=_rl.requests_per_minute, burst_size=_rl.burst_size
        )
        async with AdFlexClient(api_key=api_key, rate_limiter=limiter) as client:
            updates: dict[str, dict] = {}
            successes = 0
            failures = 0

            for i, ad_data in enumerate(to_enrich):
                ad_id = ad_data.get("ad_id", "")
                platform_raw = ad_data.get("platform", "")
                adflex_platform = _PLATFORM_TO_ADFLEX.get(platform_raw, platform_raw)

                try:
                    detail_resp = await client.get_ad_detail(int(ad_id), adflex_platform)
                    raw_ad = RawAd(**ad_data)
                    AdFlexClient._merge_detail(raw_ad, detail_resp)

                    patch = raw_ad.model_dump(
                        include={
                            "ad_copy",
                            "impressions",
                            "first_seen",
                            "last_seen",
                            "demographics",
                            "interests",
                            "devices",
                        }
                    )
                    updates[ad_id] = patch
                    enriched_ids.add(ad_id)
                    successes += 1

                except Exception as e:
                    log.warning(f"Detail fetch failed for {adflex_platform}/{ad_id}: {e}")
                    failures += 1

                if (i + 1) % 20 == 0:
                    console.print(
                        f"  [{i + 1}/{len(to_enrich)}] {successes} enriched, {failures} failed"
                    )

            if updates:
                jsonl_path = raw_dir / "adflex_ads.jsonl"
                updated = update_jsonl_records(jsonl_path, updates)
                console.print(f"\n[green]Updated {updated} records in {jsonl_path.name}[/green]")

            save_enriched_ids(raw_dir, enriched_ids)

            stats = {
                "total_candidates": len(candidates),
                "already_enriched": len(enriched_ids) - successes,
                "attempted": len(to_enrich),
                "succeeded": successes,
                "failed": failures,
                "records_updated": len(updates),
            }
            stats_path = save_enrich_stats(raw_dir, stats)
            console.print(f"Stats saved to {stats_path}")

    asyncio.run(_run())
    console.print("\nRe-run [bold]score score_all[/bold] to re-score with enriched data.")


@app.command()
def reset(
    plan_path: str = typer.Option("configs/sweep_plans.yaml"),
    filters_dir: str = typer.Option("configs/filters"),
    output_dir: str = typer.Option("data/raw"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen without writing"),
) -> None:
    """Rebuild collection_state.json from scratch.

    Computes stats from adflex_ads.jsonl (source of truth), resets all query
    cursors to page 0, and archives legacy state files. Dedup via seen_ids
    ensures no duplicate ads on re-fetch.
    """
    raw_dir = Path(output_dir)

    # Compute stats from JSONL
    seen_ids = load_seen_ids(raw_dir)
    console.print(f"Existing ads in JSONL: {len(seen_ids)}")

    # Build stats from actual data
    checkpoint = CollectionCheckpoint(raw_dir)
    stats = checkpoint.rebuild_stats()

    # Show platform breakdown
    table = Table(title="Ads by Platform (from JSONL)")
    table.add_column("Platform")
    table.add_column("Unique", justify="right")
    for platform in sorted(stats.by_platform):
        p = stats.by_platform[platform]
        table.add_row(platform, str(p["unique"]))
    table.add_row("[bold]Total[/bold]", f"[bold]{stats.total_ads_unique}[/bold]")
    console.print(table)

    # Show sweep breakdown
    if stats.by_sweep:
        table2 = Table(title="Ads by Sweep (from JSONL)")
        table2.add_column("Sweep")
        table2.add_column("Unique", justify="right")
        for sweep_key in sorted(
            stats.by_sweep, key=lambda k: stats.by_sweep[k]["unique"], reverse=True
        ):
            table2.add_row(sweep_key, str(stats.by_sweep[sweep_key]["unique"]))
        console.print(table2)

    # Generate all queries from sweep plan
    planner = _init_planner(plan_path, filters_dir)
    all_queries = []
    for platform in planner.get_platform_names():
        for sweep_name in planner.get_sweep_names(platform):
            queries = planner.generate_queries(platform, sweep_name)
            all_queries.extend(queries)

    console.print(f"\nTotal queries from sweep plan: {len(all_queries)}")
    console.print("All query cursors will be reset to page 0.")

    # Identify legacy files to archive
    legacy_patterns = [
        "loop_*.json",
        "loop_*.json.archived",
        "adflex_api_audit.json",
        "adflex_exploration_raw.json",
        "meta_exploratory.json",
        "run_stats.json",
    ]
    legacy_files: list[Path] = []
    for pattern in legacy_patterns:
        legacy_files.extend(raw_dir.glob(pattern))

    if legacy_files:
        console.print(f"\nLegacy files to archive ({len(legacy_files)}):")
        for f in sorted(legacy_files):
            size_kb = f.stat().st_size / 1024
            console.print(f"  {f.name} ({size_kb:.0f} KB)")

    if dry_run:
        console.print("\n[yellow]Dry run — no files written.[/yellow]")
        return

    # Reset all query progress (cursors cleared, stats preserved from rebuild)
    checkpoint.reset()
    n_queries = len(all_queries)
    console.print(f"\n[green]Reset collection_state.json — {n_queries} queries at page 0[/green]")

    # Archive legacy files
    if legacy_files:
        archive_dir = raw_dir / "_archive"
        archive_dir.mkdir(exist_ok=True)
        for f in legacy_files:
            dest = archive_dir / f.name
            f.rename(dest)
            console.print(f"  Archived {f.name}")

    console.print("\nRun [bold]collect plan[/bold] to verify.")


@app.command()
def reparse(
    output_dir: str = typer.Option("data/raw"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Re-parse all ads from raw_data using the current parser.

    Fixes fields that the parser previously missed without re-fetching
    from the API. Preserves vertical, raw_data, and other collection
    metadata. Writes updated records back to the JSONL file.
    """
    import json

    raw_dir = Path(output_dir)
    jsonl_path = raw_dir / "adflex_ads.jsonl"
    if not jsonl_path.exists():
        console.print("[red]No adflex_ads.jsonl found.[/red]")
        raise typer.Exit(1)

    records: list[dict[str, object]] = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    console.print(f"Loaded {len(records):,} ads for reparsing.")

    updated = 0
    for rec in records:
        raw_data = rec.get("raw_data")
        if not isinstance(raw_data, dict) or not raw_data:
            continue

        # Detect original platform from the record
        platform = str(rec.get("platform", "facebook"))
        # Map back to API platform names for the parser
        platform_map = {"twitter": "x"}
        api_platform = platform_map.get(platform, platform)

        reparsed = AdFlexClient._parse_ad(raw_data, api_platform)

        # Preserve collection metadata that doesn't come from the API
        reparsed.vertical = str(rec.get("vertical", ""))

        records[records.index(rec)] = json.loads(reparsed.model_dump_json())
        updated += 1

    console.print(f"Reparsed {updated:,} ads.")

    if dry_run:
        # Show a sample of changes
        console.print("\n[yellow]Dry run — no files written.[/yellow]")
        console.print("Sample changes (first 3):")
        for _i, rec in enumerate(records[:3]):
            console.print(
                f"  {rec.get('ad_id')}: "
                f"last_seen={rec.get('last_seen')}, "
                f"landing_page={rec.get('landing_page_url', '')!r}, "
                f"comments={rec.get('comments')}, "
                f"description={str(rec.get('ad_copy', {}).get('description', ''))[:50]!r}"  # type: ignore[union-attr]
            )
        return

    # Write back
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")

    console.print(f"[green]Wrote {len(records):,} reparsed ads to {jsonl_path}[/green]")


if __name__ == "__main__":
    app()
