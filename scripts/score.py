"""Scoring CLI — scores raw ads and writes scored Parquet + distributions."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date as _date
from datetime import timedelta
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from draper.scoring.composite_scorer import CompositeScorer
from draper.scoring.schemas import ScoredAd, ScoringConfig
from draper.scoring.snorkel_scorer import SnorkelScorer
from draper.scoring.tier_assigner import TierAssigner
from draper.scraping.schemas import RawAd
from draper.utils.io import jsonl_to_parquet, read_jsonl, write_jsonl
from draper.utils.logging import setup_logging

# AdFlex search endpoint never returns these fields — only the (expensive)
# detail/enrich endpoint does. Until enrichment runs, persist them as null
# instead of the schema defaults ("", 0, [], False) so downstream consumers
# can't mistake "no data" for "real zero".
_ADFLEX_UNAVAILABLE_AD_FIELDS = (
    "impressions",
    "interests",
    "devices",
    "is_redelivered",
    "spend_lower",
    "spend_upper",
    "impression_lower",
    "impression_upper",
)

app = typer.Typer(help="Score raw ads and assign performance tiers")
console = Console()
log = setup_logging()


def _load_raw_ads(raw_dir: Path) -> list[RawAd]:
    """Load all RawAd objects from JSONL files in the raw directory."""
    ads: list[RawAd] = []
    for path in sorted(raw_dir.glob("adflex_*.jsonl")):
        records = read_jsonl(path)
        for rec in records:
            try:
                ads.append(RawAd(**rec))
            except Exception as e:
                log.warning(f"Failed to parse ad from {path.name}: {e}")
    return ads


@app.command()
def score_all(
    raw_dir: str = typer.Option("data/raw", help="Directory with raw JSONL files"),
    output_dir: str | None = typer.Option(
        None, help="Output directory (default: data/scored/{version})"
    ),
    config_path: str = typer.Option("configs/scoring.yaml", help="Scoring config"),
    scorer_version: str = typer.Option(
        "v1", help="Scorer version: v1 (composite), v2 (snorkel), or v3 (hybrid)"
    ),
) -> None:
    """Score all raw ads and write scored output as Parquet + distributions JSON."""
    raw_path = Path(raw_dir)
    resolved_output = output_dir if output_dir is not None else f"data/scored/{scorer_version}"
    out_path = Path(resolved_output)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load config and ads
    config = ScoringConfig.from_yaml(config_path)
    console.print(
        f"Loaded scoring config: {len(config.signals)} signals, "
        f"total weight = {config.total_weight:.2f}"
    )

    ads = _load_raw_ads(raw_path)
    if not ads:
        console.print("[yellow]No raw ads found in {raw_dir}[/yellow]")
        raise typer.Exit(1)
    console.print(f"Loaded {len(ads)} raw ads")

    # Score
    if scorer_version == "v3":
        from draper.scoring.hybrid_scorer import HybridScorer

        scorer: CompositeScorer | SnorkelScorer | HybridScorer = HybridScorer(config)  # type: ignore[no-redef]
        console.print("[bold green]Using v3 (hybrid) scorer[/bold green]")
    elif scorer_version == "v2":
        scorer = SnorkelScorer(
            config,
            high_pct=config.snorkel.high_pct,
            low_pct=config.snorkel.low_pct,
            early_death_days=config.snorkel.early_death_days,
            long_runner_days=config.snorkel.long_runner_days,
        )
        console.print("[bold green]Using v2 (Snorkel) scorer[/bold green]")
    else:
        scorer = CompositeScorer(config)
        console.print("Using v1 (composite) scorer")

    scored = scorer.score_batch(ads)
    console.print(f"Scored {len(scored)} ads")

    # Assign tiers
    assigner = TierAssigner(config)
    scored = assigner.assign_tiers(scored)
    tier_counts = assigner.tier_summary(scored)
    console.print(f"Tiers: {tier_counts}")

    # Write scored ads as JSONL (then convert to Parquet)
    scored_jsonl = out_path / "scored_ads.jsonl"
    prepared = _prepare_for_output(scored, ads)
    written = write_jsonl(prepared, scored_jsonl)
    backfill_summary = _summarize_backfill(prepared)
    console.print(f"Wrote {written} scored ads to {scored_jsonl}")
    console.print(
        f"Backfilled: advertiser_ad_count populated for "
        f"{backfill_summary['advertiser_ad_count_nonzero']:,} ads, "
        f"first_seen derived for {backfill_summary['first_seen_derived']:,} ads"
    )

    scored_parquet = out_path / "scored_ads.parquet"
    try:
        row_count = jsonl_to_parquet(scored_jsonl, scored_parquet)
        console.print(f"Converted to Parquet: {scored_parquet} ({row_count} rows)")
    except Exception as e:
        console.print(f"[yellow]Parquet conversion failed: {e}[/yellow]")
        console.print("JSONL output is the canonical scored format.")
        if scored_parquet.exists() and scored_parquet.stat().st_size == 0:
            scored_parquet.unlink()

    # Compute and write distributions
    scores = [s.composite_score for s in scored]
    scores.sort()
    n = len(scores)

    distributions = {
        "total_ads": n,
        "tiers": tier_counts,
        "score_stats": {
            "min": scores[0],
            "p10": scores[int(n * 0.10)],
            "p25": scores[int(n * 0.25)],
            "median": scores[int(n * 0.50)],
            "p75": scores[int(n * 0.75)],
            "p90": scores[int(n * 0.90)],
            "max": scores[-1],
            "mean": sum(scores) / n,
        },
        "per_vertical": _per_group_stats(scored, lambda s: s.ad.vertical),
        "per_platform": _per_group_stats(scored, lambda s: s.ad.platform.value),
    }

    dist_path = out_path / "distributions.json"
    with dist_path.open("w") as f:
        json.dump(distributions, f, indent=2)
    console.print(f"Wrote distributions to {dist_path}")

    # Display summary table
    _print_summary(distributions)


def _prepare_for_output(
    scored: list[ScoredAd], all_ads: list[RawAd]
) -> list[dict[str, Any]]:
    """Backfill derivable fields and null-out AdFlex-unavailable ones.

    Mutations applied to each ScoredAd dict before JSONL serialization:
    - ``ad.advertiser_ad_count``: set to true count from groupby over the pool
      (AdFlex returns 0 here regardless).
    - ``ad.first_seen``: derived as ``last_seen - active_days`` when null
      (AdFlex search endpoint never returns first_seen).
    - AdFlex-unavailable fields (cta, impressions, interests, devices,
      is_redelivered, spend/impression bounds): set to None for AdFlex-source
      ads so consumers see "missing" instead of fake zeros/empties.
    """
    advertiser_counts = Counter(
        ad.advertiser_id for ad in all_ads if ad.advertiser_id
    )

    out: list[dict[str, Any]] = []
    for s in scored:
        rec = s.model_dump(mode="json")
        ad = rec.get("ad", {})

        # Backfill: real advertiser_ad_count from the pool
        if ad.get("advertiser_id"):
            ad["advertiser_ad_count"] = int(advertiser_counts.get(ad["advertiser_id"], 0))

        # Backfill: first_seen from last_seen - active_days when missing
        if not ad.get("first_seen") and ad.get("last_seen") and ad.get("active_days"):
            try:
                last = _date.fromisoformat(str(ad["last_seen"]))
                ad["first_seen"] = (
                    last - timedelta(days=int(ad["active_days"]))
                ).isoformat()
            except (ValueError, TypeError):
                pass

        # Null-out AdFlex-unavailable fields so they are honest about being missing
        if ad.get("source") == "adflex":
            for k in _ADFLEX_UNAVAILABLE_AD_FIELDS:
                if k in ad:
                    ad[k] = None
            # Nested: ad_copy.cta is also AdFlex-unavailable
            if isinstance(ad.get("ad_copy"), dict) and "cta" in ad["ad_copy"]:
                ad["ad_copy"]["cta"] = None

        out.append(rec)
    return out


def _summarize_backfill(prepared: list[dict[str, Any]]) -> dict[str, int]:
    """Count rows where each backfilled field landed, for the run log."""
    n_adv = 0
    n_fs = 0
    for rec in prepared:
        ad = rec.get("ad", {})
        if int(ad.get("advertiser_ad_count") or 0) > 0:
            n_adv += 1
        if ad.get("first_seen"):
            n_fs += 1
    return {
        "advertiser_ad_count_nonzero": n_adv,
        "first_seen_derived": n_fs,
    }


def _per_group_stats(scored: list, key_fn) -> dict:
    """Compute score stats grouped by a key function."""
    from collections import defaultdict

    groups: dict[str, list[float]] = defaultdict(list)
    for s in scored:
        k = key_fn(s) or "unknown"
        groups[k].append(s.composite_score)

    stats = {}
    for group, group_scores in sorted(groups.items()):
        group_scores.sort()
        n = len(group_scores)
        tiers = {"high": 0, "medium": 0, "low": 0}
        for s_item in scored:
            k = key_fn(s_item) or "unknown"
            if k == group:
                tiers[s_item.tier] += 1

        stats[group] = {
            "count": n,
            "mean": round(sum(group_scores) / n, 4),
            "median": round(group_scores[n // 2], 4),
            "tiers": tiers,
        }
    return stats


def _print_summary(distributions: dict) -> None:
    """Print a summary table of scoring distributions."""
    table = Table(title="Scoring Summary")
    table.add_column("Group")
    table.add_column("Count", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("High", justify="right")
    table.add_column("Medium", justify="right")
    table.add_column("Low", justify="right")

    for group, stats in distributions.get("per_vertical", {}).items():
        tiers = stats.get("tiers", {})
        table.add_row(
            group,
            str(stats["count"]),
            f"{stats['mean']:.4f}",
            f"{stats['median']:.4f}",
            str(tiers.get("high", 0)),
            str(tiers.get("medium", 0)),
            str(tiers.get("low", 0)),
        )

    console.print(table)


if __name__ == "__main__":
    app()
