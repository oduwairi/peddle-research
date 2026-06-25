"""Proxy score validation CLI — multi-source validation for RQ3.

Validation streams (revised after Meta-EU Apify and AdFlex-detail dead ends):
- Stream A: Upworthy A/B winner prediction (creative-feature complementary)
- Stream B: Google Political Ads BigQuery (PRIMARY — real spend/impression buckets)
- Stream C: IRA dataset (secondary out-of-domain political robustness)
- Stream D: Internal consistency + WordStream benchmark calibration

Workflow:
  # 1. Set up BigQuery auth (one-time)
  gcloud auth application-default login

  # 2. Fetch Google Political Ads (free; counts against your BQ free tier)
  python scripts/validate.py collect-google-political --limit 1000

  # 3. Run validation (Upworthy + IRA already on disk)
  python scripts/validate.py run

Output paths (separate from training data):
  data/validation/upworthy/                         — already on disk
  data/validation/ira/                              — already on disk
  data/validation/google_political/google_political_ads.jsonl  — BQ output
  data/validation/meta_eu/meta_eu_ads.jsonl         — Apify (parked, dead end)
  data/validation/adflex_meta/adflex_meta_ads.jsonl — AdFlex detail (parked)
  data/validation/validation_report.json            — final report

Notes on parked sources:
- Apify Meta EU: spend/impression fields are null for commercial ads
  (EU DSA disclosure only applies to political/issue ads).
- AdFlex detail endpoint: costs 100 credits/call AND the "impressions"
  field appears to be AdFlex's internal scrape count, not real Meta delivery.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from draper.evaluation.adflex_loader import AdFlexImpressionsLoader
from draper.evaluation.google_political_loader import GooglePoliticalLoader
from draper.evaluation.ira_loader import IRALoader
from draper.evaluation.meta_eu_loader import MetaEULoader
from draper.evaluation.proxy_validation import (
    ConsistencyResult,
    PairwiseValidationResult,
    ProxyValidator,
    ValidationResult,
    headline_text_score,
)
from draper.evaluation.upworthy_loader import UpworthyLoader
from draper.scoring.benchmark_calibrator import BenchmarkCalibrator
from draper.scoring.schemas import ScoredAd, ScoringConfig
from draper.scraping.adflex import AdFlexClient
from draper.scraping.meta_library import MetaLibraryClient
from draper.scraping.rate_limiter import RateLimiter
from draper.scraping.schemas import Platform, RawAd
from draper.utils.io import append_jsonl, read_jsonl, write_jsonl
from draper.utils.logging import setup_logging

app = typer.Typer(help="Proxy score validation for RQ3")
console = Console()
log = setup_logging()

# AdFlex platform name mapping (canonical Platform → AdFlex API platform string)
_PLATFORM_TO_ADFLEX = {
    "facebook": "facebook",
    "tiktok": "tiktok",
    "twitter": "x",
    "reddit": "reddit",
    "pinterest": "pinterest",
}


@app.command("collect-adflex-meta")
def collect_adflex_meta(
    scored_path: str = typer.Option(
        "data/scored/scored_ads.jsonl",
        help="Source of Facebook ad IDs to enrich (read-only)",
    ),
    output_path: str = typer.Option(
        "data/validation/adflex_meta/adflex_meta_ads.jsonl",
        help="Output JSONL for enriched Meta ads (separate from training data)",
    ),
    budget: int = typer.Option(
        1,
        help="Max detail API calls (start with 1 to verify zero credit cost)",
    ),
    auto: bool = typer.Option(False, help="Skip confirmation prompt"),
) -> None:
    """Fetch detail (impressions) for AdFlex Facebook ads to validation dir.

    Reads ad_ids from existing scored data, calls AdFlex detail endpoint
    for each (free for Meta), and writes enriched RawAd records to a
    SEPARATE validation file. Training data is never modified.

    Resumable: skips ad_ids already present in the output file.
    """
    load_dotenv()
    api_key = os.environ.get("ADFLEX_API_KEY")
    if not api_key:
        console.print("[red]ADFLEX_API_KEY not set in .env[/red]")
        raise typer.Exit(1)

    src = Path(scored_path)
    if not src.exists():
        console.print(f"[red]Scored ads not found at {src}[/red]")
        raise typer.Exit(1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load Facebook ad_ids from scored data (read-only)
    scored_records = read_jsonl(src)
    fb_ads: list[dict[str, object]] = []
    for rec in scored_records:
        ad = rec.get("ad", {})
        if isinstance(ad, dict) and ad.get("platform") == "facebook":
            fb_ads.append(ad)

    console.print(f"Found {len(fb_ads)} Facebook ads in scored data")

    # Resume support: skip ad_ids already in the output file
    already_enriched: set[str] = set()
    if out.exists():
        existing = read_jsonl(out)
        already_enriched = {str(r.get("ad_id", "")) for r in existing if r.get("ad_id")}
        console.print(f"  {len(already_enriched)} already enriched in {out.name}")

    # Filter out already-enriched and apply budget
    candidates = [
        ad
        for ad in fb_ads
        if str(ad.get("ad_id", "")) and str(ad.get("ad_id")) not in already_enriched
    ]
    to_fetch = candidates[:budget]

    if not to_fetch:
        console.print("[yellow]Nothing to enrich (all done or no candidates).[/yellow]")
        return

    console.print(
        f"\n[bold]Will fetch detail for {len(to_fetch)} Facebook ads[/bold]\n"
        f"  Output: {out}\n"
        f"  Note: AdFlex Meta is advertised as FREE (0 credits per call).\n"
        f"  Verify credit balance at https://app.adflex.io after this run.\n"
    )

    if not auto:
        typer.confirm("Proceed?", abort=True)

    async def _run() -> None:
        limiter = RateLimiter(requests_per_minute=30, burst_size=5)
        successes = 0
        failures = 0
        enriched_batch: list[RawAd] = []

        async with AdFlexClient(api_key=api_key, rate_limiter=limiter) as client:
            for i, ad_data in enumerate(to_fetch):
                ad_id = str(ad_data.get("ad_id", ""))
                try:
                    detail_resp = await client.get_ad_detail(int(ad_id), "facebook")
                    raw_ad = RawAd(**ad_data)
                    AdFlexClient._merge_detail(raw_ad, detail_resp)

                    # Force platform (in case enum got dropped through dict)
                    raw_ad.platform = Platform.FACEBOOK
                    enriched_batch.append(raw_ad)
                    successes += 1

                    console.print(
                        f"  [{i + 1}/{len(to_fetch)}] ad_id={ad_id} "
                        f"impressions={raw_ad.impressions}"
                    )
                except Exception as e:
                    log.warning(f"Detail fetch failed for {ad_id}: {e}")
                    failures += 1
                    console.print(
                        f"  [{i + 1}/{len(to_fetch)}] ad_id={ad_id} [red]FAILED: {e}[/red]"
                    )

                # Flush every 20 to avoid losing progress
                if len(enriched_batch) >= 20:
                    append_jsonl(enriched_batch, out)
                    enriched_batch = []

            # Final flush
            if enriched_batch:
                append_jsonl(enriched_batch, out)

        console.print(f"\n[green]Done: {successes} enriched, {failures} failed[/green]")
        console.print(f"  Output: {out}")
        console.print("  [bold yellow]Verify your AdFlex credit balance now.[/bold yellow]")

    asyncio.run(_run())


@app.command("collect-meta-eu")
def collect_meta_eu(
    output_path: str = typer.Option(
        "data/validation/meta_eu/meta_eu_ads.jsonl",
        help="Output JSONL for scraped EU Meta ads (separate from training data)",
    ),
    countries: str = typer.Option(
        "DE,FR,NL,ES",
        help="Comma-separated EU country codes",
    ),
    verticals: str = typer.Option(
        "ecommerce,technology,finance,health,education,retail,travel,food",
        help="Comma-separated vertical search terms",
    ),
    max_per_query: int = typer.Option(
        10,
        help="Max ads per country+vertical query (start small for pilot)",
    ),
    include_inactive: bool = typer.Option(
        True,
        help="Include inactive ads (for longevity variation)",
    ),
    auto: bool = typer.Option(False, help="Skip confirmation prompt"),
) -> None:
    """Scrape EU ads from Meta Ad Library via Apify (free credits, $ via Apify).

    EU transparency regulations require Meta to disclose spend ranges and
    impression ranges for ads targeting EU countries. This is the primary
    ground-truth source for proxy validation.

    Output is written to a SEPARATE validation directory; training data is
    never modified.
    """
    load_dotenv()
    apify_token = os.environ.get("APIFY_API_TOKEN", "")
    if not apify_token:
        console.print("[red]APIFY_API_TOKEN not set in .env[/red]")
        raise typer.Exit(1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    country_list = [c.strip().upper() for c in countries.split(",") if c.strip()]
    vertical_list = [v.strip() for v in verticals.split(",") if v.strip()]

    n_queries = len(country_list) * len(vertical_list)
    est_total = n_queries * max_per_query

    console.print(
        f"\n[bold]Collecting Meta EU ads via Apify[/bold]\n"
        f"  Countries: {country_list}\n"
        f"  Verticals: {vertical_list}\n"
        f"  Max per query: {max_per_query}\n"
        f"  Total queries: {n_queries}\n"
        f"  Estimated max ads: {est_total}\n"
        f"  Output: {out}\n"
        f"  [dim]Apify billing applies (~$0.005/ad on Apify platform).[/dim]\n"
    )

    if not auto:
        typer.confirm("Proceed?", abort=True)

    client = MetaLibraryClient(apify_token)
    all_ads: list[RawAd] = []

    async def _collect() -> None:
        for country in country_list:
            for vertical in vertical_list:
                console.print(f"  Collecting: {country} / {vertical} (max {max_per_query})...")
                try:
                    ads = await client.search_ads(
                        search_term=vertical,
                        country=country,
                        max_results=max_per_query,
                        active_only=not include_inactive,
                    )
                    for ad in ads:
                        ad.vertical = vertical
                    all_ads.extend(ads)
                    console.print(f"    -> {len(ads)} ads")
                except Exception as e:
                    console.print(f"    [yellow]Failed: {e}[/yellow]")

    asyncio.run(_collect())

    if not all_ads:
        console.print("[red]No ads collected[/red]")
        raise typer.Exit(1)

    # Deduplicate by ad_id
    seen: set[str] = set()
    unique_ads: list[RawAd] = []
    for ad in all_ads:
        if ad.ad_id and ad.ad_id not in seen:
            seen.add(ad.ad_id)
            unique_ads.append(ad)

    written = write_jsonl(unique_ads, out)
    console.print(
        f"\n[green]Collected {written} unique ads (from {len(all_ads)} total) -> {out}[/green]"
    )

    # Report ground-truth coverage
    with_spend = sum(
        1 for ad in unique_ads if ad.spend_lower is not None or ad.spend_upper is not None
    )
    with_impressions = sum(
        1 for ad in unique_ads if ad.impression_lower is not None or ad.impression_upper is not None
    )
    console.print(
        f"  With spend data:       {with_spend}/{written} ({with_spend / written * 100:.0f}%)"
    )
    console.print(
        f"  With impression data:  {with_impressions}/{written} "
        f"({with_impressions / written * 100:.0f}%)"
    )


@app.command("collect-google-political")
def collect_google_political(
    output_path: str = typer.Option(
        "data/validation/google_political/google_political_ads.jsonl",
        help="Output JSONL for Google Political Ads",
    ),
    limit: int = typer.Option(
        1000,
        help="Max rows to fetch from BigQuery (start small for pilot)",
    ),
    min_first_served: str = typer.Option(
        "2022-01-01",
        help="Filter ads first served on or after this date (YYYY-MM-DD)",
    ),
    project: str = typer.Option(
        "",
        help="GCP project ID for billing (defaults to ADC project)",
    ),
    auto: bool = typer.Option(False, help="Skip confirmation prompt"),
) -> None:
    """Fetch Google Political Ads from BigQuery to validation dir.

    Requires gcloud Application Default Credentials. Run once first:
        gcloud auth application-default login

    Free public dataset; query bills against your BQ free tier (1 TB/month).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[bold]Fetching Google Political Ads from BigQuery[/bold]\n"
        f"  Limit: {limit} rows\n"
        f"  Min first served: {min_first_served}\n"
        f"  Output: {out}\n"
        f"  [dim]Bills against your BQ free tier (1 TB/month free).[/dim]\n"
    )

    if not auto:
        typer.confirm("Proceed?", abort=True)

    loader = GooglePoliticalLoader(project=project or None)
    try:
        written = loader.collect(
            output_path=out,
            limit=limit,
            min_first_served=min_first_served,
        )
    except Exception as e:
        console.print(f"[red]BigQuery query failed: {e}[/red]")
        console.print(
            "[dim]Hint: run 'gcloud auth application-default login' "
            "and ensure you have a GCP project enabled for BigQuery.[/dim]"
        )
        raise typer.Exit(1) from e

    console.print(f"\n[green]Wrote {written} ads -> {out}[/green]")


@app.command()
def run(
    upworthy_path: str = typer.Option(
        "data/validation/upworthy/exploratory.csv",
        help="Path to Upworthy CSV (Stream A creative validation)",
    ),
    google_political_path: str = typer.Option(
        "data/validation/google_political/google_political_ads.jsonl",
        help="Path to Google Political Ads JSONL (Stream B primary)",
    ),
    meta_eu_path: str = typer.Option(
        "data/validation/meta_eu/meta_eu_ads.jsonl",
        help="Path to Meta EU ads (parked — null spend on commercial)",
    ),
    adflex_meta_path: str = typer.Option(
        "data/validation/adflex_meta/adflex_meta_ads.jsonl",
        help="Path to enriched AdFlex Meta ads (parked — invalid impressions)",
    ),
    scored_path: str = typer.Option(
        "data/scored/scored_ads.jsonl",
        help="Path to scored AdFlex ads (Stream D internal consistency)",
    ),
    ira_path: str = typer.Option(
        "data/validation/ira/ira_ads.csv",
        help="Path to IRA dataset CSV (Stream C secondary)",
    ),
    output_dir: str = typer.Option(
        "data/validation",
        help="Output directory for validation report",
    ),
    config_path: str = typer.Option(
        "configs/scoring.yaml",
        help="Scoring config path",
    ),
    n_bootstrap: int = typer.Option(1000, help="Bootstrap resamples for CIs"),
    scorer_version: str = typer.Option(
        "v1", help="Scorer version: v1 (composite), v2 (snorkel), or v3 (hybrid)"
    ),
) -> None:
    """Run two-level proxy validation (AdFlex + IRA) and write report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = ScoringConfig.from_yaml(config_path)
    validator = ProxyValidator()
    results: dict[str, object] = {}
    console.print(f"Scorer version: [bold]{scorer_version}[/bold]")

    # --- Stream A: Upworthy creative-feature validation ---
    upworthy_file = Path(upworthy_path)
    if upworthy_file.exists():
        console.print(
            "\n[bold]Stream A: Upworthy creative validation (complementary to RQ3)[/bold]"
        )
        up_loader = UpworthyLoader()
        tests = up_loader.load(upworthy_path)
        pairs = up_loader.to_pairs(tests, only_significant=True)
        console.print(
            f"  {len(tests)} tests, "
            f"{sum(1 for t in tests if t.has_significant_winner)} significant, "
            f"{len(pairs)} (winner, loser) pairs"
        )

        if len(pairs) >= 10:
            # Naive text-feature baseline. The AdFlex composite scorer
            # is NOT tested on Upworthy because it would require mapping
            # clicks→likes (different constructs), which both creates
            # circular validation (clicks are also the winner label) and
            # violates the principle of only mapping fields that measure
            # the same thing.
            result_up_text = ProxyValidator.validate_pairwise_winners(
                pairs=list(pairs),
                score_fn=headline_text_score,
                source="upworthy_text_features",
                limitations=[
                    "Naive text-feature scorer (length, digits, questions, "
                    "clickbait words). NOT the AdFlex composite scorer.",
                    "News headlines, not full ads.",
                    "Upworthy archive is 2013–2015.",
                ],
            )
            results["stream_a_upworthy_text_features"] = result_up_text.summary()
            _print_pairwise_result(result_up_text)
        else:
            console.print("[yellow]Not enough significant pairs[/yellow]")
    else:
        console.print(f"[yellow]Upworthy CSV not found at {upworthy_path}[/yellow]")

    # --- Stream B: Google Political Ads (BigQuery) — PRIMARY ---
    gpa_file = Path(google_political_path)
    if gpa_file.exists():
        console.print("\n[bold]Stream B: Google Political Ads (PRIMARY RQ3)[/bold]")
        gpa_loader = GooglePoliticalLoader(config, scorer_version=scorer_version)
        gpa_scored, gpa_gt = gpa_loader.load_and_score(google_political_path)

        if gpa_scored and len(gpa_gt) > 0:
            # Spend is the sole primary label for Stream B.
            # We don't validate against impressions because impressions are
            # conceptually closer to a scorer input (passive delivery), not
            # an outcome — and we don't validate against engagement signals
            # since Google Political doesn't have engagement data either.
            log_spend = gpa_gt["log_spend_mid"].to_list()
            spend_indices = [i for i, v in enumerate(log_spend) if v is not None]
            if len(spend_indices) >= 10:
                spend_scored = [gpa_scored[i] for i in spend_indices]
                spend_vals = [log_spend[i] for i in spend_indices]
                result_b_spend = validator.validate_stream(
                    scored_ads=spend_scored,
                    ground_truth=spend_vals,
                    source="google_political",
                    target_metric="log_spend_midpoint",
                    limitations=[
                        "Spend values are bucket midpoints (e.g. $100-$200), not exact",
                        "Political ads only (commercial spend not disclosed "
                        "by Google or Meta in transparency reports)",
                        "Validates longevity + early_death sub-score only; "
                        "engagement signals are not testable on this dataset "
                        "(Google does not report likes/comments/shares)",
                    ],
                    n_bootstrap=n_bootstrap,
                )
                results["stream_b_google_political_spend"] = result_b_spend.summary()
                _print_validation_result(result_b_spend)
        else:
            console.print(
                "[yellow]No valid Google Political ads loaded. "
                "Run: python scripts/validate.py collect-google-political[/yellow]"
            )
    else:
        console.print(
            f"[yellow]Google Political file not found at {google_political_path}\n"
            "  Run: python scripts/validate.py collect-google-political[/yellow]"
        )

    # Parked sources — keep imports alive but do not run
    _ = MetaEULoader  # parked: spend null on commercial ads
    _ = AdFlexImpressionsLoader  # parked: detail endpoint not free for Meta
    _ = meta_eu_path  # parked
    _ = adflex_meta_path  # parked

    # --- Stream C: IRA dataset (SECONDARY out-of-domain) ---
    ira_file = Path(ira_path)
    if ira_file.exists():
        console.print("\n[bold]Stream C: IRA dataset (secondary)[/bold]")
        ira_loader = IRALoader(config, scorer_version=scorer_version)
        scored_ira, ira_gt = ira_loader.load_and_score(ira_path)

        if scored_ira and len(ira_gt) > 0:
            # NOTE: IRA impressions are now fed into views as a scorer input,
            # so validating against log_impressions would be circular. Skipped.

            # Against log(cost)
            log_cost = ira_gt["log_cost"].to_list()
            cost_indices = [i for i, v in enumerate(log_cost) if v is not None]
            if len(cost_indices) >= 10:
                cost_scored = [scored_ira[i] for i in cost_indices]
                cost_vals = [log_cost[i] for i in cost_indices]
                result_b_cost = validator.validate_stream(
                    scored_ads=cost_scored,
                    ground_truth=cost_vals,
                    source="ira",
                    target_metric="log_cost_usd",
                    limitations=[
                        "Political ads with extremely cheap CPCs ($0.003–0.005)",
                        "Cost in RUB converted at approximate 2016 rate",
                    ],
                    n_bootstrap=n_bootstrap,
                )
                results["stream_c_ira_cost"] = result_b_cost.summary()
                _print_validation_result(result_b_cost)

            # Also against log(clicks)
            log_clicks = ira_gt["log_clicks"].to_list()
            click_indices = [i for i, v in enumerate(log_clicks) if v > 0]
            if len(click_indices) >= 10:
                click_scored = [scored_ira[i] for i in click_indices]
                click_vals = [log_clicks[i] for i in click_indices]
                result_b_clicks = validator.validate_stream(
                    scored_ads=click_scored,
                    ground_truth=click_vals,
                    source="ira",
                    target_metric="log_clicks",
                    limitations=[
                        "Clicks mechanically correlated with impressions",
                    ],
                    n_bootstrap=n_bootstrap,
                )
                results["stream_c_ira_clicks"] = result_b_clicks.summary()
                _print_validation_result(result_b_clicks)
    else:
        console.print(f"[yellow]IRA data not found at {ira_path}[/yellow]")

    # --- Stream D: Internal consistency + benchmark calibration ---
    scored_file = Path(scored_path)
    if scored_file.exists():
        console.print("\n[bold]Stream D: Internal consistency + benchmarks[/bold]")
        records = read_jsonl(scored_path)
        scored_adflex: list[ScoredAd] = []
        for rec in records:
            try:
                scored_adflex.append(ScoredAd(**rec))
            except Exception:
                continue

        if scored_adflex:
            consistency = validator.validate_internal_consistency(scored_adflex)
            results["stream_d_consistency"] = {
                "tier_separation": {
                    "h_statistic": round(consistency.tier_separation.h_statistic, 4),
                    "p_value": consistency.tier_separation.p_value,
                    "n_per_tier": consistency.tier_separation.n_per_tier,
                    "median_per_tier": consistency.tier_separation.median_per_tier,
                    "effect_sizes": consistency.tier_separation.effect_sizes,
                },
                "platform_homogeneity": {
                    "chi2": round(consistency.platform_homogeneity.chi2, 4),
                    "p_value": consistency.platform_homogeneity.p_value,
                    "dof": consistency.platform_homogeneity.dof,
                    "tier_counts_by_platform": (
                        consistency.platform_homogeneity.tier_counts_by_platform
                    ),
                },
                "signal_contributions": consistency.signal_contributions,
            }
            _print_consistency_result(consistency)

            # Benchmark calibration
            calibrator = BenchmarkCalibrator()
            cal_report = calibrator.calibrate(scored_adflex)
            results["benchmark_calibration"] = cal_report.summary()

            if cal_report.overall_notes:
                for note in cal_report.overall_notes:
                    console.print(f"  [yellow]{note}[/yellow]")

    # Write report
    report_path = out / "validation_report.json"
    with report_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[green]Validation report written to {report_path}[/green]")


@app.command()
def report(
    report_path: str = typer.Option(
        "data/validation/validation_report.json",
        help="Path to validation report JSON",
    ),
) -> None:
    """Print a summary of a previously generated validation report."""
    path = Path(report_path)
    if not path.exists():
        console.print(f"[red]Report not found: {path}[/red]")
        raise typer.Exit(1)

    with path.open() as f:
        data = json.load(f)

    for stream, result in data.items():
        console.print(f"\n[bold]{stream}[/bold]")
        if isinstance(result, dict):
            if "spearman_rho" in result:
                _print_stream_summary(result)
            else:
                console.print(json.dumps(result, indent=2, default=str)[:500])


def _print_validation_result(result: ValidationResult) -> None:
    """Print a ValidationResult to console."""
    table = Table(title=f"{result.source} vs {result.target_metric} (n={result.n_ads})")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    c = result.correlation
    table.add_row("Spearman rho", f"{c.rho:.4f}")
    table.add_row("p-value", f"{c.p_value:.2e}")
    table.add_row("95% CI", f"[{c.ci_lower:.4f}, {c.ci_upper:.4f}]")

    ts = result.tier_separation
    table.add_row("Kruskal-Wallis H", f"{ts.h_statistic:.4f}")
    table.add_row("KW p-value", f"{ts.p_value:.2e}")

    for pair, delta in ts.effect_sizes.items():
        table.add_row(f"Cliff's d ({pair})", f"{delta:.4f}")

    for k, prec in result.ranking.precision_at_k.items():
        table.add_row(f"Precision@{k}%", f"{prec:.4f}")

    for k, ndcg in result.ranking.ndcg_at_k.items():
        table.add_row(f"NDCG@{k}%", f"{ndcg:.4f}")

    console.print(table)

    if result.limitations:
        console.print("  [dim]Limitations:[/dim]")
        for lim in result.limitations:
            console.print(f"    [dim]- {lim}[/dim]")


def _print_pairwise_result(result: PairwiseValidationResult) -> None:
    """Print a PairwiseValidationResult to console."""
    table = Table(title=f"{result.source} pairwise winner prediction")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("N pairs", str(result.n_pairs))
    table.add_row("N correct", str(result.n_correct))
    table.add_row("N ties", str(result.n_ties))
    table.add_row("Accuracy", f"{result.accuracy:.4f}")
    table.add_row(
        "95% CI",
        f"[{result.accuracy_ci[0]:.4f}, {result.accuracy_ci[1]:.4f}]",
    )
    table.add_row("Binomial p (vs 0.5)", f"{result.binomial_p_value:.2e}")
    console.print(table)
    if result.limitations:
        console.print("  [dim]Limitations:[/dim]")
        for lim in result.limitations:
            console.print(f"    [dim]- {lim}[/dim]")


def _print_consistency_result(result: ConsistencyResult) -> None:
    """Print ConsistencyResult to console."""
    ts = result.tier_separation
    console.print(f"  Tier separation (engagement): H={ts.h_statistic:.2f}, p={ts.p_value:.2e}")
    for pair, delta in ts.effect_sizes.items():
        console.print(f"    Cliff's delta ({pair}): {delta:.4f}")

    ph = result.platform_homogeneity
    console.print(f"  Platform homogeneity: chi2={ph.chi2:.2f}, p={ph.p_value:.2e}, dof={ph.dof}")

    console.print("  Signal contributions by tier:")
    for tier in ["high", "medium", "low"]:
        if tier in result.signal_contributions:
            signals = result.signal_contributions[tier]
            sig_str = ", ".join(f"{k}={v:.3f}" for k, v in signals.items())
            console.print(f"    {tier}: {sig_str}")


def _print_stream_summary(data: dict[str, object]) -> None:
    """Print a stream summary from the report JSON."""
    console.print(f"  rho={data.get('spearman_rho')}, p={data.get('spearman_p')}")
    console.print(f"  CI={data.get('spearman_ci')}")
    console.print(f"  KW H={data.get('kruskal_h')}, p={data.get('kruskal_p')}")
    prec = data.get("precision_at_k", {})
    if isinstance(prec, dict):
        for k, v in prec.items():
            console.print(f"  P@{k}%={v}")


if __name__ == "__main__":
    app()
