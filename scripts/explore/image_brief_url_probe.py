"""HEAD-probe AdFlex creative URLs to measure hotlink decay.

Phase 0 of the image-brief skill plan. Samples ~100 ads stratified by
(creative_format x ad age bucket) and HEADs each creative_url to check
liveness. Writes a JSON report and prints a status-code histogram.

Decision threshold: if >5% of URLs return 4xx/5xx, the caption corpus
build (Phase 2) needs a creative-snapshot step before captioning.

Run with: uv run python scripts/explore/image_brief_url_probe.py
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import typer

app = typer.Typer(no_args_is_help=False, add_completion=False)

SCORED_ADS = Path("data/scored/v3/scored_ads.parquet")
REPORT_OUT = Path("data/explore/image_brief_url_probe.json")
TODAY = date(2026, 5, 26)
SAMPLE_PER_CELL = 8  # 3 ages x 4 formats x 8 = 96 probes
HEAD_TIMEOUT = 10.0
HEAD_CONCURRENCY = 16


def _age_bucket(first_seen: str) -> str:
    """Bucket an ad by months since first_seen relative to TODAY."""
    try:
        seen = date.fromisoformat(first_seen)
    except ValueError:
        return "unknown"
    if seen.year < 2000:  # 1970-01-01 sentinels
        return "unknown"
    delta_days = (TODAY - seen).days
    if delta_days < 0:
        return "unknown"
    if delta_days <= 180:
        return "recent_le_6mo"
    if delta_days <= 365:
        return "mid_6_to_12mo"
    return "old_gt_12mo"


def _sample(df: pl.DataFrame, seed: int) -> list[dict[str, Any]]:
    """Pick SAMPLE_PER_CELL ads per (format, age_bucket) cell."""
    random.seed(seed)
    eligible = df.filter(
        (pl.col("creative_url").is_not_null())
        & (pl.col("creative_url") != "")
    ).with_columns(
        pl.col("first_seen").map_elements(_age_bucket, return_dtype=pl.String).alias("age_bucket")
    )

    rows: list[dict[str, Any]] = []
    formats = ["image", "carousel", "video", "other"]
    ages = ["recent_le_6mo", "mid_6_to_12mo", "old_gt_12mo"]

    for fmt in formats:
        for age in ages:
            cell = eligible.filter(
                (pl.col("creative_format") == fmt) & (pl.col("age_bucket") == age)
            )
            if cell.is_empty():
                continue
            n = min(SAMPLE_PER_CELL, cell.height)
            picks = cell.sample(n=n, seed=seed)
            for row in picks.iter_rows(named=True):
                rows.append(
                    {
                        "ad_id": row["ad_id"],
                        "creative_url": row["creative_url"],
                        "creative_format": row["creative_format"],
                        "age_bucket": row["age_bucket"],
                        "first_seen": row["first_seen"],
                    }
                )
    return rows


async def _head_one(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, row: dict[str, Any]
) -> dict[str, Any]:
    async with sem:
        try:
            r = await client.head(row["creative_url"], follow_redirects=True, timeout=HEAD_TIMEOUT)
            return {
                **row,
                "status": r.status_code,
                "content_type": r.headers.get("content-type", ""),
                "content_length": int(r.headers.get("content-length", "0") or 0),
                "error": None,
            }
        except httpx.HTTPError as exc:
            return {
                **row,
                "status": None,
                "content_type": "",
                "content_length": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }


async def _probe_all(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(HEAD_CONCURRENCY)
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*(_head_one(client, sem, r) for r in rows))


def _classify(status: int | None) -> str:
    if status is None:
        return "network_error"
    if 200 <= status < 300:
        return "ok"
    if 300 <= status < 400:
        return "redirect"
    if 400 <= status < 500:
        return "4xx"
    if 500 <= status < 600:
        return "5xx"
    return "other"


@app.command()
def main(
    seed: int = typer.Option(42, help="RNG seed for sampling"),
    output: Path = typer.Option(REPORT_OUT, help="Where to write the JSON report"),
) -> None:
    typer.echo(f"Loading {SCORED_ADS}...")
    df = pl.read_parquet(SCORED_ADS)
    typer.echo(f"Loaded {df.height} rows")

    rows = _sample(df, seed=seed)
    typer.echo(f"Sampled {len(rows)} URLs across {len({(r['creative_format'], r['age_bucket']) for r in rows})} cells")

    typer.echo(f"HEAD-probing (concurrency={HEAD_CONCURRENCY}, timeout={HEAD_TIMEOUT}s)...")
    results = asyncio.run(_probe_all(rows))

    overall = Counter(_classify(r["status"]) for r in results)
    total = len(results)
    failed = sum(overall[c] for c in ("4xx", "5xx", "network_error"))
    fail_rate = failed / total if total else 0.0

    # Breakdown by format
    by_format: dict[str, Counter[str]] = {}
    for r in results:
        by_format.setdefault(r["creative_format"], Counter())[_classify(r["status"])] += 1

    # Breakdown by age
    by_age: dict[str, Counter[str]] = {}
    for r in results:
        by_age.setdefault(r["age_bucket"], Counter())[_classify(r["status"])] += 1

    report = {
        "probed_at": TODAY.isoformat(),
        "seed": seed,
        "total_probed": total,
        "overall": dict(overall),
        "failure_rate": round(fail_rate, 4),
        "decision": "snapshot_required" if fail_rate > 0.05 else "hotlinks_ok",
        "by_format": {k: dict(v) for k, v in by_format.items()},
        "by_age": {k: dict(v) for k, v in by_age.items()},
        "rows": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))

    typer.echo("")
    typer.echo("=== Overall ===")
    for k, v in sorted(overall.items()):
        typer.echo(f"  {k:14s} {v:3d} ({v / total:.1%})")
    typer.echo(f"  failure_rate: {fail_rate:.2%}")
    typer.echo(f"  decision:     {report['decision']}")
    typer.echo("")
    typer.echo("=== By format ===")
    for fmt, counts in sorted(by_format.items()):
        tot = sum(counts.values())
        ok = counts.get("ok", 0) + counts.get("redirect", 0)
        typer.echo(f"  {fmt:10s} n={tot:3d}  ok={ok:3d} ({ok / tot:.0%})  {dict(counts)}")
    typer.echo("")
    typer.echo("=== By age ===")
    for age, counts in sorted(by_age.items()):
        tot = sum(counts.values())
        ok = counts.get("ok", 0) + counts.get("redirect", 0)
        typer.echo(f"  {age:14s} n={tot:3d}  ok={ok:3d} ({ok / tot:.0%})  {dict(counts)}")
    typer.echo("")
    typer.echo(f"Report: {output}")


if __name__ == "__main__":
    app()
