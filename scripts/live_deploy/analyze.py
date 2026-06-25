"""Analyze live Meta Ads results for the RQ1 two-cell test.

Ingests a per-ad CSV exported from Meta Ads Manager and computes the
pre-registered contrast (Draper+agent CTR vs GPT-5.5 CTR), plus secondary
endpoints.

Expected CSV columns (Meta Ads Manager export defaults — confirm names on
export):
- ``Ad name`` — used to assign each row to a cell + brief. We expect the
  user to follow a naming convention like ``<cell>__<brief_id>``, e.g.
  ``gpt55__indie_hacker`` or ``draper_agent__d2c_marketer``.
- ``Impressions``
- ``Link clicks`` (or ``Clicks (all)`` as fallback)
- ``Amount spent (USD)``
- Optional: ``CPM (cost per 1,000 impressions)``, ``CTR (link click-through rate)``,
  ``Post engagements``, ``Reach``.

Usage:
    python scripts/live_deploy/analyze.py \\
        --csv data/live_deploy/2026-05-11/meta_export.csv \\
        --out data/live_deploy/2026-05-11/results.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import typer

CELL_GPT = "gpt55"
CELL_DRAPER = "draper_agent"

# Statistical bootstrap. 1000 resamples matches configs/eval.yaml's bootstrap_n.
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class CellAggregate:
    cell: str
    impressions: int
    clicks: int
    spend_usd: float

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions > 0 else 0.0

    @property
    def cpc(self) -> float:
        return self.spend_usd / self.clicks if self.clicks > 0 else math.nan

    @property
    def cpm(self) -> float:
        return (
            (self.spend_usd / self.impressions) * 1000.0
            if self.impressions > 0
            else math.nan
        )


def _split_ad_name(ad_name: str) -> tuple[str | None, str | None]:
    """Parse 'cell__brief_id' naming convention."""
    if "__" not in ad_name:
        return None, None
    cell, brief_id = ad_name.split("__", 1)
    return cell.strip(), brief_id.strip()


def _aggregate_by_cell(df: pl.DataFrame) -> dict[str, CellAggregate]:
    by_cell: dict[str, CellAggregate] = {}
    for cell in df["cell"].unique().to_list():
        if cell is None:
            continue
        sub = df.filter(pl.col("cell") == cell)
        by_cell[cell] = CellAggregate(
            cell=cell,
            impressions=int(sub["impressions"].sum()),
            clicks=int(sub["clicks"].sum()),
            spend_usd=float(sub["spend_usd"].sum()),
        )
    return by_cell


def _two_proportion_z(
    clicks_a: int, impressions_a: int, clicks_b: int, impressions_b: int
) -> tuple[float, float]:
    """Returns (z, two-sided p-value). Standard pooled-proportion test."""
    if impressions_a <= 0 or impressions_b <= 0:
        return float("nan"), float("nan")
    p_a = clicks_a / impressions_a
    p_b = clicks_b / impressions_b
    p_pool = (clicks_a + clicks_b) / (impressions_a + impressions_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / impressions_a + 1 / impressions_b))
    if se == 0:
        return float("nan"), float("nan")
    z = (p_a - p_b) / se
    # Two-sided p via erf
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return z, p_value


def _bootstrap_ctr_ci(
    clicks: int, impressions: int, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Bootstrap a 95% CI on CTR by resampling impressions with replacement.

    Models each impression as a Bernoulli(p_hat) trial where p_hat = ctr.
    """
    if impressions <= 0:
        return float("nan"), float("nan")
    import numpy as np

    rng = np.random.default_rng(seed)
    p_hat = clicks / impressions
    # Binomial samples are the sum of `impressions` independent Bernoullis.
    samples = rng.binomial(n=impressions, p=p_hat, size=n_boot) / impressions
    lo = float(np.quantile(samples, 0.025))
    hi = float(np.quantile(samples, 0.975))
    return lo, hi


def load_meta_csv(csv_path: Path) -> pl.DataFrame:
    """Load and normalize a Meta Ads Manager CSV export."""
    raw = pl.read_csv(csv_path)
    column_map = {}
    for col in raw.columns:
        lower = col.lower().strip()
        if lower == "ad name":
            column_map[col] = "ad_name"
        elif lower == "impressions":
            column_map[col] = "impressions"
        elif lower == "link clicks" or (
            lower == "clicks (all)" and "clicks" not in column_map.values()
        ):
            column_map[col] = "clicks"
        elif lower in ("amount spent (usd)", "amount spent"):
            column_map[col] = "spend_usd"

    df = raw.rename(column_map)
    missing = {"ad_name", "impressions", "clicks", "spend_usd"} - set(df.columns)
    if missing:
        raise SystemExit(
            f"Meta CSV is missing required columns after normalization: {missing}. "
            f"Available columns: {df.columns}"
        )

    parsed = df["ad_name"].map_elements(_split_ad_name, return_dtype=pl.Object)
    cells = parsed.map_elements(lambda t: t[0] if t else None, return_dtype=pl.Utf8)
    briefs = parsed.map_elements(lambda t: t[1] if t else None, return_dtype=pl.Utf8)
    df = df.with_columns([cells.alias("cell"), briefs.alias("brief_id")])
    df = df.with_columns(
        pl.col("impressions").cast(pl.Int64),
        pl.col("clicks").cast(pl.Int64),
        pl.col("spend_usd").cast(pl.Float64),
    )
    return df


def run_analysis(df: pl.DataFrame) -> dict[str, Any]:
    by_cell = _aggregate_by_cell(df)
    if CELL_GPT not in by_cell or CELL_DRAPER not in by_cell:
        raise SystemExit(
            f"Expected both cells {CELL_GPT} and {CELL_DRAPER} in CSV — "
            f"got {sorted(by_cell)}. Check ad naming convention 'cell__brief_id'."
        )

    gpt = by_cell[CELL_GPT]
    draper = by_cell[CELL_DRAPER]

    z, p_value = _two_proportion_z(
        draper.clicks, draper.impressions, gpt.clicks, gpt.impressions
    )

    gpt_lo, gpt_hi = _bootstrap_ctr_ci(gpt.clicks, gpt.impressions)
    draper_lo, draper_hi = _bootstrap_ctr_ci(draper.clicks, draper.impressions)

    # Per-brief breakdown
    per_brief: list[dict[str, Any]] = []
    for brief_id in sorted(b for b in df["brief_id"].unique().to_list() if b):
        sub = df.filter(pl.col("brief_id") == brief_id)
        entry: dict[str, Any] = {"brief_id": brief_id, "cells": {}}
        for cell in (CELL_GPT, CELL_DRAPER):
            cell_sub = sub.filter(pl.col("cell") == cell)
            imp = int(cell_sub["impressions"].sum())
            clk = int(cell_sub["clicks"].sum())
            entry["cells"][cell] = {
                "impressions": imp,
                "clicks": clk,
                "ctr": clk / imp if imp > 0 else None,
            }
        per_brief.append(entry)

    return {
        "primary_contrast": {
            "name": "draper_agent vs gpt55 — CTR (RQ1 live)",
            "test": "two-sided two-proportion z-test on pooled clicks/impressions",
            "draper_agent": {
                "impressions": draper.impressions,
                "clicks": draper.clicks,
                "spend_usd": draper.spend_usd,
                "ctr": draper.ctr,
                "ctr_95ci": [draper_lo, draper_hi],
                "cpc": draper.cpc,
                "cpm": draper.cpm,
            },
            "gpt55": {
                "impressions": gpt.impressions,
                "clicks": gpt.clicks,
                "spend_usd": gpt.spend_usd,
                "ctr": gpt.ctr,
                "ctr_95ci": [gpt_lo, gpt_hi],
                "cpc": gpt.cpc,
                "cpm": gpt.cpm,
            },
            "ctr_diff_pp": (draper.ctr - gpt.ctr) * 100.0,
            "z_stat": z,
            "p_value": p_value,
            "significant_at_005": (p_value < 0.05) if not math.isnan(p_value) else None,
        },
        "per_brief": per_brief,
    }


app = typer.Typer(add_completion=False)


@app.command()
def main(
    csv: Path = typer.Option(..., "--csv", help="Meta Ads Manager per-ad CSV export."),  # noqa: B008
    out: Path = typer.Option(..., "--out", help="Output JSON path."),  # noqa: B008
) -> None:
    df = load_meta_csv(csv)
    results = run_analysis(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    pc = results["primary_contrast"]
    typer.echo("\nRQ1 Live Results")
    typer.echo("================")
    typer.echo(
        f"draper_agent: {pc['draper_agent']['clicks']} clicks / "
        f"{pc['draper_agent']['impressions']} impressions "
        f"= CTR {pc['draper_agent']['ctr']:.4f} "
        f"[{pc['draper_agent']['ctr_95ci'][0]:.4f}, {pc['draper_agent']['ctr_95ci'][1]:.4f}]"
    )
    typer.echo(
        f"gpt55:        {pc['gpt55']['clicks']} clicks / "
        f"{pc['gpt55']['impressions']} impressions "
        f"= CTR {pc['gpt55']['ctr']:.4f} "
        f"[{pc['gpt55']['ctr_95ci'][0]:.4f}, {pc['gpt55']['ctr_95ci'][1]:.4f}]"
    )
    typer.echo(
        f"Delta:        {pc['ctr_diff_pp']:+.3f} pp  "
        f"(z={pc['z_stat']:.3f}, p={pc['p_value']:.4f}, "
        f"sig@0.05={pc['significant_at_005']})"
    )

    typer.echo(f"\nFull results written to {out}")


if __name__ == "__main__":
    app()
