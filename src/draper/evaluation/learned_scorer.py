"""Absolute-scorer arm — Phase 2 wiring of the trained scoring predictor.

Consumes the same per-config Inference JSONs the LLM-judge arms use, plus the
LLM-extracted clean copy from ``data/eval/inferences_clean/``, and produces a
per-config Parquet of calibrated 4-head scores. Pairwise judges produce
*relative* signal (config A vs config B win-rate); this arm produces
*absolute* signal (mean composite over the test set), and the two are
designed to disagree usefully.

Public surface:
- :func:`score_configs` — runs the predictor across one or more configs and
  writes one Parquet per config.
- :func:`load_scores` — reads a previously-written per-config Parquet.
- :func:`summarize` — per-config (and optionally per-segment) aggregates.

See ``docs/project/`` for the broader design and the plan in
``~/.claude/plans/recursive-launching-hartmanis.md``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from draper.scoring_predictor import ScoringPredictor
from draper.scoring_predictor.data import HEAD_NAMES, WEAK_ENGAGEMENT_PLATFORMS

from ._inference_io import config_example_ids as _config_example_ids
from ._inference_io import resolve_text as _resolve_text
from .schemas import Brief

logger = logging.getLogger(__name__)


# Heads whose targets are masked at training time for weak-engagement
# platforms (Reddit, "other"). Predictions for those rows are unreliable —
# we surface them as null in the per-row output rather than letting them
# pollute downstream means.
_MASKED_HEADS: tuple[str, ...] = ("engagement_volume", "engagement_velocity")

# Per-row output column order — also used by :func:`summarize` to enumerate
# heads that get aggregated.
SCORE_COLUMNS: tuple[str, ...] = HEAD_NAMES


def _row_for_score(
    *,
    example_id: str,
    config: str,
    brief: Brief,
    text: str,
    used_clean: bool,
    scores: dict[str, float],
) -> dict[str, object]:
    """Build one output row, applying the Reddit/other engagement mask."""
    platform_norm = (brief.platform or "").strip().lower()
    is_weak = platform_norm in WEAK_ENGAGEMENT_PLATFORMS

    row: dict[str, object] = {
        "example_id": example_id,
        "config": config,
        "platform": brief.platform,
        "vertical": brief.vertical,
        "source_tier_first": brief.source_tiers[0] if brief.source_tiers else None,
        "text_len": len(text),
        "used_clean": used_clean,
        "created_at": datetime.now(UTC).isoformat(),
    }
    for head in HEAD_NAMES:
        if is_weak and head in _MASKED_HEADS:
            row[head] = None
        else:
            row[head] = float(scores[head])
    return row


def score_configs(
    *,
    predictor: ScoringPredictor,
    briefs_by_id: dict[str, Brief],
    configs: Sequence[str],
    inferences_clean_dir: Path,
    inferences_raw_dir: Path,
    out_dir: Path,
    batch_size: int = 64,
) -> dict[str, Path]:
    """Score one or more configs, writing one Parquet per config.

    Returns ``{config_name: parquet_path}``. Writes ``{out_dir}/{config}.parquet``;
    parents are created on demand. Skips example_ids missing from
    ``briefs_by_id`` — without platform/vertical we can't run the predictor
    consistently with how it was trained.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for config in configs:
        ids = _config_example_ids(
            config=config,
            inferences_clean_dir=inferences_clean_dir,
            inferences_raw_dir=inferences_raw_dir,
        )
        if not ids:
            logger.warning("No inferences found for config %r — skipping.", config)
            continue

        items: list[dict[str, str | None]] = []
        sidecars: list[tuple[str, str, bool]] = []  # (example_id, text, used_clean)
        skipped_no_brief = 0
        skipped_empty_text = 0

        for ex_id in ids:
            brief = briefs_by_id.get(ex_id)
            if brief is None:
                skipped_no_brief += 1
                continue
            text, used_clean = _resolve_text(
                config=config,
                example_id=ex_id,
                inferences_clean_dir=inferences_clean_dir,
                inferences_raw_dir=inferences_raw_dir,
            )
            if not text.strip():
                skipped_empty_text += 1
                continue
            # Feed the entire blob as ``body`` — the clean inferences and raw
            # outputs are monolithic strings, not split into headline / body /
            # description. ``build_text`` skips empty fields, so the resulting
            # input is ``[platform=X] [vertical=Y]\n<blob>``.
            items.append(
                {
                    "platform": brief.platform,
                    "vertical": brief.vertical,
                    "headline": None,
                    "body": text,
                    "description": None,
                }
            )
            sidecars.append((ex_id, text, used_clean))

        if skipped_no_brief:
            logger.warning(
                "config=%s: skipped %d example(s) with no matching Brief.",
                config,
                skipped_no_brief,
            )
        if skipped_empty_text:
            logger.warning(
                "config=%s: skipped %d example(s) with empty / extraction-failed text.",
                config,
                skipped_empty_text,
            )

        if not items:
            logger.warning("config=%s: nothing to score after filtering.", config)
            continue

        score_dicts = predictor.score_many(items, batch_size=batch_size)
        rows: list[dict[str, object]] = []
        for (ex_id, text, used_clean), scores in zip(sidecars, score_dicts, strict=True):
            brief = briefs_by_id[ex_id]
            rows.append(
                _row_for_score(
                    example_id=ex_id,
                    config=config,
                    brief=brief,
                    text=text,
                    used_clean=used_clean,
                    scores=scores,
                )
            )

        df = pl.DataFrame(rows)
        out_path = out_dir / f"{config}.parquet"
        df.write_parquet(out_path)
        written[config] = out_path
        composite_mean = df["composite"].mean()
        composite_mean_f = float(composite_mean) if isinstance(composite_mean, int | float) else 0.0
        logger.info(
            "config=%s: wrote %d rows to %s (composite mean=%.3f)",
            config,
            df.height,
            out_path,
            composite_mean_f,
        )

    return written


def load_scores(out_dir: Path, config: str) -> pl.DataFrame:
    """Read the per-config Parquet written by :func:`score_configs`."""
    path = out_dir / f"{config}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No learned-score parquet at {path}")
    return pl.read_parquet(path)


def _summary_aggs() -> list[pl.Expr]:
    """Aggregation expressions for per-head distribution summaries."""
    aggs: list[pl.Expr] = [pl.len().alias("n")]
    for head in SCORE_COLUMNS:
        col = pl.col(head)
        # ``mean`` / ``median`` / ``quantile`` skip nulls by default in Polars,
        # which is what we want — Reddit/other rows have null engagement_*.
        aggs.extend(
            [
                col.count().alias(f"{head}_n"),
                col.mean().alias(f"{head}_mean"),
                col.median().alias(f"{head}_median"),
                col.quantile(0.25).alias(f"{head}_p25"),
                col.quantile(0.75).alias(f"{head}_p75"),
                col.quantile(0.90).alias(f"{head}_p90"),
            ]
        )
    return aggs


def summarize(
    *,
    out_dir: Path,
    configs: Sequence[str],
    by: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Per-config (and optionally per-segment) summary across heads.

    Reads ``{out_dir}/{config}.parquet`` for each named config, concatenates,
    then groups by ``("config", *by)`` and computes count + mean/median/p25/p75/p90
    of each head. Polars' aggregations skip nulls, so engagement-head means
    automatically exclude weak-platform rows.
    """
    frames: list[pl.DataFrame] = []
    for config in configs:
        path = out_dir / f"{config}.parquet"
        if not path.exists():
            logger.warning("No learned-score parquet for config=%s at %s", config, path)
            continue
        frames.append(pl.read_parquet(path))
    if not frames:
        raise FileNotFoundError(
            f"No learned-score parquets found for configs {list(configs)} in {out_dir}"
        )
    df = pl.concat(frames, how="vertical_relaxed")

    group_cols = ["config", *list(by or [])]
    return df.group_by(group_cols).agg(_summary_aggs()).sort(group_cols)
