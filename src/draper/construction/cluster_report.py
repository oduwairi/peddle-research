"""Capacity study for copywriting training-data construction.

Read-only pre-flight check: given the copywriting score threshold and raw-
generation target from ``ConstructionConfig``, report the fingerprint-unique
capacity we can actually hit. Writes nothing to disk — the goal is to inform
threshold/target choices before paying teacher-LLM credits.

Run via::

    python scripts/construct.py cluster-report --config configs/construction.yaml

All thresholds live in ``configs/construction.yaml`` under
``construction.clustering`` (``FormatClusteringConfig``) and
``construction.formats.copywriting.score_min``. This module is a pure consumer.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from draper.construction.clusterer import _vertical_key
from draper.construction.schemas import ConstructionConfig, TaskFormat
from draper.scoring.schemas import ScoredAd
from draper.utils.io import read_jsonl

logger = logging.getLogger("draper")


@dataclass
class FormatCapacity:
    """Simulated strict-pass capacity for one format."""

    format_name: str
    unique_ad_pool: int
    bundles_available: int
    raw_target: int
    platforms: Counter[str] = field(default_factory=Counter)
    verticals: Counter[str] = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)

    @property
    def pct_of_target(self) -> float:
        if self.raw_target == 0:
            return 0.0
        return 100.0 * self.bundles_available / self.raw_target


@dataclass
class CapacityReport:
    """All-formats capacity summary."""

    formats: list[FormatCapacity]
    cross_format_overlap: dict[tuple[str, str], int]
    total_ads: int


def _has_copy(ad: ScoredAd, min_chars: int) -> bool:
    """Return True if total ad copy (headline+body+description+cta) ≥ min_chars."""
    copy = ad.ad.ad_copy
    total = (
        len((copy.headline or "").strip())
        + len((copy.body or "").strip())
        + len((copy.description or "").strip())
        + len((copy.cta or "").strip())
    )
    return total >= min_chars


def load_ads(scored_ads_path: str | Path) -> list[ScoredAd]:
    """Load scored ads from JSONL for simulation."""
    records = read_jsonl(scored_ads_path)
    return [ScoredAd(**r) for r in records]


def _score_min(cfg: ConstructionConfig, task_format: TaskFormat) -> float:
    """Format-level score floor from the config."""
    return cfg.format_config(task_format).score_min


def _copywriting_capacity(
    ads: list[ScoredAd],
    cfg: ConstructionConfig,
    raw_target: int,
) -> tuple[FormatCapacity, set[str]]:
    """Single high-score ads with body ≥ min chars → 1-ad bundles."""
    score_min = _score_min(cfg, TaskFormat.COPYWRITING)
    copy_min = cfg.clustering.format.copywriting_min_copy_chars
    eligible = [
        ad
        for ad in ads
        if ad.composite_score >= score_min and _has_copy(ad, copy_min)
    ]
    used_ids = {ad.ad.ad_id for ad in eligible}
    platforms: Counter[str] = Counter(ad.ad.platform.value for ad in eligible)
    verticals: Counter[str] = Counter(
        _vertical_key(ad) for ad in eligible if _vertical_key(ad)
    )

    return (
        FormatCapacity(
            format_name="copywriting",
            unique_ad_pool=len(eligible),
            bundles_available=len(eligible),
            raw_target=raw_target,
            platforms=platforms,
            verticals=verticals,
            notes=[
                f"score_min={score_min}",
                f"copy_min_chars={copy_min}",
            ],
        ),
        used_ids,
    )


def build_report(
    ads: list[ScoredAd],
    cfg: ConstructionConfig,
    raw_targets: dict[str, int] | None = None,
) -> CapacityReport:
    """Simulate copywriting strict-pass capacity and return the report.

    ``raw_targets`` maps format name → raw-generation target. If omitted,
    copywriting defaults to ``cfg.raw_target_for(TaskFormat.COPYWRITING)``.
    """
    defaults = {fmt.value: cfg.raw_target_for(fmt) for fmt in TaskFormat}
    targets = {**defaults, **(raw_targets or {})}

    cw, cw_ids = _copywriting_capacity(ads, cfg, targets["copywriting"])

    return CapacityReport(
        formats=[cw],
        cross_format_overlap={},
        total_ads=len(ads),
    )
