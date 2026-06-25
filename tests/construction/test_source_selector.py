"""Tests for the source ad selector — copywriting dedup guarantees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from draper.construction.schemas import (
    ConstructionConfig,
    FormatConfig,
    PromptStyle,
    TaskFormat,
)
from draper.construction.source_selector import SourceSelector
from draper.scoring.schemas import ScoredAd
from draper.scraping.schemas import AdCopy, AdSource, Platform, RawAd


def _make_scored(
    ad_id: str,
    advertiser: str = "BrandA",
    platform: Platform = Platform.FACEBOOK,
    vertical: str = "facebook:saas",
    tier: str = "high",
) -> ScoredAd:
    return ScoredAd(
        ad=RawAd(
            ad_id=ad_id,
            source=AdSource.ADFLEX,
            platform=platform,
            advertiser_name=advertiser,
            vertical=vertical,
            ad_copy=AdCopy(headline="h", body="b"),
        ),
        composite_score=0.9 if tier == "high" else 0.2,
        tier=tier,
    )


def _seed_clusters_dir(
    base: Path,
    ads: list[ScoredAd],
    copywriting_ads: list[str],
) -> tuple[Path, Path]:
    """Create the copywriting manifest ``SourceSelector`` expects."""
    scored_path = base / "scored_ads.jsonl"
    with scored_path.open("w") as f:
        for ad in ads:
            f.write(ad.model_dump_json() + "\n")

    clusters_dir = base / "_clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    with (clusters_dir / "copywriting_ads.jsonl").open("w") as f:
        for aid in copywriting_ads:
            f.write(json.dumps({"ad_id": aid, "score": 0.85}) + "\n")

    return scored_path, clusters_dir


def _config(scored: Path, clusters: Path) -> ConstructionConfig:
    return ConstructionConfig(
        scored_ads_path=str(scored),
        output_dir=str(clusters.parent / "constructed"),
        clusters_dir=str(clusters),
        formats={
            "copywriting": FormatConfig(
                target=10,
                valid_styles=[PromptStyle.BACKTRANSLATION],
                style_ratios={PromptStyle.BACKTRANSLATION.value: 1.0},
            ),
        },
    )


def _flatten(batches: list[list[ScoredAd]]) -> set[str]:
    return {a.ad.ad_id for b in batches for a in b}


class TestCopywritingNoReuse:
    def test_consumed_blocks_emission(self, tmp_path: Path) -> None:
        ad_ids = [f"c{i}" for i in range(5)]
        ads = [_make_scored(aid) for aid in ad_ids]
        scored, cdir = _seed_clusters_dir(tmp_path, ads, copywriting_ads=ad_ids)
        cfg = _config(scored, cdir)
        sel = SourceSelector(cfg)
        batches = sel.select_batches(TaskFormat.COPYWRITING, {"c0", "c2"}, count=5)
        emitted = _flatten(batches)
        assert "c0" not in emitted and "c2" not in emitted

    def test_fresh_selection_fills_requested_count(self, tmp_path: Path) -> None:
        ad_ids = [f"c{i}" for i in range(10)]
        ads = [_make_scored(aid) for aid in ad_ids]
        scored, cdir = _seed_clusters_dir(tmp_path, ads, copywriting_ads=ad_ids)
        cfg = _config(scored, cdir)
        sel = SourceSelector(cfg)
        batches = sel.select_batches(TaskFormat.COPYWRITING, set(), count=5)
        emitted = _flatten(batches)
        assert len(emitted) == 5

    def test_bundles_are_single_ad(self, tmp_path: Path) -> None:
        ad_ids = [f"c{i}" for i in range(3)]
        ads = [_make_scored(aid) for aid in ad_ids]
        scored, cdir = _seed_clusters_dir(tmp_path, ads, copywriting_ads=ad_ids)
        cfg = _config(scored, cdir)
        sel = SourceSelector(cfg)
        batches: Any = sel.select_batches(TaskFormat.COPYWRITING, set(), count=3)
        for batch in batches:
            assert len(batch) == 1
