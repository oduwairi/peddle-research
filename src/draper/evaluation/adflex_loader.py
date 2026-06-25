"""Load AdFlex Facebook ads with impressions for proxy validation.

AdFlex's free Meta tier provides exact impression counts via the detail
endpoint. This loader reads enriched RawAd records from the validation
directory (separate from training data), scores them through the
CompositeScorer, and aligns them with ground-truth impression values.

Prerequisite: Run `python scripts/validate.py collect-adflex-meta` to
fetch detail data into data/validation/adflex_meta/adflex_meta_ads.jsonl.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import polars as pl

from draper.scoring.composite_scorer import CompositeScorer
from draper.scoring.schemas import ScoredAd, ScoringConfig
from draper.scoring.tier_assigner import TierAssigner
from draper.scraping.schemas import Platform, RawAd
from draper.utils.io import read_jsonl

logger = logging.getLogger("draper")


class AdFlexImpressionsLoader:
    """Load enriched AdFlex Facebook ads and prepare for validation."""

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or ScoringConfig.from_yaml()

    def load_and_score(
        self,
        path: str | Path,
    ) -> tuple[list[ScoredAd], pl.DataFrame]:
        """Load AdFlex Meta ads from validation JSONL, score, and align.

        Filters to Facebook ads with impressions > 0.

        Returns:
            Tuple of (scored_ads, ground_truth_df) where ground_truth_df
            has columns: ad_id, impressions, log_impressions.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("AdFlex validation file not found: %s", path)
            return [], pl.DataFrame()

        records = read_jsonl(path)
        if not records:
            logger.warning("No records found in %s", path)
            return [], pl.DataFrame()

        valid_ads: list[RawAd] = []
        for rec in records:
            try:
                ad = RawAd(**rec)
            except Exception as e:
                logger.debug("Skipping invalid record: %s", e)
                continue

            # Filter to Facebook ads with impressions
            if ad.platform != Platform.FACEBOOK:
                continue
            if ad.impressions is None or ad.impressions <= 0:
                continue

            valid_ads.append(ad)

        if not valid_ads:
            logger.warning(
                "No Facebook ads with impressions found. "
                "Run 'python scripts/validate.py collect-adflex-meta' first."
            )
            return [], pl.DataFrame()

        logger.info(
            "Loaded %d Facebook ads with impressions (of %d records)",
            len(valid_ads),
            len(records),
        )

        # Score through CompositeScorer
        scorer = CompositeScorer(self.config)
        scored = scorer.score_batch(valid_ads)
        assigner = TierAssigner(self.config)
        scored = assigner.assign_tiers(scored)

        # Build ground truth aligned with scored order
        gt_rows = [
            {
                "ad_id": ad.ad_id,
                "impressions": ad.impressions,
                "log_impressions": math.log1p(ad.impressions or 0),
            }
            for ad in valid_ads
        ]

        return scored, pl.DataFrame(gt_rows)
