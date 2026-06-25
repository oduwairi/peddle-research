"""Load Meta Ad Library EU ads (via Apify) for proxy validation.

EU transparency regulations require Meta to disclose spend ranges and
impression ranges for ads targeting EU countries. This loader reads
scraped Meta EU ads from the validation directory, computes range
midpoints, and scores them through the CompositeScorer for comparison
against real spend/reach data.

Prerequisite: Run `python scripts/validate.py collect-meta-eu` to fetch
EU ads via Apify into data/validation/meta_eu/meta_eu_ads.jsonl.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from draper.scoring.composite_scorer import CompositeScorer
from draper.scoring.schemas import ScoredAd, ScoringConfig
from draper.scoring.tier_assigner import TierAssigner
from draper.scraping.schemas import RawAd
from draper.utils.io import read_jsonl

logger = logging.getLogger("draper")


class MetaEULoader:
    """Load Apify-scraped Meta Ad Library EU ads and prepare for validation."""

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or ScoringConfig.from_yaml()

    def load_and_score(
        self,
        path: str | Path,
    ) -> tuple[list[ScoredAd], pl.DataFrame]:
        """Load Meta EU ads, score, and align with spend/impression ground truth.

        Filters to ads with valid spend or impression range data and dates.

        Returns:
            Tuple of (scored_ads, ground_truth_df) where ground_truth_df
            has columns: ad_id, spend_lower, spend_upper, spend_mid,
            impression_lower, impression_upper, impression_mid.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Meta EU validation file not found: %s", path)
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

            # Require either spend or impression range, plus dates
            has_spend = ad.spend_lower is not None or ad.spend_upper is not None
            has_imp = ad.impression_lower is not None or ad.impression_upper is not None
            has_dates = ad.first_seen is not None and ad.last_seen is not None

            if (has_spend or has_imp) and has_dates:
                valid_ads.append(ad)

        if not valid_ads:
            logger.warning(
                "No valid Meta EU ads with spend/impression + date data. "
                "Run 'python scripts/validate.py collect-meta-eu' first."
            )
            return [], pl.DataFrame()

        logger.info(
            "Loaded %d Meta EU ads with ground truth (of %d total records)",
            len(valid_ads),
            len(records),
        )

        # Score through CompositeScorer
        scorer = CompositeScorer(self.config)
        scored = scorer.score_batch(valid_ads)
        assigner = TierAssigner(self.config)
        scored = assigner.assign_tiers(scored)

        # Build ground truth aligned with scored order
        gt_rows = [self._extract_ground_truth(ad) for ad in valid_ads]
        return scored, pl.DataFrame(gt_rows)

    @staticmethod
    def _extract_ground_truth(ad: RawAd) -> dict[str, Any]:
        """Extract ground truth row with midpoints from spend/impression ranges."""
        spend_lower = float(ad.spend_lower) if ad.spend_lower is not None else None
        spend_upper = float(ad.spend_upper) if ad.spend_upper is not None else None
        imp_lower = float(ad.impression_lower) if ad.impression_lower is not None else None
        imp_upper = float(ad.impression_upper) if ad.impression_upper is not None else None

        spend_mid = _midpoint(spend_lower, spend_upper)
        imp_mid = _midpoint(imp_lower, imp_upper)

        return {
            "ad_id": ad.ad_id,
            "platform": ad.platform.value if ad.platform else "other",
            "country": ad.country[0] if ad.country else None,
            "spend_lower": spend_lower,
            "spend_upper": spend_upper,
            "spend_mid": spend_mid,
            "impression_lower": imp_lower,
            "impression_upper": imp_upper,
            "impression_mid": imp_mid,
        }


def _midpoint(lower: float | None, upper: float | None) -> float | None:
    """Compute midpoint of a range, falling back to either bound if one is missing."""
    if lower is not None and upper is not None:
        return (lower + upper) / 2.0
    if lower is not None:
        return lower
    if upper is not None:
        return upper
    return None
