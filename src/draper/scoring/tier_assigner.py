"""Tier assignment based on composite score distribution.

Assigns tiers using percentile-based cutoffs from the actual score distribution,
not fixed thresholds. The config thresholds (high: 0.80, medium: 0.30) represent
percentile boundaries: top 20% → high, middle 50% → medium, bottom 30% → low.
"""

from __future__ import annotations

from draper.scoring.schemas import ScoredAd, ScoringConfig


class TierAssigner:
    """Assigns performance tiers based on score distribution percentiles."""

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config

    def assign_tiers(self, scored_ads: list[ScoredAd]) -> list[ScoredAd]:
        """Assign tiers to scored ads based on their score distribution.

        Tiers are assigned by percentile rank within the batch:
        - high: top 20% (scores above the 80th percentile)
        - medium: middle 50% (between 30th and 80th percentile)
        - low: bottom 30% (below the 30th percentile)

        Args:
            scored_ads: Ads with composite_score already computed.

        Returns:
            Same ads with tier field updated.
        """
        if not scored_ads:
            return scored_ads

        # Sort scores to find percentile cutoffs
        scores = sorted(ad.composite_score for ad in scored_ads)
        n = len(scores)

        high_cutoff = scores[int(n * self.config.tiers.high)]
        medium_cutoff = scores[int(n * self.config.tiers.medium)]

        for ad in scored_ads:
            if ad.composite_score >= high_cutoff:
                ad.tier = "high"
            elif ad.composite_score >= medium_cutoff:
                ad.tier = "medium"
            else:
                ad.tier = "low"

        return scored_ads

    def tier_summary(self, scored_ads: list[ScoredAd]) -> dict[str, int]:
        """Return count of ads per tier."""
        counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for ad in scored_ads:
            counts[ad.tier] = counts.get(ad.tier, 0) + 1
        return counts
