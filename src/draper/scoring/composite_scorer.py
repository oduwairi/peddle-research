"""Composite performance scorer for ads.

Two-pass scoring:
1. Compute raw signal values for each ad.
2. Normalize via percentile ranks — per-platform for engagement signals,
   global for platform-agnostic signals (longevity, early_death).
3. Weighted sum using config weights (with redistribution for missing signals).
"""

from __future__ import annotations

import math
from collections import defaultdict

from draper.scoring.schemas import ScoredAd, ScoringConfig, Transform
from draper.scraping.schemas import RawAd

# Platforms with weak engagement data — engagement signals are dropped
# entirely on these platforms and the remaining weights are renormalized.
# Pinterest was excluded after data review (median 148 engagements,
# 3.5% zero) — its engagement is usable under per-platform normalization.
_WEAK_ENGAGEMENT_PLATFORMS = {"reddit", "other"}

# Signals that rely on engagement metrics — normalized per-platform
_ENGAGEMENT_SIGNALS = {"engagement_volume", "engagement_velocity"}


class CompositeScorer:
    """Scores ads using a weighted combination of normalized signals."""

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config

    def score_batch(self, ads: list[RawAd]) -> list[ScoredAd]:
        """Score a batch of ads. Requires the full batch for percentile normalization.

        Args:
            ads: All ads to score. Must be the complete dataset (normalization
                 is relative to this batch).

        Returns:
            List of ScoredAd with composite scores and tiers unset (use
            TierAssigner to assign tiers after scoring).
        """
        if not ads:
            return []

        # Pass 1: compute raw signal values per ad
        raw_signals: list[dict[str, float | None]] = []
        for ad in ads:
            raw_signals.append(self._compute_raw_signals(ad))

        # Pass 2: normalize signals via percentile ranks.
        # Engagement signals are normalized per-platform (a top Facebook ad
        # should rank equally with a top TikTok ad regardless of raw scale).
        # Platform-agnostic signals (longevity, early_death) are normalized
        # globally since a day is a day on every platform.
        signal_names = list(self.config.signals.keys())
        normalized: list[dict[str, float | None]] = [{} for _ in ads]

        # Build platform index for per-platform normalization
        platform_indices: dict[str, list[int]] = defaultdict(list)
        for i, ad in enumerate(ads):
            plat = ad.platform.value if ad.platform else "other"
            platform_indices[plat].append(i)

        for signal_name in signal_names:
            if signal_name in _ENGAGEMENT_SIGNALS:
                # Per-platform normalization
                for indices in platform_indices.values():
                    plat_values = [raw_signals[i].get(signal_name) for i in indices]
                    plat_norm = _percentile_normalize(plat_values)
                    for idx, norm_val in zip(indices, plat_norm, strict=True):
                        normalized[idx][signal_name] = norm_val
            else:
                # Global normalization
                raw_values = [rs.get(signal_name) for rs in raw_signals]
                norm_values = _percentile_normalize(raw_values)
                for i, val in enumerate(norm_values):
                    normalized[i][signal_name] = val

        # Pass 3: weighted sum with missing-data redistribution
        scored: list[ScoredAd] = []
        for ad, norm_sigs in zip(ads, normalized, strict=True):
            composite, final_signals = self._weighted_sum(ad, norm_sigs)
            scored.append(
                ScoredAd(
                    ad=ad,
                    composite_score=composite,
                    signal_scores=final_signals,
                )
            )

        return scored

    def _compute_raw_signals(self, ad: RawAd) -> dict[str, float | None]:
        """Compute raw (pre-normalization) signal values for a single ad."""
        signals: dict[str, float | None] = {}

        for name, cfg in self.config.signals.items():
            raw = self._extract_raw_value(ad, name)
            if raw is None:
                signals[name] = None
                continue

            match cfg.transform:
                case Transform.LOG:
                    signals[name] = math.log1p(raw)
                case Transform.LINEAR:
                    signals[name] = raw
                case Transform.BINARY:
                    signals[name] = raw  # already 0 or 1

        return signals

    def _extract_raw_value(self, ad: RawAd, signal_name: str) -> float | None:
        """Extract the raw value for a signal from an ad."""
        match signal_name:
            case "longevity":
                days = ad.longevity_days
                return float(days) if days is not None else None

            case "engagement_volume":
                total = ad.weighted_engagement
                return total if total > 0 else None

            case "engagement_velocity":
                # Treat zero engagement as missing data, not as a measured 0.
                # We can't distinguish "ad has 0 likes" from "we don't know
                # this ad's likes" — both look like 0 in our schema.
                if ad.weighted_engagement <= 0:
                    return None
                return ad.weighted_engagement_velocity

            case "early_death":
                threshold = self.config.signals[signal_name].threshold_days or 3
                days = ad.longevity_days
                if days is None:
                    return None
                # 1.0 = survived (good), 0.0 = died early (penalty)
                return 0.0 if days < threshold else 1.0

            case _:
                return None

    def _weighted_sum(
        self, ad: RawAd, normalized_signals: dict[str, float | None]
    ) -> tuple[float, dict[str, float]]:
        """Compute weighted sum from available signals only.

        Missing signals (None) are dropped entirely. The remaining signal
        weights are renormalized so they sum to 1.0. This is more honest
        than fudging missing signals to a "neutral" 0.5, which artificially
        drags scores toward the middle and treats absence as an assertion.

        Engagement signals are also dropped on weak-engagement platforms
        (Pinterest/Reddit/Other) where their semantics differ.

        Returns:
            (composite_score, signal_scores_dict) where signal_scores_dict
            contains only the signals that actually contributed.
        """
        platform = ad.platform.value if ad.platform else "other"
        is_weak_engagement = platform in _WEAK_ENGAGEMENT_PLATFORMS

        # Collect signals that have actual values (drop None + weak-engagement)
        available: dict[str, float] = {}
        for name in self.config.signals:
            val = normalized_signals.get(name)
            if val is None:
                continue
            if is_weak_engagement and name in _ENGAGEMENT_SIGNALS:
                continue
            available[name] = val

        if not available:
            return 0.5, {}  # No data → neutral fallback

        # Renormalize: divide each weight by the sum of available weights
        available_weight_total = sum(self.config.signals[n].weight for n in available)
        if available_weight_total <= 0:
            return 0.5, {}

        composite = 0.0
        signal_scores: dict[str, float] = {}
        for name, val in available.items():
            weight = self.config.signals[name].weight
            renormalized_weight = weight / available_weight_total
            composite += val * renormalized_weight
            signal_scores[name] = round(val, 4)

        composite = max(0.0, min(1.0, composite))
        return round(composite, 6), signal_scores


def _percentile_normalize(values: list[float | None]) -> list[float | None]:
    """Normalize values to [0, 1] using percentile ranks.

    None values remain None. Ties receive the same percentile rank.

    Args:
        values: Raw signal values (may contain None).

    Returns:
        Percentile-normalized values in [0, 1].
    """
    # Collect non-None values with original indices
    indexed = [(i, v) for i, v in enumerate(values) if v is not None]
    if not indexed:
        return [None] * len(values)

    n = len(indexed)
    if n == 1:
        # Single value normalizes to 0.5
        single_result: list[float | None] = [None] * len(values)
        single_result[indexed[0][0]] = 0.5
        return single_result

    # Sort by value and assign percentile ranks
    sorted_items = sorted(indexed, key=lambda x: x[1])

    # Handle ties: group identical values and assign average rank
    result: list[float | None] = [None] * len(values)
    i = 0
    while i < n:
        # Find run of equal values
        j = i
        while j < n and sorted_items[j][1] == sorted_items[i][1]:
            j += 1
        # Average rank for this group (0-indexed)
        avg_rank = (i + j - 1) / 2.0
        percentile = avg_rank / (n - 1)
        for k in range(i, j):
            orig_idx = sorted_items[k][0]
            result[orig_idx] = round(percentile, 6)
        i = j

    return result
