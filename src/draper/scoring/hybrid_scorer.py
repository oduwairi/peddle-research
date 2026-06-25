"""Hybrid scorer v3: v1's continuous weighted sum + v2's KM survival curves.

Replaces v1's ``longevity`` + ``early_death`` pair (2 signals, global
percentile) with a single ``survivability`` signal from per-platform
Kaplan-Meier curves. Keeps v1's continuous ``engagement_volume`` and
``engagement_velocity`` signals with per-platform percentile normalization.

Result: continuous score output (no discretization) with platform-aware
longevity modeling.
"""

from __future__ import annotations

import math
from collections import defaultdict

from draper.scoring.schemas import ScoredAd, ScoringConfig
from draper.scoring.survival import compute_survivability
from draper.scraping.schemas import RawAd

# Platforms with truly unreliable engagement (median ~5, ~37% zero-engagement
# on Reddit). Pinterest was previously included but has usable engagement
# distribution under per-platform percentile normalization, so it now uses
# the standard scoring path.
_WEAK_ENGAGEMENT_PLATFORMS = {"reddit", "other"}
_ENGAGEMENT_SIGNALS = {"engagement_volume", "engagement_velocity"}


class HybridScorer:
    """Scores ads using continuous weighted sum with KM survival curves."""

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config
        self._weights = config.hybrid_v3.as_dict()

    def score_batch(self, ads: list[RawAd]) -> list[ScoredAd]:
        """Score a batch of ads.

        Args:
            ads: All ads to score (full batch for percentile normalization).

        Returns:
            List of ScoredAd with continuous composite scores.
        """
        if not ads:
            return []

        # Compute KM survivability per ad
        survivabilities = compute_survivability(ads)

        # Compute raw engagement signals.
        #
        # Zero engagement is REAL data (the ad got 0 likes), not missing
        # data. We surface it as 0.0 so it pins to the bottom of the
        # per-platform percentile distribution, instead of None which
        # excludes the ad from engagement entirely and lets survivability
        # carry the whole composite — the zombie-ad pathology.
        raw_signals: list[dict[str, float | None]] = []
        for i, ad in enumerate(ads):
            signals: dict[str, float | None] = {}
            signals["survivability"] = survivabilities[i]

            eng = ad.weighted_engagement
            signals["engagement_volume"] = math.log1p(eng) if eng > 0 else 0.0
            signals["engagement_velocity"] = (
                ad.weighted_engagement_velocity if eng > 0 else 0.0
            )

            raw_signals.append(signals)

        # Normalize: survivability is already [0,1] from KM, just percentile-rank it globally.
        # Engagement signals: per-platform percentile normalization.
        platform_indices: dict[str, list[int]] = defaultdict(list)
        for i, ad in enumerate(ads):
            plat = ad.platform.value if ad.platform else "other"
            platform_indices[plat].append(i)

        normalized: list[dict[str, float | None]] = [{} for _ in ads]

        for signal_name in self._weights:
            if signal_name in _ENGAGEMENT_SIGNALS:
                for indices in platform_indices.values():
                    plat_values = [raw_signals[i].get(signal_name) for i in indices]
                    plat_norm = _percentile_normalize(plat_values)
                    for idx, norm_val in zip(indices, plat_norm, strict=True):
                        normalized[idx][signal_name] = norm_val
            else:
                raw_values = [rs.get(signal_name) for rs in raw_signals]
                norm_values = _percentile_normalize(raw_values)
                for i, val in enumerate(norm_values):
                    normalized[i][signal_name] = val

        # Weighted sum with missing-data redistribution
        scored: list[ScoredAd] = []
        for ad, norm_sigs in zip(ads, normalized, strict=True):
            composite, final_signals = _weighted_sum(ad, norm_sigs, self._weights)
            scored.append(
                ScoredAd(
                    ad=ad,
                    composite_score=composite,
                    signal_scores=final_signals,
                    scoring_version="v3",
                )
            )

        return scored


def _weighted_sum(
    ad: RawAd,
    normalized_signals: dict[str, float | None],
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Compute weighted sum from available signals with redistribution."""
    platform = ad.platform.value if ad.platform else "other"
    is_weak = platform in _WEAK_ENGAGEMENT_PLATFORMS

    available: dict[str, float] = {}
    for name in weights:
        val = normalized_signals.get(name)
        if val is None:
            continue
        if is_weak and name in _ENGAGEMENT_SIGNALS:
            continue
        available[name] = val

    if not available:
        return 0.5, {}

    weight_total = sum(weights[n] for n in available)
    if weight_total <= 0:
        return 0.5, {}

    composite = 0.0
    signal_scores: dict[str, float] = {}
    for name, val in available.items():
        w = weights[name] / weight_total
        composite += val * w
        signal_scores[name] = round(val, 4)

    composite = max(0.0, min(1.0, composite))
    return round(composite, 6), signal_scores


def _percentile_normalize(values: list[float | None]) -> list[float | None]:
    """Normalize values to [0, 1] using percentile ranks. Ties share avg rank."""
    indexed = [(i, v) for i, v in enumerate(values) if v is not None]
    if not indexed:
        return [None] * len(values)

    n = len(indexed)
    if n == 1:
        result: list[float | None] = [None] * len(values)
        result[indexed[0][0]] = 0.5
        return result

    sorted_items = sorted(indexed, key=lambda x: x[1])
    result = [None] * len(values)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_items[j][1] == sorted_items[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        percentile = avg_rank / (n - 1)
        for k in range(i, j):
            result[sorted_items[k][0]] = round(percentile, 6)
        i = j

    return result
