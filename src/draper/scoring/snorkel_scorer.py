"""Snorkel-based composite scorer for ads (v2).

Replaces the v1 ``CompositeScorer``'s hand-tuned weighted sum with a Snorkel
``LabelModel`` that learns to combine multiple noisy labeling functions (LFs)
by observing their agreement patterns across the batch. No ground-truth labels
are required.

Each LF votes HIGH (1), LOW (0), or ABSTAIN (-1). Engagement LFs abstain on
weak-engagement platforms (Pinterest/Reddit/Other) where likes and shares are
unreliable. The ``LabelModel`` outputs ``P(HIGH)`` per ad, which becomes
``ScoredAd.composite_score``. Tier assignment is handled by the existing
``TierAssigner`` downstream, preserving the v1 tier boundary logic.

The 2-class design (HIGH vs LOW, no explicit MEDIUM class) is intentional:
no LF can reliably vote "this ad is mediocre", and Snorkel's ``LabelModel``
assigns near-zero prior to classes that no LF ever votes for. MEDIUM emerges
naturally as the tier for ads where ``P(HIGH)`` falls between the high/low
percentile boundaries — exactly how v1 works.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from snorkel.labeling.model import LabelModel

from draper.scoring.schemas import ScoredAd, ScoringConfig
from draper.scoring.survival import compute_survivability
from draper.scraping.schemas import RawAd

logger = logging.getLogger(__name__)

ABSTAIN = -1
LOW = 0
HIGH = 1

_WEAK_ENGAGEMENT_PLATFORMS = {"reddit", "other"}


@dataclass
class _AdFeatures:
    """Pre-computed per-ad features consumed by labeling functions."""

    survivability: float | None
    longevity_days: int | None
    platform: str
    engagement_volume_pct: float | None
    velocity_pct: float | None


class SnorkelScorer:
    """Score ads using a Snorkel LabelModel over noisy labeling functions."""

    def __init__(
        self,
        config: ScoringConfig,
        *,
        high_pct: float = 0.80,
        low_pct: float = 0.20,
        early_death_days: int = 3,
        long_runner_days: int = 90,
    ) -> None:
        self.config = config
        self.high_pct = high_pct
        self.low_pct = low_pct
        self.early_death_days = early_death_days
        self.long_runner_days = long_runner_days

    def score_batch(self, ads: list[RawAd]) -> list[ScoredAd]:
        """Score a batch of ads using Snorkel labeling functions.

        The full batch is required because percentile ranks and the survival
        curve are computed relative to the cohort.

        Args:
            ads: All ads to score.

        Returns:
            List of ``ScoredAd`` with ``composite_score = P(HIGH)`` and
            ``tier_probs`` populated. Tier is left as ``"low"`` (use
            ``TierAssigner`` downstream to assign tiers from the score
            distribution).
        """
        if not ads:
            return []

        features = self._compute_features(ads)
        l_matrix = self._apply_lfs(features)
        probs = self._fit_and_predict(l_matrix)

        scored: list[ScoredAd] = []
        for ad, prob in zip(ads, probs, strict=True):
            p_high = float(prob[HIGH])
            p_low = float(prob[LOW])
            scored.append(
                ScoredAd(
                    ad=ad,
                    composite_score=round(p_high, 6),
                    tier_probs={"high": round(p_high, 4), "low": round(p_low, 4)},
                    scoring_version="v2",
                )
            )
        return scored

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_features(self, ads: list[RawAd]) -> list[_AdFeatures]:
        """Build per-ad features: survivability + per-platform engagement percentiles."""
        survivabilities = compute_survivability(ads)

        platform_indices: dict[str, list[int]] = defaultdict(list)
        for i, ad in enumerate(ads):
            platform = ad.platform.value if ad.platform else "other"
            platform_indices[platform].append(i)

        engagement_volume_pct: list[float | None] = [None] * len(ads)
        velocity_pct: list[float | None] = [None] * len(ads)

        for _platform, indices in platform_indices.items():
            vol_raw: list[float | None] = []
            vel_raw: list[float | None] = []
            for idx in indices:
                ad = ads[idx]
                if ad.weighted_engagement > 0:
                    vol_raw.append(ad.weighted_engagement)
                    vel_raw.append(ad.weighted_engagement_velocity)
                else:
                    vol_raw.append(None)
                    vel_raw.append(None)

            vol_ranks = _percentile_rank(vol_raw)
            vel_ranks = _percentile_rank(vel_raw)

            for local_i, global_i in enumerate(indices):
                engagement_volume_pct[global_i] = vol_ranks[local_i]
                velocity_pct[global_i] = vel_ranks[local_i]

        return [
            _AdFeatures(
                survivability=survivabilities[i],
                longevity_days=ads[i].longevity_days,
                platform=ads[i].platform.value if ads[i].platform else "other",
                engagement_volume_pct=engagement_volume_pct[i],
                velocity_pct=velocity_pct[i],
            )
            for i in range(len(ads))
        ]

    # ------------------------------------------------------------------
    # Labeling functions
    # ------------------------------------------------------------------

    def _lfs(self) -> list[Callable[[_AdFeatures], int]]:
        """Return the ordered list of labeling functions."""
        return [
            self._lf_survivability_high,
            self._lf_survivability_low,
            self._lf_engagement_volume_high,
            self._lf_engagement_volume_low,
            self._lf_velocity_high,
            self._lf_velocity_low,
            self._lf_early_death_hard,
            self._lf_long_runner_hard,
        ]

    def _lf_survivability_high(self, f: _AdFeatures) -> int:
        if f.survivability is None:
            return ABSTAIN
        return HIGH if f.survivability > self.high_pct else ABSTAIN

    def _lf_survivability_low(self, f: _AdFeatures) -> int:
        if f.survivability is None:
            return ABSTAIN
        return LOW if f.survivability < self.low_pct else ABSTAIN

    def _lf_engagement_volume_high(self, f: _AdFeatures) -> int:
        if f.platform in _WEAK_ENGAGEMENT_PLATFORMS or f.engagement_volume_pct is None:
            return ABSTAIN
        return HIGH if f.engagement_volume_pct > self.high_pct else ABSTAIN

    def _lf_engagement_volume_low(self, f: _AdFeatures) -> int:
        if f.platform in _WEAK_ENGAGEMENT_PLATFORMS or f.engagement_volume_pct is None:
            return ABSTAIN
        return LOW if f.engagement_volume_pct < self.low_pct else ABSTAIN

    def _lf_velocity_high(self, f: _AdFeatures) -> int:
        if f.platform in _WEAK_ENGAGEMENT_PLATFORMS or f.velocity_pct is None:
            return ABSTAIN
        return HIGH if f.velocity_pct > self.high_pct else ABSTAIN

    def _lf_velocity_low(self, f: _AdFeatures) -> int:
        if f.platform in _WEAK_ENGAGEMENT_PLATFORMS or f.velocity_pct is None:
            return ABSTAIN
        return LOW if f.velocity_pct < self.low_pct else ABSTAIN

    def _lf_early_death_hard(self, f: _AdFeatures) -> int:
        if f.longevity_days is None:
            return ABSTAIN
        return LOW if f.longevity_days < self.early_death_days else ABSTAIN

    def _lf_long_runner_hard(self, f: _AdFeatures) -> int:
        if f.longevity_days is None:
            return ABSTAIN
        return HIGH if f.longevity_days > self.long_runner_days else ABSTAIN

    # ------------------------------------------------------------------
    # LF application + model fitting
    # ------------------------------------------------------------------

    def _apply_lfs(self, features: list[_AdFeatures]) -> NDArray[np.int64]:
        """Apply all LFs to every ad, producing the (n_ads, n_lfs) label matrix."""
        lfs = self._lfs()
        n = len(features)
        m = len(lfs)
        l_matrix: NDArray[np.int64] = np.full((n, m), ABSTAIN, dtype=np.int64)
        for i, feat in enumerate(features):
            for j, lf in enumerate(lfs):
                l_matrix[i, j] = lf(feat)
        return l_matrix

    def _fit_and_predict(self, l_matrix: NDArray[np.int64]) -> NDArray[np.float64]:
        """Fit a Snorkel LabelModel on the L matrix and return P(class) per ad.

        Returns an (n_ads, 2) array with columns [P(LOW), P(HIGH)].
        Falls back to majority-vote probabilities when the LabelModel cannot
        converge (e.g. batch too small or all LFs abstaining).
        """
        n_ads = l_matrix.shape[0]
        has_any_vote = (l_matrix != ABSTAIN).any(axis=1)

        if not has_any_vote.any():
            return np.full((n_ads, 2), 0.5)

        try:
            label_model = LabelModel(cardinality=2, verbose=False)
            label_model.fit(L_train=l_matrix, n_epochs=500, log_freq=500, seed=42)
            probs: NDArray[np.float64] = label_model.predict_proba(L=l_matrix)
        except Exception:  # noqa: BLE001
            logger.warning("LabelModel fit failed; falling back to majority vote")
            probs = _majority_vote_proba(l_matrix)

        return probs


def _majority_vote_proba(l_matrix: NDArray[np.int64]) -> NDArray[np.float64]:
    """Simple majority-vote fallback producing soft P(LOW), P(HIGH).

    For each ad: count HIGH votes vs LOW votes among non-abstaining LFs.
    Convert to a probability via (n_high + 1) / (n_high + n_low + 2)
    (Laplace smoothing toward 0.5).
    """
    n_ads = l_matrix.shape[0]
    probs = np.full((n_ads, 2), 0.5)
    for i in range(n_ads):
        votes = l_matrix[i]
        non_abstain = votes[votes != ABSTAIN]
        if len(non_abstain) == 0:
            continue
        n_high = int((non_abstain == HIGH).sum())
        n_low = int((non_abstain == LOW).sum())
        total = n_high + n_low + 2  # Laplace smoothing
        p_high = (n_high + 1) / total
        probs[i] = [1.0 - p_high, p_high]
    return probs


def _percentile_rank(values: list[float | None]) -> list[float | None]:
    """Percentile-rank non-None values into [0, 1]. Ties share average rank."""
    indexed = [(i, v) for i, v in enumerate(values) if v is not None]
    if not indexed:
        return [None] * len(values)

    n = len(indexed)
    if n == 1:
        out: list[float | None] = [None] * len(values)
        out[indexed[0][0]] = 0.5
        return out

    sorted_items = sorted(indexed, key=lambda x: x[1])
    result: list[float | None] = [None] * len(values)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_items[j][1] == sorted_items[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        pct = avg_rank / (n - 1)
        for k in range(i, j):
            result[sorted_items[k][0]] = round(pct, 6)
        i = j
    return result
