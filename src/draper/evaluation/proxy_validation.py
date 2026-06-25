"""Multi-source proxy score validation for RQ3.

Validation streams:
- Stream A: Upworthy A/B winner prediction (creative-feature validation)
- Stream B: Google Political Ads BigQuery — real spend/impression buckets
- Stream C: IRA dataset — out-of-domain political robustness
- Stream D: Internal consistency — tier separation, cross-platform homogeneity
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from draper.scoring.schemas import ScoredAd


@dataclass
class CorrelationResult:
    """Result of a Spearman rank correlation with bootstrap CI."""

    rho: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n: int


@dataclass
class TierSeparationResult:
    """Result of Kruskal-Wallis tier separation test."""

    h_statistic: float
    p_value: float
    n_per_tier: dict[str, int]
    median_per_tier: dict[str, float]
    effect_sizes: dict[str, float]  # e.g. high_vs_low, high_vs_medium


@dataclass
class RankingResult:
    """Precision@K and NDCG@K results."""

    precision_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]


@dataclass
class ValidationResult:
    """Complete validation result for a single data stream."""

    source: str
    n_ads: int
    target_metric: str
    correlation: CorrelationResult
    tier_separation: TierSeparationResult
    ranking: RankingResult
    limitations: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        """Return a JSON-serializable summary."""
        return {
            "source": self.source,
            "n_ads": self.n_ads,
            "target_metric": self.target_metric,
            "spearman_rho": round(self.correlation.rho, 4),
            "spearman_p": self.correlation.p_value,
            "spearman_ci": [
                round(self.correlation.ci_lower, 4),
                round(self.correlation.ci_upper, 4),
            ],
            "kruskal_h": round(self.tier_separation.h_statistic, 4),
            "kruskal_p": self.tier_separation.p_value,
            "n_per_tier": self.tier_separation.n_per_tier,
            "median_per_tier": self.tier_separation.median_per_tier,
            "effect_sizes": self.tier_separation.effect_sizes,
            "precision_at_k": self.ranking.precision_at_k,
            "ndcg_at_k": self.ranking.ndcg_at_k,
            "limitations": self.limitations,
        }


@dataclass
class ConsistencyResult:
    """Internal consistency validation result."""

    tier_separation: TierSeparationResult
    platform_homogeneity: PlatformHomogeneityResult
    signal_contributions: dict[str, dict[str, float]]


@dataclass
class PlatformHomogeneityResult:
    """Chi-squared test for tier distribution homogeneity across platforms."""

    chi2: float
    p_value: float
    dof: int
    tier_counts_by_platform: dict[str, dict[str, int]]


@dataclass
class PairwiseValidationResult:
    """Pairwise winner-prediction result (e.g., Upworthy A/B tests).

    Tests whether a scoring function ranks the winning variant higher than
    the losing variant in matched pairs. Uses a binomial test against the
    50% chance baseline.
    """

    source: str
    n_pairs: int
    n_correct: int
    accuracy: float
    binomial_p_value: float
    accuracy_ci: tuple[float, float]
    n_ties: int
    limitations: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "source": self.source,
            "n_pairs": self.n_pairs,
            "n_correct": self.n_correct,
            "n_ties": self.n_ties,
            "accuracy": round(self.accuracy, 4),
            "accuracy_ci": [
                round(self.accuracy_ci[0], 4),
                round(self.accuracy_ci[1], 4),
            ],
            "binomial_p_value": self.binomial_p_value,
            "limitations": self.limitations,
        }


class ProxyValidator:
    """Multi-source proxy score validation for RQ3."""

    def validate_stream(
        self,
        scored_ads: list[ScoredAd],
        ground_truth: list[float],
        source: str,
        target_metric: str,
        limitations: list[str] | None = None,
        n_bootstrap: int = 1000,
    ) -> ValidationResult:
        """Run the full statistical validation battery on a scored dataset.

        Args:
            scored_ads: Ads scored by CompositeScorer + TierAssigner.
            ground_truth: Matching ground-truth values (e.g. log spend).
            source: Label for this stream (e.g. "meta_eu", "ira").
            target_metric: Name of the ground-truth metric.
            limitations: Known limitations of this validation stream.
            n_bootstrap: Number of bootstrap resamples for CIs.
        """
        proxy_scores = np.array([ad.composite_score for ad in scored_ads], dtype=np.float64)
        truth = np.array(ground_truth, dtype=np.float64)
        tiers = [ad.tier for ad in scored_ads]

        correlation = self.spearman_with_bootstrap(proxy_scores, truth, n_bootstrap)
        tier_sep = self.kruskal_wallis_by_tier(truth, tiers)
        ranking = self.ranking_metrics(proxy_scores, truth)

        return ValidationResult(
            source=source,
            n_ads=len(scored_ads),
            target_metric=target_metric,
            correlation=correlation,
            tier_separation=tier_sep,
            ranking=ranking,
            limitations=limitations or [],
        )

    def validate_internal_consistency(
        self,
        scored_ads: list[ScoredAd],
    ) -> ConsistencyResult:
        """Stream C: Internal consistency checks on AdFlex scored data."""
        # Tier separation on raw engagement
        engagement_values = [ad.ad.weighted_engagement for ad in scored_ads]
        tiers = [ad.tier for ad in scored_ads]
        tier_sep = self.kruskal_wallis_by_tier(np.array(engagement_values, dtype=np.float64), tiers)

        # Cross-platform tier homogeneity
        platform_homogeneity = self.platform_tier_homogeneity(scored_ads)

        # Signal contribution by tier
        signal_contribs = self._signal_contributions_by_tier(scored_ads)

        return ConsistencyResult(
            tier_separation=tier_sep,
            platform_homogeneity=platform_homogeneity,
            signal_contributions=signal_contribs,
        )

    # --- Statistical primitives ---

    @staticmethod
    def spearman_with_bootstrap(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> CorrelationResult:
        """Spearman rank correlation with bootstrap confidence interval."""
        rho_val, p_val = sp_stats.spearmanr(x, y)

        # Bootstrap CI
        rng = np.random.default_rng(seed)
        n = len(x)
        boot_rhos: list[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            r, _ = sp_stats.spearmanr(x[idx], y[idx])
            boot_rhos.append(float(r))

        boot_arr = np.array(boot_rhos)
        ci_lower = float(np.percentile(boot_arr, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))

        return CorrelationResult(
            rho=float(rho_val),
            p_value=float(p_val),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n=n,
        )

    @staticmethod
    def kruskal_wallis_by_tier(
        values: NDArray[np.float64],
        tiers: list[str],
    ) -> TierSeparationResult:
        """Kruskal-Wallis H-test for tier separation + pairwise effect sizes."""
        tier_groups: dict[str, list[float]] = {"high": [], "medium": [], "low": []}
        for val, tier in zip(values, tiers, strict=True):
            if tier in tier_groups:
                tier_groups[tier].append(float(val))

        # Filter to tiers with data
        available = {t: np.array(v) for t, v in tier_groups.items() if len(v) >= 2}
        tier_arrays = list(available.values())

        if len(tier_arrays) < 2:
            return TierSeparationResult(
                h_statistic=0.0,
                p_value=1.0,
                n_per_tier={t: len(v) for t, v in available.items()},
                median_per_tier={t: float(np.median(v)) for t, v in available.items()},
                effect_sizes={},
            )

        # Guard against degenerate input where all values are identical
        # (Kruskal-Wallis returns NaN due to ties correction divide-by-zero)
        all_values = np.concatenate(tier_arrays)
        if np.all(all_values == all_values[0]):
            h_stat: float = 0.0
            p_val: float = 1.0
        else:
            h_stat, p_val = sp_stats.kruskal(*tier_arrays)

        # Pairwise Cliff's delta effect sizes
        effect_sizes: dict[str, float] = {}
        pairs = [("high", "low"), ("high", "medium"), ("medium", "low")]
        for t1, t2 in pairs:
            if t1 in available and t2 in available:
                delta = _cliff_delta(available[t1], available[t2])
                effect_sizes[f"{t1}_vs_{t2}"] = round(delta, 4)

        return TierSeparationResult(
            h_statistic=float(h_stat),
            p_value=float(p_val),
            n_per_tier={t: len(v) for t, v in available.items()},
            median_per_tier={t: round(float(np.median(v)), 4) for t, v in available.items()},
            effect_sizes=effect_sizes,
        )

    @staticmethod
    def ranking_metrics(
        proxy_scores: NDArray[np.float64],
        truth_values: NDArray[np.float64],
        k_percentiles: list[int] | None = None,
    ) -> RankingResult:
        """Precision@K and NDCG@K at various percentile cutoffs."""
        if k_percentiles is None:
            k_percentiles = [10, 20, 30]

        n = len(proxy_scores)
        proxy_ranking = np.argsort(-proxy_scores)  # descending
        truth_ranking = np.argsort(-truth_values)

        precision_at_k: dict[int, float] = {}
        ndcg_at_k: dict[int, float] = {}

        for k_pct in k_percentiles:
            k = max(1, int(n * k_pct / 100))
            proxy_top_k = set(proxy_ranking[:k].tolist())
            truth_top_k = set(truth_ranking[:k].tolist())
            overlap = len(proxy_top_k & truth_top_k)
            precision_at_k[k_pct] = round(overlap / k, 4)

            # NDCG@K
            ndcg = _ndcg_at_k(proxy_scores, truth_values, k)
            ndcg_at_k[k_pct] = round(ndcg, 4)

        return RankingResult(
            precision_at_k=precision_at_k,
            ndcg_at_k=ndcg_at_k,
        )

    @staticmethod
    def platform_tier_homogeneity(
        scored_ads: list[ScoredAd],
        min_platform_size: int = 30,
    ) -> PlatformHomogeneityResult:
        """Chi-squared test for tier distribution homogeneity across platforms."""
        from collections import defaultdict

        platform_tiers: dict[str, dict[str, int]] = defaultdict(
            lambda: {"high": 0, "medium": 0, "low": 0}
        )
        for ad in scored_ads:
            plat = ad.ad.platform.value if ad.ad.platform else "other"
            platform_tiers[plat][ad.tier] += 1

        # Filter to platforms with enough data
        valid_platforms = {
            p: counts
            for p, counts in platform_tiers.items()
            if sum(counts.values()) >= min_platform_size
        }

        if len(valid_platforms) < 2:
            return PlatformHomogeneityResult(
                chi2=0.0,
                p_value=1.0,
                dof=0,
                tier_counts_by_platform=dict(valid_platforms),
            )

        # Build contingency table
        tier_order = ["high", "medium", "low"]
        observed = np.array(
            [[counts[t] for t in tier_order] for counts in valid_platforms.values()]
        )

        chi2_val, p_val, dof, _ = sp_stats.chi2_contingency(observed)

        return PlatformHomogeneityResult(
            chi2=float(chi2_val),
            p_value=float(p_val),
            dof=int(dof),
            tier_counts_by_platform=dict(valid_platforms),
        )

    @staticmethod
    def _signal_contributions_by_tier(
        scored_ads: list[ScoredAd],
    ) -> dict[str, dict[str, float]]:
        """Average signal score per tier."""
        from collections import defaultdict

        tier_signals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for ad in scored_ads:
            for signal_name, score in ad.signal_scores.items():
                tier_signals[ad.tier][signal_name].append(score)

        result: dict[str, dict[str, float]] = {}
        for tier, signals in tier_signals.items():
            result[tier] = {
                name: round(float(np.mean(values)), 4) for name, values in signals.items()
            }
        return result

    @staticmethod
    def validate_pairwise_winners(
        pairs: list[tuple[object, object]],
        score_fn: Callable[[object], float],
        source: str,
        limitations: list[str] | None = None,
    ) -> PairwiseValidationResult:
        """Test whether a scoring function predicts A/B winners (vs losers).

        Args:
            pairs: List of (winner, loser) tuples.
            score_fn: Function that maps a variant object to a numeric score.
                A higher score should mean "more likely to win."
            source: Label for the validation source (e.g. "upworthy").
            limitations: Known limitations of this validation.

        Returns:
            PairwiseValidationResult with accuracy, binomial p-value, and CI.
        """
        if not pairs:
            return PairwiseValidationResult(
                source=source,
                n_pairs=0,
                n_correct=0,
                accuracy=0.0,
                binomial_p_value=1.0,
                accuracy_ci=(0.0, 0.0),
                n_ties=0,
                limitations=limitations or [],
            )

        n_correct = 0
        n_ties = 0
        for winner, loser in pairs:
            w_score = score_fn(winner)
            l_score = score_fn(loser)
            if w_score > l_score:
                n_correct += 1
            elif w_score == l_score:
                n_ties += 1

        n_decisive = len(pairs) - n_ties
        accuracy = n_correct / n_decisive if n_decisive > 0 else 0.0

        # Binomial test against 0.5 chance baseline (only on decisive pairs)
        if n_decisive > 0:
            try:
                p_val = float(
                    sp_stats.binomtest(n_correct, n_decisive, p=0.5, alternative="greater").pvalue
                )
            except AttributeError:
                # Fallback for older scipy
                p_val = float(
                    sp_stats.binom_test(n_correct, n_decisive, p=0.5, alternative="greater")
                )
        else:
            p_val = 1.0

        # Wilson 95% CI for proportion
        ci_lower, ci_upper = _wilson_ci(n_correct, n_decisive)

        return PairwiseValidationResult(
            source=source,
            n_pairs=len(pairs),
            n_correct=n_correct,
            accuracy=accuracy,
            binomial_p_value=p_val,
            accuracy_ci=(ci_lower, ci_upper),
            n_ties=n_ties,
            limitations=limitations or [],
        )


# --- Utility functions ---


def _wilson_ci(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def headline_text_score(variant: object) -> float:
    """Simple text-feature score for a headline-bearing object.

    Combines features that prior work (Upworthy A/B research) has shown to
    correlate with click-through:
    - Headline length (tokens) — moderate length wins
    - Has a number/digit — boosts CTR
    - Has a question mark — boosts CTR
    - Has emotional/clickbait words

    The variant must expose `.headline` and (optionally) `.excerpt`.
    """
    headline = str(getattr(variant, "headline", "") or "")
    excerpt = str(getattr(variant, "excerpt", "") or "")
    text = (headline + " " + excerpt).lower()
    if not headline:
        return 0.0

    score = 0.0

    # Length feature: log of word count, peaks around 12-18 words
    words = headline.split()
    n_words = len(words)
    if n_words > 0:
        # Quadratic peak at ~15 words
        length_score = -((n_words - 15) ** 2) / 200.0
        score += max(-1.0, length_score)

    # Has digit
    if any(c.isdigit() for c in headline):
        score += 0.3

    # Has question mark
    if "?" in headline:
        score += 0.2

    # Has clickbait emotional words
    clickbait = (
        "you",
        "your",
        "this",
        "why",
        "how",
        "what",
        "when",
        "actually",
        "really",
        "surprising",
        "amazing",
        "shocking",
        "wait",
        "even",
    )
    cb_count = sum(1 for w in clickbait if w in text.split())
    score += 0.05 * cb_count

    return score


def _cliff_delta(
    group_a: NDArray[np.float64],
    group_b: NDArray[np.float64],
) -> float:
    """Cliff's delta effect size between two groups.

    Returns a value in [-1, 1]:
    - +1: all values in A > all values in B
    - -1: all values in B > all values in A
    - 0: no difference
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a == 0 or n_b == 0:
        return 0.0

    # Vectorized pairwise comparison
    diff = np.subtract.outer(group_a, group_b)
    greater = int(np.sum(diff > 0))
    less = int(np.sum(diff < 0))
    return (greater - less) / (n_a * n_b)


def _dcg_at_k(
    relevances: NDArray[np.float64],
    k: int,
) -> float:
    """Discounted Cumulative Gain at position k."""
    relevances = relevances[:k]
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances / discounts))


def _ndcg_at_k(
    proxy_scores: NDArray[np.float64],
    truth_values: NDArray[np.float64],
    k: int,
) -> float:
    """Normalized DCG@K — how well proxy ranking matches truth ranking."""
    # Order by proxy scores (what we predicted)
    proxy_order = np.argsort(-proxy_scores)
    truth_in_proxy_order = truth_values[proxy_order]

    dcg = _dcg_at_k(truth_in_proxy_order, k)

    # Ideal ordering (perfect ranking)
    ideal_order = np.sort(truth_values)[::-1]
    idcg = _dcg_at_k(ideal_order, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_log_midpoint(
    lower: float | int | None,
    upper: float | int | None,
) -> float | None:
    """Compute log of the midpoint of a range. Returns None if both bounds are missing."""
    if lower is not None and upper is not None:
        mid = (float(lower) + float(upper)) / 2.0
        return math.log1p(mid) if mid > 0 else None
    if lower is not None:
        return math.log1p(float(lower)) if lower > 0 else None
    if upper is not None:
        return math.log1p(float(upper)) if upper > 0 else None
    return None
