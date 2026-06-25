"""Tests for proxy validation statistical functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from draper.evaluation.proxy_validation import (
    ProxyValidator,
    _cliff_delta,
    _ndcg_at_k,
    _wilson_ci,
    compute_log_midpoint,
    headline_text_score,
)
from draper.scoring.schemas import ScoredAd
from draper.scraping.schemas import AdCopy, AdSource, Platform, RawAd


def _make_scored_ad(score: float, tier: str, days: int = 10) -> ScoredAd:
    """Create a minimal ScoredAd for testing."""
    ad = RawAd(
        ad_id=f"test_{score}_{tier}",
        source=AdSource.META_LIBRARY,
        platform=Platform.FACEBOOK,
        ad_copy=AdCopy(headline="test", body="test"),
        active_days=days,
    )
    return ScoredAd(
        ad=ad,
        composite_score=score,
        signal_scores={"longevity": score, "early_death": 1.0},
        tier=tier,
    )


class TestSpearmanWithBootstrap:
    def test_perfect_positive_correlation(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ProxyValidator.spearman_with_bootstrap(x, y, n_bootstrap=100)
        assert result.rho == pytest.approx(1.0)
        assert result.p_value < 0.05
        assert result.ci_lower > 0.8
        assert result.n == 5

    def test_perfect_negative_correlation(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = ProxyValidator.spearman_with_bootstrap(x, y, n_bootstrap=100)
        assert result.rho == pytest.approx(-1.0)

    def test_no_correlation(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.random(100)
        y = rng.random(100)
        result = ProxyValidator.spearman_with_bootstrap(x, y, n_bootstrap=100)
        assert abs(result.rho) < 0.3  # weak at most
        assert result.ci_lower < result.ci_upper

    def test_bootstrap_ci_contains_point_estimate(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.random(50)
        y = x + rng.normal(0, 0.3, 50)
        result = ProxyValidator.spearman_with_bootstrap(x, y, n_bootstrap=500)
        # CI should contain the point estimate (or be very close)
        assert result.ci_lower <= result.rho + 0.05
        assert result.ci_upper >= result.rho - 0.05


class TestKruskalWallisByTier:
    def test_clearly_separated_tiers(self) -> None:
        values = np.array([10.0, 11.0, 12.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0])
        tiers = ["high", "high", "high", "medium", "medium", "medium", "low", "low", "low"]
        result = ProxyValidator.kruskal_wallis_by_tier(values, tiers)
        assert result.h_statistic > 0
        assert result.p_value < 0.05
        assert result.n_per_tier["high"] == 3
        assert result.median_per_tier["high"] > result.median_per_tier["low"]
        assert result.effect_sizes["high_vs_low"] > 0

    def test_identical_tiers(self) -> None:
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        tiers = ["high", "high", "medium", "medium", "low", "low"]
        result = ProxyValidator.kruskal_wallis_by_tier(values, tiers)
        assert result.h_statistic == pytest.approx(0.0, abs=0.1)

    def test_single_tier_returns_empty(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        tiers = ["high", "high", "high"]
        result = ProxyValidator.kruskal_wallis_by_tier(values, tiers)
        assert result.p_value == 1.0
        assert result.effect_sizes == {}


class TestCliffDelta:
    def test_complete_dominance(self) -> None:
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([1.0, 2.0, 3.0])
        assert _cliff_delta(a, b) == pytest.approx(1.0)

    def test_reverse_dominance(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        assert _cliff_delta(a, b) == pytest.approx(-1.0)

    def test_no_difference(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert _cliff_delta(a, b) == pytest.approx(0.0)

    def test_empty_group(self) -> None:
        a = np.array([1.0, 2.0])
        b = np.array([])
        assert _cliff_delta(a, b) == 0.0


class TestRankingMetrics:
    def test_perfect_ranking(self) -> None:
        proxy = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        truth = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = ProxyValidator.ranking_metrics(proxy, truth, k_percentiles=[20, 40])
        assert result.precision_at_k[20] == pytest.approx(1.0)
        assert result.ndcg_at_k[20] == pytest.approx(1.0)

    def test_reversed_ranking(self) -> None:
        proxy = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        truth = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        result = ProxyValidator.ranking_metrics(proxy, truth, k_percentiles=[10, 20])
        assert result.precision_at_k[10] == pytest.approx(0.0)


class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        proxy = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        truth = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _ndcg_at_k(proxy, truth, 3) == pytest.approx(1.0)

    def test_all_zeros(self) -> None:
        proxy = np.array([1.0, 2.0, 3.0])
        truth = np.array([0.0, 0.0, 0.0])
        assert _ndcg_at_k(proxy, truth, 3) == pytest.approx(0.0)


class TestComputeLogMidpoint:
    def test_both_bounds(self) -> None:
        result = compute_log_midpoint(100, 200)
        assert result == pytest.approx(math.log1p(150.0))

    def test_lower_only(self) -> None:
        result = compute_log_midpoint(100, None)
        assert result == pytest.approx(math.log1p(100.0))

    def test_upper_only(self) -> None:
        result = compute_log_midpoint(None, 200)
        assert result == pytest.approx(math.log1p(200.0))

    def test_both_none(self) -> None:
        assert compute_log_midpoint(None, None) is None

    def test_zero_value(self) -> None:
        assert compute_log_midpoint(0, 0) is None


class TestValidateStream:
    def test_end_to_end(self) -> None:
        """Test the full validate_stream pipeline with synthetic data."""
        scored_ads = [
            _make_scored_ad(0.9, "high", days=30),
            _make_scored_ad(0.85, "high", days=28),
            _make_scored_ad(0.6, "medium", days=15),
            _make_scored_ad(0.55, "medium", days=12),
            _make_scored_ad(0.5, "medium", days=10),
            _make_scored_ad(0.2, "low", days=3),
            _make_scored_ad(0.15, "low", days=2),
            _make_scored_ad(0.1, "low", days=1),
        ]
        # Ground truth that correlates with proxy scores
        ground_truth = [8.0, 7.5, 5.0, 4.5, 4.0, 1.5, 1.0, 0.5]

        validator = ProxyValidator()
        result = validator.validate_stream(
            scored_ads=scored_ads,
            ground_truth=ground_truth,
            source="test",
            target_metric="synthetic",
            n_bootstrap=100,
        )

        assert result.source == "test"
        assert result.n_ads == 8
        assert result.correlation.rho > 0.8  # strong correlation
        assert result.tier_separation.h_statistic > 0
        assert 10 in result.ranking.precision_at_k


class _StubVariant:
    """Test stub for headline-bearing object."""

    def __init__(self, headline: str, excerpt: str = "") -> None:
        self.headline = headline
        self.excerpt = excerpt


class TestWilsonCI:
    def test_zero_n(self) -> None:
        assert _wilson_ci(0, 0) == (0.0, 0.0)

    def test_perfect_success(self) -> None:
        lower, upper = _wilson_ci(10, 10)
        assert lower > 0.5
        assert upper == pytest.approx(1.0, abs=0.01)

    def test_half_success(self) -> None:
        lower, upper = _wilson_ci(50, 100)
        # 95% CI for 50/100 should bracket 0.5
        assert lower < 0.5 < upper
        assert upper - lower < 0.25  # reasonable width


class TestHeadlineTextScore:
    def test_empty_returns_zero(self) -> None:
        assert headline_text_score(_StubVariant("")) == 0.0

    def test_question_boost(self) -> None:
        a = headline_text_score(_StubVariant("Is this a great headline"))
        b = headline_text_score(_StubVariant("Is this a great headline?"))
        assert b > a

    def test_digit_boost(self) -> None:
        a = headline_text_score(_StubVariant("Top tips for marketing"))
        b = headline_text_score(_StubVariant("Top 5 tips for marketing"))
        assert b > a


class TestPairwiseValidation:
    def test_perfect_predictor(self) -> None:
        # Build pairs where score_fn always predicts the winner
        pairs: list[tuple[object, object]] = [
            (_StubVariant("winner one"), _StubVariant("loser")),
            (_StubVariant("winner two"), _StubVariant("loser")),
            (_StubVariant("winner three"), _StubVariant("loser")),
        ]

        # Score function: longer headline wins
        result = ProxyValidator.validate_pairwise_winners(
            pairs=pairs,
            score_fn=lambda v: len(v.headline),
            source="test",
        )
        assert result.n_pairs == 3
        assert result.n_correct == 3
        assert result.accuracy == 1.0

    def test_random_predictor_around_chance(self) -> None:
        # All variants identical → all ties
        identical = _StubVariant("same headline")
        pairs: list[tuple[object, object]] = [(identical, identical) for _ in range(10)]
        result = ProxyValidator.validate_pairwise_winners(
            pairs=pairs,
            score_fn=lambda v: 1.0,
            source="test",
        )
        assert result.n_ties == 10
        assert result.n_correct == 0

    def test_empty_pairs(self) -> None:
        result = ProxyValidator.validate_pairwise_winners(
            pairs=[],
            score_fn=lambda v: 0.0,
            source="test",
        )
        assert result.n_pairs == 0
        assert result.accuracy == 0.0


class TestPlatformTierHomogeneity:
    def test_with_multiple_platforms(self) -> None:
        ads: list[ScoredAd] = []
        # Create 50 ads per platform with same tier distribution
        for platform in [Platform.FACEBOOK, Platform.TIKTOK]:
            for i in range(50):
                ad = RawAd(
                    ad_id=f"{platform.value}_{i}",
                    source=AdSource.ADFLEX,
                    platform=platform,
                )
                tier = "high" if i < 10 else ("medium" if i < 35 else "low")
                ads.append(ScoredAd(ad=ad, composite_score=float(i) / 50, tier=tier))

        validator = ProxyValidator()
        result = validator.platform_tier_homogeneity(ads)
        assert result.chi2 >= 0
        assert len(result.tier_counts_by_platform) == 2
