"""Tests for the Snorkel-based v2 scorer."""

from __future__ import annotations

import numpy as np

from draper.scoring.schemas import ScoringConfig
from draper.scoring.snorkel_scorer import (
    ABSTAIN,
    HIGH,
    LOW,
    SnorkelScorer,
    _majority_vote_proba,
    _percentile_rank,
)
from draper.scraping.schemas import AdSource, Platform, RawAd


def _config() -> ScoringConfig:
    return ScoringConfig.from_yaml("configs/scoring.yaml")


def _ad(
    ad_id: str,
    *,
    platform: Platform = Platform.FACEBOOK,
    likes: int = 0,
    comments: int = 0,
    shares: int = 0,
    views: int = 0,
    active_days: int | None = None,
) -> RawAd:
    return RawAd(
        ad_id=ad_id,
        source=AdSource.ADFLEX,
        platform=platform,
        likes=likes,
        comments=comments,
        shares=shares,
        views=views,
        active_days=active_days,
    )


# --- End-to-end scoring ---


def test_score_batch_empty() -> None:
    scorer = SnorkelScorer(_config())
    assert scorer.score_batch([]) == []


def test_score_batch_produces_valid_scores() -> None:
    scorer = SnorkelScorer(_config())
    ads = [_ad(str(i), likes=i * 50, active_days=i * 5 + 1) for i in range(30)]
    scored = scorer.score_batch(ads)
    assert len(scored) == 30
    for s in scored:
        assert 0.0 <= s.composite_score <= 1.0
        assert s.scoring_version == "v2"
        assert "high" in s.tier_probs
        assert "low" in s.tier_probs


def test_high_quality_ad_scores_higher_than_low() -> None:
    """An ad with high engagement + long lifespan should outscore a weak ad."""
    scorer = SnorkelScorer(_config())
    ads = [_ad(f"weak_{i}", likes=0, active_days=1) for i in range(15)] + [
        _ad(f"strong_{i}", likes=1000, shares=200, active_days=120) for i in range(15)
    ]
    scored = scorer.score_batch(ads)
    weak_scores = [s.composite_score for s in scored[:15]]
    strong_scores = [s.composite_score for s in scored[15:]]
    assert max(weak_scores) < min(strong_scores), (
        f"Strong ads should all outscore weak ads.\n"
        f"  weak max={max(weak_scores)}, strong min={min(strong_scores)}"
    )


def test_weak_engagement_platforms_abstain() -> None:
    """Pinterest/Reddit ads should still get scores (from survivability LFs)."""
    scorer = SnorkelScorer(_config())
    ads = [_ad(f"fb_{i}", likes=100, active_days=30) for i in range(20)] + [
        _ad("pn_long", platform=Platform.PINTEREST, active_days=100),
        _ad("pn_short", platform=Platform.PINTEREST, active_days=1),
    ]
    scored = scorer.score_batch(ads)
    pn_long = scored[-2]
    pn_short = scored[-1]
    assert pn_long.composite_score is not None
    assert pn_short.composite_score is not None
    # Long-running Pinterest ad should score higher than short-lived one
    assert pn_long.composite_score > pn_short.composite_score


def test_all_sparse_ads_get_neutral_scores() -> None:
    """Ads with no signals at all should get a neutral ~0.5 score."""
    scorer = SnorkelScorer(_config())
    ads = [_ad(f"sparse_{i}") for i in range(10)]
    scored = scorer.score_batch(ads)
    for s in scored:
        assert 0.3 <= s.composite_score <= 0.7, (
            f"Sparse ad should be near 0.5, got {s.composite_score}"
        )


# --- Percentile rank ---


def test_percentile_rank_basic() -> None:
    values: list[float | None] = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = _percentile_rank(values)
    assert result == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_percentile_rank_with_nones() -> None:
    values: list[float | None] = [None, 10.0, 30.0, None, 20.0]
    result = _percentile_rank(values)
    assert result[0] is None
    assert result[3] is None
    assert result[1] == 0.0
    assert result[4] == 0.5
    assert result[2] == 1.0


def test_percentile_rank_single() -> None:
    result = _percentile_rank([42.0])
    assert result == [0.5]


def test_percentile_rank_all_none() -> None:
    result = _percentile_rank([None, None])
    assert result == [None, None]


# --- Majority vote fallback ---


def test_majority_vote_all_abstain() -> None:
    L = np.full((3, 4), ABSTAIN, dtype=np.int64)
    probs = _majority_vote_proba(L)
    assert probs.shape == (3, 2)
    np.testing.assert_allclose(probs, 0.5)


def test_majority_vote_unanimous_high() -> None:
    L = np.full((1, 4), HIGH, dtype=np.int64)
    probs = _majority_vote_proba(L)
    p_high = probs[0, HIGH]
    assert p_high > 0.7, f"Unanimous HIGH should give high P(HIGH), got {p_high}"


def test_majority_vote_unanimous_low() -> None:
    L = np.full((1, 4), LOW, dtype=np.int64)
    probs = _majority_vote_proba(L)
    p_high = probs[0, HIGH]
    assert p_high < 0.3, f"Unanimous LOW should give low P(HIGH), got {p_high}"
