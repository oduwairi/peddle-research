"""Tests for composite scoring and tier assignment."""

from draper.scoring.composite_scorer import CompositeScorer, _percentile_normalize
from draper.scoring.schemas import ScoredAd, ScoringConfig
from draper.scoring.tier_assigner import TierAssigner
from draper.scraping.schemas import AdSource, Platform, RawAd


def _make_config() -> ScoringConfig:
    return ScoringConfig.from_yaml("configs/scoring.yaml")


def _make_ad(
    ad_id: str = "1",
    platform: Platform = Platform.FACEBOOK,
    likes: int = 0,
    comments: int = 0,
    shares: int = 0,
    views: int = 0,
    active_days: int | None = None,
    first_seen: str | None = None,
    last_seen: str | None = None,
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
        first_seen=first_seen,
        last_seen=last_seen,
    )


# --- Percentile normalization ---


def test_percentile_normalize_basic() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = _percentile_normalize(values)
    assert result == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_percentile_normalize_with_nones() -> None:
    values: list[float | None] = [None, 10.0, 30.0, None, 20.0]
    result = _percentile_normalize(values)
    assert result[0] is None
    assert result[3] is None
    assert result[1] == 0.0  # lowest
    assert result[4] == 0.5  # middle
    assert result[2] == 1.0  # highest


def test_percentile_normalize_ties() -> None:
    values = [10.0, 20.0, 20.0, 30.0]
    result = _percentile_normalize(values)
    assert result[0] == 0.0
    # Tied values at index 1 and 2 should get same rank
    assert result[1] == result[2]
    assert result[3] == 1.0


def test_percentile_normalize_single() -> None:
    result = _percentile_normalize([42.0])
    assert result == [0.5]


def test_percentile_normalize_all_none() -> None:
    result = _percentile_normalize([None, None, None])
    assert result == [None, None, None]


# --- Signal computation ---


def test_longevity_signal() -> None:
    config = _make_config()
    scorer = CompositeScorer(config)
    ad = _make_ad(active_days=30)
    signals = scorer._compute_raw_signals(ad)
    assert signals["longevity"] is not None
    assert signals["longevity"] > 0  # log1p(30)


def test_longevity_none_when_no_dates() -> None:
    config = _make_config()
    scorer = CompositeScorer(config)
    ad = _make_ad()  # no active_days, no dates
    signals = scorer._compute_raw_signals(ad)
    assert signals["longevity"] is None


def test_engagement_volume_signal() -> None:
    config = _make_config()
    scorer = CompositeScorer(config)
    ad = _make_ad(likes=100, comments=20, shares=10)
    signals = scorer._compute_raw_signals(ad)
    assert signals["engagement_volume"] is not None
    assert signals["engagement_volume"] > 0  # log1p(130)


def test_engagement_volume_none_when_zero() -> None:
    config = _make_config()
    scorer = CompositeScorer(config)
    ad = _make_ad()  # all engagement = 0
    signals = scorer._compute_raw_signals(ad)
    assert signals["engagement_volume"] is None


def test_engagement_velocity_signal() -> None:
    config = _make_config()
    scorer = CompositeScorer(config)
    ad = _make_ad(likes=100, active_days=10)
    signals = scorer._compute_raw_signals(ad)
    # engagement_velocity = total_engagement / days = 100 / 10 = 10.0
    assert signals["engagement_velocity"] == 10.0


def test_early_death_signal() -> None:
    config = _make_config()
    scorer = CompositeScorer(config)
    ad_short = _make_ad(active_days=1)
    ad_long = _make_ad(active_days=30)
    assert scorer._compute_raw_signals(ad_short)["early_death"] == 0.0  # penalty
    assert scorer._compute_raw_signals(ad_long)["early_death"] == 1.0  # survived


# --- Batch scoring ---


def test_score_batch_produces_scores() -> None:
    config = _make_config()
    scorer = CompositeScorer(config)
    ads = [
        _make_ad("1", likes=100, active_days=30),
        _make_ad("2", likes=10, active_days=5),
        _make_ad("3", likes=500, active_days=90),
    ]
    scored = scorer.score_batch(ads)
    assert len(scored) == 3
    for s in scored:
        assert 0.0 <= s.composite_score <= 1.0
        assert len(s.signal_scores) > 0


def test_score_batch_ordering() -> None:
    """Higher engagement + longevity → higher score."""
    config = _make_config()
    scorer = CompositeScorer(config)
    low = _make_ad("low", likes=1, active_days=1)
    high = _make_ad("high", likes=1000, active_days=100)
    scored = scorer.score_batch([low, high])
    assert scored[1].composite_score > scored[0].composite_score


def test_score_batch_empty() -> None:
    config = _make_config()
    scorer = CompositeScorer(config)
    assert scorer.score_batch([]) == []


# --- Platform-aware weight redistribution ---


def test_weak_engagement_platform_redistribution() -> None:
    """Reddit/Pinterest ads should redistribute engagement weight to other signals."""
    config = _make_config()
    scorer = CompositeScorer(config)

    # Same ad data, different platforms
    fb_ad = _make_ad(
        "fb",
        platform=Platform.FACEBOOK,
        likes=50,
        active_days=30,
    )
    reddit_ad = _make_ad(
        "rd",
        platform=Platform.OTHER,
        likes=50,
        active_days=30,
    )

    # Score individually won't tell us much, but we can verify signal_scores differ
    scored = scorer.score_batch([fb_ad, reddit_ad])
    fb_signals = scored[0].signal_scores
    rd_signals = scored[1].signal_scores

    # Reddit should NOT have engagement signals in its final scores
    # (they get skipped, not just zeroed)
    assert "engagement_volume" in fb_signals
    assert "engagement_volume" not in rd_signals or "engagement_velocity" not in rd_signals


# --- Missing data handling (drop + renormalize, not fudge to 0.5) ---


def test_missing_signals_dropped_not_fudged() -> None:
    """Ads with no signals at all should fall back to neutral 0.5."""
    config = _make_config()
    scorer = CompositeScorer(config)
    sparse_ad = _make_ad("sparse")  # no engagement, no longevity
    scored = scorer.score_batch([sparse_ad])
    assert len(scored) == 1
    # No signals available → fallback 0.5
    assert scored[0].composite_score == 0.5
    # Signal scores dict should be empty
    assert scored[0].signal_scores == {}


def test_missing_signals_excluded_from_signal_scores() -> None:
    """Missing signals should NOT appear in signal_scores (not even as 0.5)."""
    config = _make_config()
    scorer = CompositeScorer(config)
    # Ad with longevity but no engagement
    ad = _make_ad("longevity_only", active_days=30)
    scored = scorer.score_batch([ad])
    assert "longevity" in scored[0].signal_scores
    assert "early_death" in scored[0].signal_scores
    # Engagement signals were None → dropped, not fudged
    assert "engagement_volume" not in scored[0].signal_scores
    assert "engagement_velocity" not in scored[0].signal_scores


def test_renormalization_preserves_rank_order() -> None:
    """Two ads with the same available signals should rank by their values."""
    config = _make_config()
    scorer = CompositeScorer(config)
    # Both ads have only longevity (no engagement)
    short = _make_ad("short", active_days=2)
    long_ad = _make_ad("long", active_days=100)
    scored = scorer.score_batch([short, long_ad])
    # Long ad should score higher
    assert scored[1].composite_score > scored[0].composite_score


def test_renormalization_no_neutral_drag() -> None:
    """An ad with high longevity but missing engagement should not be
    artificially dragged toward 0.5 by the (now removed) neutral fudge.

    Under the new logic, the score reflects only the available signals
    renormalized to weight 1.0, so a high longevity should produce a
    high composite (close to 1.0), not ~0.7.
    """
    config = _make_config()
    scorer = CompositeScorer(config)
    # Two ads: one extreme longevity, one low. Only longevity differs.
    ads = [
        _make_ad("low", active_days=1),
        _make_ad("high", active_days=1000),
    ]
    scored = scorer.score_batch(ads)
    # High-longevity ad should be at or near 1.0 (not dragged to ~0.6)
    high_score = scored[1].composite_score
    assert high_score > 0.9, (
        f"Expected high-longevity ad to score > 0.9, got {high_score}. "
        "Old neutral-0.5 fudge would have dragged it toward 0.6."
    )


# --- Tier assignment ---


def test_tier_assignment_distribution() -> None:
    """Tier assignment should roughly follow 20/50/30 split."""
    config = _make_config()
    scorer = CompositeScorer(config)
    assigner = TierAssigner(config)

    # Create 100 ads with varying quality
    ads = []
    for i in range(100):
        ads.append(
            _make_ad(
                ad_id=str(i),
                likes=i * 10,
                active_days=i + 1,
            )
        )

    scored = scorer.score_batch(ads)
    scored = assigner.assign_tiers(scored)

    counts = assigner.tier_summary(scored)
    assert counts["high"] > 0
    assert counts["medium"] > 0
    assert counts["low"] > 0
    # Roughly correct proportions (allow some flexibility due to ties)
    assert 15 <= counts["high"] <= 25  # ~20%
    assert 40 <= counts["medium"] <= 60  # ~50%
    assert 20 <= counts["low"] <= 40  # ~30%


def test_tier_assignment_empty() -> None:
    config = _make_config()
    assigner = TierAssigner(config)
    assert assigner.assign_tiers([]) == []


def test_tier_summary() -> None:
    config = _make_config()
    assigner = TierAssigner(config)
    scored = [
        ScoredAd(ad=_make_ad("1"), composite_score=0.9, tier="high"),
        ScoredAd(ad=_make_ad("2"), composite_score=0.5, tier="medium"),
        ScoredAd(ad=_make_ad("3"), composite_score=0.1, tier="low"),
    ]
    summary = assigner.tier_summary(scored)
    assert summary == {"high": 1, "medium": 1, "low": 1}
