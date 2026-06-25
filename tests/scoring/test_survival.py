"""Tests for the Kaplan-Meier survivability signal."""

from __future__ import annotations

from datetime import date, timedelta

from draper.scoring.survival import (
    CENSORING_WINDOW_DAYS,
    _cohort_horizon,
    _event_observed,
    compute_survivability,
)
from draper.scraping.schemas import AdSource, Platform, RawAd


def _ad(
    ad_id: str,
    *,
    active_days: int | None = None,
    first_seen: date | None = None,
    last_seen: date | None = None,
    platform: Platform = Platform.FACEBOOK,
) -> RawAd:
    return RawAd(
        ad_id=ad_id,
        source=AdSource.ADFLEX,
        platform=platform,
        active_days=active_days,
        first_seen=first_seen,
        last_seen=last_seen,
    )


def test_empty_input() -> None:
    assert compute_survivability([]) == []


def test_all_missing_longevity_returns_none() -> None:
    ads = [_ad("a"), _ad("b"), _ad("c")]  # no active_days, no dates
    result = compute_survivability(ads)
    assert result == [None, None, None]


def test_long_runner_scores_higher_than_short() -> None:
    ads = [_ad(f"short_{i}", active_days=1) for i in range(10)] + [
        _ad(f"long_{i}", active_days=100) for i in range(10)
    ]
    result = compute_survivability(ads)
    assert all(r is not None for r in result)
    short_scores = [r for r in result[:10] if r is not None]
    long_scores = [r for r in result[10:] if r is not None]
    assert max(short_scores) <= min(long_scores), (
        f"long-runners should outscore short-lived ads: shorts={short_scores}, longs={long_scores}"
    )


def test_score_range_within_unit_interval() -> None:
    ads = [_ad(str(i), active_days=i + 1) for i in range(20)]
    result = compute_survivability(ads)
    for r in result:
        assert r is not None
        assert 0.0 <= r <= 1.0


def test_skips_zero_or_negative_durations() -> None:
    ads = [
        _ad("zero", active_days=0),
        _ad("ok", active_days=10),
        _ad("ok2", active_days=20),
    ]
    result = compute_survivability(ads)
    assert result[0] is None  # zero duration → skipped
    assert result[1] is not None
    assert result[2] is not None


def test_per_platform_stratification_uses_global_fallback() -> None:
    """Sparse platforms (< MIN_PLATFORM_COHORT) fall back to the global KM."""
    facebook_ads = [_ad(f"fb_{i}", active_days=10 * (i + 1)) for i in range(25)]
    pinterest_ads = [
        _ad("pn_short", active_days=1, platform=Platform.PINTEREST),
        _ad("pn_long", active_days=100, platform=Platform.PINTEREST),
    ]
    ads = facebook_ads + pinterest_ads
    result = compute_survivability(ads)
    assert all(r is not None for r in result)
    # The two Pinterest ads should still rank correctly relative to each other
    pn_short_score = result[-2]
    pn_long_score = result[-1]
    assert pn_short_score is not None and pn_long_score is not None
    assert pn_long_score > pn_short_score


def test_single_valid_ad_returns_fallback() -> None:
    """One ad cannot fit a KM curve; the fallback returns 0.5."""
    result = compute_survivability([_ad("only", active_days=15)])
    assert result == [0.5]


def test_event_observed_with_recent_last_seen_is_censored() -> None:
    horizon = date(2026, 4, 10)
    ad = _ad("recent", last_seen=horizon - timedelta(days=2))
    assert _event_observed(ad, horizon) is False  # within censoring window


def test_event_observed_with_old_last_seen_is_observed() -> None:
    horizon = date(2026, 4, 10)
    ad = _ad("old", last_seen=horizon - timedelta(days=CENSORING_WINDOW_DAYS + 5))
    assert _event_observed(ad, horizon) is True


def test_event_observed_with_no_dates_assumes_observed() -> None:
    assert _event_observed(_ad("nodates"), None) is True


def test_cohort_horizon_picks_latest_last_seen() -> None:
    ads = [
        _ad("a", last_seen=date(2026, 1, 1)),
        _ad("b", last_seen=date(2026, 4, 1)),
        _ad("c", last_seen=date(2026, 2, 15)),
        _ad("d"),  # no last_seen
    ]
    assert _cohort_horizon(ads) == date(2026, 4, 1)


def test_cohort_horizon_none_when_no_dates() -> None:
    assert _cohort_horizon([_ad("a"), _ad("b")]) is None


def test_censoring_changes_score() -> None:
    """A long-running censored ad should still score high.

    The censored ad outlived its peers up to the last observation, so KM
    correctly assigns it a high 1 - S(d) value even though we don't know
    its final lifespan.
    """
    horizon = date(2026, 4, 10)
    ads = [
        _ad(
            f"dead_{i}",
            first_seen=date(2026, 1, 1),
            last_seen=date(2026, 1, 1) + timedelta(days=i + 1),
        )
        for i in range(20)
    ]
    censored_ad = _ad(
        "alive",
        first_seen=date(2026, 1, 1),
        last_seen=horizon - timedelta(days=1),  # within censoring window → censored
    )
    ads.append(censored_ad)
    result = compute_survivability(ads)
    censored_score = result[-1]
    assert censored_score is not None
    assert censored_score > 0.5, f"censored long-runner should score high, got {censored_score}"
