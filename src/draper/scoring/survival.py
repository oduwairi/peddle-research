"""Survival-based longevity signal for ad scoring.

Replaces the v1 ``longevity`` + ``early_death`` pair with a single per-ad
``survivability`` signal that respects right-censoring (ads still running on
the last scrape have unknown final lifespans). Per-ad score is
``1 - S_platform(observed_duration)``, where ``S_platform`` is a Kaplan-Meier
survival curve fit from all ads on the same platform.

Long-running ads outlast their platform peers, so ``S(d)`` is small and the
score is close to 1.0. Short-lived ads die before most peers, so ``S(d)`` is
close to 1.0 and the score is close to 0.0. Censored observations contribute
to the at-risk denominator without inflating the death count, which is the
whole reason this replaces a raw ``days``-based percentile.

Per-platform stratification is used instead of Cox-with-platform-covariate
because (a) it produces meaningful per-ad variation from the observed duration
rather than collapsing to one value per platform, and (b) it does not require
the cohort to span multiple platforms before any signal can be produced.
"""

from __future__ import annotations

import math
from datetime import date

from lifelines import KaplanMeierFitter

from draper.scraping.schemas import RawAd

# Days from the cohort's max ``last_seen`` within which an ad is treated as
# still running (right-censored). Ads outside this window are treated as
# fully observed deaths.
CENSORING_WINDOW_DAYS = 7

# Minimum number of ads on a single platform before we trust its own KM fit.
# Below this threshold the platform's ads are scored against a global KM curve
# fit on all valid ads in the cohort.
MIN_PLATFORM_COHORT = 20

# Upper bound on observed ad lifespan, in days. Any source value beyond this
# is treated as a data error (corrupt timestamps, Unix-epoch artifacts) and
# clamped before scoring. 3650d = 10 years, comfortably past the oldest
# legitimate ad lifetime on any platform we scrape.
MAX_ACTIVE_DAYS = 3650


def compute_survivability(ads: list[RawAd]) -> list[float | None]:
    """Return a per-ad survivability score in ``[0, 1]``.

    Args:
        ads: A batch of ads. The KM curve(s) are fit on this cohort.

    Returns:
        A list aligned with ``ads``. Values close to 1.0 mean the ad outlived
        almost all of its platform peers; values close to 0.0 mean it died very
        early. ``None`` for ads with no usable longevity data.
    """
    if not ads:
        return []

    cohort_horizon = _cohort_horizon(ads)

    valid: list[tuple[int, float, int, str]] = []
    for i, ad in enumerate(ads):
        days = ad.longevity_days
        if days is None or days <= 0:
            continue
        d = float(days)
        # Defend against NaN/Inf and clamp implausibly long durations.
        # AdFlex occasionally serves marketplace listings with Unix-epoch
        # artifacts (active_days in the tens of thousands) that would
        # otherwise dominate the right tail of the KM curve.
        if not math.isfinite(d):
            continue
        if d > MAX_ACTIVE_DAYS:
            d = float(MAX_ACTIVE_DAYS)
        platform = ad.platform.value if ad.platform else "other"
        event_observed = int(_event_observed(ad, cohort_horizon))
        valid.append((i, d, event_observed, platform))

    result: list[float | None] = [None] * len(ads)
    if not valid:
        return result

    by_platform: dict[str, list[tuple[int, float, int]]] = {}
    for orig_idx, dur, event, platform in valid:
        by_platform.setdefault(platform, []).append((orig_idx, dur, event))

    global_km = _fit_km(
        durations=[v[1] for v in valid],
        events=[v[2] for v in valid],
    )

    for _platform, rows in by_platform.items():
        if len(rows) >= MIN_PLATFORM_COHORT:
            km = _fit_km(
                durations=[r[1] for r in rows],
                events=[r[2] for r in rows],
            )
        else:
            km = global_km

        for orig_idx, dur, _ in rows:
            result[orig_idx] = _score_from_km(km, dur)

    return result


def _fit_km(durations: list[float], events: list[int]) -> KaplanMeierFitter | None:
    """Fit a Kaplan-Meier curve, returning ``None`` on failure or tiny inputs."""
    if len(durations) < 2:
        return None
    try:
        km = KaplanMeierFitter()
        km.fit(durations=durations, event_observed=events)
    except (ValueError, ZeroDivisionError):
        return None
    return km


def _score_from_km(km: KaplanMeierFitter | None, duration: float) -> float:
    """Compute ``1 - S(duration)`` against a fitted KM curve.

    Falls back to 0.5 (uninformative) when no fit is available.
    """
    if km is None:
        return 0.5
    sf = km.survival_function_at_times([duration])
    s_val = float(sf.iloc[0])
    return max(0.0, min(1.0, 1.0 - s_val))


def _cohort_horizon(ads: list[RawAd]) -> date | None:
    """Return the latest ``last_seen`` date in the cohort, or ``None``."""
    last_seens = [ad.last_seen for ad in ads if ad.last_seen is not None]
    return max(last_seens) if last_seens else None


def _event_observed(ad: RawAd, cohort_horizon: date | None) -> bool:
    """Return ``True`` if the ad's death has been observed.

    An ad is right-censored (``False``) when its ``last_seen`` falls within
    ``CENSORING_WINDOW_DAYS`` of the cohort's max ``last_seen`` — we cannot
    yet tell whether it has actually stopped running. Ads with no date
    information are conservatively treated as observed.
    """
    if ad.last_seen is None or cohort_horizon is None:
        return True
    return (cohort_horizon - ad.last_seen).days > CENSORING_WINDOW_DAYS
