"""Leak guard: 5-gram bridge ↔ ad overlap rejection."""

from __future__ import annotations

import pytest

from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.ingest.leak_guard import check_bridge_leak
from draper.construction_v2.schemas.brief import Brief, BriefBridge, BriefProduct


def test_clean_bridge_passes(sample_brief: Brief, sample_source_ad: SourceAd) -> None:
    result = check_bridge_leak(sample_brief, sample_source_ad)
    assert result.passed
    assert result.reason == ""


def test_leak_rejects_when_bridge_quotes_ad(
    sample_source_ad: SourceAd,
) -> None:
    leaking = Brief(
        task="Write ad copy.",
        product=BriefProduct(name="x", description="y", tone_signals=["crisp"]),
        bridge=BriefBridge(
            positioning="legit ops",
            target_audience="ops people",
            # Pulls 5 contiguous words from the ad body.
            angle="weekly compliance reviews into a 72-hour background check pipeline",
            buyer_pain="generic pain",
        ),
        platform="meta",
    )
    result = check_bridge_leak(leaking, sample_source_ad)
    assert not result.passed
    assert result.reason.endswith("gram_leak")
    assert result.offending_field == "bridge.angle"


def test_leak_n_parameter_respected(sample_source_ad: SourceAd) -> None:
    """At n=3 the same brief should also leak; at n=12 nothing leaks."""
    leaking = Brief(
        task="Write ad copy.",
        product=BriefProduct(name="x", description="y", tone_signals=["crisp"]),
        bridge=BriefBridge(
            positioning="weekly compliance reviews",
            target_audience="ops people",
            angle="cool",
            buyer_pain="generic pain",
        ),
        platform="meta",
    )
    at_3 = check_bridge_leak(leaking, sample_source_ad, n=3)
    assert not at_3.passed
    at_12 = check_bridge_leak(leaking, sample_source_ad, n=12)
    assert at_12.passed


def test_empty_tone_signals_caught_by_schema() -> None:
    # The Brief schema validator catches the empty tone_signals case
    # before the leak guard ever runs — verify the schema enforcement.
    with pytest.raises(ValueError, match="tone_signals"):
        BriefProduct(name="x", description="y", tone_signals=[])
