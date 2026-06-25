"""Fidelity + grounding checks."""

from __future__ import annotations

from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.ingest.fidelity import (
    check_deliverable_fidelity,
    check_think_grounding,
)
from draper.construction_v2.schemas.brief import Brief


def test_fidelity_passes_on_verbatim(sample_source_ad: SourceAd) -> None:
    verbatim = sample_source_ad.ad_copy_text
    result = check_deliverable_fidelity(verbatim, sample_source_ad)
    assert result.passed
    assert result.coverage >= 0.6


def test_fidelity_fails_on_paraphrase(sample_source_ad: SourceAd) -> None:
    paraphrase = (
        "We help startups onboard new hires quickly with a fast compliance workflow service."
    )
    result = check_deliverable_fidelity(paraphrase, sample_source_ad)
    assert not result.passed
    assert "word_coverage" in result.reason or "signature" in result.reason


def test_fidelity_short_ad_auto_pass() -> None:
    short_ad = SourceAd(
        ad_id="x",
        platform="meta",
        composite_score=0.5,
        headline="Try it.",
        body="",
        description="",
        cta="",
        raw={},
    )
    result = check_deliverable_fidelity("anything goes", short_ad)
    assert result.passed
    assert result.reason == "short_ad_skip"


def test_fidelity_fails_when_signature_missing(
    sample_source_ad: SourceAd,
) -> None:
    # All content words present but not contiguous → signature fails.
    scrambled = (
        "service. Compliantly review reviews fast turns into weekly "
        "compliance hire 72-hour pipeline check background ops a teams"
    )
    result = check_deliverable_fidelity(scrambled, sample_source_ad)
    # Coverage may pass; signature must fail.
    assert not result.passed or not result.signature_passed


def test_grounding_passes_when_both_referenced(sample_brief: Brief) -> None:
    think = (
        "The 72-hour turnaround claim in the brief lets me lead with "
        "speed; the audience of Series-A operations leads explains why "
        "I keep it crisp and outcome-led."
    )
    result = check_think_grounding(think, sample_brief)
    assert result.passed


def test_grounding_passes_on_bridge_only(sample_brief: Brief) -> None:
    # Product-fact requirement was dropped (2026-05-22); bridge anchor
    # is the only required ground. Strategically-grounded think that
    # paraphrases the product name must still pass.
    think = (
        "I focused on the aspirational founder identity angle because "
        "Series-A ops leads respond to identity-led copy."
    )
    result = check_think_grounding(think, sample_brief)
    assert result.passed


def test_grounding_fails_without_bridge_field(sample_brief: Brief) -> None:
    think = (
        "Compliantly's background-checks-as-a-service positioning is "
        "front and center. The SOC 2 audit trail is the proof point."
    )
    result = check_think_grounding(think, sample_brief)
    assert not result.passed
    assert "no_bridge_field_ref" in result.reason
