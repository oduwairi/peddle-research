"""Unit tests for ``draper.construction_v2.teacher.rationale_prompt``."""

from __future__ import annotations

import re

from draper.construction_v2.config import RationaleConfig
from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.schemas.brief import Brief, canonical_json
from draper.construction_v2.teacher.rationale_prompt import (
    RATIONALE_TEACHER_SYSTEM,
    build_rationale_messages,
    build_rationale_request,
)


def test_build_rationale_messages_shape(sample_brief: Brief, sample_source_ad: SourceAd) -> None:
    """The user turn must carry the canonical brief JSON + verbatim ad copy."""
    messages = build_rationale_messages(sample_brief, sample_source_ad)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    # Brief is serialized canonically and embedded.
    assert canonical_json(sample_brief) in content
    # Ad copy fields are concatenated into the prompt.
    assert sample_source_ad.headline in content
    assert sample_source_ad.body in content
    # The user turn mentions the <think> tag in the instruction text but
    # must NOT carry a populated example block — the demonstration would
    # bias the teacher toward the specific reasoning we showed.
    assert "</think>\n\n" not in content
    assert "<think>\n" not in content


def test_build_rationale_request_shape(sample_brief: Brief, sample_source_ad: SourceAd) -> None:
    cfg = RationaleConfig(model="claude-haiku-4-5", max_tokens=600, temperature=0.4)
    req = build_rationale_request(sample_brief, sample_source_ad, cfg)
    assert req.custom_id == f"rationale-{sample_source_ad.ad_id}"
    assert req.system == RATIONALE_TEACHER_SYSTEM
    assert req.model == "claude-haiku-4-5"
    assert req.max_tokens == 600
    assert req.temperature == 0.4
    assert req.messages == build_rationale_messages(sample_brief, sample_source_ad)


def test_build_rationale_request_custom_id_safe_charset(
    sample_brief: Brief, sample_source_ad: SourceAd
) -> None:
    """custom_id must satisfy Anthropic's ``^[a-zA-Z0-9_-]{1,64}$``."""
    cfg = RationaleConfig()
    req = build_rationale_request(sample_brief, sample_source_ad, cfg)
    assert re.match(r"^[a-zA-Z0-9_-]{1,64}$", req.custom_id), req.custom_id


def test_rationale_teacher_system_invariants() -> None:
    """The system prompt encodes the locked v2 contract; guard the key clauses.

    A change here is intentional — bump the assertions to match — but a
    silent edit elsewhere shouldn't go unnoticed because the prompt is
    fine-tuning-byte-coupled.
    """
    s = RATIONALE_TEACHER_SYSTEM
    # Single structural anchor: <think> + verbatim ad. No third slot.
    assert "<think>...</think>" in s
    assert "<note>" not in s
    # Verbatim discipline (the cardinal rule borrowed from v1).
    assert "word-for-word" in s
    # No prose-fence wrapping of the ad.
    assert "no fences" in s.lower()
    # First-person decisional voice (the v2 lock from 2026-05-19).
    assert "first-person" in s.lower()
