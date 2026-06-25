"""Unit tests for the single-pass smoke teacher prompts + parser."""

from __future__ import annotations

import json

from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.teacher.single_pass import (
    SINGLE_PASS_TEACHER_SYSTEM,
    build_single_pass_request,
    build_single_pass_user_message,
    parse_single_pass_response,
)

_MODEL_TASK = "Draft a Meta ad for our payroll-compliance tool aimed at HR leads."
_VALID_BRIEF: dict[str, object] = {
    "task": _MODEL_TASK,
    "product": {
        "name": "Compliantly",
        "tone_signals": ["clipped"],
    },
    "bridge": {
        "angle": "problem-aware skeptic",
        "buyer_pain": "compliance review blocks weekly hires",
    },
    "platform": "meta",
}


def _fake_response(brief: dict[str, object], think: str, deliverable: str) -> str:
    return f"<brief>\n{json.dumps(brief)}\n</brief>\n\n<think>\n{think}\n</think>\n\n{deliverable}"


def test_single_pass_request_shape(sample_source_ad: SourceAd) -> None:
    req = build_single_pass_request(sample_source_ad, model="claude-haiku-4-5")
    assert req.custom_id == f"teacher-{sample_source_ad.ad_id}"
    assert req.system == SINGLE_PASS_TEACHER_SYSTEM
    assert req.model == "claude-haiku-4-5"
    assert len(req.messages) == 1
    assert req.messages[0]["role"] == "user"
    assert sample_source_ad.ad_id in req.messages[0]["content"]


def test_single_pass_user_message_lists_copy_fields(sample_source_ad: SourceAd) -> None:
    msg = build_single_pass_user_message(sample_source_ad)
    assert sample_source_ad.headline in msg
    assert sample_source_ad.body in msg
    assert "platform_hint:" in msg


def test_parse_single_pass_response_happy_path(sample_source_ad: SourceAd) -> None:
    content = _fake_response(
        _VALID_BRIEF,
        think="I lead with the speed claim because the brief flags compliance friction.",
        deliverable=f"{sample_source_ad.headline}\n\n{sample_source_ad.body}",
    )
    result = parse_single_pass_response(content)
    assert result.brief is not None
    assert result.brief["task"] == _MODEL_TASK
    assert result.brief["platform"] == "meta"
    assert result.think and "speed claim" in result.think
    assert result.deliverable and sample_source_ad.headline in result.deliverable
    assert not result.errors


def test_parse_single_pass_response_flags_missing_task(sample_source_ad: SourceAd) -> None:
    brief_no_task = {k: v for k, v in _VALID_BRIEF.items() if k != "task"}
    content = _fake_response(
        brief_no_task,
        think="I lead with the speed claim because the brief flags compliance friction.",
        deliverable=f"{sample_source_ad.headline}\n\n{sample_source_ad.body}",
    )
    result = parse_single_pass_response(content)
    assert result.brief is not None
    assert "task" not in result.brief
    assert any("missing `task`" in e for e in result.errors)


def test_parse_single_pass_response_missing_brief() -> None:
    content = (
        "<think>\nthought process is decently long here to clear the min-char gate\n</think>\n\n"
        "the ad copy verbatim goes here"
    )
    result = parse_single_pass_response(content)
    assert result.brief is None
    assert any("missing <brief>" in e for e in result.errors)
    # think + deliverable still parse via response_parser.
    assert result.think is not None
    assert result.deliverable is not None


def test_parse_single_pass_response_malformed_brief_json() -> None:
    content = (
        "<brief>\n{not: valid json}\n</brief>\n\n"
        "<think>\nthought process is decently long here to clear the min-char gate\n</think>\n\n"
        "the ad copy verbatim goes here"
    )
    result = parse_single_pass_response(content)
    assert result.brief is None
    assert any("brief JSON parse" in e for e in result.errors)


def test_parse_single_pass_response_empty_content() -> None:
    result = parse_single_pass_response("")
    assert result.brief is None
    assert result.think is None
    assert result.deliverable is None
    assert result.errors == ["empty content"]


def test_single_pass_system_prompt_invariants() -> None:
    """The smoke prompt must stay aligned with production prompts in spirit."""
    s = SINGLE_PASS_TEACHER_SYSTEM
    # Three regions: brief, think, verbatim ad.
    assert "<brief>...</brief>" in s
    assert "<think>...</think>" in s
    assert "character-for-character" in s
    # Grounding contract clauses (shared with production brief extractor).
    assert "World knowledge about the brand is FORBIDDEN" in s
    assert "tone_signals" in s
    # Task is now emitted by the model, not injected.
    assert "``task`` (required string)" in s
    assert "Do NOT emit a ``task`` field" not in s
    # Deliverable wrap is natural prose, no hard "nothing after the ad" rule.
    assert "actually answering" in s
    assert "nothing after the verbatim ad" not in s.lower()
