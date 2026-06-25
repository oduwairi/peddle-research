"""Response parser: <think> + freeform deliverable extraction."""

from __future__ import annotations

from draper.construction_v2.ingest.response_parser import (
    MIN_DELIVERABLE_CHARS,
    MIN_THINK_CHARS,
    ParsedResponse,
    ParseRejection,
    parse_response,
)


def _long_think(suffix: str = "") -> str:
    """A think block comfortably above MIN_THINK_CHARS."""
    base = "I want to lead with the speed claim — compliance review blocks "
    return base + base + suffix


def _wrap(think: str, deliverable: str) -> str:
    """Build a canonical teacher response string."""
    return f"<think>\n{think}\n</think>\n\n{deliverable}"


def test_parse_response_happy_path() -> None:
    text = _wrap(_long_think(), "Hire fast. Stay compliant.")
    parsed = parse_response(text)
    assert isinstance(parsed, ParsedResponse)
    assert parsed.think.startswith("I want")
    assert parsed.deliverable == "Hire fast. Stay compliant."


def test_parse_response_multiline_deliverable() -> None:
    deliverable = "Hire fast.\n\nStay compliant.\n\nFree 14-day trial."
    text = _wrap(_long_think(), deliverable)
    parsed = parse_response(text)
    assert isinstance(parsed, ParsedResponse)
    assert parsed.deliverable == deliverable


def test_parse_response_missing_think() -> None:
    text = "Hire fast. Stay compliant."
    assert parse_response(text) == ParseRejection.MISSING_THINK


def test_parse_response_missing_deliverable() -> None:
    text = f"<think>{_long_think()}</think>"
    assert parse_response(text) == ParseRejection.MISSING_DELIVERABLE


def test_parse_response_deliverable_too_short() -> None:
    text = _wrap(_long_think(), "hi")
    assert parse_response(text) == ParseRejection.MISSING_DELIVERABLE


def test_parse_response_teacher_failed_refusal() -> None:
    text = "I'm sorry, but I can't help with that request."
    assert parse_response(text) == ParseRejection.TEACHER_FAILED


def test_parse_response_teacher_failed_sentinel() -> None:
    text = _wrap(_long_think(), "<EXTRACTION_FAILED>")
    assert parse_response(text) == ParseRejection.TEACHER_FAILED


def test_parse_response_think_too_short() -> None:
    text = _wrap("ok", "Hire fast. Stay compliant.")
    assert parse_response(text) == ParseRejection.THINK_TOO_SHORT


def test_parse_response_pre_think_noise() -> None:
    text = f"Preamble prose.\n\n<think>\n{_long_think()}\n</think>\n\nAd copy here."
    assert parse_response(text) == ParseRejection.PRE_THINK_NOISE


def test_parse_response_strips_code_fence_around_deliverable() -> None:
    deliverable_fenced = "```\nHire fast. Stay compliant.\n```"
    text = _wrap(_long_think(), deliverable_fenced)
    parsed = parse_response(text)
    assert isinstance(parsed, ParsedResponse)
    assert parsed.deliverable == "Hire fast. Stay compliant."


def test_parse_response_empty() -> None:
    assert parse_response("") == ParseRejection.MISSING_THINK
    assert parse_response("   \n   ") == ParseRejection.MISSING_THINK


def test_parsed_response_assistant_content_format() -> None:
    parsed = ParsedResponse(
        think="r" * MIN_THINK_CHARS,
        deliverable="Hire fast. Stay compliant.",
    )
    content = parsed.assistant_content
    assert content.startswith("<think>\n")
    assert "</think>" in content
    # Deliverable follows </think> with no wrapping tags.
    think_end = content.index("</think>")
    after_think = content[think_end + len("</think>") :]
    assert "Hire fast. Stay compliant." in after_think
    assert "<ad>" not in content
    assert "<deliverable>" not in content


def test_min_constants_are_positive() -> None:
    assert MIN_THINK_CHARS > 0
    assert MIN_DELIVERABLE_CHARS > 0
