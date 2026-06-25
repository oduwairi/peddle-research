"""Tests for ``draper.evaluation.judge.extract.clean_copy``.

The extractor normalizes raw model output before judging so single-shot
configs (which leak ``<think>`` blocks and markdown chrome) and the
frontend pipeline configs are scored on the same surface.
"""

from __future__ import annotations

import pytest

from draper.evaluation.judge.extract import clean_copy


def test_pure_copy_passes_through_unchanged() -> None:
    raw = (
        "Try Before You Buy\n\n"
        "Get ready for a flawless finish with IL MAKIAGE NY No Filter Primer."
    )
    assert clean_copy(raw) == raw


def test_strips_empty_think_block() -> None:
    raw = "<think>\n\n</think>\nTry Before You Buy\n\nFlawless finish."
    out = clean_copy(raw)
    assert "<think>" not in out
    assert out.startswith("Try Before You Buy")


def test_strips_nonempty_think_block() -> None:
    raw = (
        "<think>let me brainstorm three angles…</think>\n"
        "Headline: Glow All Day"
    )
    out = clean_copy(raw)
    assert "<think>" not in out
    assert "brainstorm" not in out
    assert "Glow All Day" in out


def test_strips_leading_ad_copy_header() -> None:
    raw = "**Ad Copy:**\n\n🌟 Unlock Flawless Skin with IL MAKIAGE 🌟"
    out = clean_copy(raw)
    assert not out.startswith("**Ad Copy")
    assert out.startswith("🌟 Unlock")


def test_strips_leading_copy_header_no_colon() -> None:
    raw = "**Copy**\n\nPay less, do more."
    out = clean_copy(raw)
    assert "**Copy**" not in out
    assert out.startswith("Pay less")


def test_keeps_inline_field_label() -> None:
    """Inline labels like ``Headline:`` on a single line stay intact —
    they're part of the platform-shaped output for some Google RSA copy.
    """
    raw = "**Headline:** Glow All Day Primer\n**Body:** No filter needed."
    out = clean_copy(raw)
    assert "Headline:" in out
    assert "Body:" in out


def test_strips_heres_an_preamble() -> None:
    raw = "Here's a strong ad execution:\n\nGlow All Day."
    out = clean_copy(raw)
    assert not out.lower().startswith("here")
    assert out.startswith("Glow All Day")


def test_strips_sure_preamble() -> None:
    raw = "Sure! Here's a Meta ad for the brief:\n\nFlawless finish."
    out = clean_copy(raw)
    assert "Sure!" not in out
    assert out.startswith("Flawless")


def test_strips_ill_write_preamble() -> None:
    raw = "I'll craft a Meta ad that hooks first-time skincare buyers:\n\nGlow All Day."
    out = clean_copy(raw)
    assert "craft" not in out
    assert out.startswith("Glow All Day")


def test_strips_below_is_preamble() -> None:
    raw = "Below is a short ad:\n\nFlawless skin in 30 seconds."
    out = clean_copy(raw)
    assert "Below" not in out
    assert out.startswith("Flawless")


def test_strips_combined_think_then_header() -> None:
    raw = (
        "<think>\n\n</think>\n"
        "**Ad Copy:**\n\n"
        "Here's the hook:\n\n"
        "Glow All Day."
    )
    out = clean_copy(raw)
    assert "<think>" not in out
    assert "**Ad Copy" not in out
    assert not out.lower().startswith("here")
    assert out.startswith("Glow All Day")


def test_strips_trailing_rationale_block() -> None:
    raw = (
        "Glow All Day Primer.\n"
        "Try it tonight.\n\n"
        "Why this works: it leads with the outcome and ends with a CTA."
    )
    out = clean_copy(raw)
    assert "Why this works" not in out
    assert "Try it tonight" in out


def test_strips_strategy_trailing_block() -> None:
    raw = (
        "Pay less, do more.\n\n"
        "Strategy: lead with savings, close with urgency."
    )
    out = clean_copy(raw)
    assert "Strategy:" not in out
    assert out.endswith("do more.")


def test_idempotent() -> None:
    raw = (
        "<think>x</think>\n"
        "**Ad Copy:**\n\n"
        "Glow All Day Primer.\n\n"
        "Why this works: outcome-led."
    )
    once = clean_copy(raw)
    twice = clean_copy(once)
    assert once == twice


def test_empty_string() -> None:
    assert clean_copy("") == ""


def test_only_think_block() -> None:
    assert clean_copy("<think>\n\n</think>") == ""


@pytest.mark.parametrize(
    "raw",
    [
        "Headline starting with the word here is fine.",
        "Here's a deal you'll love. Limited time only.",
    ],
)
def test_does_not_eat_inline_here_in_copy(raw: str) -> None:
    """Real ads sometimes start with 'Here's …' as a hook. We only strip
    when the preamble is followed by a blank line + new content — i.e.
    looks structurally like a preamble, not part of the copy itself.
    """
    out = clean_copy(raw)
    # The first sentence should still be present.
    assert "Here" in out or raw.startswith("Headline")
