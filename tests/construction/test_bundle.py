"""Tests for the copywriting bundle builder + parser."""

from __future__ import annotations

from draper.construction.bundle import (
    ASSISTANT_RESPONSE_CLOSE,
    ASSISTANT_RESPONSE_OPEN,
    USER_PROMPT_CLOSE,
    USER_PROMPT_OPEN,
    BundleContext,
    build_bundle,
    parse_bundle_output,
)
from draper.construction.formats.copywriting.dice import CopywritingContext
from draper.construction.personas import Persona, PersonaLibrary
from draper.construction.schemas import PromptStyle, TaskFormat


def _make_persona() -> Persona:
    return Persona(
        id="smb_founder_first_ads",
        role="SMB founder, first time running ads",
        tone="casual",
        sophistication="low",
        budget="small",
        industry="online store",
    )


def _make_ctx(
    with_context: bool = True,
    formatted_ads: str = "Ad 1: sample ad content",
) -> BundleContext:
    ctx = BundleContext(
        task_format=TaskFormat.COPYWRITING,
        style=PromptStyle.BACKTRANSLATION,
        persona=_make_persona(),
        seed_idx=-1,
        seed_text="",
        evol_op=None,
        source_ads=[],
        formatted_ads=formatted_ads,
        response_format="",
        difficulty="standard",
        provider="claude",
    )
    if with_context:
        ctx.copywriting_context = CopywritingContext(
            source_ad_shape="has_body",
        )
    return ctx


class TestBundleBuilder:
    def test_includes_style_rules(self) -> None:
        bundle = build_bundle(_make_ctx())
        assert "copywriting" in bundle.lower()
        assert "User prompt" in bundle
        assert "Assistant response" in bundle

    def test_source_ad_labeled_as_gold_target(self) -> None:
        bundle = build_bundle(_make_ctx())
        assert "Source ad" in bundle
        assert "gold target" in bundle
        assert "Ad 1: sample ad content" in bundle

    def test_requires_all_tags(self) -> None:
        bundle = build_bundle(_make_ctx())
        for tag in (
            USER_PROMPT_OPEN,
            USER_PROMPT_CLOSE,
            ASSISTANT_RESPONSE_OPEN,
            ASSISTANT_RESPONSE_CLOSE,
        ):
            assert tag in bundle
        # Self-rating and multi-turn tags must not appear in the bundle.
        assert "<self_rating>" not in bundle
        assert "<user_followup>" not in bundle
        assert "<assistant_response_2>" not in bundle

    def test_no_copywriting_context_emits_no_directive(self) -> None:
        bundle = build_bundle(_make_ctx(with_context=False))
        assert USER_PROMPT_OPEN in bundle


class TestBundleParser:
    def test_roundtrip_extracts_prompt_and_response(self) -> None:
        response = (
            f"{USER_PROMPT_OPEN}\nHow do I advertise?\n{USER_PROMPT_CLOSE}\n"
            f"{ASSISTANT_RESPONSE_OPEN}\nUse multiple channels.\n"
            f"{ASSISTANT_RESPONSE_CLOSE}"
        )
        parsed = parse_bundle_output(response)
        assert parsed.user_prompt == "How do I advertise?"
        assert parsed.assistant_response == "Use multiple channels."

    def test_missing_tags_return_empty_strings(self) -> None:
        parsed = parse_bundle_output("just plain text, no tags")
        assert parsed.user_prompt == ""
        assert parsed.assistant_response == ""


class TestPersonas:
    def test_persona_library_loads(self) -> None:
        lib = PersonaLibrary.from_yaml("configs/personas.yaml")
        assert len(lib) >= 15  # minimum persona count per plan
        assert lib.by_id("smb_founder_first_ads") is not None
