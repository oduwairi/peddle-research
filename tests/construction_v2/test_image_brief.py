"""Tests for the image-brief skill modules (Phase 4)."""

from __future__ import annotations

import json

import pytest

from draper.construction_v2.ingest.image_brief_fidelity import (
    MIN_CAPTION_OVERLAP,
    check_image_brief_brief_alignment,
    check_image_brief_content_bridge,
    check_image_brief_fidelity,
)
from draper.construction_v2.ingest.skills import get_bundle
from draper.construction_v2.schemas.image_brief import (
    ImageBrief,
    ImageBriefInput,
    aspect_ratio_for_platform,
    canonical_image_brief_input_json,
    canonical_image_brief_json,
)
from draper.construction_v2.teacher.image_brief_single_pass import (
    CAPTION_RAW_KEY,
    build_image_brief_user_message,
    parse_image_brief_response,
)


# Stand-in for SourceAd to drive the fidelity gate without pulling in the
# full Pydantic model. The gate only reads `.raw` (a dict).
class _FakeAd:
    def __init__(self, raw: dict[str, object]) -> None:
        self.raw = raw
        self.ad_id = "test-ad-0"
        self.platform = "meta"
        self.headline = "Hero headline"
        self.body = "Body copy."
        self.description = "Desc."
        self.cta = "Shop now"

    @property
    def ad_copy_text(self) -> str:
        return self.headline + " " + self.body


# ---------------------------------------------------------------------------
# Schema — ImageBrief (the parsed deliverable view)
# ---------------------------------------------------------------------------


class TestImageBriefSchema:
    def _ok_dict(self) -> dict[str, object]:
        return {
            "brief": (
                "Create an overhead product shot of the bottle centered on a "
                "warm wooden desk, founder hands holding it in natural morning "
                "light. Avoid: text overlay; busy background"
            ),
            "negative": ["text overlay", "busy background"],
        }

    def test_round_trip(self) -> None:
        ib = ImageBrief(**self._ok_dict())
        s = canonical_image_brief_json(ib)
        ib2 = ImageBrief(**json.loads(s))
        assert ib == ib2

    def test_brief_must_be_non_empty(self) -> None:
        bad = self._ok_dict()
        bad["brief"] = "   "
        with pytest.raises(ValueError, match="non-empty"):
            ImageBrief(**bad)

    def test_negative_defaults_empty(self) -> None:
        d = self._ok_dict()
        d.pop("negative")
        ib = ImageBrief(**d)
        assert ib.negative == []

    def test_extra_fields_forbidden(self) -> None:
        d = self._ok_dict()
        d["bogus"] = "x"
        with pytest.raises(ValueError):
            ImageBrief(**d)


# ---------------------------------------------------------------------------
# Aspect ratio derivation
# ---------------------------------------------------------------------------


class TestAspectRatioForPlatform:
    def test_known_platforms(self) -> None:
        assert aspect_ratio_for_platform("meta") == "square"
        assert aspect_ratio_for_platform("tiktok") == "portrait"
        assert aspect_ratio_for_platform("x") == "landscape"
        assert aspect_ratio_for_platform("pinterest") == "portrait"
        assert aspect_ratio_for_platform("reddit") == "square"

    def test_unknown_platform_defaults_square(self) -> None:
        assert aspect_ratio_for_platform("nonexistent") == "square"


# ---------------------------------------------------------------------------
# Teacher parser
# ---------------------------------------------------------------------------


class TestImageBriefTeacherParse:
    def _content(self, image_brief_body: str) -> str:
        return (
            "<brief>"
            + json.dumps(
                {
                    "task": "Give me the image brief for our Meta ad below.",
                    "objective": "awareness",
                    "product": {"tone_signals": ["warm-but-direct"]},
                    "creative": {
                        "brand_guidelines": (
                            "Calm, premium founder-brand feel; soft natural-light "
                            "product photography; understated serif type."
                        ),
                        "on_creative_text": [],
                        "key_facts": [],
                    },
                }
            )
            + "</brief>\n"
            "<think>I'm leaning on the brand's calm, natural-light photography "
            "feel; the bottle in founder hands is the hero so the premium "
            "register reads at a glance.</think>\n"
            "<image_brief>" + image_brief_body + "</image_brief>"
        )

    def test_extracts_all_three_regions(self) -> None:
        body = (
            "Create an overhead product shot of the bottle centered on a warm "
            "wooden desk, founder hands holding it in natural morning light.\n"
            "Avoid: x; y"
        )
        result = parse_image_brief_response(self._content(body))
        # A well-behaved teacher omits the injected `platform` field, so the
        # baseline fixture parses cleanly. Stray-field stripping is covered by
        # test_parser_drops_stray_injected_fields.
        assert result.errors == []
        assert result.brief is not None
        assert result.brief["task"].startswith("Give me the image brief")
        assert result.brief["objective"] == "awareness"
        assert "brand_guidelines" in result.brief["creative"]
        # The teacher must NOT author copy or canvas — none injected at parse time.
        assert "ad_copy" not in result.brief
        assert "copy" not in result.brief
        assert "aspect_ratio" not in result.brief
        assert "orientation" not in result.brief["creative"]
        assert result.think is not None
        assert "photography" in result.think
        assert result.exclusions == ["x", "y"]
        assert result.deliverable is not None
        assert "<image_brief>" in result.deliverable

    def test_prose_deliverable_extracted_verbatim(self) -> None:
        body = (
            "Create a vertical split-collage product shot of an iced boba milk "
            'tea. Center a small white label reading "50% less sugar" across the '
            "seam.\nAvoid: real cafe brand logos; human hands"
        )
        result = parse_image_brief_response(self._content(body))
        assert result.deliverable is not None
        # The text between the <image_brief> tags survives character-for-character.
        assert f"<image_brief>{body}</image_brief>" in result.deliverable

    def test_exclusions_parsed_from_avoid_line(self) -> None:
        body = (
            "Create a hero shot of the product on a clean field.\n"
            "Avoid: real brand logos; human hands; cluttered background"
        )
        result = parse_image_brief_response(self._content(body))
        assert result.exclusions == [
            "real brand logos",
            "human hands",
            "cluttered background",
        ]

    def test_no_avoid_line_yields_empty_exclusions(self) -> None:
        body = "Create a hero shot of the product on a clean pastel field."
        result = parse_image_brief_response(self._content(body))
        # Baseline fixture parses cleanly; no Avoid line means no exclusions.
        assert result.errors == []
        assert result.exclusions == []

    def test_missing_image_brief_region_reports_error(self) -> None:
        # Valid <brief> + valid <think> length, but trailing prose has no
        # <image_brief> region. Should pass response_parser and surface
        # the missing-region error from the image-brief stage.
        long_think = (
            "Anchoring the image to the calm premium photography feel, since "
            "the brand register is understated, calls for a hero composition."
        )
        long_deliverable = (
            "The campaign goes Meta, hero shot, calm register — but I never "
            "produced the structured brief tag here."
        )
        bad = (
            "<brief>"
            + json.dumps(
                {
                    "task": "x",
                    "objective": "awareness",
                    "product": {"tone_signals": ["warm"]},
                    "creative": {"brand_guidelines": "calm premium photography feel"},
                    "platform": "meta",
                }
            )
            + f"</brief>\n<think>{long_think}</think>\n{long_deliverable}"
        )
        result = parse_image_brief_response(bad)
        assert result.exclusions == []
        assert any("missing <image_brief>" in e for e in result.errors)

    def test_parser_flags_missing_brand_guidelines(self) -> None:
        body = "Create a hero shot.\nAvoid: clutter"
        content = (
            "<brief>"
            + json.dumps(
                {
                    "task": "Give me the image brief.",
                    "objective": "conversion",
                    "product": {"tone_signals": ["warm"]},
                    "creative": {},  # creative present but no brand_guidelines
                    "platform": "meta",
                }
            )
            + "</brief>\n<think>calm hero composition reasoning here for length.</think>\n"
            "<image_brief>" + body + "</image_brief>"
        )
        result = parse_image_brief_response(content)
        assert any("brand_guidelines" in e for e in result.errors)

    def test_parser_flags_missing_creative_object(self) -> None:
        body = "Create a hero shot.\nAvoid: clutter"
        content = (
            "<brief>"
            + json.dumps(
                {
                    "task": "Give me the image brief.",
                    "objective": "conversion",
                    "product": {"tone_signals": ["warm"]},
                    "platform": "meta",
                }
            )
            + "</brief>\n<think>calm hero composition reasoning here for length.</think>\n"
            "<image_brief>" + body + "</image_brief>"
        )
        result = parse_image_brief_response(content)
        assert any("missing `creative` object" in e for e in result.errors)

    def test_parser_flags_missing_or_invalid_objective(self) -> None:
        body = "Create a hero shot.\nAvoid: clutter"
        content = (
            "<brief>"
            + json.dumps(
                {
                    "task": "Give me the image brief.",
                    "objective": "not_a_real_objective",
                    "product": {"tone_signals": ["warm"]},
                    "creative": {"brand_guidelines": "clean modern startup aesthetic"},
                    "platform": "meta",
                }
            )
            + "</brief>\n<think>clean modern reasoning here for length.</think>\n"
            "<image_brief>" + body + "</image_brief>"
        )
        result = parse_image_brief_response(content)
        assert any("objective" in e for e in result.errors)

    def test_parser_drops_stray_injected_fields(self) -> None:
        content = (
            "<brief>"
            + json.dumps(
                {
                    "task": "Give me the image brief.",
                    "objective": "promo_offer",
                    "ad_copy": "teacher should not author this",
                    "aspect_ratio": "portrait",
                    "product": {"tone_signals": ["warm"]},
                    "creative": {
                        "brand_guidelines": "bold graphic promo feel",
                        "orientation": "portrait",  # injected — must be dropped
                    },
                    "platform": "meta",
                }
            )
            + "</brief>\n<think>bold graphic reasoning here for length.</think>\n"
            "<image_brief>Create a hero shot.</image_brief>"
        )
        result = parse_image_brief_response(content)
        assert result.brief is not None
        assert "ad_copy" not in result.brief
        assert "aspect_ratio" not in result.brief
        assert "platform" not in result.brief
        assert "orientation" not in result.brief["creative"]
        assert any("ad_copy" in e for e in result.errors)
        assert any("aspect_ratio" in e for e in result.errors)
        assert any("platform" in e for e in result.errors)
        assert any("creative.orientation" in e for e in result.errors)


class TestImageBriefUserMessage:
    def test_includes_caption_copy_and_canvas(self) -> None:
        ad = _FakeAd(raw={"advertiser_name": "Acme", "landing_page_url": "https://acme.test/"})
        caption = "A black greyhound in a colorful crocheted sweater on artificial grass."
        msg = build_image_brief_user_message(ad, caption=caption)
        assert "Creative description" in msg
        assert "greyhound" in msg
        # The finished copy is now shown so the visual pairs with it.
        assert "Finished ad copy" in msg
        assert "Hero headline" in msg  # _FakeAd.headline → Meta "Primary text"
        # The platform-derived canvas is shown so <think> can compose for it.
        assert "Target canvas" in msg
        assert "square" in msg  # _FakeAd.platform == "meta" → square

    def test_canvas_normalizes_raw_platform_alias(self) -> None:
        """A raw platform alias must be normalized before the aspect lookup.

        ``platform_group_for("twitter")`` is the X group, whose canvas is
        landscape. Without normalization, ``aspect_ratio_for_platform`` sees the
        unknown literal "twitter" and wrongly defaults to "square" — diverging
        from the ``creative.orientation`` injected at ingest. Locks in the
        divergent-twin fix at the canvas call site.
        """
        ad = _FakeAd(raw={"advertiser_name": "Acme", "landing_page_url": "https://acme.test/"})
        ad.platform = "twitter"
        msg = build_image_brief_user_message(ad, caption="A neon X logo on a billboard at dusk.")
        assert "aspect_ratio: landscape" in msg
        assert "aspect_ratio: square" not in msg

    def test_rejects_empty_caption(self) -> None:
        ad = _FakeAd(raw={})
        with pytest.raises(ValueError, match="non-empty VLM caption"):
            build_image_brief_user_message(ad, caption="   ")


# ---------------------------------------------------------------------------
# Fidelity gate
# ---------------------------------------------------------------------------


class TestImageBriefFidelity:
    def _valid_ib_block(self, *, prose: str) -> str:
        return f"<image_brief>{prose}</image_brief>"

    def test_rejects_missing_block(self) -> None:
        r = check_image_brief_fidelity("no block here", _FakeAd(raw={}))
        assert not r.passed
        assert r.reason == "image_brief_missing_or_empty"

    def test_rejects_when_no_caption_present(self) -> None:
        """At ingest, missing caption is a pipeline bug — not benign.

        Selection gates on caption availability and submit's prepare hook
        enriches ads with captions before request build, so an ingested
        response without a caption indicates upstream drift.
        """
        block = self._valid_ib_block(
            prose="Create a founder-hands hero shot of the bottle over a wooden desk."
        )
        r = check_image_brief_fidelity(block, _FakeAd(raw={}))
        assert not r.passed
        assert r.reason == "image_brief_missing_caption"
        assert r.signature_passed is True

    def test_rejects_low_caption_overlap(self) -> None:
        # Prose content words share little with the caption.
        block = self._valid_ib_block(
            prose="Create an abstract neon swirl over geometric floating tiles."
        )
        ad = _FakeAd(
            raw={
                CAPTION_RAW_KEY: (
                    "smiling person holding a green smoothie bottle in a sunlit kitchen"
                )
            }
        )
        r = check_image_brief_fidelity(block, ad)
        assert not r.passed
        assert r.reason == "image_brief_low_caption_overlap"
        assert r.coverage < MIN_CAPTION_OVERLAP

    def test_passes_with_aligned_caption(self) -> None:
        # Caption + prose overlap is well above 30%.
        block = self._valid_ib_block(
            prose=(
                "Create a shot of founder hands holding the bottle over a wooden "
                "desk in warm morning light."
            )
        )
        ad = _FakeAd(
            raw={CAPTION_RAW_KEY: ("founder hands holding bottle over wooden desk in warm light")}
        )
        r = check_image_brief_fidelity(block, ad)
        assert r.passed
        assert r.coverage >= MIN_CAPTION_OVERLAP


# ---------------------------------------------------------------------------
# Skill bundle registration
# ---------------------------------------------------------------------------


class TestImageBriefBundleRegistration:
    def test_bundle_registered_with_no_labels_or_leak(self) -> None:
        bundle = get_bundle("image_brief")
        assert bundle.name == "image_brief"
        # Image briefs have no platform-native field labels.
        assert bundle.labels is None
        # The copy is a legitimate verbatim brief input — no leak guard.
        assert bundle.leak is None
        # Fidelity + grounding + build_brief are wired.
        assert bundle.fidelity is not None
        assert bundle.grounding is not None
        assert callable(bundle.build_brief)
        # The factual content bridge is verified for this skill.
        assert bundle.content_bridge is not None


# ---------------------------------------------------------------------------
# ImageBriefInput schema (the brief the writer conditions on)
# ---------------------------------------------------------------------------


class TestImageBriefInputSchema:
    def _ok(self) -> dict[str, object]:
        return {
            "task": "Give me the image brief for the reddit ad below.",
            "objective": "launch",
            "product": {
                "name": "Clozemaster",
                "category": "language-learning game",
                "tone_signals": ["bold", "playful"],
            },
            "creative": {
                "orientation": "square",
                "brand_guidelines": (
                    "Bold, graphic, retro-game brand feel; flat high-contrast "
                    "illustration; chunky pixel-friendly display type."
                ),
                "on_creative_text": ["GET FLUENT FASTER", "PLAY CLOZEMASTER"],
                "key_facts": ["a retro arcade title screen"],
            },
            "ad_copy": "**Headline:** Get fluent faster\n\n**CTA:** Install",
            "platform": "reddit",
        }

    def test_round_trip_byte_stable(self) -> None:
        ib = ImageBriefInput.model_validate(self._ok())
        s = canonical_image_brief_input_json(ib)
        ib2 = ImageBriefInput.model_validate(json.loads(s))
        assert ib == ib2
        # Re-serializing the reparsed model is byte-identical (sorted-key
        # canonical form), the property the frontend twin must match.
        assert canonical_image_brief_input_json(ib2) == s

    def test_ad_copy_required_non_empty(self) -> None:
        bad = self._ok()
        bad["ad_copy"] = "   "
        with pytest.raises(ValueError, match="ad_copy"):
            ImageBriefInput.model_validate(bad)

    def test_brand_guidelines_required_non_empty(self) -> None:
        bad = self._ok()
        bad["creative"]["brand_guidelines"] = "  "  # type: ignore[index]
        with pytest.raises(ValueError, match="brand_guidelines"):
            ImageBriefInput.model_validate(bad)

    def test_content_bridge_lists_default_empty(self) -> None:
        d = self._ok()
        d["creative"].pop("on_creative_text", None)  # type: ignore[attr-defined]
        d["creative"].pop("key_facts", None)  # type: ignore[attr-defined]
        ib = ImageBriefInput.model_validate(d)
        assert ib.creative.on_creative_text == []
        assert ib.creative.key_facts == []

    def test_content_bridge_coerces_null_and_strips_blanks(self) -> None:
        d = self._ok()
        d["creative"]["on_creative_text"] = None  # type: ignore[index]
        d["creative"]["key_facts"] = ["  ", "START NOW", ""]  # type: ignore[index]
        ib = ImageBriefInput.model_validate(d)
        assert ib.creative.on_creative_text == []
        assert ib.creative.key_facts == ["START NOW"]

    def test_objective_must_be_in_enum(self) -> None:
        bad = self._ok()
        bad["objective"] = "viral"
        with pytest.raises(ValueError):
            ImageBriefInput.model_validate(bad)

    def test_orientation_must_be_in_enum(self) -> None:
        bad = self._ok()
        bad["creative"]["orientation"] = "16:9"  # type: ignore[index]
        with pytest.raises(ValueError):
            ImageBriefInput.model_validate(bad)

    def test_unknown_top_level_field_rejected(self) -> None:
        # extra="forbid": stray top-level keys (e.g. the old creative_direction)
        # are rejected.
        bad = self._ok()
        bad["creative_direction"] = {"visual_angle": "a"}
        with pytest.raises(ValueError):
            ImageBriefInput.model_validate(bad)

    def test_unknown_creative_field_rejected(self) -> None:
        # CreativeDirection is also extra="forbid".
        bad = self._ok()
        bad["creative"]["bogus"] = "x"  # type: ignore[index]
        with pytest.raises(ValueError):
            ImageBriefInput.model_validate(bad)


# ---------------------------------------------------------------------------
# Grounding gate — keys on brand_guidelines (the bridging strategic field)
# ---------------------------------------------------------------------------


class TestImageBriefGrounding:
    def _brief(self) -> ImageBriefInput:
        return ImageBriefInput.model_validate(
            {
                "task": "Give me the image brief.",
                "objective": "awareness",
                "product": {"tone_signals": ["bold"]},
                "creative": {
                    "orientation": "square",
                    "brand_guidelines": (
                        "Bright, playful, appetizing brand feel; high-key studio "
                        "product photography; flat modern sans-serif type."
                    ),
                },
                "ad_copy": "**Headline:** Get fluent faster",
                "platform": "reddit",
            }
        )

    def test_passes_when_think_references_brand_guidelines(self) -> None:
        think = "I'll shoot this as bright high-key studio product photography, hero the box."
        r = check_image_brief_brief_alignment(think, self._brief())
        assert r.passed
        assert r.bridge_match  # carries the matched brand_guidelines token

    def test_fails_when_think_ignores_brand_guidelines(self) -> None:
        think = "A calm minimalist composition with lots of soft negative space."
        r = check_image_brief_brief_alignment(think, self._brief())
        assert not r.passed
        assert r.reason == "no_brand_guidelines_ref"


# ---------------------------------------------------------------------------
# build_brief — injects the verbatim labeled copy + platform-derived canvas
# ---------------------------------------------------------------------------


class TestImageBriefBuildBrief:
    def _teacher_brief(self) -> dict[str, object]:
        """The brief shape the teacher actually emits (pre-injection).

        The teacher authors ``task`` + ``objective`` + ``product`` + the
        ``creative`` block's authored keys. ``ad_copy``, ``creative.orientation``,
        and ``platform`` are injected from the source ad.
        """
        return {
            "task": "Give me the image brief for the meta ad below.",
            "objective": "conversion",
            "product": {"tone_signals": ["warm-but-direct"]},
            "creative": {
                "brand_guidelines": "Calm premium feel; soft natural-light product photography.",
                "on_creative_text": [],
                "key_facts": [],
            },
        }

    def test_injects_copy_canvas_and_platform(self) -> None:
        bundle = get_bundle("image_brief")
        ad = _FakeAd(raw={"advertiser_name": "Acme"})
        brief = bundle.build_brief(self._teacher_brief(), ad)
        # The injected copy is the platform-labeled render of the source ad
        # (Meta maps headline → "Primary text").
        assert "Hero headline" in brief.ad_copy
        assert "Primary text" in brief.ad_copy
        # The canvas + platform are injected from the source ad (meta → square).
        assert brief.creative.orientation == "square"
        assert brief.platform == "meta"

    def test_hoists_stray_top_level_visual_fields(self) -> None:
        # A teacher that places brand_guidelines / content bridges at the top
        # level (drift) has them hoisted into the creative block.
        bundle = get_bundle("image_brief")
        ad = _FakeAd(raw={})
        brief_dict = {
            "task": "Give me the image brief for the meta ad below.",
            "objective": "conversion",
            "product": {"tone_signals": ["warm"]},
            "brand_guidelines": "Calm premium feel.",
            "on_creative_text": ["SAVE 20%"],
            "key_facts": ["a hero bottle"],
        }
        brief = bundle.build_brief(brief_dict, ad)
        assert brief.creative.brand_guidelines == "Calm premium feel."
        assert brief.creative.on_creative_text == ["SAVE 20%"]
        assert brief.creative.key_facts == ["a hero bottle"]
        assert brief.creative.orientation == "square"

    def test_overwrites_any_teacher_authored_known_facts(self) -> None:
        bundle = get_bundle("image_brief")
        ad = _FakeAd(raw={})
        teacher = self._teacher_brief()
        # Teacher strays: stale copy/platform at top level + a wrong orientation
        # nested in creative — all must be overwritten by the injected facts.
        teacher["creative"]["orientation"] = "portrait"  # type: ignore[index]
        brief_dict = {
            **teacher,
            "ad_copy": "STALE teacher copy that must be overwritten",
            "aspect_ratio": "portrait",  # wrong — meta should resolve to square
            "platform": "tiktok",  # wrong — must be overwritten with ad.platform
        }
        brief = bundle.build_brief(brief_dict, ad)
        assert "STALE" not in brief.ad_copy
        assert "Hero headline" in brief.ad_copy
        assert brief.creative.orientation == "square"
        assert brief.platform == "meta"


# ---------------------------------------------------------------------------
# Content-bridge gate — factuality + over/under-report consistency
# ---------------------------------------------------------------------------


class TestImageBriefContentBridge:
    def _brief(
        self,
        *,
        on_text: list[str] | None = None,
        key_facts: list[str] | None = None,
        ad_copy: str = "**Headline:** Get fluent faster",
    ) -> ImageBriefInput:
        return ImageBriefInput.model_validate(
            {
                "task": "Give me the image brief.",
                "objective": "conversion",
                "product": {"tone_signals": ["bold"]},
                "creative": {
                    "orientation": "landscape",
                    "brand_guidelines": "Flat graphic, high-contrast, bold sans-serif.",
                    "on_creative_text": on_text or [],
                    "key_facts": key_facts or [],
                },
                "ad_copy": ad_copy,
                "platform": "x",
            }
        )

    @staticmethod
    def _deliv(prose: str) -> str:
        return f"<image_brief>{prose}</image_brief>"

    def test_clean_bridge_passes(self) -> None:
        brief = self._brief(
            on_text=["START NOW", "LEARN AI IN 30 DAYS"],
            key_facts=["a sequence of AI-tool tiles"],
        )
        prose = (
            'Create a flat graphic roadmap. Place bold text reading "LEARN AI IN 30 DAYS" '
            'at the top and a green button reading "START NOW", surrounding a sequence of '
            "AI-tool tiles."
        )
        ad = _FakeAd(
            raw={
                CAPTION_RAW_KEY: (
                    'a flat graphic 30-day roadmap with text "LEARN AI IN 30 DAYS" and a '
                    'button "START NOW" and a sequence of AI-tool tiles'
                )
            }
        )
        r = check_image_brief_content_bridge(brief, self._deliv(prose), ad)
        assert r.passed
        assert r.reason == "ok"

    def test_rejects_ungrounded_bridge_item(self) -> None:
        # key_facts claims content the caption never shows.
        brief = self._brief(key_facts=["a purple unicorn mascot"])
        prose = "Create a flat graphic with a purple unicorn mascot front and center."
        ad = _FakeAd(
            raw={CAPTION_RAW_KEY: "a flat graphic roadmap with green tiles and bold white text"}
        )
        r = check_image_brief_content_bridge(brief, self._deliv(prose), ad)
        assert not r.passed
        assert r.reason == "content_bridge_ungrounded"
        assert r.detail == "a purple unicorn mascot"

    def test_rejects_under_reported_on_image_text(self) -> None:
        # Deliverable quotes on-image text that is neither the copy nor bridged.
        brief = self._brief(on_text=[])
        prose = 'Create a flat graphic with a green button reading "START NOW".'
        ad = _FakeAd(raw={CAPTION_RAW_KEY: 'a flat graphic with a button reading "START NOW"'})
        r = check_image_brief_content_bridge(brief, self._deliv(prose), ad)
        assert not r.passed
        assert r.reason == "content_bridge_text_under_reported"
        assert "START NOW" in r.detail

    def test_rejects_over_reported_on_image_text(self) -> None:
        # Bridge claims on-image text the deliverable never renders.
        brief = self._brief(on_text=["FREE SHIPPING"])
        prose = "Create a clean product hero shot on a white field."
        ad = _FakeAd(
            raw={CAPTION_RAW_KEY: "a clean product hero shot with the words free shipping"}
        )
        r = check_image_brief_content_bridge(brief, self._deliv(prose), ad)
        assert not r.passed
        assert r.reason == "content_bridge_text_missing_from_deliverable"

    def test_quoted_copy_in_deliverable_is_not_under_report(self) -> None:
        # On-image text that IS part of the ad copy needs no bridge entry.
        brief = self._brief(on_text=[], ad_copy="**Headline:** Get fluent faster")
        prose = 'Create a bold title card reading "Get fluent faster".'
        ad = _FakeAd(raw={CAPTION_RAW_KEY: 'a bold title card reading "Get fluent faster"'})
        r = check_image_brief_content_bridge(brief, self._deliv(prose), ad)
        assert r.passed

    def test_avoid_line_quotes_are_ignored(self) -> None:
        # A quoted brand name on the trailing Avoid line is not on-image text.
        brief = self._brief(on_text=[])
        prose = (
            "Create a clean product hero shot on a white field.\n"
            'Avoid: real "Acme Corp" logos; clutter'
        )
        ad = _FakeAd(raw={CAPTION_RAW_KEY: "a clean product hero shot on a white field"})
        r = check_image_brief_content_bridge(brief, self._deliv(prose), ad)
        assert r.passed
