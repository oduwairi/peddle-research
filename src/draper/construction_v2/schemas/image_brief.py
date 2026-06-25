"""ImageBrief schema + canonical JSON serializer.

The image-brief skill's deliverable is FREEFORM ART-DIRECTION PROSE — after
``<think>`` the teacher emits an ``<image_brief>...</image_brief>`` region whose
body is directive prose (one or more paragraphs) that re-registers every visible
fact of the real winning creative, with one optional trailing ``Avoid:`` line
carrying exclusions. The on-wire deliverable is NEVER this JSON.

:class:`ImageBrief` is a small optional parsed/cache view only: the prose brief
plus the parsed exclusion list. It exists so callers that want a structured
handle on a parsed deliverable have one; it is not the wire format.

:class:`ImageBriefInput` is the BRIEF the writer conditions on. The image skill
runs AFTER the copy is written, so the brief carries the finished ad copy
verbatim and **grounds the visual decision without prescribing it** — composition
is the writer's job (it lives in the weights), never the brief. Fields:

- ``task`` — the founder-voiced ask (teacher-authored).
- ``objective`` — what the ad is trying to DO (awareness / promo_offer / launch
  / social_proof / conversion). The ad's purpose shapes the creative even when
  it isn't obvious from the copy. Teacher-authored (reverse-engineered).
- ``product`` — the FULL copywriting :class:`BriefProduct` (the rest of the
  grounding: subject, USPs, tone). Teacher-authored.
- ``creative`` — the :class:`CreativeDirection` block grouping the canvas + the
  two bridges (see that class). It carries WHAT the creative must look like and
  contain, never HOW it is composed (layout/framing/lighting/placement stay the
  deliverable's job).
- ``ad_copy`` — the finished ad copy, VERBATIM, platform-labeled exactly as the
  copywriting skill emits it (``render_labeled_ad``). Injected by the pipeline at
  ingest, never authored by the teacher, so it is byte-exact and the image-skill
  input equals the copy-skill output. (Named ``ad_copy`` rather than ``copy`` to
  avoid shadowing pydantic's ``BaseModel.copy``.)
- ``platform`` — one of :data:`SUPPORTED_PLATFORMS`. Injected by the pipeline
  at ingest from the source ad's platform (normalized to the canonical
  vocabulary via ``platform_group_for``), never authored by the teacher (the
  teacher prompt forbids emitting it; ``_image_brief_build_brief`` drops any
  stray value and reinjects it).

:class:`CreativeDirection` is the one nested block holding all visual direction:

- ``orientation`` — the canvas the creative must fill (was top-level
  ``aspect_ratio``). The orchestrator picks it from a fixed list (mirrors the
  frontend ``generate_image`` ``ImageSize`` enum) derived from platform, so it is
  INJECTED at ingest from :func:`aspect_ratio_for_platform`, never authored.
- ``brand_guidelines`` — the STYLE bridge: the brand's recurring VISUAL IDENTITY
  / feel (aesthetic register + art-style/medium + type feel). A brand like Nike
  has a feel the model must be *given*, not invent. Stated at the reusable-brand
  level, never this ad's composition, never world knowledge. Teacher-authored.
- ``on_creative_text`` — the CONTENT bridge (text): verbatim strings burned into
  the creative that are NOT part of the ad copy. The student cannot derive them
  from product facts, so the brief must carry them. The string only — never where
  it sits (placement is the deliverable's job). ``[]`` when the only text in the
  creative is the copy. Teacher-authored from the caption.
- ``key_facts`` — the CONTENT bridge (facts): the facts the writer would need to
  build this creative but could not work out from the copy + product. Membership
  is a two-part test: a caption fact is bridged iff substituting a different
  tasteful choice in its place would change what the ad claims AND it does not
  already follow from the brief. Whatever the creative is free to choose without
  changing the ad's meaning is the writer's to invent and stays out; whatever the
  ad genuinely makes a claim about, and the copy omits, must be carried or the
  writer hallucinates it at inference. ``[]`` only when copy + product already
  imply all the content. Teacher-authored from the caption.

The rule for the two content bridges: ``on_creative_text`` carries the literal
text that appears on the creative; ``key_facts`` carries what the creative
depends on that the copy omits and the writer could not infer. Anything that
follows from the brief, or that the writer is free to choose without changing the
ad's claim, is not bridged — the model confabulates it. Anything the ad genuinely
makes a claim about, which the copy never states, must be bridged, or the model
learns to hallucinate it.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from draper.construction_v2.schemas.brief import (
    BriefProduct,
    canonical_dict_json,
)

# The canvas the creative must fill. Mirrors the frontend ``generate_image``
# ``ImageSize`` enum exactly (square=1024², landscape=1536×1024,
# portrait=1024×1536) — the backend is the source of truth and the frontend is
# fitted to it. Injected from platform at ingest, never authored by the teacher.
AspectRatio = Literal["square", "landscape", "portrait"]

# What the ad is trying to DO. Drives the creative archetype the writer chooses,
# even when the objective isn't obvious from the finished copy. Teacher-authored.
AdObjective = Literal["awareness", "promo_offer", "launch", "social_proof", "conversion"]

# Deterministic platform -> default canvas. Matches the orchestrator's
# inference-time choice ("square = Meta feed; landscape = X card; portrait =
# TikTok/Reels") so training and serving agree. Real advertisers also follow
# these placement conventions, so the default tracks the actual creative shape.
PLATFORM_ASPECT_RATIO: dict[str, AspectRatio] = {
    "meta": "square",
    "tiktok": "portrait",
    "x": "landscape",
    "google": "square",
    "pinterest": "portrait",
    "reddit": "square",
}


def aspect_ratio_for_platform(platform: str) -> AspectRatio:
    """Return the canonical canvas for ``platform`` (defaults to ``square``)."""
    return PLATFORM_ASPECT_RATIO.get(platform, "square")


class CreativeDirection(BaseModel):
    """The one nested block holding all visual direction for the image brief.

    Three kinds of fact, deliberately grouped together: the canvas
    (``orientation``), the STYLE bridge (``brand_guidelines`` — how it looks),
    and the CONTENT bridge (``on_creative_text`` + ``key_facts`` — what it must
    convey). All three describe WHAT the creative must look like and carry;
    none describes HOW it is composed. Composition is the writer's job and stays
    in the deliverable.
    """

    model_config = ConfigDict(extra="forbid")

    orientation: AspectRatio = Field(
        ...,
        description=(
            "The canvas the creative must fill. Derived from platform; injected "
            "at ingest, never authored by the teacher."
        ),
    )
    brand_guidelines: str = Field(
        ...,
        description=(
            "STYLE bridge — the brand's recurring visual identity / feel: "
            "aesthetic register, art-style/medium (photographic vs illustrated "
            "vs 3D vs flat), and type feel. Reverse-engineered at the "
            "reusable-brand level — never this ad's composition, never world "
            "knowledge of the brand."
        ),
    )
    on_creative_text: list[str] = Field(
        default_factory=list,
        description=(
            "CONTENT bridge (text) — verbatim strings burned into the creative "
            "that are NOT part of the ad copy. The string only, never its "
            "placement; every visible non-copy string belongs here. Empty when "
            "the only text in the creative is the copy."
        ),
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description=(
            "CONTENT bridge (facts) — the facts the writer would need to build "
            "this creative but could not work out from the copy + product. "
            "Two-part test: bridge a caption fact iff substituting a different "
            "tasteful choice would change what the ad claims AND it does not "
            "already follow from the brief. What the creative is free to choose "
            "without changing the claim stays out; what the ad makes a claim "
            "about and the copy omits must be carried, else the model "
            "hallucinates it. Empty only when copy + product already imply the "
            "content."
        ),
    )

    @field_validator("brand_guidelines")
    @classmethod
    def _brand_guidelines_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            msg = "CreativeDirection.brand_guidelines must be non-empty."
            raise ValueError(msg)
        return s

    @field_validator("on_creative_text", "key_facts", mode="before")
    @classmethod
    def _coerce_null_list(cls, v: Any) -> Any:
        # Teachers correctly emit null for these when the creative carries no
        # non-copy text / the content is fully supplied by copy+product. Coerce
        # to empty list so a null answer parses identically to the default.
        if v is None:
            return []
        return v

    @field_validator("on_creative_text", "key_facts")
    @classmethod
    def _clean_items(cls, v: list[str]) -> list[str]:
        # Strip and drop blanks; the lists are optional content atoms, so an
        # empty list is the honest answer when nothing must be bridged.
        return [s.strip() for s in v if s and s.strip()]


class ImageBriefInput(BaseModel):
    """The brief the image-brief writer conditions on.

    GROUNDING only — never composition. ``objective`` narrows the visual
    *strategically* (what the ad is for); ``product`` + ``ad_copy`` carry the
    facts; ``creative`` carries the canvas, the style bridge, and the factual
    content bridge (see :class:`CreativeDirection`). The composition is the
    writer's job and stays out of the brief (it lives in the weights).
    """

    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        ...,
        description=(
            "Natural-language description of the image/creative-brief work the "
            "caller is asking for, referencing the campaign whose copy is in "
            "``ad_copy``."
        ),
    )
    objective: AdObjective = Field(
        ...,
        description=(
            "What the ad is trying to do — its marketing purpose. Shapes the "
            "creative archetype even when the copy is subtle."
        ),
    )
    product: BriefProduct
    creative: CreativeDirection
    ad_copy: str = Field(
        ...,
        description=(
            "The finished ad copy, verbatim, platform-labeled exactly as the "
            "copywriting skill emits it. Injected from the source ad."
        ),
    )
    platform: Literal["meta", "tiktok", "x", "google", "pinterest", "reddit"]

    @field_validator("ad_copy")
    @classmethod
    def _ad_copy_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            msg = "ImageBriefInput.ad_copy must be non-empty (the labeled ad copy)."
            raise ValueError(msg)
        return s


def canonical_image_brief_input_json(brief: ImageBriefInput) -> str:
    """Serialize an :class:`ImageBriefInput` to canonical JSON.

    Same byte contract as :func:`draper.construction_v2.schemas.brief.canonical_json`
    (delegates to the shared ``canonical_dict_json``), so the frontend's
    image-brief serializer must produce byte-identical output.
    """
    return canonical_dict_json(brief.model_dump(mode="json"))


class ImageBrief(BaseModel):
    """Prose art-direction brief — an optional parsed view of the deliverable.

    Two fields: ``brief`` carries the directive art-direction prose; ``negative``
    carries the parsed exclusion list (the items off the deliverable's trailing
    ``Avoid:`` line). The wire deliverable is prose, not this object.
    """

    model_config = ConfigDict(extra="forbid")

    brief: str = Field(
        ...,
        description=(
            "Directive art-direction prose re-registering every visible fact "
            "of the creative (observational -> directive), with bindings intact."
        ),
    )
    negative: list[str] = Field(
        default_factory=list,
        description=(
            "Things to exclude — real brand logos, hands in frame, text overlay, "
            "clutter, etc. Empty when nothing must be excluded."
        ),
    )

    @field_validator("brief")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            msg = "image brief prose must be non-empty"
            raise ValueError(msg)
        return s


def canonical_image_brief_json(brief: ImageBrief) -> str:
    """Serialize an :class:`ImageBrief` to canonical JSON.

    Mirrors :func:`draper.construction_v2.schemas.brief.canonical_json`:
    sorted keys, ``null`` fields preserved, UTF-8 source bytes, no extra
    whitespace, no trailing newline.
    """
    payload: dict[str, Any] = brief.model_dump(mode="json")
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "PLATFORM_ASPECT_RATIO",
    "AdObjective",
    "AspectRatio",
    "CreativeDirection",
    "ImageBrief",
    "ImageBriefInput",
    "aspect_ratio_for_platform",
    "canonical_image_brief_input_json",
    "canonical_image_brief_json",
]
