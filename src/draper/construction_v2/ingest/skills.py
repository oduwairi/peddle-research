"""Skill-keyed registry that dispatches the full v2 construction pipeline.

The v2 pipeline runs four ordered stages per source ad: select → submit
→ collect → ingest. Every stage that has skill-specific behavior routes
through this registry. The bundle exposes these callables matching the
per-stage extension points:

- ``prepare_source_ads`` — submit-time enrichment (e.g. join VLM
  captions onto image-brief source ads). Copywriting is identity.
- ``build_request`` — submit-time teacher request builder.
- ``parse_response`` — collect-time response parser. Both skills return
  a result that satisfies the :class:`TeacherParseResult` protocol
  (brief / think / deliverable / errors).
- ``build_brief`` — ingest-time validator that turns the cached brief
  dict into the skill's brief model, injecting any skill-specific field
  the teacher does not author. Copywriting validates a :class:`Brief`;
  image_brief validates an :class:`ImageBriefInput` after injecting the
  verbatim platform-labeled ``ad_copy`` and the platform-derived
  ``creative.orientation`` (the canvas, via
  :func:`aspect_ratio_for_platform`). May raise on an invalid brief — the
  ingest loop converts that to a rejection.
- ``fidelity`` — ingest gate enforcing the verbatim / structural
  contract on the deliverable.
- ``grounding`` — ingest gate ensuring ``<think>`` references the brief
  (not the source ad).
- ``leak`` — ingest gate rejecting briefs that paraphrase the ad copy
  into strategic fields. Optional: skills where the copy is a legitimate
  verbatim brief input (e.g. image_brief) set this to ``None`` and the
  stage is skipped.
- ``labels`` — ingest gate validating platform-native field labels.
  Optional: skills whose deliverable carries no platform labels (e.g.
  image_brief) set this to ``None`` and the stage is skipped.

Symmetry property: changing ``config.skill`` is the only thing that
needs to change between runs of the two skills. ``submit_single_pass``,
``collect_batch``, and ``ingest_responses`` all look up the bundle and
delegate; they contain no per-skill branches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from draper.construction_v2.ingest.fidelity import (
    FidelityResult,
    GroundingResult,
    check_deliverable_fidelity,
    check_think_grounding,
)
from draper.construction_v2.platform_labels import (
    LabelResult,
    check_platform_labels_present,
)

if TYPE_CHECKING:
    from draper.construction.batch.types import BatchRequest
    from draper.construction_v2.config import ConstructionV2Config
    from draper.construction_v2.dataset.source_selector import SourceAd
    from draper.construction_v2.ingest.image_brief_fidelity import ContentBridgeResult
    from draper.construction_v2.ingest.leak_guard import LeakResult


class TeacherParseResult(Protocol):
    """Duck-typed result of parsing one teacher response.

    Both :class:`SinglePassParseResult` (copywriting) and
    :class:`ImageBriefParseResult` (image-brief) structurally satisfy
    this protocol. The collect stage only reads these four attributes,
    so they are declared as read-only properties to keep the Protocol
    covariant (mypy refuses to substitute concrete dataclass returns
    for an invariant Protocol return type).
    """

    @property
    def brief(self) -> dict[str, Any] | None: ...
    @property
    def think(self) -> str | None: ...
    @property
    def deliverable(self) -> str | None: ...
    @property
    def errors(self) -> list[str]: ...


# Submit-time hooks.
PrepareSourceAds = Callable[
    [list["SourceAd"], "ConstructionV2Config"],
    tuple[list["SourceAd"], list[str]],
]
BuildRequest = Callable[..., "BatchRequest"]
ParseResponse = Callable[[str], TeacherParseResult]

# Ingest-time hooks + gates. ``build_brief`` returns the skill's brief
# model (``Brief`` or ``ImageBriefInput``); the gates that consume it are
# typed over ``Any`` so a concrete per-skill check assigns cleanly.
BuildBrief = Callable[[dict[str, Any], "SourceAd"], Any]
FidelityCheck = Callable[[str, "SourceAd"], FidelityResult]
GroundingCheck = Callable[[str, Any], GroundingResult]
LeakCheck = Callable[..., "LeakResult"]
LabelsCheck = Callable[[str, "SourceAd"], LabelResult]
# (brief, deliverable, source_ad) -> result. Verifies the factual content
# bridge (image_brief only); copywriting leaves this ``None``.
ContentBridgeCheck = Callable[[Any, str, "SourceAd"], "ContentBridgeResult"]


@dataclass(frozen=True)
class SkillGateBundle:
    """Per-skill dispatch table for the full construction pipeline.

    ``leak``, ``labels``, and ``content_bridge`` are optional: skills where
    the copy is a legitimate verbatim brief input (e.g. image_brief) leave
    ``leak`` ``None``; skills whose deliverable has no platform-native field
    labels leave ``labels`` ``None``; ``content_bridge`` is set only by skills
    that carry a factual content bridge in the brief (image_brief), verifying
    the bridge is grounded in the caption and consistent with the deliverable.
    ingest skips any stage whose callable is ``None``.

    ``build_brief`` turns the cached brief dict into the skill's brief
    model, injecting any field the teacher does not author (image_brief
    injects the verbatim ``ad_copy`` + the platform-derived
    ``creative.orientation`` canvas, via :func:`aspect_ratio_for_platform`).
    It may raise on an invalid brief.

    ``system_prompt`` is the verbatim teacher SYSTEM string for the skill.
    The submit path doesn't read it (``build_request`` already bakes it
    into the :class:`BatchRequest`), but smoke/render tools need it to
    display the prompt next to the responses for auditing.
    """

    name: str
    prepare_source_ads: PrepareSourceAds
    build_request: BuildRequest
    parse_response: ParseResponse
    build_brief: BuildBrief
    fidelity: FidelityCheck
    grounding: GroundingCheck
    leak: LeakCheck | None
    labels: LabelsCheck | None
    content_bridge: ContentBridgeCheck | None
    system_prompt: str


_REGISTRY: dict[str, SkillGateBundle] = {}


def register(bundle: SkillGateBundle) -> None:
    """Add a skill bundle to the registry. Overwrites by name on conflict."""
    _REGISTRY[bundle.name] = bundle


def get_bundle(skill: str) -> SkillGateBundle:
    """Return the registered :class:`SkillGateBundle` for ``skill``.

    Raises :class:`KeyError` with the available skills in the message so
    a typo in ``config.skill`` produces an actionable error. Also logs
    the error to stderr before raising so it surfaces in batch logs.
    """
    bundle = _REGISTRY.get(skill)
    if bundle is None:
        available = sorted(_REGISTRY.keys())
        msg = (
            f"No skill bundle registered for {skill!r}. "
            f"Available: {available}. Add it via "
            f"draper.construction_v2.ingest.skills.register(...)."
        )
        import logging

        logging.getLogger("draper").error(msg)
        raise KeyError(msg)
    return bundle


def registered_skills() -> list[str]:
    """List the registered skill names (useful for CLI help)."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Identity prepare hook for skills that need no submit-time enrichment.
# ---------------------------------------------------------------------------


def _identity_prepare(
    ads: list[SourceAd], _config: ConstructionV2Config
) -> tuple[list[SourceAd], list[str]]:
    """Pass ads through unchanged; nothing dropped. Used by copywriting."""
    return ads, []


# ---------------------------------------------------------------------------
# Copywriting skill — wraps the single-pass teacher + existing gates.
# ---------------------------------------------------------------------------


def _copywriting_build_brief(brief_dict: dict[str, Any], _ad: SourceAd) -> Any:
    """Validate the cached brief dict into a copywriting :class:`Brief`."""
    from draper.construction_v2.schemas.brief import Brief

    return Brief.model_validate(brief_dict)


def _register_copywriting_bundle() -> None:
    from draper.construction_v2.ingest.leak_guard import check_bridge_leak
    from draper.construction_v2.teacher.single_pass import (
        SINGLE_PASS_TEACHER_SYSTEM,
        build_single_pass_request,
        parse_single_pass_response,
    )

    register(
        SkillGateBundle(
            name="copywriting",
            prepare_source_ads=_identity_prepare,
            build_request=build_single_pass_request,
            parse_response=parse_single_pass_response,
            build_brief=_copywriting_build_brief,
            fidelity=check_deliverable_fidelity,
            grounding=check_think_grounding,
            leak=check_bridge_leak,
            labels=check_platform_labels_present,
            content_bridge=None,
            system_prompt=SINGLE_PASS_TEACHER_SYSTEM,
        )
    )


# ---------------------------------------------------------------------------
# Image-brief skill — caption-enriched submit + image-brief teacher +
# looser fidelity + no platform labels.
# ---------------------------------------------------------------------------


def _image_brief_prepare(
    ads: list[SourceAd], _config: ConstructionV2Config
) -> tuple[list[SourceAd], list[str]]:
    """Join VLM captions onto each ad's ``raw[CAPTION_RAW_KEY]``.

    Ads without a caption are dropped; their ad_ids are returned so
    submit_single_pass can log the count. ``require_caption=True``
    matches the contract that the image-brief teacher rejects empty
    captions at request build.
    """
    from draper.construction_v2.captions.builder import (
        enrich_source_ads_with_captions,
    )

    kept, missing = enrich_source_ads_with_captions(
        ads, captions_parquet=None, require_caption=True
    )
    return kept, missing


def _image_brief_build_brief(brief_dict: dict[str, Any], ad: SourceAd) -> Any:
    """Validate into an :class:`ImageBriefInput`, injecting the non-authored fields.

    The teacher authors ``task`` + ``objective`` + ``product`` + the ``creative``
    block's three authored keys (``brand_guidelines`` + ``on_creative_text`` +
    ``key_facts``). Three fields are KNOWN FACTS about the source ad, injected
    here, never authored, so they are deterministic and match what the
    orchestrator supplies at inference:

    - ``ad_copy`` — the finished copy in the exact platform-labeled form the
      copywriting skill emits (``render_labeled_ad``), byte-exact from the
      source ad.
    - ``creative.orientation`` — the canvas, derived from the canonical platform
      (:func:`aspect_ratio_for_platform`).
    - ``platform`` — the source ad's platform, normalized to the canonical
      vocabulary (``platform_group_for`` folds source-native names like
      ``facebook`` -> ``meta`` and ``twitter`` -> ``x``).

    Robustness: stray top-level ``brand_guidelines`` / ``on_creative_text`` /
    ``key_facts`` (teacher placing them outside ``creative``) are hoisted into
    the block; any teacher-authored orientation/platform/copy is overwritten.
    """
    from draper.construction_v2.platform_labels import (
        platform_group_for,
        render_labeled_ad,
    )
    from draper.construction_v2.schemas.image_brief import (
        ImageBriefInput,
        aspect_ratio_for_platform,
    )

    platform = platform_group_for(ad.platform).value

    creative = dict(brief_dict.get("creative") or {})
    # Hoist stray top-level visual fields into the block without clobbering
    # values the teacher already nested under ``creative``.
    for key in ("brand_guidelines", "on_creative_text", "key_facts"):
        if key in brief_dict and key not in creative:
            creative[key] = brief_dict[key]
    # The canvas is injected, never authored — drop any stray copy, then set it.
    creative.pop("aspect_ratio", None)
    creative["orientation"] = aspect_ratio_for_platform(platform)

    # Rebuild the top level without the fields that live in ``creative`` or are
    # injected, so ``extra="forbid"`` validation can't trip on a stray copy.
    drop = {
        "creative",
        "brand_guidelines",
        "on_creative_text",
        "key_facts",
        "ad_copy",
        "aspect_ratio",
        "platform",
    }
    payload = {k: v for k, v in brief_dict.items() if k not in drop}
    payload["creative"] = creative
    payload["ad_copy"] = render_labeled_ad(ad)
    payload["platform"] = platform
    return ImageBriefInput.model_validate(payload)


def _register_image_brief_bundle() -> None:
    from draper.construction_v2.ingest.image_brief_fidelity import (
        check_image_brief_brief_alignment,
        check_image_brief_content_bridge,
        check_image_brief_fidelity,
    )
    from draper.construction_v2.teacher.image_brief_single_pass import (
        IMAGE_BRIEF_TEACHER_SYSTEM,
        build_image_brief_request,
        parse_image_brief_response,
    )

    register(
        SkillGateBundle(
            name="image_brief",
            prepare_source_ads=_image_brief_prepare,
            build_request=build_image_brief_request,
            parse_response=parse_image_brief_response,
            build_brief=_image_brief_build_brief,
            fidelity=check_image_brief_fidelity,
            grounding=check_image_brief_brief_alignment,
            leak=None,
            labels=None,
            content_bridge=check_image_brief_content_bridge,
            system_prompt=IMAGE_BRIEF_TEACHER_SYSTEM,
        )
    )


_register_copywriting_bundle()
_register_image_brief_bundle()
