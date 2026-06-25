"""Fidelity gate for image-brief deliverables.

The copywriting skill's fidelity gate (``check_deliverable_fidelity``)
enforces that the deliverable reproduces the source ad verbatim. That
contract does NOT apply to image-brief: the deliverable is freeform
art-direction PROSE inside ``<image_brief>...</image_brief>`` that describes
a visual, not the source ad's words. We replace that gate with two checks:

1. **Prose extraction** — the deliverable's ``<image_brief>`` region must be
   present and non-empty after stripping. ``signature_passed`` carries this
   bit ("prose region extracted non-empty").

2. **Caption alignment** — the prose shares at least ``MIN_CAPTION_OVERLAP``
   content-word overlap with the source VLM caption joined onto
   ``source_ad.raw[CAPTION_RAW_KEY]``. Analogue of the copywriting fidelity
   gate — the teacher is supposed to be re-registering the real creative's
   visible facts, not inventing a visual. The caption is required at ingest:
   the selection step (``source_selector``) already filters out ads without a
   caption, and ``submit_single_pass`` enriches the remaining ads via the
   skill bundle's prepare hook. If an ad reaches ingest without a caption the
   gate fails — that scenario indicates a pipeline bug, not a benign smoke
   run.

The gate returns a :class:`FidelityResult` so it slots into the existing
:class:`SkillGateBundle` shape without changes to the ingest dispatch.
``coverage`` carries the caption-overlap fraction and ``signature_passed``
carries the prose-extraction bit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.ingest.fidelity import (
    MIN_WORD_LEN,
    FidelityResult,
    GroundingResult,
    _content_words,
    _first_match,
)
from draper.construction_v2.schemas.image_brief import ImageBriefInput
from draper.construction_v2.teacher.image_brief_single_pass import CAPTION_RAW_KEY

# Minimum caption-overlap fraction (content words from the source VLM
# caption that also appear in the image brief prose). 30% is conservative —
# the brief re-registers the caption observational -> directive, not a
# verbatim paraphrase of it.
MIN_CAPTION_OVERLAP: float = 0.30

_IMAGE_BRIEF_RE = re.compile(r"<image_brief>(.*?)</image_brief>", re.DOTALL | re.IGNORECASE)
# Double-quoted strings in the deliverable prose — the teacher quotes
# on-creative text in situ ('a label reading "50% less sugar"'). Used by the
# content-bridge gate to find on-image text the deliverable renders.
_QUOTED_RE = re.compile(r'"([^"]{2,})"')


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+", text))


def _extract_image_brief_prose(deliverable: str) -> str | None:
    """Pull the prose body out of the deliverable's ``<image_brief>`` region.

    Returns the stripped region body (including any trailing ``Avoid:`` line),
    or ``None`` when the region is missing or empty after stripping.
    """
    m = _IMAGE_BRIEF_RE.search(deliverable)
    if not m:
        return None
    prose = m.group(1).strip()
    return prose or None


def check_image_brief_fidelity(deliverable: str, source_ad: SourceAd) -> FidelityResult:
    """Two-stage fidelity gate for the image-brief prose deliverable.

    Returns a :class:`FidelityResult` so the existing :class:`SkillGateBundle`
    interface stays uniform across skills. ``coverage`` is the caption-overlap
    fraction (0.0 when no caption is available); ``signature_passed`` is True
    iff the ``<image_brief>`` prose region was extracted non-empty.
    """
    # Stage 1: prose extraction.
    prose = _extract_image_brief_prose(deliverable)
    if prose is None:
        return FidelityResult(
            passed=False,
            coverage=0.0,
            ad_word_count=0,
            signature_passed=False,
            reason="image_brief_missing_or_empty",
        )

    # Stage 2: caption alignment. The caption is required — selection
    # and submit_single_pass together guarantee it's attached for every
    # ingested ad. Missing here means a pipeline bug, not a benign case.
    caption = source_ad.raw.get(CAPTION_RAW_KEY) if isinstance(source_ad.raw, dict) else None
    if not isinstance(caption, str) or not caption.strip():
        return FidelityResult(
            passed=False,
            coverage=0.0,
            ad_word_count=0,
            signature_passed=True,
            reason="image_brief_missing_caption",
        )

    cap_words = _content_words(caption, min_len=MIN_WORD_LEN)
    if not cap_words:
        return FidelityResult(
            passed=False,
            coverage=0.0,
            ad_word_count=0,
            signature_passed=True,
            reason="image_brief_empty_caption_after_tokenize",
        )

    prose_words = _content_words(prose, min_len=MIN_WORD_LEN)
    shared = cap_words & prose_words
    coverage = len(shared) / len(cap_words)
    if coverage < MIN_CAPTION_OVERLAP:
        return FidelityResult(
            passed=False,
            coverage=coverage,
            ad_word_count=len(cap_words),
            signature_passed=True,
            reason="image_brief_low_caption_overlap",
        )

    return FidelityResult(
        passed=True,
        coverage=coverage,
        ad_word_count=_word_count(caption),
        signature_passed=True,
        reason="ok",
    )


def check_image_brief_brief_alignment(think: str, brief: ImageBriefInput) -> GroundingResult:
    """Grounding gate for the image-brief skill.

    ``creative.brand_guidelines`` (the brand's reusable visual identity / feel)
    is the load-bearing strategic anchor that grounds the visual decision. The
    gate passes iff a content word from ``brand_guidelines`` surfaces in the
    ``<think>`` trace — proving the rationale grounds its visual decision in the
    brief rather than free-associating from the supplied caption.

    ``GroundingResult.bridge_match`` carries the matched token (field name kept
    for cross-skill symmetry of the result type).
    """
    match = _first_match(think, [brief.creative.brand_guidelines])
    if match:
        return GroundingResult(passed=True, bridge_match=match, reason="")
    return GroundingResult(
        passed=False,
        bridge_match="",
        reason="no_brand_guidelines_ref",
    )


@dataclass(frozen=True)
class ContentBridgeResult:
    """Outcome of :func:`check_image_brief_content_bridge`.

    ``reason`` names the failing check (or ``"ok"``); ``detail`` carries the
    offending bridge item / quoted string so the audit log is actionable.
    """

    passed: bool
    reason: str
    detail: str


def _norm_for_match(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding quotes/punctuation.

    So a quoting/casing variant ('"GET FLUENT FASTER."' vs ``GET FLUENT
    FASTER``) compares equal under substring checks.
    """
    collapsed = re.sub(r"\s+", " ", text).strip().lower()
    return collapsed.strip(" .!?,;:\"'")


def check_image_brief_content_bridge(
    brief: ImageBriefInput, deliverable: str, source_ad: SourceAd
) -> ContentBridgeResult:
    """Verify the factual content bridge against the caption and the deliverable.

    The ``creative`` block carries two CONTENT-bridge lists the student
    conditions on: ``on_creative_text`` (verbatim non-copy on-image text) and
    ``key_facts`` (the non-inferable specifics the creative depends on and the
    copy omits — never what follows logically from the brief or is freely
    invented). They exist so the student reasons TO the deliverable's specifics
    instead of hallucinating them. Three checks:

    1. **Factuality** — every bridged atom (both lists) must be grounded in the
       real creative: its content words must overlap the source VLM caption.
       Rejects invented bridge content. (Skipped when no caption is present — the
       fidelity gate, run first, already rejects that.)
    2. **Over-report** — every ``on_creative_text`` string must actually appear
       in the deliverable prose (the teacher quotes on-image text in situ); a
       bridge that claims text the deliverable never renders is inconsistent.
    3. **Under-report (anti-hallucination)** — every double-quoted on-image
       string in the deliverable that is NOT part of the ad copy must be covered
       by an ``on_creative_text`` item. Otherwise the student would have to
       invent that text from nothing at inference. ``key_facts`` is deliberately
       NOT credited here: it carries facts (paraphrased into the visual by the
       deliverable), not verbatim text — every visible on-image string belongs in
       ``on_creative_text``, so a quoted string covered only by ``key_facts``
       means the teacher mis-slotted it. The trailing ``Avoid:`` exclusion line
       is excluded from this scan (its quotes name things to keep OUT, not
       on-image text).

    The necessity rule (``key_facts`` carries only the non-inferable specifics
    the creative depends on, never what follows from the brief or is freely
    invented) is NOT machine-checked here: an n-gram guard would false-positive
    because the deliverable legitimately renders the bridged facts. It is enforced
    structurally (flat atom lists) + by the teacher prompt + by audit. Returns
    :class:`ContentBridgeResult`; never raises.
    """
    creative = brief.creative
    on_text = creative.on_creative_text
    facts = creative.key_facts
    prose = _extract_image_brief_prose(deliverable) or ""

    # 1. Factuality — bridge atoms must be grounded in the caption.
    caption = source_ad.raw.get(CAPTION_RAW_KEY) if isinstance(source_ad.raw, dict) else None
    if isinstance(caption, str) and caption.strip():
        cap_words = _content_words(caption, min_len=MIN_WORD_LEN)
        for item in (*on_text, *facts):
            item_words = _content_words(item, min_len=MIN_WORD_LEN)
            if item_words and not (item_words & cap_words):
                return ContentBridgeResult(
                    passed=False, reason="content_bridge_ungrounded", detail=item
                )

    norm_prose = _norm_for_match(prose)
    norm_copy = _norm_for_match(brief.ad_copy)

    # 2. Over-report — each on_creative_text string must appear in the prose.
    for item in on_text:
        norm_item = _norm_for_match(item)
        if norm_item and norm_item not in norm_prose:
            return ContentBridgeResult(
                passed=False,
                reason="content_bridge_text_missing_from_deliverable",
                detail=item,
            )

    # 3. Under-report — quoted on-image text in the prose (excluding the Avoid
    #    line) that is neither the copy nor a bridged item is a hallucination.
    body = "\n".join(
        ln for ln in prose.splitlines() if not re.match(r"^\s*avoid:", ln, re.IGNORECASE)
    )
    norm_on_text = [t for t in (_norm_for_match(x) for x in on_text) if t]
    for raw_q in _QUOTED_RE.findall(body):
        q = _norm_for_match(raw_q)
        if not q or q in norm_copy:
            continue
        if not any(t in q or q in t for t in norm_on_text):
            return ContentBridgeResult(
                passed=False,
                reason="content_bridge_text_under_reported",
                detail=raw_q,
            )

    return ContentBridgeResult(passed=True, reason="ok", detail="")


__all__ = [
    "MIN_CAPTION_OVERLAP",
    "ContentBridgeResult",
    "check_image_brief_brief_alignment",
    "check_image_brief_content_bridge",
    "check_image_brief_fidelity",
]
