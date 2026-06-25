"""Single-pass teacher: production prompt + parser for the v2 pipeline.

One API call per source ad emits three regions:

1. ``<brief>...</brief>`` — JSON with ``task``, ``product``,
   ``bridge``, and ``platform``.
2. ``<think>...</think>`` — first-person reasoning grounded in the
   brief the teacher just emitted (not in the source ad).
3. Freeform deliverable — the source ad reproduced verbatim, optionally
   wrapped with short framing prose.

This is the **production teacher**: ``pipeline.submit_single_pass``
calls :func:`build_single_pass_request` for every source ad;
``pipeline.collect_batch`` calls :func:`parse_single_pass_response`
to split the collected content into ``briefs.jsonl`` rows and
``responses_raw.jsonl`` rows for downstream ingest.

``scripts/explore/single_pass_smoke.py`` and
``scripts/explore/render_single_pass_md.py`` also import from here
for manual review workflows.

The legacy two-stage teachers (``brief_extractor.py``,
``rationale_prompt.py``) are retained for reference until Phase 4.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from draper.construction.batch.types import BatchRequest
from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.ingest.response_parser import (
    ParsedResponse,
    ParseRejection,
    parse_response,
)
from draper.construction_v2.platform_labels import render_labeled_ad
from draper.construction_v2.schemas.brief import SUPPORTED_PLATFORMS

DEFAULT_MAX_TOKENS: int = 4000
DEFAULT_TEMPERATURE: float = 0.4

SINGLE_PASS_TEACHER_SYSTEM: str = (
    "You are preparing training data for Draper, a marketing "
    "specialist. This smoke covers the ad-copy slice of that role. "
    "Given one real, high-performing ad, produce a single response "
    "containing a ``<brief>`` JSON region followed by the canonical "
    "Draper assistant turn: a ``<think>`` block followed by Draper's "
    "response to the user, which contains the source ad verbatim "
    "within it.\n"
    "\n"
    "## Grounding contract (read this first)\n"
    "\n"
    "Every field in the ``<brief>`` JSON MUST be derivable from the "
    "source you are given: the ad's copy text plus the supplied "
    "metadata (advertiser_name, landing_page_url, platform). World "
    "knowledge about the brand is FORBIDDEN. If the ad does not state "
    "or clearly imply a fact, you MUST leave that field null or empty. "
    "A blank field is the correct, honest answer — not a failure. "
    "Read the ad. For each field, ask: 'is this stated or clearly "
    "readable from the ad text or metadata?' If yes, fill it. If no, "
    "leave it null/empty and MOVE ON. Do not reach for world knowledge "
    "about the advertiser to backfill features.\n"
    "\n"
    "Why: at inference time a downstream agent fills these same fields "
    "from real research. If you fabricate features, USPs, or proof "
    "points the ad doesn't mention, the model learns to ignore those "
    "fields entirely. Hallucinated facts here directly damage the "
    "product.\n"
    "\n"
    "## Regions\n"
    "\n"
    "1. ``<brief>...</brief>`` — a JSON object with these keys:\n"
    "     - ``task`` (required string): a one-sentence natural-language "
    "query that a founder might plausibly have written to ask Draper "
    "for this exact piece of work. Phrase it the way a real user would "
    "type it in chat — concrete about the product and the platform, "
    'no jargon, no enum slugs. Examples: "Write a Reddit post warning '
    'trailer haulers about tongue-weight failures.", "Draft a Meta ad '
    "for our payroll-compliance tool aimed at HR leads who hate "
    'spreadsheets." Infer it from the source ad — do not hardcode.\n'
    "     - ``product`` (required object). Only ``tone_signals`` is "
    "required (string array, non-empty — voice cues read off the ad's "
    'natural voice, e.g. "playful", "clipped", "warm-but-direct"). '
    "These are voice cues, not topic descriptors. "
    '"safety-focused", "community-oriented", "low-friction" are NOT '
    "valid — those describe subject matter or positioning, not voice. "
    "All other product fields are OPTIONAL and should be null/empty "
    "when the ad doesn't support them: ``name``, ``description``, "
    "``category``, ``key_features``, ``unique_selling_points``, "
    "``price_info``, ``category_context``, ``proof_points``, ``offer``.\n"
    "     - ``bridge`` (required object). REQUIRED: ``angle`` (short "
    'label, e.g. "problem-aware skeptic") and ``buyer_pain`` (concrete '
    "friction). OPTIONAL — leave null when the ad alone doesn't reveal "
    "them: ``positioning``, ``target_audience``.\n"
    "     - ``platform`` (required string): exactly one of "
    f"{', '.join(SUPPORTED_PLATFORMS)}.\n"
    "\n"
    "2. ``<think>...</think>`` — **Persona switch: for this block "
    "you ARE Draper. Reason ONLY from the brief you JUST EMITTED in "
    "``<brief>``, not from the source ad in your input.** The ad does "
    "not exist yet from Draper's seat — he is about to write it. "
    "**The source ad is not yet known to Draper. Referencing its "
    "specific phrases, emoji, or punctuation inside ``<think>`` is "
    "wrong by construction.** First-person decisional voice, anchored "
    "to populated brief fields.\n"
    "\n"
    "3. Draper's response to the user. This region IS the response, "
    "end to end. The source ad appears verbatim somewhere within it, "
    "carrying the SAME platform-native field labels shown in the input "
    "block (preserve emojis, punctuation, capitalization, line breaks "
    "of each value; no tags, no fences, no draft framing). Compose the "
    "response as Draper actually answering the `task` — not as "
    "wrapping around an artifact, but as the actual reply, which "
    "contains the labeled ad.\n"
    "\n"
    "## Platform-native field labels\n"
    "\n"
    "The source ad in the input block is laid out with the field "
    "labels the platform uses natively. Reproduce ONLY the labels "
    "shown in the input — same Title Case, same bold markdown, same "
    "order, same content byte-for-byte. Do NOT invent additional "
    "labels (no fabricated CTAs, taglines, or empty slots). The "
    "vocabulary per platform:\n"
    "- ``meta``: ``Primary text``, ``Headline``, ``Description``\n"
    "- ``tiktok``: ``Caption``\n"
    "- ``x``: ``Tweet``, ``Card title``, ``CTA``\n"
    "- ``pinterest``: ``Title``, ``Description``\n"
    "- ``reddit``: ``Headline``\n"
    "\n"
    "Field labels are part of the response shape, not observations of "
    "the ad. Do NOT mention ``Primary text``, ``Headline``, ``CTA``, "
    "or any other label name inside ``<think>``.\n"
    "\n"
    "## Hard rules\n"
    "- Inside ``<think>``, Draper writes from inside the act — he "
    "doesn't reference his own response or the ad as an artifact "
    "he's about to deliver.\n"
    "- Bridge fields must NEVER quote or paraphrase the ad's copy. "
    "Restate the strategic intent in your own words — no verbatim "
    "phrases from the ad copy may appear in any bridge field.\n"
    "- ``tone_signals`` must be non-empty.\n"
    "- Use the EXACT brief field names listed above; do NOT invent "
    "shorter aliases like ``what_it_is`` or ``usps``.\n"
    "- The source ad MUST appear character-for-character within "
    "Draper's response — same field labels, same values, byte-for-byte.\n"
    "- No content before ``<brief>``. No markdown fences around the "
    "regions themselves."
)


_BRIEF_RE = re.compile(r"<brief>(.*?)</brief>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fence(text: str) -> str:
    m = _FENCE_RE.search(text.strip())
    return m.group(1).strip() if m else text.strip()


def build_single_pass_user_message(ad: SourceAd) -> str:
    """Render the user turn for a single-pass teacher request.

    The ad copy is laid out with platform-native field labels (Meta's
    ``**Primary text:**``, Pinterest's ``**Title:**``, etc.) via
    :func:`render_labeled_ad`. The teacher reproduces these labels
    verbatim in its deliverable, training the student to emit the
    same structure the frontend's ``emit_campaign`` parser expects.
    """
    labeled = render_labeled_ad(ad)
    parts: list[str] = [
        "# Source ad",
        "",
        f"- ad_id: {ad.ad_id}",
        f"- platform_hint: {ad.platform}",
        "",
        "## Source ad copy (platform-native field labels — reproduce "
        "VERBATIM in the response, preserving the same labels and values)",
        "",
        labeled if labeled else "(empty)",
    ]
    advertiser = ad.raw.get("advertiser_name") if isinstance(ad.raw, dict) else None
    landing = ad.raw.get("landing_page_url") if isinstance(ad.raw, dict) else None
    if isinstance(advertiser, str) and advertiser:
        parts.extend(["", f"- advertiser_name: {advertiser!r}"])
    if isinstance(landing, str) and landing:
        parts.append(f"- landing_page_url: {landing!r}")
    parts.extend(
        [
            "",
            "Produce the response now. Begin with `<brief>`, then "
            "`<think>`, then Draper's response — composed as the actual "
            "reply to the `task` you wrote, containing the source ad "
            "verbatim somewhere within it.",
        ]
    )
    return "\n".join(parts)


def build_single_pass_request(
    ad: SourceAd,
    *,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> BatchRequest:
    """Build a ``BatchRequest`` for the single-pass teacher."""
    return BatchRequest(
        custom_id=f"teacher-{ad.ad_id}",
        system=SINGLE_PASS_TEACHER_SYSTEM,
        messages=[{"role": "user", "content": build_single_pass_user_message(ad)}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


@dataclass(frozen=True)
class SinglePassParseResult:
    """Outcome of parsing one single-pass teacher response.

    ``brief`` carries the parsed JSON (with ``task`` injected) or
    ``None`` when the region was missing or malformed. ``think`` and
    ``deliverable`` come from composing :func:`parse_response` on the
    content after ``</brief>``. ``errors`` is an ordered list of all
    issues encountered so the smoke can render them next to the raw
    response.
    """

    brief: dict[str, Any] | None
    think: str | None
    deliverable: str | None
    errors: list[str] = field(default_factory=list)


def parse_single_pass_response(content: str) -> SinglePassParseResult:
    """Parse a single-pass teacher response.

    The output is ``<brief>...</brief>`` followed by the canonical
    ``<think>`` + freeform deliverable. We extract the brief JSON
    separately, then delegate to :func:`parse_response` on the
    remainder so parsers share the same think/deliverable rules.
    ``task`` is emitted by the model as part of the brief — we do not
    inject it here.
    """
    errors: list[str] = []

    if not content or not content.strip():
        return SinglePassParseResult(
            brief=None, think=None, deliverable=None, errors=["empty content"]
        )

    brief_payload: dict[str, Any] | None = None
    bm = _BRIEF_RE.search(content)
    if bm:
        try:
            payload = json.loads(_strip_fence(bm.group(1)))
        except json.JSONDecodeError as e:
            errors.append(f"brief JSON parse: {e}")
        else:
            if isinstance(payload, dict):
                if not isinstance(payload.get("task"), str) or not payload["task"].strip():
                    errors.append("brief missing `task` string")
                brief_payload = payload
            else:
                errors.append("brief JSON is not an object")
        tail = content[bm.end() :]
    else:
        errors.append("missing <brief> region")
        tail = content

    parsed = parse_response(tail)
    if isinstance(parsed, ParseRejection):
        errors.append(f"response_parser: {parsed.value}")
        return SinglePassParseResult(
            brief=brief_payload, think=None, deliverable=None, errors=errors
        )
    if isinstance(parsed, ParsedResponse):
        return SinglePassParseResult(
            brief=brief_payload,
            think=parsed.think,
            deliverable=parsed.deliverable,
            errors=errors,
        )
    # Defensive: parse_response should only ever return the two types above.
    errors.append("unexpected response_parser output")
    return SinglePassParseResult(brief=brief_payload, think=None, deliverable=None, errors=errors)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "SINGLE_PASS_TEACHER_SYSTEM",
    "SinglePassParseResult",
    "build_single_pass_request",
    "build_single_pass_user_message",
    "parse_single_pass_response",
]
