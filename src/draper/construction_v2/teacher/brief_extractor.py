"""LEGACY (two-stage) — Stage 1 brief extractor.

The production v2 pipeline uses the unified single-pass teacher in
``teacher/single_pass.py``. This module is retained until Phase 4
removes the two-stage path.

Two execution modes (legacy):

- **Chat mode** (:func:`extract_brief`) — synchronous Anthropic call,
  used for smoke testing and small N. ~1s/ad.
- **Batch mode** (:func:`build_brief_batch_requests`) — emits
  :class:`BatchRequest` records suitable for ``BatchClient.submit``.

The teacher emits structured output: the Brief schema is passed as a
tool definition with ``tool_choice`` forcing the call, so the model can
only respond with a valid Brief JSON payload.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic
from pydantic import ValidationError

from draper.construction.batch.types import BatchRequest
from draper.construction_v2.config import BriefExtractionConfig
from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.schemas.brief import (
    SUPPORTED_PLATFORMS,
    Brief,
    BriefBridge,
    BriefProduct,
)
from draper.utils.llm_client import _get_anthropic

logger = logging.getLogger("draper")


# The task field on Brief is the discriminator between Draper skills.
# For the ad-copy slice the value is fixed — injected here, not asked of
# the teacher. When a second slice is added (positioning, diagnostics,
# etc.) it'll parameterize this constant.
AD_COPY_TASK = "Write ad copy for the platform below."


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


BRIEF_EXTRACTION_SYSTEM_PROMPT = (
    "You are a marketing analyst building structured briefs from "
    "real, high-performing ads.\n"
    "\n"
    "## Grounding contract (read this first)\n"
    "\n"
    "Every field you fill MUST be derivable from the source you are "
    "given: the ad's copy text plus the supplied metadata "
    "(advertiser_name, landing_page_url, platform). World knowledge "
    "about the brand is FORBIDDEN. If the ad does not state or "
    "clearly imply a fact, you MUST leave that field null or empty. "
    "A blank field is the correct, honest answer — not a failure.\n"
    "\n"
    "Why: at inference time a downstream agent fills these same "
    "fields from real research. If you fabricate features, USPs, or "
    "proof points the ad doesn't mention, the model learns to ignore "
    "those fields entirely, and the trained writer becomes "
    "unsteerable. Hallucinated facts here directly damage the "
    "product. Treat null as a first-class answer.\n"
    "\n"
    "## How to decide per field\n"
    "\n"
    "Read the ad. For each product field, ask: 'is this stated or "
    "clearly readable from the ad text or metadata?' If yes, fill it. "
    "If no, leave it null/empty and MOVE ON. Do not reach for "
    "world knowledge about the advertiser to backfill features.\n"
    "\n"
    "Examples of correct restraint:\n"
    "- Ad is a single hook headline ('should we drop the corn "
    "borger?'): ``key_features`` = [], ``unique_selling_points`` = "
    "[], ``proof_points`` = [], ``description`` = null, "
    "``price_info`` = null. ``name`` may use the advertiser when it "
    "is clearly the product name; otherwise null.\n"
    "- Ad lists specs in the body (modem with port counts, antenna "
    "gain, etc.): fill ``key_features`` from what the ad actually "
    "lists. Do not add features the ad doesn't mention.\n"
    "- Ad never mentions price: ``price_info`` = null. Even if the "
    "category typically has a known price band.\n"
    "\n"
    "## Sections\n"
    "\n"
    "1. **product** — grounded facts about the product. The ONLY "
    "required product field is ``tone_signals`` (voice cues you read "
    "off the ad's natural voice — always derivable). All other "
    "product fields are optional and should be null/empty when the "
    "ad doesn't support them.\n"
    "\n"
    "2. **bridge** — strategic interpretation of the ad. ``angle`` "
    "and ``buyer_pain`` are required (a successful ad always carries "
    "these). ``positioning`` and ``target_audience`` are optional — "
    "leave null when the ad alone doesn't reveal them.\n"
    "\n"
    "## Hard rules\n"
    "\n"
    "- Bridge fields must NEVER quote or paraphrase the ad's copy. "
    "Restate the strategic intent in your own words — no verbatim "
    "phrases from the ad copy may appear in any bridge field.\n"
    '- ``tone_signals`` is REQUIRED, non-empty (e.g. ``"crisp"``, '
    '``"warm-but-direct"``, ``"playful"``). These are voice cues, '
    'not topic descriptors. ``"safety-focused"``, '
    '``"community-oriented"``, ``"low-friction"`` are NOT valid — '
    "those describe subject matter or positioning, not voice.\n"
    '- ``angle`` is a short LABEL like "problem-aware skeptic" or '
    '"aspirational founder identity" — NOT a hook line.\n'
    '- ``buyer_pain`` is a description like "Sundays lost to '
    'payroll spreadsheets" — NOT a tagline.\n'
    "- ``platform`` must be exactly one of: "
    + ", ".join(SUPPORTED_PLATFORMS)
    + ". Use the supplied platform hint."
)


# The Anthropic tool schema mirrors :class:`Brief`. Two-way binding via a
# unit test that diffs this schema against `Brief.model_json_schema()`.
_BRIEF_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "product": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
                "category": {"type": ["string", "null"]},
                "key_features": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "unique_selling_points": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "price_info": {"type": ["string", "null"]},
                "tone_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "category_context": {"type": ["string", "null"]},
                "proof_points": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "offer": {"type": ["string", "null"]},
            },
            "required": ["tone_signals"],
        },
        "bridge": {
            "type": "object",
            "properties": {
                "positioning": {"type": ["string", "null"]},
                "target_audience": {"type": ["string", "null"]},
                "angle": {"type": "string"},
                "buyer_pain": {"type": "string"},
            },
            "required": ["angle", "buyer_pain"],
        },
        "platform": {"type": "string", "enum": list(SUPPORTED_PLATFORMS)},
    },
    "required": ["product", "bridge", "platform"],
}

_PLATFORM_ALIASES: dict[str, str] = {
    "facebook": "meta",
    "instagram": "meta",
    "twitter": "x",
    "youtube": "google",
    "linkedin": "meta",
}


def _normalize_platform(platform: str | None) -> str:
    """Coerce a source-ad platform string to the v2 surface enum."""
    if not platform:
        return "meta"
    key = platform.lower().strip()
    if key in SUPPORTED_PLATFORMS:
        return key
    return _PLATFORM_ALIASES.get(key, "meta")


def _user_message(ad: SourceAd) -> str:
    """Build the user-turn payload describing the source ad."""
    platform_hint = _normalize_platform(ad.platform)
    # Ad copy is passed as a single concatenated text rather than split into
    # API-level fields (headline / body / description / cta). The scraper's
    # field categorization is unreliable and the teacher's output doesn't
    # carry that structure either, so feeding the labels through invites
    # the model to reason against inaccurate cues.
    parts: list[str] = [
        "# Source ad",
        "",
        f"- ad_id: {ad.ad_id}",
        f"- platform_hint: {platform_hint}",
        "",
        "## Ad copy (verbatim)",
        "",
        ad.ad_copy_text if ad.ad_copy_text else "(empty)",
    ]
    advertiser = ad.raw.get("advertiser_name") if isinstance(ad.raw, dict) else None
    if isinstance(advertiser, str) and advertiser:
        parts.extend(["", f"- advertiser_name: {advertiser!r}"])
    landing = ad.raw.get("landing_page_url") if isinstance(ad.raw, dict) else None
    if isinstance(landing, str) and landing:
        parts.append(f"- landing_page_url: {landing!r}")
    parts.extend(
        [
            "",
            "Produce a Brief grounded ONLY in the source above. Leave "
            "fields null/empty when the ad does not support them — "
            "blank is the correct answer. Bridge fields must never "
            "quote the ad copy verbatim.",
        ]
    )
    return "\n".join(parts)


def _build_messages(ad: SourceAd) -> list[dict[str, str]]:
    return [{"role": "user", "content": _user_message(ad)}]


def _parse_brief_tool_input(tool_input: Any, *, ad: SourceAd) -> Brief:
    """Coerce a teacher's tool-call payload into a :class:`Brief`."""
    if not isinstance(tool_input, dict):
        msg = f"Brief tool_input not a dict for ad {ad.ad_id}: {type(tool_input)!r}"
        raise ValueError(msg)
    # Coerce platform so the v2 enum is satisfied even if the teacher
    # returns the source-ad platform key (e.g. "facebook" → "meta").
    platform = tool_input.get("platform")
    tool_input = {**tool_input, "platform": _normalize_platform(platform)}
    try:
        product = BriefProduct(**tool_input.get("product", {}))
        bridge = BriefBridge(**tool_input.get("bridge", {}))
        return Brief(
            task=AD_COPY_TASK,
            product=product,
            bridge=bridge,
            platform=tool_input["platform"],
        )
    except (ValidationError, TypeError, KeyError) as e:
        msg = f"Brief validation failed for ad {ad.ad_id}: {e}"
        raise ValueError(msg) from e


# Pre-compiled JSON-fence stripper for the chat-mode fallback path.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse_brief_text(text: str, *, ad: SourceAd) -> Brief:
    """Fallback parser for non-tool-using teachers (e.g. raw JSON return)."""
    stripped = text.strip()
    match = _JSON_FENCE_RE.search(stripped)
    if match:
        stripped = match.group(1).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as e:
        msg = f"Brief teacher returned non-JSON for ad {ad.ad_id}: {e}"
        raise ValueError(msg) from e
    return _parse_brief_tool_input(payload, ad=ad)


# ---------------------------------------------------------------------------
# Chat-mode extraction (smoke / integration tests)
# ---------------------------------------------------------------------------


async def extract_brief(
    ad: SourceAd,
    config: BriefExtractionConfig,
    *,
    client: anthropic.AsyncAnthropic | None = None,
) -> Brief:
    """Synchronously extract a Brief from one source ad.

    Used by the chat-mode CLI path for smoke testing. Production runs
    submit batch requests via :func:`build_brief_batch_requests`.
    """
    client = client or _get_anthropic()
    # Build kwargs as Any so mypy doesn't try to match `tools`/`tool_choice`
    # against the Anthropic SDK's deeply parameterised TypedDicts. The
    # SDK validates at runtime; we trust the schema we shipped above.
    create_kwargs: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "system": BRIEF_EXTRACTION_SYSTEM_PROMPT,
        "tools": [
            {
                "name": "emit_brief",
                "description": (
                    "Emit a structured Brief describing the source ad's product + strategic bridge."
                ),
                "input_schema": _BRIEF_TOOL_SCHEMA,
            }
        ],
        "tool_choice": {"type": "tool", "name": "emit_brief"},
        "messages": _build_messages(ad),
    }
    response = await client.messages.create(**create_kwargs)
    for block in response.content:
        if getattr(block, "type", "") == "tool_use":
            return _parse_brief_tool_input(block.input, ad=ad)
    # Fallback for teachers that ignored tool_choice and returned text.
    text_parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            text_parts.append(text)
    if not text_parts:
        msg = f"Brief teacher returned no content for ad {ad.ad_id}"
        raise ValueError(msg)
    return _parse_brief_text("\n".join(text_parts), ad=ad)


# ---------------------------------------------------------------------------
# Batch-mode requests
# ---------------------------------------------------------------------------


def build_brief_batch_requests(
    ads: list[SourceAd], config: BriefExtractionConfig
) -> list[BatchRequest]:
    """Build :class:`BatchRequest` records, one per source ad.

    custom_id format: ``"brief-{ad_id}"``. The ``-`` separator (vs
    ``:``) is required by Anthropic's Batch API custom_id pattern
    ``^[a-zA-Z0-9_-]{1,64}$``. Stage 2 references the same ad_id so
    briefs and rationales reconcile by direct lookup.
    """
    out: list[BatchRequest] = []
    for ad in ads:
        out.append(
            BatchRequest(
                custom_id=f"brief-{ad.ad_id}",
                system=BRIEF_EXTRACTION_SYSTEM_PROMPT,
                messages=_build_messages(ad),
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        )
    return out


def parse_brief_response_content(content: str, *, ad: SourceAd) -> Brief:
    r"""Parse a batch :attr:`BatchResponse.content` into a :class:`Brief`.

    Anthropic batch responses serialize all message blocks (including
    tool-use blocks) into the response ``content`` string as JSON in the
    raw API format. For our purposes, we accept either:

    - a raw JSON object body, or
    - a tool_use block whose ``input`` is the Brief payload, or
    - a fenced ``\`\`\`json ... \`\`\`\`` block.
    """
    stripped = content.strip()
    if not stripped:
        msg = f"Empty brief response for ad {ad.ad_id}"
        raise ValueError(msg)
    return _parse_brief_text(stripped, ad=ad)


__all__ = [
    "AD_COPY_TASK",
    "BRIEF_EXTRACTION_SYSTEM_PROMPT",
    "build_brief_batch_requests",
    "extract_brief",
    "parse_brief_response_content",
]
