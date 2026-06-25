"""LLM-based ad-copy extractor — strips rationale/chrome uniformly across configs.

The regex extractor in ``extract.clean_copy`` only catches trailing meta blocks
that use a narrow header allowlist (``rationale|why this works|analysis|notes|
strategy|approach|angle|audience``). It systematically misses:

  - Draper-r16's ``**Hook.** **Structure and sequence.** **Word choice.**``
    craft-analysis paragraphs (the model's training format includes them).
  - GOLD's unmarked pedagogical prose in ``Brief.reference_assistant`` —
    no header at all, just blank-line-separated rationale.
  - Config-B emoji-spam hallucinations on long-form briefs.

The asymmetry contaminates pairwise judging: the judge sees pure ad copy
for some configs and ad-copy-plus-rationale for others, then marks the
latter down for being "meta-instructions." This module wraps each
candidate in a single cheap LLM extraction pass that emits just the
user-facing ad copy (headline + body + CTA, verbatim) — same shape
across every config including GOLD.

Cleaned outputs are cached per ``(config, example_id)`` under
``data/eval/inferences_clean/<config>/<example_id>.json`` so judging
stays cheap on rerun and so the raw inferences on disk stay untouched
for forensics.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..schemas import Inference
from .extract import clean_copy

logger = logging.getLogger(__name__)

EXTRACTION_FAILED = "<EXTRACTION_FAILED>"
"""Sentinel returned when the response is incoherent (emoji spam, no copy)."""

DEFAULT_EXTRACTOR_MODEL = "claude-haiku-4-5"
"""Cheap extractor — passed our judge-validation gate at >80% on Upworthy."""

# Surfaced to the judge in place of the sentinel so the judge sees a clear
# non-response rather than an opaque magic string.
NO_COPY_PLACEHOLDER = "[Model produced no usable ad copy.]"

EXTRACTION_SYSTEM_PROMPT = (
    "You extract the user-facing copy from a model's response.\n"
    "\n"
    "Return ONLY what would actually appear when the copy is published on"
    " the platform — the headline, body text, and call-to-action label, in"
    " the order the writer presented them. This includes ad copy, product"
    " descriptions, listing copy, landing page copy — any user-facing"
    " marketing text. The format may be a single sentence, a multi-line"
    " ad, or a structured product page; preserve the writer's structure.\n"
    "\n"
    "Strip:\n"
    '- preambles like "Here\'s the ad:", "Sure, I\'ll write...", "ok, i\'d'
    ' keep it really direct..."\n'
    "- format chrome and section labels like **Headline:**, **Ad Copy:**,"
    " **CTA:**, ## Why This Works, **Hook.**, **Structure.**, **Word"
    " choice.**, **Punctuation and line breaks.**\n"
    '- analytical or pedagogical commentary (rationale, "Why this works",'
    ' "This works because...", craft notes, strategy explanations,'
    " breakdowns of why each line/word was chosen)\n"
    "- thinking blocks like <think>...</think>\n"
    "- meta-instructions about how to deploy the copy\n"
    "\n"
    "Keep verbatim:\n"
    "- the actual headline, body text, and CTA exactly as written\n"
    "- emojis, punctuation, line breaks within the copy\n"
    "- capitalization and word choice, even if clunky\n"
    "\n"
    "Rules:\n"
    "- Do not rewrite or paraphrase. If the writer used an awkward phrase,"
    " keep it.\n"
    "- Do not invent a CTA the writer did not write.\n"
    "- If the response contains multiple variant ads, return only the FIRST"
    " complete variant.\n"
    "- If the response went into a loop (the same line, paragraph, or"
    " sentence repeated 3+ times verbatim) AFTER producing real copy,"
    " include only the FIRST coherent instance and stop. Do not include"
    " the repeated chunks.\n"
    "- If the response is *purely* incoherent — pure emoji spam with no"
    " words, only the product title repeated 5+ times with no surrounding"
    " sentences, or empty — output the literal string <EXTRACTION_FAILED>"
    " on its own line and nothing else. A response with at least one full"
    " coherent sentence of marketing copy is NOT incoherent.\n"
    "- Output only the copy text — no quotes, backticks, JSON, or"
    " commentary.\n"
)


def _user_prompt(raw: str, *, platform: str | None) -> str:
    plat = platform or "unspecified"
    return f"Platform: {plat}\n\nResponse to extract from:\n---\n{raw}\n---"


def _strip_wrapping(text: str) -> str:
    """Strip code fences or matched-quote wrappers the model sometimes adds.

    Conservative: only strips when both ends match. We don't unwrap quotes that
    appear only on one side because real ad copy frequently opens with a quote
    ("…the best yogurt I've ever had!" — User).
    """
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


# ---------------------------------------------------------------------------
# Arm-2 (agent-pipeline) campaign flattening
# ---------------------------------------------------------------------------

# Keys in an emitted ``campaign`` dict that are NOT user-facing published copy:
# destination URLs, asset/video briefs, account handles, and scoring metadata.
# Everything else that is string- or list-of-string-valued is treated as copy.
_CAMPAIGN_NON_COPY_KEYS = frozenset(
    {
        "platform",
        "final_url",
        "destination_url",
        "url",
        "asset_brief",
        "asset_image_url",
        "asset_url",
        "image_url",
        "video_brief",
        "image_brief",
        "display_name",
        "account",
        "handle",
        "username",
        "business_name",
        "path1",
        "path2",
        "meta",
        "requirements_check",
        "craft",
        "predicted_score",
        "predicted_scores",
        "violations",
        "brand",
        "brand_assets",
        "score",
        "scores",
    }
)

# Preferred ordering so flattened copy reads headline -> body -> CTA, matching
# how the extractor presents published copy. Unlisted copy keys are appended in
# dict order.
_CAMPAIGN_COPY_ORDER = (
    "title",
    "headline",
    "headlines",
    "long_headline",
    "primary_text",
    "text",
    "body",
    "description",
    "descriptions",
    "caption",
    "cta_label",
    "cta",
)


def campaign_published_copy(campaign: Mapping[str, Any]) -> str:
    """Flatten an emitted ``campaign`` dict to its user-facing published copy.

    Arm-2 (agent-pipeline) inferences carry the canonical ad copy in the
    structured ``campaign`` object emitted by ``emit_campaign``; the raw
    ``assistant_text`` is frequently just a one-line summary
    ("Drafted a PINTEREST campaign: ..."). This joins the copy-bearing fields
    (headline, body, CTA, in reading order) and drops asset/video briefs,
    account handles, destination URLs, and scoring metadata — yielding the same
    shape the LLM extractor produces for single-shot configs. Returns ``""`` if
    no copy-bearing field is present.
    """
    parts: list[str] = []
    seen: set[str] = set()

    def _add(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                parts.append(text)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())

    for key in _CAMPAIGN_COPY_ORDER:
        if key in campaign and key not in _CAMPAIGN_NON_COPY_KEYS:
            _add(campaign[key])
            seen.add(key)
    for key, value in campaign.items():
        if key in seen or key in _CAMPAIGN_NON_COPY_KEYS:
            continue
        _add(value)
    return "\n\n".join(parts)


async def extract_ad_copy(
    raw: str,
    *,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    platform: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """Extract just the user-facing ad copy from a model's response.

    Returns the ad copy text, or the ``EXTRACTION_FAILED`` sentinel when
    the input is incoherent / empty / has no usable ad copy.
    """
    if not raw or not raw.strip():
        return EXTRACTION_FAILED

    # Lazy import — utils.llm_client pulls in google-genai which is heavy
    # for callers that only want the path helpers (e.g., judge-time reads).
    from ...utils.llm_client import complete_with_usage

    result = await complete_with_usage(
        messages=[{"role": "user", "content": _user_prompt(raw, platform=platform)}],
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        system=EXTRACTION_SYSTEM_PROMPT,
    )
    text = (result.content or "").strip()
    if not text:
        return EXTRACTION_FAILED
    # The model is told to emit the literal sentinel on incoherent input.
    if EXTRACTION_FAILED in text:
        return EXTRACTION_FAILED
    text = _strip_wrapping(text)
    # Defense in depth: any residual headers/preambles still get the regex pass.
    text = clean_copy(text)
    if not text:
        return EXTRACTION_FAILED
    return text


# ---------------------------------------------------------------------------
# On-disk cache: data/eval/inferences_clean/<config>/<example_id>.json
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanedRecord:
    """One cached extraction result, persisted alongside the raw inference."""

    example_id: str
    config: str
    assistant_text_clean: str
    extractor_model: str
    extracted_at: str
    raw_text_sha256: str
    """SHA256 of the source text — invalidates cache when the source changes."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_path(root: Path, config: str, example_id: str) -> Path:
    """Return the on-disk path for a cleaned record."""
    return root / config / f"{example_id}.json"


def load_clean(root: Path, config: str, example_id: str) -> CleanedRecord | None:
    """Load a cached cleaned record, or None if missing / unreadable."""
    path = clean_path(root, config, example_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return CleanedRecord(
            example_id=data["example_id"],
            config=data["config"],
            assistant_text_clean=data["assistant_text_clean"],
            extractor_model=data["extractor_model"],
            extracted_at=data["extracted_at"],
            raw_text_sha256=data.get("raw_text_sha256", ""),
        )
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        logger.warning(f"Failed to read cleaned record at {path}: {exc}")
        return None


def save_clean(record: CleanedRecord, root: Path) -> None:
    """Persist a cleaned record. Creates parent dirs as needed."""
    path = clean_path(root, record.config, record.example_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "example_id": record.example_id,
                "config": record.config,
                "assistant_text_clean": record.assistant_text_clean,
                "extractor_model": record.extractor_model,
                "extracted_at": record.extracted_at,
                "raw_text_sha256": record.raw_text_sha256,
            },
            f,
            indent=2,
        )


async def extract_and_cache(
    *,
    raw: str,
    config: str,
    example_id: str,
    clean_root: Path,
    platform: str | None = None,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    force: bool = False,
) -> CleanedRecord:
    """Extract ad copy and persist the result. Skips when cache is valid.

    Cache is invalidated when the SHA256 of ``raw`` differs from the stored
    hash (e.g., the inference was re-run with new model weights).
    """
    raw_hash = _sha256(raw)
    if not force:
        existing = load_clean(clean_root, config, example_id)
        if existing is not None and existing.raw_text_sha256 == raw_hash:
            return existing
    cleaned = await extract_ad_copy(raw, model=model, platform=platform)
    record = CleanedRecord(
        example_id=example_id,
        config=config,
        assistant_text_clean=cleaned,
        extractor_model=model,
        extracted_at=datetime.now(UTC).isoformat(),
        raw_text_sha256=raw_hash,
    )
    save_clean(record, clean_root)
    return record


# ---------------------------------------------------------------------------
# Judge-time integration
# ---------------------------------------------------------------------------


def judge_input_text(
    inference: Inference,
    *,
    clean_root: Path | None,
) -> str:
    """Resolve the text the judge should score for an inference.

    Preference order:
      1. Cached cleaned text from the LLM extractor (uniform across configs).
      2. ``clean_copy(raw assistant_text)`` — regex fallback (asymmetric).

    When the cleaned text is the ``EXTRACTION_FAILED`` sentinel, returns a
    human-readable placeholder so the judge sees a clear non-response.
    """
    if clean_root is not None:
        rec = load_clean(clean_root, inference.config, inference.example_id)
        if rec is not None:
            if rec.assistant_text_clean == EXTRACTION_FAILED:
                return NO_COPY_PLACEHOLDER
            return rec.assistant_text_clean
    return clean_copy(inference.assistant_text)
