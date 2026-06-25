"""Copywriting-specific quality filters.

Two extra stages that only apply to backtranslation responses:

- **Schema-leak** — the teacher used to see the source ad as a labeled
  block (``headline:``, ``body:``, ``cta:``). Weaker/literal models
  mirrored that scaffold into their prose (review batch Ex 6/7/10). The
  ad_facts block is now prose, but this filter stays as a defensive
  backstop: any line-starting "Label:" from the scaffold vocabulary is
  rejected.

- **Ad-centrality** — the grounding rule says the real ad IS the
  answer. Hedges like "you could also try a different angle" or "an
  alternative would be" signal the teacher diverged into forward-mode
  speculation. Only applies to backtranslation; forward styles
  legitimately discuss alternatives.

Also exposes a lower min-length floor for backtranslation responses: the
real ad + short rationale is naturally tight and the default 200-char
floor (tuned for forward-mode multi-variant responses) false-rejects.
"""

from __future__ import annotations

import re

from draper.construction.schemas import PromptStyle, TrainingExample

BACKTRANSLATION_MIN_LENGTH_FLOOR: int = 80

_SCHEMA_LEAK_PATTERN = re.compile(
    r"^\s*(?:advertiser|platform|business\s+vertical|landing\s+page|"
    r"creative\s+format|headline|body|description|cta|ad\s+copy|"
    r"business\s+category)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

_AD_CENTRALITY_HEDGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\byou could also try\b", re.IGNORECASE),
    re.compile(r"\byou (?:might|may) (?:also )?want to try\b", re.IGNORECASE),
    re.compile(r"\ban alternative (?:would be|is|could be)\b", re.IGNORECASE),
    re.compile(r"\banother (?:approach|option) (?:would be|is|could be)\b", re.IGNORECASE),
    re.compile(r"\bone option (?:would be|is|could be)\b", re.IGNORECASE),
    re.compile(r"\ba better (?:approach|version|headline|option) would be\b", re.IGNORECASE),
    re.compile(r"\binstead,? you could\b", re.IGNORECASE),
    re.compile(r"\bi(?:'d| would) (?:also )?suggest (?:trying|writing|testing) a\b", re.IGNORECASE),
    re.compile(r"\bmy recommendation would be to (?:rewrite|rework|revise)\b", re.IGNORECASE),
    re.compile(r"\bif i were (?:rewriting|reworking) this\b", re.IGNORECASE),
]


def _get_assistant_content(example: TrainingExample) -> str:
    for msg in example.messages:
        if msg.role == "assistant":
            return msg.content
    return ""


def check_schema_leak(example: TrainingExample) -> str:
    """Return the leaked label, or empty string if clean."""
    if example.metadata.prompt_style != PromptStyle.BACKTRANSLATION:
        return ""
    response = _get_assistant_content(example)
    match = _SCHEMA_LEAK_PATTERN.search(response)
    if not match:
        return ""
    return match.group(0).strip().rstrip(":").strip().lower()


def check_ad_centrality(example: TrainingExample) -> str:
    """Return the matched hedge phrase, or empty string if clean."""
    if example.metadata.prompt_style != PromptStyle.BACKTRANSLATION:
        return ""
    response = _get_assistant_content(example)
    for pat in _AD_CENTRALITY_HEDGE_PATTERNS:
        match = pat.search(response)
        if match:
            return match.group(0).lower()
    return ""


def min_length_floor(example: TrainingExample, default_floor: int) -> int:
    """Lower floor for backtranslation responses; default otherwise."""
    if example.metadata.prompt_style == PromptStyle.BACKTRANSLATION:
        return BACKTRANSLATION_MIN_LENGTH_FLOOR
    return default_floor


def extra_filter_reasons(example: TrainingExample) -> list[tuple[str, str]]:
    """Return ``(example_id, reason)`` pairs for any violations."""
    reasons: list[tuple[str, str]] = []
    leak = check_schema_leak(example)
    if leak:
        reasons.append((example.example_id, f"schema_leak:{leak}"))
    hedge = check_ad_centrality(example)
    if hedge:
        reasons.append((example.example_id, f"ad_centrality:{hedge}"))
    return reasons
