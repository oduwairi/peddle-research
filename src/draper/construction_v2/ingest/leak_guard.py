"""Bridge-field leakage guard.

The cardinal v2 rule: bridge fields describe marketing INTENT, never the
ad's surface copy. If a teacher paraphrases the headline into ``angle``
or quotes the body in ``buyer_pain``, the model learns to copy bridge →
ad instead of reasoning from facts → ad.

Enforcement is double-locked:

- The brief-extraction system prompt forbids quoting (see
  :data:`brief_extractor.BRIEF_EXTRACTION_SYSTEM_PROMPT`).
- This guard rejects briefs whose any bridge field shares an n-gram of
  length ``n`` (default 5) with the source ad copy.

Also enforces the secondary v2 contract that ``tone_signals`` is
non-empty — the schema-side validator catches missing field, this guard
re-asserts it after the brief has cleared all other parse paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.schemas.brief import Brief

# n-gram length over which bridge → ad overlap is forbidden.
DEFAULT_NGRAM_N: int = 5

_WORD_RE = re.compile(r"\b[\w'-]+", re.UNICODE)


@dataclass(frozen=True)
class LeakResult:
    """Outcome of :func:`check_bridge_leak`."""

    passed: bool
    offending_field: str
    offending_ngram: str
    reason: str


def _tokenize(text: str) -> list[str]:
    """Tokenize text into words, normalized to lowercase.

    Handles Unicode word boundaries, accented characters, contractions,
    and both straight ('') and smart ("") quotes by normalizing quotes
    to their ASCII equivalents before tokenizing.
    """
    # Normalize smart quotes and dashes to ASCII equivalents for consistency
    normalized = text.replace('"', '"').replace('"', '"').replace(""", "'").replace(""", "'")
    return [m.group(0).lower() for m in _WORD_RE.finditer(normalized)]


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _bridge_fields(brief: Brief) -> dict[str, str]:
    """Bridge fields that have non-null values (optional fields skipped)."""
    raw: dict[str, str | None] = {
        "positioning": brief.bridge.positioning,
        "target_audience": brief.bridge.target_audience,
        "angle": brief.bridge.angle,
        "buyer_pain": brief.bridge.buyer_pain,
    }
    return {k: v for k, v in raw.items() if v}


def check_bridge_leak(brief: Brief, source_ad: SourceAd, *, n: int = DEFAULT_NGRAM_N) -> LeakResult:
    """Reject the brief if any bridge field shares an n-gram with the ad."""
    if not brief.product.tone_signals:
        # Belt-and-suspenders: the schema validator should already have
        # rejected this. Still surface it as a leak-guard reason so the
        # ingestion stats name the cause clearly.
        return LeakResult(
            passed=False,
            offending_field="product.tone_signals",
            offending_ngram="",
            reason="empty_tone_signals",
        )

    ad_tokens = _tokenize(source_ad.ad_copy_text)
    ad_ngrams = _ngrams(ad_tokens, n)
    if not ad_ngrams:
        return LeakResult(passed=True, offending_field="", offending_ngram="", reason="")
    for field_name, field_value in _bridge_fields(brief).items():
        bridge_tokens = _tokenize(field_value)
        bridge_ngrams = _ngrams(bridge_tokens, n)
        intersection = bridge_ngrams & ad_ngrams
        if intersection:
            sample = next(iter(intersection))
            return LeakResult(
                passed=False,
                offending_field=f"bridge.{field_name}",
                offending_ngram=" ".join(sample),
                reason=f"{n}gram_leak",
            )
    return LeakResult(passed=True, offending_field="", offending_ngram="", reason="")


__all__ = [
    "DEFAULT_NGRAM_N",
    "LeakResult",
    "check_bridge_leak",
]
