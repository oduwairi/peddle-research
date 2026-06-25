"""Copywriting ingestion validation.

Backtranslation's premise is that the real ad's copy IS the assistant
response. Two fidelity checks defend that premise at ingest time:

1. **Word coverage** — a floor fraction of the source ad's content
   words must reappear in the response. Catches teachers that fabricated
   fresh copy instead of reproducing the ad.
2. **Verbatim signature** — a contiguous k-word phrase from the source
   ad's longest field must appear in the response (modulo formatting).
   Catches pure-rationale responses that talk about the ad without
   quoting it (review batch Ex 8 pattern).

Both checks skip short / low-signal ads so they don't false-reject.
"""

from __future__ import annotations

import re

from draper.scoring.schemas import ScoredAd
from draper.utils.language import detect_language

# Word-coverage check.
BACKTRANS_MIN_WORD_COVERAGE: float = 0.60
BACKTRANS_MIN_WORD_LEN: int = 4
BACKTRANS_MIN_AD_WORDS: int = 5

# Verbatim signature check.
_VERBATIM_MIN_FIELD_LEN: int = 15
_VERBATIM_SIGNATURE_WORDS: int = 6

# Per-field langdetect floor — strings shorter than this are ambiguous to
# the detector and are kept regardless of outcome. Mirrors the floor used
# by the bundle formatter (see ``constructor._ad_facts``); kept here so
# both the formatter and the ingestion checks agree on which source-ad
# fields the teacher actually saw.
_LANG_DETECT_MIN_LEN = 20

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def field_is_english(text: str) -> bool:
    """True if ``text`` is short, empty, or detected as English.

    Single source of truth for the English-only filter applied to source
    ad fields. The bundle formatter uses it to decide which fields to
    show the teacher; the ingestion checks use it to decide which fields
    to verify against. Keeping the two in sync prevents false rejections
    where the teacher correctly reproduced what we showed it but our
    coverage check expected text we hid.
    """
    if not text or len(text) < _LANG_DETECT_MIN_LEN:
        return True
    code = detect_language(text, "")
    return not code or code == "en"


def _content_words(text: str, min_len: int = BACKTRANS_MIN_WORD_LEN) -> set[str]:
    """Extract lowercased content words (≥ ``min_len`` chars, alpha-only)."""
    return {
        m.group(0).lower()
        for m in _WORD_RE.finditer(text)
        if len(m.group(0)) >= min_len
    }


def _normalize_for_match(text: str) -> str:
    """Collapse to lowercase alphanumeric-only, single-spaced.

    Tolerates formatting variation (emoji, markdown, quote styles, extra
    whitespace) so a headline quoted in the response still matches its
    source even if the teacher wrapped it in quotes or bolded it.
    """
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


def _signature_phrase(text: str, word_count: int = _VERBATIM_SIGNATURE_WORDS) -> str:
    """First ``word_count`` alphanumeric words of ``text`` as a signature."""
    tokens = _normalize_for_match(text).split()
    if len(tokens) < word_count:
        return ""
    return " ".join(tokens[:word_count])


def check_word_coverage(
    assistant_response: str, source_ads: list[ScoredAd]
) -> tuple[bool, float, int]:
    """Return ``(passed, coverage, ad_word_count)`` for the coverage check.

    Unions all text fields from the source ads (``ad_copy.headline +
    body + description + cta``), extracts content words, and measures
    what fraction appears in the assistant_response. Ads shorter than
    ``BACKTRANS_MIN_AD_WORDS`` content words skip the check (returns
    ``(True, 1.0, n)`` so short ads don't false-reject).
    """
    ad_text_parts: list[str] = []
    for ad in source_ads:
        copy = ad.ad.ad_copy
        for field in (copy.headline, copy.body, copy.description, copy.cta):
            if field and field_is_english(field):
                ad_text_parts.append(field)
    ad_words = _content_words(" ".join(ad_text_parts))
    if len(ad_words) < BACKTRANS_MIN_AD_WORDS:
        return True, 1.0, len(ad_words)
    response_words = _content_words(assistant_response)
    overlap = ad_words & response_words
    coverage = len(overlap) / len(ad_words)
    return coverage >= BACKTRANS_MIN_WORD_COVERAGE, coverage, len(ad_words)


def check_verbatim_signature(
    assistant_response: str, source_ads: list[ScoredAd]
) -> tuple[bool, str]:
    """Check a signature phrase from the source ad appears verbatim in the response.

    Tries every usable field (headline, body, description) — passes if
    *any* one of them has its first ``_VERBATIM_SIGNATURE_WORDS`` content
    words appear consecutively in the response (modulo formatting). The
    word-coverage check catches paraphrases; this catches "rationale
    only, no ad" responses that describe the ad without quoting it.
    Checking every field (not just headline) avoids false-rejecting
    teachers who preserved the body or description but the headline was
    longer / less recognisable.
    """
    if not source_ads:
        return True, ""
    copy = source_ads[0].ad.ad_copy
    candidates = [
        f for f in (copy.headline, copy.body, copy.description)
        if f and len(f) >= _VERBATIM_MIN_FIELD_LEN and field_is_english(f)
    ]
    if not candidates:
        return True, ""
    normalized_response = _normalize_for_match(assistant_response)
    for field in candidates:
        signature = _signature_phrase(field)
        if signature and signature in normalized_response:
            return True, ""
    return False, (
        f"Verbatim signature absent: first {_VERBATIM_SIGNATURE_WORDS} "
        f"content words of every source field "
        f"(headline/body/description) failed to appear in the response "
        f"as a contiguous phrase. Teacher likely paraphrased the ad."
    )
