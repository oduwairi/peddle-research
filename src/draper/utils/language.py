"""Language detection utilities for Draper.ai ad copy."""

from __future__ import annotations

import logging

logger = logging.getLogger("draper")

# Minimum combined character length to attempt detection.
# Below this langdetect is unreliable (returns garbage for 1-3 word strings).
_MIN_TEXT_LEN = 12


def detect_language(headline: str, body: str, description: str = "") -> str:
    """Detect the primary language of an ad's copy.

    Combines headline, body, and description for a richer signal. Returns
    an ISO 639-1 language code (e.g. ``"en"``, ``"fr"``, ``"ar"``), or
    ``""`` when the text is too short or ambiguous to classify confidently.

    Including ``description`` matters: some ads have a short ALL-CAPS or
    brand-heavy body that langdetect mis-classifies as English even when
    the longer description is plainly another language.

    Args:
        headline: Ad headline copy.
        body: Ad body copy.
        description: Ad description copy (optional — folded in for richer signal).

    Returns:
        ISO 639-1 code string, or empty string if undetectable.
    """
    try:
        from langdetect import detect
    except ImportError:
        logger.warning("langdetect not installed — language detection disabled")
        return ""

    text = " ".join(s for s in (headline, body, description) if s).strip()
    if len(text) < _MIN_TEXT_LEN:
        return ""

    try:
        return str(detect(text))
    except Exception:  # LangDetectException or any other failure
        return ""
