"""Religious-scripture ad detector.

Purpose: exclude ads whose primary copy *is* religious scripture text —
Quranic verses, Bible verses, and other sacred-text quotations — from the
clustering pool. These ads don't fit Draper.ai's marketing-reasoning
fine-tune: the "copy" is ancient text, not strategic craft, so a teacher
reshaping a scenario around them produces semantic mismatches (see the
``b2b_demand_event_pipeline`` paired with a Quranic YouTube ad in the
15-example copywriting pilot).

Design goal: **high precision, moderate recall**. We specifically do
*not* want to filter out legitimate faith-based marketing — a nonprofit
charity ad mentioning "faith-based relief," a church announcing Sunday
service, a mosque hosting an event. Those are normal marketing copy.
What we filter is ads whose headline/body copy consists of scripture
itself. When in doubt, keep the ad.

Detection signals (any one is sufficient):

1. **Diacritized Arabic text.** Classical-orthography Arabic (with
   fatha/kasra/damma/sukun marks) is overwhelmingly used for Quranic
   quotation in contemporary media. Modern marketing Arabic almost
   never carries diacritics — they're a near-unique fingerprint for
   scripture. Threshold: 4+ diacritic codepoints in the ad text.

2. **Scripture-style curly braces around Arabic.** The typographic
   convention ``{ … }`` (ASCII) or ``﴿ … ﴾`` (Unicode ornate)
   wrapping Arabic text is the standard print/web convention for
   Quranic verse quotation.

3. **Bible verse citations.** Book-chapter:verse patterns
   (``John 3:16``, ``Romans 8:28-30``) matched against the full
   canonical Bible book list.

4. **Explicit scripture references.** Direct mentions of ``Quran``,
   ``Qur'an``, ``Koran``, ``Holy Bible``, ``New Testament``,
   ``Old Testament``, or ``Scripture`` in the ad copy.

5. **Canonical scripture phrases.** A small curated list of verbatim
   lines that only appear as Bible quotes (``"For God so loved the
   world"``, ``"The Lord is my shepherd"``, etc.).

Not flagged:
- Generic faith mentions ("faith-based", "God bless", "pray for")
- Church / mosque / temple event marketing
- Religious-goods retail (Christmas decor, prayer beads)
- Nonprofit copy invoking faith as audience affinity
"""

from __future__ import annotations

import re

from draper.scoring.schemas import ScoredAd

# ---------------------------------------------------------------------------
# Signal 1: diacritized Arabic
# ---------------------------------------------------------------------------

# Arabic diacritic codepoints (tashkeel): fatha, kasra, damma, sukun, shadda,
# tanwin variants, and letter-position marks. U+064B..U+065F covers the core
# set used in Quranic typography.
_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_ARABIC_DIACRITIC_THRESHOLD = 4

# ---------------------------------------------------------------------------
# Signal 2: scripture-style curly-brace quotation
# ---------------------------------------------------------------------------

# ASCII curly braces enclosing Arabic text, OR the Unicode ornate parenthesis
# pair U+FD3E / U+FD3F commonly used for Quranic verse delimitation.
_ARABIC_CHAR = r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
_BRACE_QURAN_PATTERN = re.compile(
    r"\{[^{}]*" + _ARABIC_CHAR + r"[^{}]*\}"
    r"|\uFD3F[^\uFD3F\uFD3E]*\uFD3E",
)

# ---------------------------------------------------------------------------
# Signal 3: Bible verse citation
# ---------------------------------------------------------------------------

_BIBLE_BOOKS = (
    # Old Testament
    "Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|"
    r"(?:1|2|I|II)\s*Samuel|(?:1|2|I|II)\s*Kings|(?:1|2|I|II)\s*Chronicles|"
    "Ezra|Nehemiah|Esther|Job|Psalms?|Proverbs|Ecclesiastes|"
    r"Song\s+of\s+(?:Solomon|Songs)|Isaiah|Jeremiah|Lamentations|Ezekiel|"
    "Daniel|Hosea|Joel|Amos|Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|"
    "Haggai|Zechariah|Malachi|"
    # New Testament
    "Matthew|Mark|Luke|John|Acts|Romans|"
    r"(?:1|2|I|II)\s*Corinthians|Galatians|Ephesians|Philippians|Colossians|"
    r"(?:1|2|I|II)\s*Thessalonians|(?:1|2|I|II)\s*Timothy|Titus|Philemon|"
    r"Hebrews|James|(?:1|2|I|II)\s*Peter|(?:1|2|3|I|II|III)\s*John|Jude|"
    "Revelation"
)
_BIBLE_VERSE_PATTERN = re.compile(
    rf"\b(?:{_BIBLE_BOOKS})\s+\d+:\d+(?:[-\u2013]\d+)?\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Signal 4: explicit scripture references
# ---------------------------------------------------------------------------

_EXPLICIT_SCRIPTURE_TERMS = re.compile(
    r"\b(?:qur[\u2019']?an|quran|koran|holy\s+bible|"
    r"new\s+testament|old\s+testament|holy\s+scripture|"
    r"book\s+of\s+mormon|torah|tanakh)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Signal 5: canonical scripture phrases (curated, verbatim only)
# ---------------------------------------------------------------------------

_CANONICAL_PHRASES = re.compile(
    r"(?:"
    r"for\s+god\s+so\s+loved\s+the\s+world"
    r"|the\s+lord\s+is\s+my\s+shepherd"
    r"|blessed\s+are\s+the\s+(?:meek|poor|peacemakers|merciful)"
    r"|our\s+father\s+(?:who|which)\s+art\s+in\s+heaven"
    r"|in\s+the\s+beginning\s+(?:god\s+created|was\s+the\s+word)"
    r"|the\s+word\s+became\s+flesh"
    r"|i\s+am\s+the\s+way(?:,|\s+and)\s+the\s+truth"
    r"|peace\s+be\s+upon\s+(?:him|you)"
    r"|subhan[a']?\s*allah"
    r"|alhamdulillah"
    r"|bismillah(?:ir[\s-]?rahman)?"
    r"|la\s+ilaha\s+illa(?:\s+llah|h)"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _joined_text(ad: ScoredAd) -> str:
    """Concatenate all ad-copy fields into a single searchable string."""
    copy = ad.ad.ad_copy
    parts = [copy.headline, copy.body, copy.description, copy.cta]
    return "\n".join(p for p in parts if p)


def is_religious_scripture_text(text: str) -> tuple[bool, str]:
    """Return ``(True, reason)`` when ``text`` matches a scripture signal.

    Text-only variant for callers (v2 construction) that work from raw
    JSONL rows without materializing a full :class:`ScoredAd`.
    """
    if not text.strip():
        return False, ""

    # Signal 1: diacritized Arabic.
    diacritics = _ARABIC_DIACRITICS.findall(text)
    if len(diacritics) >= _ARABIC_DIACRITIC_THRESHOLD:
        return True, "diacritized_arabic"

    # Signal 2: scripture-style braces around Arabic.
    if _BRACE_QURAN_PATTERN.search(text):
        return True, "arabic_brace_quotation"

    # Signal 3: Bible verse citation.
    if _BIBLE_VERSE_PATTERN.search(text):
        return True, "bible_verse_citation"

    # Signal 4: explicit scripture mention.
    if _EXPLICIT_SCRIPTURE_TERMS.search(text):
        return True, "explicit_scripture_term"

    # Signal 5: canonical phrases.
    if _CANONICAL_PHRASES.search(text):
        return True, "canonical_scripture_phrase"

    return False, ""


def is_religious_scripture_ad(ad: ScoredAd) -> tuple[bool, str]:
    """Return ``(True, reason)`` when the ad's copy is religious scripture.

    The reason string is a short tag identifying which signal fired,
    useful for logging and for spot-checking false positives.
    """
    return is_religious_scripture_text(_joined_text(ad))
