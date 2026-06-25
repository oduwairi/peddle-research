"""Content-safety filter for Draper.ai ad corpus.

Flags ads whose copy contains profanity, slurs, sexual/adult language, or
otherwise toxic text. Uses two complementary libraries:

    Level 1 — ``better_profanity`` wordlist match (explicit, high precision)
    Level 2 — ``alt_profanity_check`` ML probability (implicit toxicity)
    Level 0 — clean (neither signal fires)

Level 1 is a curated bad-word list with obfuscation handling (``f*ck``,
``@ss``, etc.). Level 2 is a linear SVM trained on toxic comments; it
catches phrasing that isn't a single bad word but reads as hostile or
sexual. Both libraries are English-only, so non-English ads are labelled
with ``level="L0_skip_lang"`` and flagged for review separately.

The module exposes a single entry point, :func:`classify_content_safety`,
plus the constants that the diagnostic runner pretty-prints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from better_profanity import profanity as _bp
from profanity_check import predict_prob as _pc_predict_prob

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

# better-profanity ships with a default word list; load once at import time.
_bp.load_censor_words()

# Word-like tokens including common obfuscation characters (@, *, $).
_WORD_RE = re.compile(r"[\w@*$]+", re.UNICODE)

# Default threshold on the alt-profanity-check probability. 0.5 is the
# library's own decision boundary; can be tuned via classify_content_safety().
DEFAULT_ML_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyResult:
    flag: bool
    level: str  # "L1_wordlist" | "L2_ml" | "L0_clean" | "L0_skip_lang" | "L0_empty"
    profane_words: tuple[str, ...]
    ml_score: float  # 0.0 when not evaluated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _join_copy(
    headline: str,
    body: str,
    description: str,
    cta: str,
    advertiser_name: str = "",
) -> str:
    parts = [p for p in (headline, body, description, cta, advertiser_name) if p]
    return " \n ".join(parts)


def _find_profane_words(text: str) -> list[str]:
    """Return the distinct bad words that triggered a better-profanity match."""
    tokens = {t.lower() for t in _WORD_RE.findall(text)}
    return sorted(t for t in tokens if _bp.contains_profanity(t))


@lru_cache(maxsize=1)
def _ml_available() -> bool:
    # alt-profanity-check is installed alongside better-profanity; this is a
    # cheap sanity ping so the module fails loudly if the model file is gone.
    try:
        _pc_predict_prob(["hello"])
    except Exception:  # noqa: BLE001
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_content_safety(
    headline: str,
    body: str,
    description: str = "",
    cta: str = "",
    advertiser_name: str = "",
    language: str = "",
    ml_threshold: float = DEFAULT_ML_THRESHOLD,
) -> SafetyResult:
    """Classify a single ad's copy as safe or unsafe.

    ``language`` is the detected ISO code (see ``utils.language``). Non-English
    ads are short-circuited because both libraries are English-only and would
    give a false sense of coverage otherwise.
    """
    text = _join_copy(headline, body, description, cta, advertiser_name)
    if not text.strip():
        return SafetyResult(flag=False, level="L0_empty", profane_words=(), ml_score=0.0)

    # Skip non-English ads — libraries can't read them. The caller should
    # route these to a multilingual pass (e.g. the LLM batch classifier).
    if language and language != "en":
        return SafetyResult(flag=False, level="L0_skip_lang", profane_words=(), ml_score=0.0)

    # L1 — exact/obfuscated wordlist
    hits = _find_profane_words(text)
    if hits:
        return SafetyResult(
            flag=True,
            level="L1_wordlist",
            profane_words=tuple(hits),
            ml_score=0.0,
        )

    # L2 — ML toxicity probability
    if _ml_available():
        score = float(_pc_predict_prob([text])[0])
        if score >= ml_threshold:
            return SafetyResult(
                flag=True,
                level="L2_ml",
                profane_words=(),
                ml_score=score,
            )
        return SafetyResult(flag=False, level="L0_clean", profane_words=(), ml_score=score)

    return SafetyResult(flag=False, level="L0_clean", profane_words=(), ml_score=0.0)
