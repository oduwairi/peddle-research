"""Fidelity + grounding checks for parsed v2 responses.

Two checks per (parsed, brief, source_ad):

1. **Deliverable fidelity** — the parsed deliverable text must reproduce
   the source ad verbatim. Same contract as v1 (≥60% word coverage + a
   6-word contiguous signature) but scoped to ``parsed.deliverable`` so
   the ``<think>`` trace can't trivially satisfy coverage by listing
   features.

2. **Think grounding** — the rationale must reference at least one
   populated brief field. Without grounding, the model never learns
   that brief fields constrain the ad — defeating the point of v2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.schemas.brief import Brief

# Mirror v1's tunings — these floors are well-stress-tested.
MIN_WORD_COVERAGE: float = 0.60
MIN_WORD_LEN: int = 4
MIN_AD_CONTENT_WORDS: int = 5
VERBATIM_SIGNATURE_WORDS: int = 6
MIN_FIELD_LEN_FOR_SIGNATURE: int = 15

# Think-grounding token floor. The teacher's rationale must contain a
# salient fragment from product facts AND from bridge fields.
MIN_GROUNDING_TOKEN_LEN: int = 4

_WORD_RE = re.compile(r"\b[\w'-]+", re.UNICODE)
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class FidelityResult:
    """Outcome of :func:`check_deliverable_fidelity`."""

    passed: bool
    coverage: float
    ad_word_count: int
    signature_passed: bool
    reason: str

    @property
    def is_pass(self) -> bool:
        return self.passed


@dataclass(frozen=True)
class GroundingResult:
    """Outcome of :func:`check_think_grounding`.

    Only bridge grounding is enforced — the gate dropped its
    product-fact requirement (2026-05-22) because the literal-token
    match was rejecting strategically-grounded ``<think>`` blocks that
    paraphrased the product (e.g. "the product" instead of the literal
    product name, "mint" instead of "Mints"). Bridge fields (``angle``
    and ``buyer_pain``) are the load-bearing anchors and remain
    required.
    """

    passed: bool
    bridge_match: str
    reason: str


def _content_words(text: str, min_len: int = MIN_WORD_LEN) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text) if len(m.group(0)) >= min_len}


def _normalize_for_match(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


def _signature_phrase(text: str, word_count: int = VERBATIM_SIGNATURE_WORDS) -> str:
    tokens = _normalize_for_match(text).split()
    if len(tokens) < word_count:
        return ""
    return " ".join(tokens[:word_count])


def check_deliverable_fidelity(deliverable: str, source_ad: SourceAd) -> FidelityResult:
    """Run word-coverage + 6-word verbatim signature checks on ``deliverable``.

    Scoped to the deliverable text emitted after ``</think>`` — never
    the think trace. Short ads (< ``MIN_AD_CONTENT_WORDS`` content
    words) auto-pass coverage so they don't false-reject.
    """
    fields = [source_ad.headline, source_ad.body, source_ad.description, source_ad.cta]
    joined = " ".join(f for f in fields if f)
    ad_words = _content_words(joined)
    if len(ad_words) < MIN_AD_CONTENT_WORDS:
        return FidelityResult(
            passed=True,
            coverage=1.0,
            ad_word_count=len(ad_words),
            signature_passed=True,
            reason="short_ad_skip",
        )

    response_words = _content_words(deliverable)
    overlap = ad_words & response_words
    coverage = len(overlap) / len(ad_words)
    coverage_ok = coverage >= MIN_WORD_COVERAGE

    # Gate candidates by both char-length AND normalized-token-count: a
    # field can be ≥15 chars but still resolve to <6 tokens after
    # normalization (e.g. "Help Gaza Survive" — 17 chars, 3 tokens), in
    # which case `_signature_phrase` returns empty and the signature
    # check spuriously fails on a deliverable that reproduces the ad
    # verbatim.
    candidates = [
        f
        for f in (source_ad.headline, source_ad.body, source_ad.description)
        if f
        and len(f) >= MIN_FIELD_LEN_FOR_SIGNATURE
        and len(_normalize_for_match(f).split()) >= VERBATIM_SIGNATURE_WORDS
    ]
    if candidates:
        normalized_response = _normalize_for_match(deliverable)
        signature_ok = any(
            (sig := _signature_phrase(field)) and sig in normalized_response for field in candidates
        )
    else:
        # No field long enough to anchor a signature — coverage decides.
        signature_ok = True

    if coverage_ok and signature_ok:
        return FidelityResult(
            passed=True,
            coverage=coverage,
            ad_word_count=len(ad_words),
            signature_passed=signature_ok,
            reason="",
        )
    reason_parts: list[str] = []
    if not coverage_ok:
        reason_parts.append(f"word_coverage={coverage:.2f}<{MIN_WORD_COVERAGE:.2f}")
    if not signature_ok:
        reason_parts.append(f"missing_{VERBATIM_SIGNATURE_WORDS}gram_signature")
    return FidelityResult(
        passed=False,
        coverage=coverage,
        ad_word_count=len(ad_words),
        signature_passed=signature_ok,
        reason=",".join(reason_parts),
    )


# ---------------------------------------------------------------------------
# Think grounding
# ---------------------------------------------------------------------------


def _bridge_grounding_tokens(brief: Brief) -> list[str]:
    """Collect salient surface tokens from the bridge side of the brief.

    ``positioning`` and ``target_audience`` are optional; only include
    them when populated. ``angle`` and ``buyer_pain`` are required.
    """
    out: list[str | None] = [
        brief.bridge.positioning,
        brief.bridge.target_audience,
        brief.bridge.angle,
        brief.bridge.buyer_pain,
    ]
    return [t for t in out if t]


def _first_match(think: str, candidates: list[str]) -> str:
    """Return the first candidate whose salient word appears in ``think``."""
    think_norm = think.lower()
    for candidate in candidates:
        if not candidate or len(candidate) < MIN_GROUNDING_TOKEN_LEN:
            continue
        words = [
            w.lower() for w in _WORD_RE.findall(candidate) if len(w) >= MIN_GROUNDING_TOKEN_LEN
        ]
        if not words:
            # Fall back to checking the whole candidate substring.
            if candidate.lower() in think_norm:
                return candidate
            continue
        # Accept any sufficiently-long content word from this candidate.
        # The prior longest-only rule rejected obviously-grounded rationales
        # where the long anchor word was absent but shorter specific terms
        # (e.g. "bunions", "workouts") appeared verbatim in <think>.
        if any(w in think_norm for w in words):
            return candidate
    return ""


def check_think_grounding(think: str, brief: Brief) -> GroundingResult:
    """Verify the rationale grounds itself in the brief's bridge fields.

    Passes iff at least one bridge field (``angle``, ``buyer_pain``,
    ``positioning``, or ``target_audience``) surfaces in the think
    trace. The product-fact requirement was dropped — see
    :class:`GroundingResult` docstring.
    """
    bridge_match = _first_match(think, _bridge_grounding_tokens(brief))
    if bridge_match:
        return GroundingResult(passed=True, bridge_match=bridge_match, reason="")
    return GroundingResult(
        passed=False,
        bridge_match="",
        reason="no_bridge_field_ref",
    )


__all__ = [
    "FidelityResult",
    "GroundingResult",
    "MIN_WORD_COVERAGE",
    "VERBATIM_SIGNATURE_WORDS",
    "check_deliverable_fidelity",
    "check_think_grounding",
]
