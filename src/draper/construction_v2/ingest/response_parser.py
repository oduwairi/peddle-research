"""Parse stage-2 teacher responses into ``<think>`` + freeform deliverable.

Expected output shape (only ``<think>`` is a structural anchor — the
deliverable flows freeform after ``</think>``):

::

    <think>
    {first-person internal reasoning, hidden by UI convention}
    </think>

    {deliverable — free-form text. For ad-copy training runs this is
     the source ad reproduced verbatim, possibly preceded or followed
     by short framing prose the model chose to add.}

Anything else is a parse rejection — the ingest stage feeds the reason
into the audit log so yield diagnostics can pinpoint where loss happened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

# Minimum think length we'll accept. Below this the rationale is
# vestigial (e.g., "<think></think>" or "<think>ok</think>") and won't
# teach the mapping function.
MIN_THINK_CHARS: int = 40
MIN_DELIVERABLE_CHARS: int = 10

# Teacher refusal / policy-failure sentinels we've seen in practice.
# Adding a few common ones up-front means ingest can short-circuit
# without paying the fidelity-check cost.
_TEACHER_FAIL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"I (?:can|cannot|can't) (?:help|assist|produce|comply)", re.IGNORECASE),
    re.compile(r"I'm (?:sorry|unable) (?:to|but)", re.IGNORECASE),
    re.compile(r"<EXTRACTION_FAILED>"),
)

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>(.*?)</think>", re.IGNORECASE | re.DOTALL)


class ParseRejection(StrEnum):
    """Why a teacher response could not be turned into a training example."""

    MISSING_THINK = "missing_think"
    MISSING_DELIVERABLE = "missing_deliverable"
    TEACHER_FAILED = "teacher_failed"
    THINK_TOO_SHORT = "think_too_short"
    PRE_THINK_NOISE = "pre_think_noise"


@dataclass(frozen=True)
class ParsedResponse:
    """Successfully parsed regions from a teacher response."""

    think: str
    deliverable: str

    @property
    def assistant_content(self) -> str:
        """Reassemble the assistant turn in the canonical training format."""
        return f"<think>\n{self.think}\n</think>\n\n{self.deliverable}"


def _is_teacher_failure(text: str) -> bool:
    return any(pattern.search(text) for pattern in _TEACHER_FAIL_PATTERNS)


def parse_response(text: str) -> ParsedResponse | ParseRejection:
    """Parse a stage-2 teacher response into (think, deliverable).

    Returns either a :class:`ParsedResponse` or a :class:`ParseRejection`
    enum member describing why the response was unusable.

    Contract:

    - ``<think>...</think>`` must appear and contain at least
      :data:`MIN_THINK_CHARS` characters.
    - Nothing meaningful may precede ``<think>``.
    - Everything after ``</think>`` is the deliverable (whitespace
      stripped, code fences around the entire blob removed). Must
      contain at least :data:`MIN_DELIVERABLE_CHARS` characters.
    """
    if not text or not text.strip():
        return ParseRejection.MISSING_THINK

    if _is_teacher_failure(text):
        return ParseRejection.TEACHER_FAILED

    think_match = _THINK_BLOCK_RE.search(text)
    if not think_match:
        return ParseRejection.MISSING_THINK

    # Nothing meaningful should precede `<think>`. A teacher that
    # spills prose before thinking is producing the wrong shape.
    if text[: think_match.start()].strip():
        return ParseRejection.PRE_THINK_NOISE

    think_raw = think_match.group(1).strip()
    if len(think_raw) < MIN_THINK_CHARS:
        return ParseRejection.THINK_TOO_SHORT

    deliverable_raw = text[think_match.end() :].strip()
    deliverable_raw = _strip_code_fences(deliverable_raw)
    if len(deliverable_raw) < MIN_DELIVERABLE_CHARS:
        return ParseRejection.MISSING_DELIVERABLE

    return ParsedResponse(think=think_raw, deliverable=deliverable_raw)


_CODE_FENCE_RE = re.compile(r"^```(?:[a-zA-Z]+)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Remove a single surrounding triple-backtick fence, if any.

    Preserves interior content exactly (no strip) so fidelity checks
    on parsed deliverables match the source verbatim. Only removes
    the outer fence and normalizes fence padding.
    """
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        # Return the interior with NO strip — preserve content exactly
        return match.group(1)
    return stripped


__all__ = [
    "MIN_DELIVERABLE_CHARS",
    "MIN_THINK_CHARS",
    "ParseRejection",
    "ParsedResponse",
    "parse_response",
]
