"""Copywriting rubric.

Backtranslation copywriting is freeform: the response preserves whatever
shape the real ad has (single line, paragraphs, bullets, emoji walls)
and adds a short rationale. Structural validity is already covered by
the backtranslation fidelity check in
:mod:`draper.construction.formats.copywriting.ingestion` plus the min-
length floor, so no section markers are required.

Price of Format (arXiv:2505.18949) shows rigid templates cause diversity
collapse in open-ended generation — hence the deliberately empty rubric.
"""

from __future__ import annotations

REQUIRED_SECTIONS: list[list[str]] = []


def check(assistant_response: str) -> list[str]:
    """Return missing required-section names for a copywriting response."""
    del assistant_response  # nothing to check
    return []
