"""Strengthen §6.2 Contributions + §6.4 Future Work (reviewer feedback #8).

Reviewer #8 asked the conclusion to restate *verified* contributions and give
*prioritized* next steps. These are minimal, in-place clause/phrase slot-ins
into the author's existing sentences -- no new sentences:

  §6.2  anchor each empirical contribution to its verified result
        (fine-tune beat the frontier model / 95.2% of ceiling; proxy scorer
        rho=0.749; learned scorer rho=0.722 on cheap CPU).
  §6.4  make the two next-step directions an explicit ranked list
        ("The first priority ..." / "Second, ...").

Conventions (match scripts/thesis):
  - python-docx, anchor by visible heading text (never a hardcoded index).
  - The two target body paragraphs carry NO inline formatting (every run is
    plain), so we losslessly reconstruct the full text, apply exact string
    replacements, and write it back onto run[0] (emptying the fragmented
    mid-word runs). Citation text is plain, so nothing is lost.
  - Idempotent: a slot-in whose result is already present is skipped; an old
    anchor that is missing AND not yet applied is reported as drift (no write).
"""

from __future__ import annotations

import sys

import docx

DOC_PATH = "docs/research/THESIS.docx"

# (heading, [(old, new, already_marker), ...])
EDITS: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "6.2 Contributions",
        [
            (
                "served via Modal vLLM), with a live endpoint and API,",
                "served via Modal vLLM) that beat the frontier model and reached "
                "95.2% of the real-ad ceiling, with a live endpoint and API,",
                "beat the frontier model and reached 95.2% of the real-ad ceiling",
            ),
            (
                "with proxy score labels,",
                "with proxy score labels validated at ρ = 0.749 against a "
                "ground-truth set,",
                "validated at ρ = 0.749 against a ground-truth set",
            ),
            (
                "and provides performance predictions.",
                "and provides reliable performance predictions (ρ = 0.722) at "
                "cheap CPU inference.",
                "reliable performance predictions (ρ = 0.722)",
            ),
        ],
    ),
    (
        "6.4 Recommendations for Future Work",
        [
            (
                "One of the most critical next steps is to expand the scope",
                "The first priority is to expand the scope",
                "The first priority is to expand the scope",
            ),
            (
                "Beyond this, a major encountered limitation",
                "Second, a major encountered limitation",
                "Second, a major encountered limitation",
            ),
        ],
    ),
]


def body_after(paras, heading: str):
    """First non-empty paragraph following the given heading."""
    hi = next((i for i, p in enumerate(paras) if p.text.strip() == heading), None)
    if hi is None:
        return None
    return next((p for p in paras[hi + 1 :] if p.text.strip()), None)


def main() -> int:
    doc = docx.Document(DOC_PATH)
    paras = doc.paragraphs
    dirty = False

    for heading, edits in EDITS:
        para = body_after(paras, heading)
        if para is None:
            print(f"ERROR: body paragraph for '{heading}' not found", file=sys.stderr)
            return 1
        text = "".join(r.text for r in para.runs)
        changed = False
        print(f"\n§ {heading}")
        for old, new, marker in edits:
            if marker in text:
                print(f"  SKIP   already applied: {marker[:48]!r}")
                continue
            if old not in text:
                print(f"  ERROR  anchor drift, not found: {old[:48]!r}", file=sys.stderr)
                return 1
            text = text.replace(old, new, 1)
            changed = True
            print(f"  EDIT   {old[:40]!r} -> +{len(new) - len(old)} chars")
        if changed:
            para.runs[0].text = text
            for r in para.runs[1:]:
                r.text = ""
            dirty = True

    if dirty:
        doc.save(DOC_PATH)
        print("\nSAVED", DOC_PATH)
    else:
        print("\nNo changes (idempotent no-op).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
