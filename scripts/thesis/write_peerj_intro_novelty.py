"""B3 — Insert the PeerJ-editorial-criterion novelty paragraph.

PeerJ Computer Science explicitly asks reviewers: "Has the field been
reviewed recently? If so, is there a good reason for this new review?"
The current Introduction frames a gap but doesn't directly answer this.
This script inserts one short paragraph at the END of the "Paper
Organization" subsection (immediately before the "Survey methodology"
heading) that names the closest existing reviews, what this review
uniquely covers, and the intended audience.

Idempotent: skips insertion if a paragraph beginning with the sentinel
opener "Existing reviews in adjacent areas" is already present.

Use a small writer pattern: prose is defined as a constant at the top so
the user can revise it in-place before re-running the script.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

from docx import Document

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
QN_P = f"{{{W}}}p"
QN_T = f"{{{W}}}t"
QN_R = f"{{{W}}}r"


PEERJ_DOCX = Path("docs/research/literature-review-peerj.docx")


NOVELTY_PARAGRAPH = (
    "Existing reviews in adjacent areas cover RAG architectures (Gao et al., "
    "2024; Fan et al., 2024), autonomous LLM-agent design (Xi et al., 2023; "
    "Wang et al., 2025), persuasive natural-language generation (Duerr & "
    "Gloor, 2021), and the ethics of AI-driven personalization (Karami et "
    "al., 2024), but none address the integrated question of how to "
    "specialize a small open-source LLM for commercial advertising "
    "copywriting under the dual constraints of weak engagement-derived "
    "labels and the absence of public, outcome-linked campaign data. This "
    "review consolidates the four design surfaces that practitioners must "
    "reason about jointly when building such a system — parameter-efficient "
    "fine-tuning, retrieval-augmented generation, autonomous agent "
    "architectures, and evaluation methodology for open-ended generation — "
    "and surveys the dataset-construction techniques (instruction tuning, "
    "backtranslation, weak supervision) that make the problem tractable in "
    "the absence of labelled data. The intended audience is computer-"
    "science researchers and machine-learning practitioners building "
    "domain-specialized generative systems in low-resource, outcome-driven "
    "verticals."
)


SENTINEL_OPENER = "Existing reviews in adjacent areas"


# ---------- helpers (kept local; same shape as finalize_peerj.py) ---------


def para_text(p) -> str:
    return "".join((t.text or "") for t in p.findall(f".//{QN_T}"))


def set_para_text(p, new_text: str) -> None:
    runs = p.findall(f".//{QN_R}")
    if not runs:
        raise RuntimeError("paragraph has no <w:r> to write into")
    first = runs[0]
    for r in runs[1:]:
        r.getparent().remove(r)
    ts = first.findall(f".//{QN_T}")
    if not ts:
        raise RuntimeError("first run has no <w:t> to write into")
    ts[0].text = new_text
    for t in ts[1:]:
        t.getparent().remove(t)


def find_para_by_text(body, predicate):
    for p in body.iterchildren(QN_P):
        if predicate(para_text(p)):
            return p
    return None


def first_body_paragraph_after(body, anchor):
    """Return the first non-empty paragraph after `anchor` to use as a
    cloning template for body style."""
    seen = False
    for p in body.iterchildren(QN_P):
        if p is anchor:
            seen = True
            continue
        if seen and para_text(p).strip():
            return p
    raise RuntimeError("no body paragraph found after anchor")


def main() -> int:
    if not PEERJ_DOCX.exists():
        print(f"ERROR: {PEERJ_DOCX} not found", file=sys.stderr)
        return 1

    doc = Document(str(PEERJ_DOCX))
    body = doc.element.body

    # Idempotency check.
    for p in body.iterchildren(QN_P):
        if para_text(p).startswith(SENTINEL_OPENER):
            print("Novelty paragraph already present; nothing to do.")
            return 0

    # Anchor: the "Survey methodology" heading. We insert immediately
    # before it (so the paragraph lives at the tail of the Introduction).
    anchor = find_para_by_text(body, lambda t: t.strip() == "Survey methodology")
    if anchor is None:
        print("ERROR: 'Survey methodology' heading not found", file=sys.stderr)
        return 1

    # Clone a body-style paragraph. The first body paragraph after
    # "Survey methodology" is the right template.
    body_tpl = first_body_paragraph_after(body, anchor)

    new_p = deepcopy(body_tpl)
    set_para_text(new_p, NOVELTY_PARAGRAPH)
    anchor.addprevious(new_p)

    doc.save(str(PEERJ_DOCX))
    print(f"Inserted novelty paragraph ({len(NOVELTY_PARAGRAPH)} chars) "
          f"before 'Survey methodology' anchor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
