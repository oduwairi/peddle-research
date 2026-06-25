"""C1 — Insert the PRISMA Figure 3 intro + caption paragraphs.

PeerJ requires figures to be UPLOADED AS SEPARATE FILES, not embedded
in the .docx (per the PeerJ template's removed-guidance callout: "DO NOT
embed figures or tables"). Only the in-text reference sentence and the
caption paragraph live in the manuscript; the PNG at
`docs/research/figures/fig-peerj-prisma-flow.png` is uploaded separately
in the submission portal.

Idempotent: skips if a paragraph beginning with the in-text reference
sentinel ("Figure 3 summarises") is already present.

Placement: immediately after the existing Survey methodology body
paragraph (the one ending "...to focus on recency in fast-changing
research domains.") and before the "2.1 Foundation Models..." heading.

Body order:
  1. In-text reference paragraph ("Figure 3 summarises ...")
  2. Caption paragraph ("Figure 3. PRISMA-style flow ...")
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
QN_P = f"{{{W}}}p"
QN_T = f"{{{W}}}t"
QN_R = f"{{{W}}}r"
QN_DRAWING = f"{{{W}}}drawing"


PEERJ_DOCX = Path("docs/research/literature-review-peerj.docx")
PRISMA_PNG = Path("docs/research/figures/fig-peerj-prisma-flow.png")

INTRO_SENTENCE = (
    "Figure 3 summarises the identification, screening, and inclusion flow "
    "that produced the final reference set."
)
CAPTION = (
    "Figure 3. PRISMA-style flow of the literature identification, "
    "screening, and inclusion process. Identification counts (n ≈ 612) "
    "reflect the AI-assisted broad sweep across eight academic databases; "
    "screening applied the inclusion and exclusion criteria stated above; "
    "the final included set (n = 89) is the reference list of this survey."
)

SENTINEL_OPENER = "Figure 3 summarises"


# ---------- helpers --------------------------------------------------------


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


def main() -> int:
    if not PEERJ_DOCX.exists():
        print(f"ERROR: {PEERJ_DOCX} not found", file=sys.stderr)
        return 1
    if not PRISMA_PNG.exists():
        print(f"WARN: {PRISMA_PNG} not found on disk. It must be uploaded "
              "separately at submission time per PeerJ figure rules.",
              file=sys.stderr)

    doc = Document(str(PEERJ_DOCX))
    body = doc.element.body

    # Idempotency check.
    for p in body.iterchildren(QN_P):
        if para_text(p).startswith(SENTINEL_OPENER):
            print("PRISMA figure intro already present; nothing to do.")
            return 0

    # Anchor: the Survey methodology body paragraph (the long one).
    anchor = find_para_by_text(
        body,
        lambda t: t.startswith("To conduct this survey"),
    )
    if anchor is None:
        print("ERROR: Survey methodology body paragraph not found", file=sys.stderr)
        return 1

    body_tpl = anchor  # safe to clone

    # 1. In-text reference sentence.
    intro_p_xml = deepcopy(body_tpl)
    set_para_text(intro_p_xml, INTRO_SENTENCE)
    anchor.addnext(intro_p_xml)

    # 2. Caption paragraph (centered, no embedded image — PeerJ rule).
    cap_p_xml = deepcopy(body_tpl)
    set_para_text(cap_p_xml, CAPTION)
    intro_p_xml.addnext(cap_p_xml)

    for p in doc.paragraphs:
        if p._element is cap_p_xml:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.save(str(PEERJ_DOCX))
    print("Inserted Figure 3 intro + caption (image NOT embedded — upload "
          f"{PRISMA_PNG.name} separately at submission).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
