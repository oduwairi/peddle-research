"""Remove the embedded PRISMA image from the PeerJ docx.

PeerJ requires figures to be uploaded as separate hi-res files, NOT
embedded in the manuscript .docx (per the now-removed template guidance:
"DO NOT embed figures or tables"). The in-text reference and caption
paragraphs remain so reviewers know what Figure 3 is; the PNG at
`docs/research/figures/fig-peerj-prisma-flow.png` will be uploaded
separately in the submission portal.

Idempotent: if there is no embedded drawing, prints a no-op message.
"""
from __future__ import annotations

import sys
from pathlib import Path

from docx import Document

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
QN_P = f"{{{W}}}p"
QN_T = f"{{{W}}}t"
QN_DRAWING = f"{{{W}}}drawing"


PEERJ_DOCX = Path("docs/research/literature-review-peerj.docx")


def para_text(p) -> str:
    return "".join((t.text or "") for t in p.findall(f".//{QN_T}"))


def main() -> int:
    doc = Document(str(PEERJ_DOCX))
    body = doc.element.body

    removed = 0
    for p in list(body.iterchildren(QN_P)):
        if p.find(f".//{QN_DRAWING}") is None:
            continue
        # Only remove if the paragraph has no other meaningful text content —
        # this is the dedicated image-holder paragraph we inserted.
        text = para_text(p).strip()
        if text:
            print(f"  skip drawing-bearing paragraph with text: {text[:80]!r}",
                  file=sys.stderr)
            continue
        body.remove(p)
        removed += 1

    doc.save(str(PEERJ_DOCX))
    print(f"Removed {removed} embedded drawing(s) from PeerJ docx.")
    if removed:
        print("Figure 3 intro + caption remain in-text; upload the PNG "
              "separately at submission time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
