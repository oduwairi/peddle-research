"""Dump XML + style of the template paragraphs needed to build §4.4.

Templates:
  [652] H2 heading  "4.3 MAUVE Distribution Matching"
  [653] H3 heading  "4.3.1 Arm Setup"
  [654] body prose  (MAUVE setup paragraph)
  [660] caption     "Figure 4.3.2 ..." (carries a TC field — section-based numbering)
Read-only.
"""
from __future__ import annotations

import docx
from lxml import etree

DOC = "docs/research/THESIS.docx"
TARGETS = {652: "H2 head", 653: "H3 head", 654: "body prose", 660: "caption+TC"}


def main() -> None:
    d = docx.Document(DOC)
    paras = d.paragraphs
    for i, label in TARGETS.items():
        p = paras[i]
        sid = p.style.style_id if p.style else None
        print(f"\n===== [{i}] {label}  style.name={p.style.name!r} style_id={sid!r} =====")
        print(f"TEXT: {p.text.strip()[:120]!r}")
        xml = etree.tostring(p._p, pretty_print=True).decode()
        print(xml)


if __name__ == "__main__":
    main()
