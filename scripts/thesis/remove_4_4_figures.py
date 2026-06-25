"""Remove the two §4.4 reference-overlap figures (image paragraph + caption)
from THESIS.docx so the redesigned versions can be re-inserted. Targets only
style-943 captions starting 'Figure 4.4.1:'/'Figure 4.4.2:' (the live LoF cache
uses a different style; the renamed ablation figures are now 4.5.x)."""
from __future__ import annotations

import docx
from docx.oxml.ns import qn

DOC = "docs/research/THESIS.docx"


def main() -> None:
    d = docx.Document(DOC)
    removed = 0
    for p in list(d.paragraphs):
        if not (p.style and p.style.style_id == "943"):
            continue
        if not p.text.strip().startswith(("Figure 4.4.1:", "Figure 4.4.2:")):
            continue
        cap = p._p
        img = cap.getprevious()
        parent = cap.getparent()
        if img is not None and img.tag == qn("w:p") and img.find(".//" + qn("w:drawing")) is not None:
            parent.remove(img)
            removed += 1
        parent.remove(cap)
        removed += 1
        print(f"removed: {p.text.strip()[:42]!r}")
    d.save(DOC)
    print(f"removed {removed} paragraphs (image + caption pairs)")


if __name__ == "__main__":
    main()
