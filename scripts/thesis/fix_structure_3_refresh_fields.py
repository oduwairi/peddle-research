"""Flag the document so the editor rebuilds all fields on open.

Discovery: the Table of Contents, List of Figures and List of Tables in
THESIS.docx are NOT hardcoded text -- they are live Word fields
(`TOC \\o "1-3" \\h \\z \\u`, plus PAGEREF/TC machinery). What looked stale was
the fields' *cached result*, built back when the thesis ended at Chapter IV.

After fix_structure_1_titles.py (chapter titles + Ch I numbering) and
fix_structure_2_outline_levels.py (Ch II subsection restyle + outline levels),
the underlying headings are correct. The fields just need to be recomputed.

This script:
  * sets <w:updateFields w:val="true"/> in word/settings.xml so OnlyOffice/Word
    updates every field when the file is opened, and
  * marks each TOC/LoF/LoT field's begin fldChar dirty="true" as a belt-and-
    suspenders nudge.

It does NOT recompute page numbers itself (no layout engine here) -- the editor
does that on open / F9. Idempotent.

Run from repo root:
    uv run python scripts/thesis/fix_structure_3_refresh_fields.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

THESIS = Path("docs/research/THESIS.docx")


def main() -> None:
    doc = Document(str(THESIS))

    # ---- 1. settings.xml: update all fields on open -------------------------
    settings = doc.settings.element
    for existing in settings.findall(qn("w:updateFields")):
        settings.remove(existing)
    uf = etree.Element(qn("w:updateFields"))
    uf.set(qn("w:val"), "true")
    # CT_Settings is order-sensitive; updateFields sits early -- prepend safely.
    settings.insert(0, uf)
    print('settings.xml: set <w:updateFields w:val="true"/>')

    # ---- 2. mark the big list-fields dirty ----------------------------------
    dirtied = 0
    for p in doc.paragraphs:
        instrs = [it.text or "" for it in p._p.iter(qn("w:instrText"))]
        if not any("TOC" in s for s in instrs):
            continue
        for fc in p._p.iter(qn("w:fldChar")):
            if fc.get(qn("w:fldCharType")) == "begin":
                fc.set(qn("w:dirty"), "true")
                dirtied += 1
                break
    print(f"marked {dirtied} TOC/LoF/LoT field(s) dirty")

    doc.save(str(THESIS))
    print(f"\nSaved {THESIS}")


if __name__ == "__main__":
    main()
