"""Walk the document body in true XML order (paragraphs AND tables) across the
LoT/LoF region of the reference template and ours, so we can see how the
reference actually builds its lists (tables vs tab-leader paragraphs).

Read-only. Run: uv run python scripts/thesis/inspect_lof_lot_tables.py
"""

from __future__ import annotations

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

REF = "docs/research/SURE NEU THESIS FORMAT abduallah inal PDF 12.docx"
OURS = "docs/research/THESIS.docx"


def para_style(p) -> str:
    return p.style.name if p.style else "?"


def field_kind(p) -> str:
    tags = []
    for r in p._p.iter():
        if r.tag == qn("w:fldChar"):
            tags.append("fld:" + (r.get(qn("w:fldCharType")) or "?"))
        elif r.tag == qn("w:instrText"):
            tags.append("instr:" + (r.text or "").strip()[:30])
    return " ".join(tags)


def dump_table(tbl: Table, indent="    ") -> None:
    rows = tbl.rows
    print(f"{indent}<TABLE rows={len(rows)} cols={len(tbl.columns)}>")
    # table-level properties
    tblPr = tbl._tbl.tblPr
    borders = tblPr.find(qn("w:tblBorders")) if tblPr is not None else None
    print(f"{indent}  borders={'yes' if borders is not None else 'none'}")
    for ri, row in enumerate(rows[:30]):
        cells = []
        for c in row.cells:
            txt = " / ".join(p.text for p in c.paragraphs if p.text.strip())
            cells.append(txt[:60])
        print(f"{indent}  row{ri}: {cells}")
    if len(rows) > 30:
        print(f"{indent}  ... (+{len(rows) - 30} more rows)")


def walk(path: str, label: str, anchors: tuple[str, ...]) -> None:
    doc = Document(path)
    body = doc.element.body
    print("\n" + "=" * 80)
    print(f"{label}: {path}")
    print("=" * 80)

    # Build a map of body-order elements.
    items = []
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            items.append(("p", Paragraph(child, doc)))
        elif child.tag == qn("w:tbl"):
            items.append(("tbl", Table(child, doc)))

    # Find the index of the first matching anchor heading.
    in_region = False
    region_lines = 0
    for kind, obj in items:
        if kind == "p":
            t = obj.text.strip().lower()
            if any(a in t for a in anchors) and t:
                in_region = True
                region_lines = 0
            if in_region:
                # stop once we hit CHAPTER I (after the lists/abbrev)
                if obj.text.strip().upper().startswith("CHAPTER I") and region_lines > 3:
                    print("--- end region (CHAPTER I) ---")
                    break
                style = para_style(obj)
                align = str(obj.alignment) if obj.alignment is not None else "-"
                fk = field_kind(obj)
                shown = obj.text if len(obj.text) <= 80 else obj.text[:77] + "..."
                marker = "  <<<" if t in ("list of tables", "list of figures",
                                          "list of figure", "table of contents") else ""
                print(f"  P ({style:>16}|{align:>9}) {fk}{marker}")
                if obj.text.strip():
                    print(f"      {shown!r}")
                region_lines += 1
        elif kind == "tbl" and in_region:
            dump_table(obj)
            region_lines += 1


if __name__ == "__main__":
    # Reference uses 'list of figure' (singular); include both spellings.
    walk(REF, "REFERENCE TEMPLATE",
         ("list of tables", "list of figure", "list of figures"))
    walk(OURS, "OURS (THESIS.docx)",
         ("list of tables", "list of figures"))
