"""Dump the exact structure of the reference template's List-of-Tables and
List-of-Figures tables: column count + grid widths, borders, header row,
per-cell alignment + font, so we can replicate the format exactly.

Read-only. Run: uv run python scripts/thesis/inspect_ref_list_tables.py
"""

from __future__ import annotations

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

REF = "docs/research/SURE NEU THESIS FORMAT abduallah inal PDF 12.docx"


def describe_table(tbl: Table, name: str) -> None:
    print(f"\n--- {name} ---")
    tblPr = tbl._tbl.tblPr
    # width / layout
    w = tblPr.find(qn("w:tblW")) if tblPr is not None else None
    if w is not None:
        print(f"tblW: type={w.get(qn('w:type'))} w={w.get(qn('w:w'))}")
    jc = tblPr.find(qn("w:jc")) if tblPr is not None else None
    print(f"tbl jc: {jc.get(qn('w:val')) if jc is not None else '-'}")
    look = tblPr.find(qn("w:tblStyle")) if tblPr is not None else None
    print(f"tblStyle: {look.get(qn('w:val')) if look is not None else '-'}")
    borders = tblPr.find(qn("w:tblBorders")) if tblPr is not None else None
    if borders is not None:
        for b in borders:
            tag = b.tag.split("}")[-1]
            print(f"  border {tag}: val={b.get(qn('w:val'))} sz={b.get(qn('w:sz'))} "
                  f"color={b.get(qn('w:color'))}")
    else:
        print("  borders: NONE at table level")
    # grid
    grid = tbl._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        widths = [c.get(qn("w:w")) for c in grid.findall(qn("w:gridCol"))]
        print(f"gridCols ({len(widths)}): {widths}")
    # rows
    for ri, row in enumerate(tbl.rows):
        cellinfo = []
        for c in row.cells:
            ps = c.paragraphs
            txt = " / ".join(p.text for p in ps)
            align = None
            bold = None
            sz = None
            for p in ps:
                if p.alignment is not None:
                    align = str(p.alignment)
                for r in p.runs:
                    if r.bold:
                        bold = True
                    if r.font.size:
                        sz = r.font.size.pt
            cellinfo.append(f"[{txt[:40]!r} align={align} bold={bold} sz={sz}]")
        print(f"  row{ri}: {cellinfo}")


def main() -> None:
    doc = Document(REF)
    body = doc.element.body
    items = []
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            items.append(("p", Paragraph(child, doc)))
        elif child.tag == qn("w:tbl"):
            items.append(("tbl", Table(child, doc)))

    # The first table after "List of Tables" = LoT; tables after "List of Figure" = LoF.
    section = None
    lot_idx = 0
    lof_idx = 0
    for kind, obj in items:
        if kind == "p":
            t = obj.text.strip().lower()
            if t == "list of tables":
                section = "LOT"
            elif t in ("list of figure", "list of figures"):
                section = "LOF"
            elif t == "list of abbreviations":
                section = None
        elif kind == "tbl" and section == "LOT":
            lot_idx += 1
            describe_table(obj, f"LIST OF TABLES — table #{lot_idx}")
        elif kind == "tbl" and section == "LOF":
            lof_idx += 1
            describe_table(obj, f"LIST OF FIGURES — table #{lof_idx}")


if __name__ == "__main__":
    main()
