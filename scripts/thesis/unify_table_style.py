"""Unify every data table in THESIS.docx to a single APA three-line style.

Target look (chosen by the author):
  * Horizontal rules only: a top rule, a rule under the header row, and a
    bottom rule. NO vertical lines, NO interior horizontal lines between
    body rows.
  * No shading anywhere (removes the Chapter-2 grey header band + zebra
    body striping).
  * Bold header row preserved.
  * Full text width (9926 dxa = 6.89 in), fixed layout, column proportions
    preserved by scaling each table's grid.

Mechanism:
  - table-level tblBorders: top=single, bottom=single (1pt black); left,
    right, insideH, insideV = nil. This overrides the Chapter-3 "Table
    Grid" (style 756) full grid as well as the Chapter-2 white borders.
  - header-row cells get a per-cell bottom border (the under-header rule),
    since insideH=nil suppresses the table-wide interior rule.
  - every cell: strip <w:shd>, clear stray <w:tcBorders>, set tcW to its
    scaled column width.

OOXML is order-sensitive: new children are inserted at their correct
position in the CT_TblPr / CT_TcPr sequence (see *_ORDER lists).

Safety: only tables with a "Table X.Y" caption within the 6 preceding
paragraphs are restyled (skips any layout tables). Idempotent — re-running
reproduces the same result.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree

THESIS = Path("docs/research/THESIS.docx")
LOCK = Path("docs/research/.~lock.THESIS.docx#")
BACKUP = Path("docs/research/THESIS.docx.pre-tablestyle.bak")

TEXT_WIDTH = 9926  # dxa; = pgSz.w - left - right margins
RULE_SZ = "8"      # eighths of a point -> 1.0pt
RULE_COLOR = "000000"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Schema sequence within <w:tblPr> and <w:tcPr> (subset we touch).
TBLPR_ORDER = [
    "tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
    "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd",
    "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
]
TCPR_ORDER = [
    "cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd",
    "noWrap", "tcMar", "textDirection", "tcFit", "vAlign", "hideMark",
]


def ln(el) -> str:
    return etree.QName(el).localname


def insert_ordered(parent, child, order: list[str]) -> None:
    """Insert *child* into *parent* at its correct schema position."""
    ci = order.index(ln(child))
    for existing in parent:
        name = ln(existing)
        if name in order and order.index(name) > ci:
            existing.addprevious(child)
            return
    parent.append(child)


def get_or_make(parent, tag: str, order: list[str]):
    el = parent.find(qn(f"w:{tag}"))
    if el is None:
        el = OxmlElement(f"w:{tag}")
        insert_ordered(parent, el, order)
    return el


def _border(tag: str, val: str, *, sz: str | None = None):
    e = OxmlElement(f"w:{tag}")
    e.set(qn("w:val"), val)
    e.set(qn("w:space"), "0")
    if val == "single":
        e.set(qn("w:sz"), sz or RULE_SZ)
        e.set(qn("w:color"), RULE_COLOR)
    else:  # nil
        e.set(qn("w:sz"), "0")
        e.set(qn("w:color"), "auto")
    return e


def set_borders(container, spec: dict[str, str], order: list[str], kind: str):
    """Rebuild a <w:tblBorders>/<w:tcBorders> element from *spec*.

    spec maps edge -> "single"|"nil"; edges emitted in canonical order.
    """
    old = container.find(qn(f"w:{kind}"))
    if old is not None:
        container.remove(old)
    bd = OxmlElement(f"w:{kind}")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if edge in spec:
            bd.append(_border(edge, spec[edge]))
    insert_ordered(container, bd, order)


def scale_columns(cols: list[int], target: int) -> list[int]:
    s = sum(cols)
    new = [round(c * target / s) for c in cols]
    new[-1] += target - sum(new)  # absorb rounding drift into the last column
    return new


def nearest_caption(tbl_el) -> str | None:
    el = tbl_el.getprevious()
    steps = 0
    while el is not None and steps < 6:
        if el.tag == qn("w:p"):
            txt = "".join((t.text or "") for t in el.findall(f".//{qn('w:t')}")).strip()
            if txt.startswith("Table "):
                return txt[:50]
        el = el.getprevious()
        steps += 1
    return None


def restyle(tbl) -> None:
    tEl = tbl._element
    tblPr = tEl.find(qn("w:tblPr"))

    # --- strip ALL shading in the table (table bg, cell, paragraph, run) ---
    # APA three-line wants no shading anywhere; this removes the Chapter-2
    # grey zebra (tcPr), the light-blue table background (tblPr), and any
    # transparent run-level remnants.
    for shd in tEl.findall(f".//{qn('w:shd')}"):
        shd.getparent().remove(shd)

    # --- width + layout ---
    tblW = get_or_make(tblPr, "tblW", TBLPR_ORDER)
    tblW.set(qn("w:w"), str(TEXT_WIDTH))
    tblW.set(qn("w:type"), "dxa")
    layout = get_or_make(tblPr, "tblLayout", TBLPR_ORDER)
    layout.set(qn("w:type"), "fixed")

    # --- table-level three-line borders ---
    set_borders(
        tblPr,
        {"top": "single", "bottom": "single", "left": "nil",
         "right": "nil", "insideH": "nil", "insideV": "nil"},
        TBLPR_ORDER, "tblBorders",
    )

    # --- scale the grid to full text width ---
    grid = tEl.find(qn("w:tblGrid"))
    gridcols = grid.findall(qn("w:gridCol"))
    new_w = scale_columns([int(c.get(qn("w:w"))) for c in gridcols], TEXT_WIDTH)
    for c, w in zip(gridcols, new_w):
        c.set(qn("w:w"), str(w))

    # --- per-cell pass ---
    for ri, row in enumerate(tbl.rows):
        for ci, cell in enumerate(row.cells):
            tc = cell._tc
            tcPr = tc.find(qn("w:tcPr"))
            if tcPr is None:
                tcPr = OxmlElement("w:tcPr")
                tc.insert(0, tcPr)
            # width
            tcW = get_or_make(tcPr, "tcW", TCPR_ORDER)
            tcW.set(qn("w:w"), str(new_w[ci] if ci < len(new_w) else new_w[-1]))
            tcW.set(qn("w:type"), "dxa")
            # kill shading (zebra / header band)
            shd = tcPr.find(qn("w:shd"))
            if shd is not None:
                tcPr.remove(shd)
            # borders: clear, then add under-header rule on row 0 only
            old_tcb = tcPr.find(qn("w:tcBorders"))
            if old_tcb is not None:
                tcPr.remove(old_tcb)
            if ri == 0:
                set_borders(
                    tcPr,
                    {"top": "nil", "left": "nil", "bottom": "single", "right": "nil"},
                    TCPR_ORDER, "tcBorders",
                )


def main() -> int:
    if LOCK.exists():
        print("ABORT: OnlyOffice lock present — close the editor first.", file=sys.stderr)
        return 1
    if not THESIS.exists():
        print(f"ABORT: {THESIS} not found", file=sys.stderr)
        return 1

    doc = Document(str(THESIS))
    if not BACKUP.exists():  # preserve the first (true pre-styling) backup on re-run
        shutil.copy(THESIS, BACKUP)

    done, skipped = 0, 0
    for tbl in doc.tables:
        cap = nearest_caption(tbl._element)
        if cap is None:
            skipped += 1
            print(f"  SKIP (no 'Table X.Y' caption nearby): {len(tbl.rows)}x{len(tbl.columns)} table")
            continue
        restyle(tbl)
        done += 1
        print(f"  RESTYLED  {cap!r:52}  ({len(tbl.rows)}x{len(tbl.columns)})")

    doc.save(str(THESIS))
    print(f"\nDONE: {done} tables -> APA three-line @ full width ({TEXT_WIDTH} dxa); {skipped} skipped.")
    print(f"backup: {BACKUP}")
    print("NEXT: open in OnlyOffice to visually verify (no field refresh needed for table styling).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
