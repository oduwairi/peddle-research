"""Populate the front-matter TOC, List of Figures, and List of Tables with real
Word field codes that Word will refresh on open (or via F9).

Strategy:
  * Set the empty "Table of Contents" Heading 2 (para 144) and replace the TOC
    placeholder paragraph (para 189) with a `TOC \\o "1-3" \\h \\z \\u` field —
    this picks up every Heading 1/2/3 automatically.
  * Replace the combined "List of Tables and List of Figures" placeholder (para
    212) with two Heading 2 sections and field codes: `TOC \\h \\z \\f F` (LOF)
    and `TOC \\h \\z \\f T` (LOT).
  * Append a hidden `TC "<caption>" \\f <F|T> \\l 1` field to every "Figure X.Y"
    and "Table X.Y" caption paragraph so the LOF/LOT fields above can find them
    without restyling captions (which would change their visual appearance).

Word fields are inserted as `dirty=true`, so Word will prompt to update or auto-
refresh them on first open.

Run from repo root:

    uv run python scripts/thesis/populate_toc_lof_lot.py
"""

from __future__ import annotations

import re
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

THESIS = Path("docs/research/THESIS.docx")
TMP = Path("docs/research/THESIS.docx.tmp")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NSMAP = {"w": W}

# Style IDs in this template (see scripts/thesis/restyle_h3_to_h2.py).
H2_STYLE_ID = "945"


def w(tag: str) -> str:
    return f"{{{W}}}{tag}"


def make_field_runs(instr: str, placeholder: str = "Update field (F9)") -> list:
    """Build a list of <w:r> elements implementing a complex field:

        { fldChar begin (dirty) }
        { instrText }
        { fldChar separate }
        { t placeholder }
        { fldChar end }
    """
    runs: list = []

    r1 = etree.SubElement(etree.Element(w("dummy")), w("r"))
    fc = etree.SubElement(r1, w("fldChar"))
    fc.set(w("fldCharType"), "begin")
    fc.set(w("dirty"), "true")
    runs.append(r1)

    r2 = etree.SubElement(etree.Element(w("dummy")), w("r"))
    it = etree.SubElement(r2, w("instrText"))
    it.set(qn("xml:space"), "preserve")
    it.text = instr
    runs.append(r2)

    r3 = etree.SubElement(etree.Element(w("dummy")), w("r"))
    fc3 = etree.SubElement(r3, w("fldChar"))
    fc3.set(w("fldCharType"), "separate")
    runs.append(r3)

    r4 = etree.SubElement(etree.Element(w("dummy")), w("r"))
    t = etree.SubElement(r4, w("t"))
    t.text = placeholder
    runs.append(r4)

    r5 = etree.SubElement(etree.Element(w("dummy")), w("r"))
    fc5 = etree.SubElement(r5, w("fldChar"))
    fc5.set(w("fldCharType"), "end")
    runs.append(r5)

    return runs


def make_tc_field_runs(caption: str, table_id: str) -> list:
    """Build a TC field. Word treats TC instruction text as a hidden entry in the
    table referenced by `\\f <id>`. We mark the runs as vanish so the field never
    shows in the rendered body.
    """
    safe = caption.replace('"', "'")
    instr = f' TC "{safe}" \\f {table_id} \\l 1 '
    runs: list = []

    r1 = etree.SubElement(etree.Element(w("dummy")), w("r"))
    rPr1 = etree.SubElement(r1, w("rPr"))
    etree.SubElement(rPr1, w("vanish"))
    fc = etree.SubElement(r1, w("fldChar"))
    fc.set(w("fldCharType"), "begin")
    runs.append(r1)

    r2 = etree.SubElement(etree.Element(w("dummy")), w("r"))
    rPr2 = etree.SubElement(r2, w("rPr"))
    etree.SubElement(rPr2, w("vanish"))
    it = etree.SubElement(r2, w("instrText"))
    it.set(qn("xml:space"), "preserve")
    it.text = instr
    runs.append(r2)

    r3 = etree.SubElement(etree.Element(w("dummy")), w("r"))
    rPr3 = etree.SubElement(r3, w("rPr"))
    etree.SubElement(rPr3, w("vanish"))
    fc3 = etree.SubElement(r3, w("fldChar"))
    fc3.set(w("fldCharType"), "end")
    runs.append(r3)

    return runs


def clear_runs(para_el) -> None:
    for child in list(para_el):
        if child.tag in (w("r"), w("hyperlink")):
            para_el.remove(child)


def set_paragraph_style(para_el, style_id: str) -> None:
    pPr = para_el.find(w("pPr"))
    if pPr is None:
        pPr = etree.SubElement(para_el, w("pPr"))
        para_el.insert(0, pPr)
    for existing in pPr.findall(w("pStyle")):
        pPr.remove(existing)
    pStyle = etree.SubElement(pPr, w("pStyle"))
    pStyle.set(w("val"), style_id)
    pPr.insert(0, pStyle)


def append_text_run(para_el, text: str) -> None:
    r = etree.SubElement(para_el, w("r"))
    t = etree.SubElement(r, w("t"))
    t.set(qn("xml:space"), "preserve")
    t.text = text


def safe_text(para_el) -> str:
    """python-docx `.text` chokes when a paragraph contains field runs (no <w:t>).
    Concatenate <w:t> text nodes ourselves, treating None as ''.
    """
    parts: list[str] = []
    for t in para_el.iter(w("t")):
        if t.text:
            parts.append(t.text)
    return "".join(parts)


def main() -> None:
    doc = Document(str(THESIS))

    paragraphs = doc.paragraphs

    # Pre-scan ALL targets before mutating, since adding field runs makes
    # python-docx's .text accessor crash on subsequent reads.
    toc_heading_el = None
    toc_placeholder_el = None
    lof_placeholder_el = None
    caption_targets: list[tuple[object, str, str]] = []  # (element, text, "F"|"T")
    # Require ":" or em/en-dash after the number so we don't catch narrative
    # paragraphs like "Figure 2.2 (Gao et al., 2024) shows ...".
    fig_re = re.compile(r"^Figure\s+\d+(?:\.\d+)*\s*[:—–]", re.I)
    tbl_re = re.compile(r"^Table\s+\d+(?:\.\d+)*\s*[:—–]", re.I)

    for i, p in enumerate(paragraphs):
        text = safe_text(p._element).strip()
        if (
            toc_heading_el is None
            and 140 <= i <= 200
            and p.style.name == "Heading 2"
            and not text
        ):
            toc_heading_el = p._element
        if toc_placeholder_el is None and text.startswith("[Table of Contents"):
            toc_placeholder_el = p._element
        if lof_placeholder_el is None and text.startswith(
            "[List of Tables and List of Figures"
        ):
            lof_placeholder_el = p._element
        if fig_re.match(text):
            caption_targets.append((p._element, text, "F"))
        elif tbl_re.match(text):
            caption_targets.append((p._element, text, "T"))

    if toc_heading_el is None:
        raise SystemExit("Could not find empty Heading 2 to use as 'Table of Contents'")
    if toc_placeholder_el is None:
        raise SystemExit("Could not find TOC placeholder paragraph")
    if lof_placeholder_el is None:
        raise SystemExit("Could not find LOF/LOT placeholder paragraph")

    # ----- 1. "Table of Contents" heading -----
    clear_runs(toc_heading_el)
    append_text_run(toc_heading_el, "Table of Contents")
    print("  set TOC heading text")

    # ----- 2. TOC field at the existing placeholder paragraph -----
    clear_runs(toc_placeholder_el)
    for r in make_field_runs(' TOC \\o "1-3" \\h \\z \\u '):
        toc_placeholder_el.append(r)
    print("  inserted TOC field at placeholder")

    anchor_el = lof_placeholder_el

    # Build four new <w:p> elements:
    #   List of Figures (H2) → LOF field
    #   List of Tables (H2)  → LOT field
    def make_h2_paragraph(text: str):
        p = etree.SubElement(etree.Element(w("dummy")), w("p"))
        set_paragraph_style(p, H2_STYLE_ID)
        append_text_run(p, text)
        return p

    def make_field_paragraph(instr: str):
        p = etree.SubElement(etree.Element(w("dummy")), w("p"))
        for r in make_field_runs(instr):
            p.append(r)
        return p

    new_elements = [
        make_h2_paragraph("List of Figures"),
        make_field_paragraph(' TOC \\h \\z \\f F '),
        make_h2_paragraph("List of Tables"),
        make_field_paragraph(' TOC \\h \\z \\f T '),
    ]

    # Insert before the placeholder, then remove the placeholder.
    for el in new_elements:
        anchor_el.addprevious(el)
    anchor_el.getparent().remove(anchor_el)
    print(f"  replaced LOF/LOT placeholder with 4 paragraphs (2 headings + 2 fields)")

    # ----- 4. TC fields beside each Figure / Table caption -------------------
    figs_added = 0
    tbls_added = 0
    for el, caption, kind in caption_targets:
        for r in make_tc_field_runs(caption, kind):
            el.append(r)
        if kind == "F":
            figs_added += 1
        else:
            tbls_added += 1
    print(f"  inserted {figs_added} figure TC fields, {tbls_added} table TC fields")

    doc.save(str(THESIS))
    print(f"\nSaved {THESIS}")


if __name__ == "__main__":
    main()
