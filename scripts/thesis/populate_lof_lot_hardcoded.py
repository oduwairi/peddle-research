"""Populate the List of Figures and List of Tables with hardcoded entries.

The previous approach (TOC + TC field codes; see ``populate_toc_lof_lot.py``)
depends on Word actually opening the document and refreshing every field,
which in practice leaves both lists blank in LibreOffice and on web previews.
Since the figure and table inventory is small and stable, we hardcode the
entries directly:

  * Add a ``Fig_<n>`` / ``Tbl_<n>`` bookmark to each caption paragraph (only
    if not already present).
  * Wipe whatever currently sits between the ``List of Figures`` and
    ``List of Tables`` Heading-2 paragraphs, and between ``List of Tables``
    and the next chapter heading.
  * Insert one paragraph per figure / table with the shape
    ``<label>: <title>``  TAB  ``<PAGEREF bookmark>`` and a right-aligned
    dot-leader tab stop so the page number sits flush against the right
    margin.

PAGEREF fields are the only Word-side machinery left; they resolve on first
open and survive page reflow. If figures / tables are added or renamed, edit
``FIGURES`` / ``TABLES`` below and re-run.

Run from repo root:

    uv run python scripts/thesis/populate_lof_lot_hardcoded.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

THESIS = Path("docs/research/THESIS.docx")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Hardcoded inventory. Order = order of appearance in the document.
FIGURES: list[tuple[str, str]] = [
    ("2.7", "Overview of the Self-Instruct pipeline"),
    ("3.1", "Proposed system overview"),
    ("3.2", "AdFlex-centred ad collection pipeline"),
    ("3.3", "Instruction backtranslation"),
    ("3.4", "Draper.ai agent architecture"),
    ("4.1", "Composite mean per configuration on the held-out test set"),
    ("4.2", "Composite score by platform and configuration"),
    ("4.3", "Per-head means by configuration"),
    ("4.4", "Predictor reliability on the held-out random split"),
    ("4.3.1", "UMAP projection of GPT-2 Large embeddings per configuration"),
    ("4.3.2", "Per-platform MAUVE scores"),
    ("4.4.1", "Paired contrasts on the 2×2 ablation"),
    ("4.4.2", "Per-cell composite means on the held-out test set"),
    ("5.1", "Mechanism comparison of candidate fine-tuning data strategies"),
]

TABLES: list[tuple[str, str]] = [
    ("2.1", "Open-Source Foundation Models"),
    ("2.2", "Quantitative Comparison of PEFT Methods for 7B-Class LLM Fine-Tuning"),
    ("2.3", "Domain-Specialized Small Models: Training Approach and Performance"),
    ("2.4", "Available Marketing Campaign Datasets for NLP Research"),
    ("2.5", "Comparison of Evaluation Frameworks for Generative Systems"),
    ("3.1", "Comparison of the two evaluation methods"),
]

# Right-margin position for the dot leader tab, in twips. 9000 ≈ 6.25in,
# which lands page numbers near the right margin for a standard letter page
# with 1in margins.
TAB_POS_TWIPS = "9000"

# Start IDs above anything Word might already use; the docx currently has no
# bookmarks at all, so 10000+ is safe.
BOOKMARK_ID_START = 10000


def w(tag: str) -> str:
    return f"{{{W}}}{tag}"


def safe_text(para_el) -> str:
    parts: list[str] = []
    for t in para_el.iter(w("t")):
        if t.text:
            parts.append(t.text)
    return "".join(parts)


def bookmark_name(prefix: str, label: str) -> str:
    return f"{prefix}_{label.replace('.', '_')}"


def find_existing_bookmark(para_el, name: str) -> bool:
    for el in para_el.iter(w("bookmarkStart")):
        if el.get(w("name")) == name:
            return True
    return False


def insert_bookmark(para_el, name: str, bid: int) -> None:
    """Insert an empty bookmark right after the paragraph's <w:pPr>."""
    if find_existing_bookmark(para_el, name):
        return
    start = etree.Element(w("bookmarkStart"))
    start.set(w("id"), str(bid))
    start.set(w("name"), name)
    end = etree.Element(w("bookmarkEnd"))
    end.set(w("id"), str(bid))
    pPr = para_el.find(w("pPr"))
    if pPr is not None:
        pPr.addnext(end)
        pPr.addnext(start)
    else:
        para_el.insert(0, end)
        para_el.insert(0, start)


def make_entry_paragraph(label: str, title: str, bookmark: str) -> etree._Element:
    """Build a LOF/LOT entry paragraph.

    Shape:
        <w:p>
          <w:pPr>
            <w:pStyle w:val="934"/>           <!-- "table of figures" -->
            <w:tabs>
              <w:tab w:val="right" w:leader="dot" w:pos="9000"/>
            </w:tabs>
          </w:pPr>
          <w:r><w:t>Figure 3.1: Proposed system overview</w:t></w:r>
          <w:r><w:tab/></w:r>
          <w:r><w:fldChar fldCharType="begin"/></w:r>
          <w:r><w:instrText> PAGEREF Fig_3_1 \\h </w:instrText></w:r>
          <w:r><w:fldChar fldCharType="separate"/></w:r>
          <w:r><w:t>1</w:t></w:r>
          <w:r><w:fldChar fldCharType="end"/></w:r>
        </w:p>
    """
    p = etree.Element(w("p"))

    pPr = etree.SubElement(p, w("pPr"))
    pStyle = etree.SubElement(pPr, w("pStyle"))
    pStyle.set(w("val"), "934")
    tabs = etree.SubElement(pPr, w("tabs"))
    tab = etree.SubElement(tabs, w("tab"))
    tab.set(w("val"), "right")
    tab.set(w("leader"), "dot")
    tab.set(w("pos"), TAB_POS_TWIPS)

    # Entry text run.
    r_text = etree.SubElement(p, w("r"))
    t = etree.SubElement(r_text, w("t"))
    t.set(qn("xml:space"), "preserve")
    t.text = f"{label}: {title}"

    # Tab run.
    r_tab = etree.SubElement(p, w("r"))
    etree.SubElement(r_tab, w("tab"))

    # PAGEREF field.
    r1 = etree.SubElement(p, w("r"))
    fc1 = etree.SubElement(r1, w("fldChar"))
    fc1.set(w("fldCharType"), "begin")
    fc1.set(w("dirty"), "true")

    r2 = etree.SubElement(p, w("r"))
    instr = etree.SubElement(r2, w("instrText"))
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" PAGEREF {bookmark} \\h "

    r3 = etree.SubElement(p, w("r"))
    fc3 = etree.SubElement(r3, w("fldChar"))
    fc3.set(w("fldCharType"), "separate")

    r4 = etree.SubElement(p, w("r"))
    t4 = etree.SubElement(r4, w("t"))
    t4.text = "1"  # placeholder; Word overwrites on first refresh.

    r5 = etree.SubElement(p, w("r"))
    fc5 = etree.SubElement(r5, w("fldChar"))
    fc5.set(w("fldCharType"), "end")

    return p


def is_heading_2(para_el) -> bool:
    pPr = para_el.find(w("pPr"))
    if pPr is None:
        return False
    pStyle = pPr.find(w("pStyle"))
    if pStyle is None:
        return False
    # Heading 2 style id in this template is "945" (see existing scripts).
    return pStyle.get(w("val")) == "945"


def is_heading_1_or_2(para_el) -> bool:
    pPr = para_el.find(w("pPr"))
    if pPr is None:
        return False
    pStyle = pPr.find(w("pStyle"))
    if pStyle is None:
        return False
    # Heading 1 = "944", Heading 2 = "945" in this template.
    return pStyle.get(w("val")) in ("944", "945")


def main() -> None:
    doc = Document(str(THESIS))
    paragraphs = doc.paragraphs

    # ---- 1. Caption paragraphs: ensure bookmarks ------------------------
    fig_re = re.compile(r"^Figure\s+(\d+(?:\.\d+)*)\s*[:—–]", re.I)
    tbl_re = re.compile(r"^Table\s+(\d+(?:\.\d+)*)\s*[:—–]", re.I)

    next_bid = BOOKMARK_ID_START
    fig_labels_in_doc: set[str] = set()
    tbl_labels_in_doc: set[str] = set()

    # Track the first occurrence of each (label) so duplicate caption-prose
    # references (the doc has "Figure 2.7 ... Figure 2.7" twice in the same
    # paragraph) only bookmark once.
    for p in paragraphs:
        text = safe_text(p._element).strip()
        m = fig_re.match(text)
        if m:
            label = m.group(1)
            if label in fig_labels_in_doc:
                continue
            fig_labels_in_doc.add(label)
            name = bookmark_name("Fig", label)
            insert_bookmark(p._element, name, next_bid)
            next_bid += 1
            continue
        m = tbl_re.match(text)
        if m:
            label = m.group(1)
            if label in tbl_labels_in_doc:
                continue
            tbl_labels_in_doc.add(label)
            name = bookmark_name("Tbl", label)
            insert_bookmark(p._element, name, next_bid)
            next_bid += 1

    print(f"  bookmarked {len(fig_labels_in_doc)} figure captions")
    print(f"  bookmarked {len(tbl_labels_in_doc)} table captions")

    # Cross-check: every hardcoded entry must have a caption in the doc.
    missing_figs = [lbl for lbl, _ in FIGURES if lbl not in fig_labels_in_doc]
    missing_tbls = [lbl for lbl, _ in TABLES if lbl not in tbl_labels_in_doc]
    if missing_figs:
        raise SystemExit(f"FIGURES references captions not in doc: {missing_figs}")
    if missing_tbls:
        raise SystemExit(f"TABLES references captions not in doc: {missing_tbls}")

    # ---- 2. Find LOF / LOT Heading-2 anchors ----------------------------
    lof_heading_el = None
    lot_heading_el = None
    for p in paragraphs:
        text = safe_text(p._element).strip().lower()
        if not is_heading_2(p._element):
            continue
        if text == "list of figures" and lof_heading_el is None:
            lof_heading_el = p._element
        elif text == "list of tables" and lot_heading_el is None:
            lot_heading_el = p._element

    if lof_heading_el is None:
        raise SystemExit("Could not find 'List of Figures' Heading 2")
    if lot_heading_el is None:
        raise SystemExit("Could not find 'List of Tables' Heading 2")

    body = lof_heading_el.getparent()

    # ---- 3. Wipe paragraphs between LOF heading and LOT heading ---------
    def remove_between(start_el, stop_el):
        """Remove every <w:p> sibling strictly between start_el and stop_el."""
        cursor = start_el.getnext()
        while cursor is not None and cursor is not stop_el:
            nxt = cursor.getnext()
            if cursor.tag == w("p"):
                body.remove(cursor)
            cursor = nxt

    remove_between(lof_heading_el, lot_heading_el)

    # ---- 4. Insert LOF entries before LOT heading -----------------------
    for label, title in FIGURES:
        para = make_entry_paragraph(
            f"Figure {label}", title, bookmark_name("Fig", label)
        )
        lot_heading_el.addprevious(para)

    # ---- 5. Wipe paragraphs after LOT heading up to the next H1/H2 ------
    cursor = lot_heading_el.getnext()
    while cursor is not None:
        if cursor.tag == w("p") and is_heading_1_or_2(cursor):
            break
        cursor = cursor.getnext()
    next_heading_el = cursor  # may be None (end of doc)

    if next_heading_el is not None:
        remove_between(lot_heading_el, next_heading_el)
        anchor_insert = next_heading_el
    else:
        # No following heading — drop everything until end and append.
        cursor = lot_heading_el.getnext()
        while cursor is not None:
            nxt = cursor.getnext()
            if cursor.tag == w("p"):
                body.remove(cursor)
            cursor = nxt
        anchor_insert = None

    # ---- 6. Insert LOT entries ------------------------------------------
    for label, title in TABLES:
        para = make_entry_paragraph(
            f"Table {label}", title, bookmark_name("Tbl", label)
        )
        if anchor_insert is not None:
            anchor_insert.addprevious(para)
        else:
            body.append(para)

    print(f"  inserted {len(FIGURES)} LOF entries")
    print(f"  inserted {len(TABLES)} LOT entries")

    doc.save(str(THESIS))
    print(f"\nSaved {THESIS}")


if __name__ == "__main__":
    main()
