"""Reviewer point #7 — Chapter I must start at ARABIC page "1", not a roman figure.

Root cause (diagnosed on THESIS.docx, 2026-06-09): page numbers live in a
top-right floating textbox inside the HEADERS (footers are empty). The
front-matter sections use `header2.xml`, whose textbox field is `PAGE \\* ROMAN`
with `pgNumType start=1` → front matter renders i, ii, iii … (correct). The BODY
is the final section (starts exactly at the CHAPTER I Heading-1) but has NO
header of its own — it is linked-to-previous, so it INHERITS the roman header and
continues the count → Chapter I shows a roman numeral. That is the bug.

Fix (minimal, standard thesis convention; front matter left roman, untouched):
  1. Unlink the body section's header (creates its own header part).
  2. Populate it with a clone of header2's positioned textbox, flipping
     `PAGE \\* ROMAN` → `PAGE \\* ARABIC` (both the mc:Choice drawing and the
     mc:Fallback VML), so the arabic number sits in the exact same spot.
  3. Set `pgNumType fmt="decimal" start="1"` on the body section → restart at 1.

PAGE fields (unlike the PAGEREF fields in the LoF/LoT) update automatically and
render correctly in OnlyOffice, so Chapter I will read "1" on open.

Idempotent: if the body section already has its own ARABIC header, it is a no-op.
python-docx + lxml only; no byte-slicing.

Run:  uv run python scripts/thesis/fix_chapter1_page_numbering.py
      uv run python scripts/thesis/fix_chapter1_page_numbering.py --dry-run
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

DOCX = Path("docs/research/THESIS.docx")


def find_roman_header_paragraph(doc):
    """Return a deep copy of the <w:p> from the header that carries the
    `PAGE \\* ROMAN` field (the positioned page-number textbox)."""
    for part in doc.part.package.iter_parts():
        if "header" not in part.partname:
            continue
        el = part.element
        for instr in el.iter(qn("w:instrText")):
            if "PAGE" in (instr.text or "") and "ROMAN" in (instr.text or ""):
                p = el.find(qn("w:p"))
                return copy.deepcopy(p) if p is not None else None
    return None


def romanize_to_arabic(p_el) -> int:
    """In a cloned header paragraph, flip the PAGE field to ARABIC, reset the
    cached value to '1', and bump shape/docPr ids to avoid collisions."""
    n = 0
    for instr in p_el.iter(qn("w:instrText")):
        if instr.text and "ROMAN" in instr.text:
            instr.text = instr.text.replace("\\* ROMAN", "\\* ARABIC")
            n += 1
    # cached field result: the roman numeral text run(s)
    for t in p_el.iter(qn("w:t")):
        if (t.text or "").strip().upper() in {
            "XIV", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
            "XI", "XII", "XIII", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
        }:
            t.text = "1"
    # bump ids (docPr / cNvPr / VML shape) so they do not clash with header2
    for dp in p_el.iter(qn("wp:docPr")):
        dp.set("id", "2")
        dp.set("name", "Textbox body page")
    a_ns = "{http://schemas.microsoft.com/office/word/2010/wordprocessingShape}"
    for cnv in p_el.iter(f"{a_ns}cNvPr"):
        cnv.set("id", "8")
        cnv.set("name", "Textbox body page")
    v_ns = "{urn:schemas-microsoft-com:vml}"
    o_ns = "{urn:schemas-microsoft-com:office:office}"
    for sh in p_el.iter(f"{v_ns}shape"):
        sh.set("id", "shape 1")
        sh.set(f"{o_ns}spid", "_x0000_s1026")
    return n


def set_pgnum_decimal_start1(sectPr) -> None:
    old = sectPr.find(qn("w:pgNumType"))
    if old is not None:
        sectPr.remove(old)
    pgNum = OxmlElement("w:pgNumType")
    pgNum.set(qn("w:fmt"), "decimal")
    pgNum.set(qn("w:start"), "1")
    cols = sectPr.find(qn("w:cols"))
    if cols is not None:
        cols.addprevious(pgNum)  # schema: pgNumType immediately precedes cols
    else:
        sectPr.append(pgNum)


def main() -> int:
    dry = "--dry-run" in sys.argv
    if not DOCX.exists():
        print(f"ERROR: {DOCX} not found", file=sys.stderr)
        return 1
    lock = DOCX.parent / f".~lock.{DOCX.name}#"
    if not dry and lock.exists():
        print(f"ERROR: {lock} present — close THESIS.docx in OnlyOffice first.",
              file=sys.stderr)
        return 1

    doc = Document(str(DOCX))
    body_sec = doc.sections[-1]

    # Sanity: the final section must begin at CHAPTER I.
    paras = doc.paragraphs
    ch1 = next((i for i, p in enumerate(paras)
                if p.text.strip() == "CHAPTER I"
                and (p.style.name or "").startswith("Heading 1")), None)
    prev_pPr = paras[ch1 - 1]._p.find(qn("w:pPr")) if ch1 else None
    if ch1 is None or prev_pPr is None or prev_pPr.find(qn("w:sectPr")) is None:
        print("ERROR: could not confirm CHAPTER I begins the final section; aborting.",
              file=sys.stderr)
        return 1
    print(f"Body section = final section; begins at CHAPTER I (para {ch1}).")

    # Idempotency: already has its own arabic header?
    already = (not body_sec.header.is_linked_to_previous) and any(
        "ARABIC" in (i.text or "")
        for i in body_sec.header.part.element.iter(qn("w:instrText"))
    )
    if already:
        print("No-op: body section already has its own ARABIC page-number header.")
        return 0

    roman_p = find_roman_header_paragraph(doc)
    if roman_p is None:
        print("ERROR: could not locate the roman PAGE-field header to clone.",
              file=sys.stderr)
        return 1

    if dry:
        print("DRY-RUN plan:")
        print("  - unlink body section header (new header part)")
        print("  - inject cloned textbox with PAGE \\* ARABIC (cached '1')")
        print("  - set body sectPr pgNumType fmt=decimal start=1 (before <w:cols>)")
        print("  Front-matter sections (roman) left untouched.")
        print("DRY-RUN: no changes written.")
        return 0

    # 1) Unlink → create the body section's own header part.
    body_sec.header.is_linked_to_previous = False
    hdr_el = body_sec.header.part.element

    # 2) Replace its content with the arabic clone.
    flipped = romanize_to_arabic(roman_p)
    for child in list(hdr_el):
        hdr_el.remove(child)
    hdr_el.append(roman_p)

    # 3) Restart numbering at arabic 1 on the body section.
    set_pgnum_decimal_start1(body_sec._sectPr)

    doc.save(str(DOCX))
    print(f"saved {DOCX}  (flipped {flipped} PAGE fields ROMAN->ARABIC)")
    print("FOLLOW-UP in OnlyOffice: Ctrl+A → F9 to refresh the TOC (body page "
          "numbers now restart at 1; front matter stays roman).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
