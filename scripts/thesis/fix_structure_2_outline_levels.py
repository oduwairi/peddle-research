"""Normalize heading OUTLINE LEVELS so an auto-generated TOC field renders a
clean three-level hierarchy.

Problem: the body uses only two heading styles -- 944 ("Heading 1") for the
"CHAPTER N" line and 945 ("Heading 2") for *everything else*: front-matter
section headings, chapter titles, X.Y sections AND X.Y.Z subsections. Style 944
sits at outline level 2 and 945 at level 3, so a TOC field would indent chapters
and flatten sections with subsections, and would also emit blank entries for the
many empty Heading-styled spacer paragraphs.

There is also a body inconsistency: Chapter II styles its X.Y.Z subsections as
Heading 2 (945) -- so "2.1.1" renders at the same visual level as "2.1" -- while
Chapters III-IV correctly use Heading 3 (882). This script first restyles the 26
Ch II subsections to Heading 3 so the whole thesis uses chapter=944 / section=945
/ subsection=882 uniformly, then sets outline levels.

Fix: write a *direct* paragraph-level <w:outlineLvl> override on each heading
paragraph, classified by visible text and front-matter/body position. Styles are
left untouched otherwise (no document-wide ripple). Mapping (w:val is 0-indexed;
val=0 -> TOC level 1):

  empty 944/945 spacer .................... 9  (excluded from TOC)
  "Table of Contents" heading ............. 9  (no self-reference)
  front-matter section heading ............ 0  (Approval, Declaration,
                                               Acknowledgments, Abstract, Ozet,
                                               List of Figures/Tables/Abbreviations)
  "CHAPTER N" .............................. 0
  "References" ............................. 0
  chapter title (no number) ............... 1  (Introduction, Methodology, ...)
  X.Y section .............................. 1
  X.Y.Z subsection ......................... 2

Idempotent. Run AFTER fix_structure_1_titles.py and BEFORE the TOC-field script.

Run from repo root:
    uv run python scripts/thesis/fix_structure_2_outline_levels.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

THESIS = Path("docs/research/THESIS.docx")

HEADING_STYLES = {"944", "945", "882"}  # Heading 1 / Heading 2 / Heading 3
H2_STYLE = "945"
H3_STYLE = "882"
SUBSECTION_RE = re.compile(r"^\d+\.\d+\.\d+")
SECTION_RE = re.compile(r"^\d+\.\d+(?!\d)")


def set_style(p_el, sid: str) -> None:
    pPr = p_el.find(qn("w:pPr"))
    if pPr is None:
        pPr = etree.SubElement(p_el, qn("w:pPr"))
        p_el.insert(0, pPr)
    for existing in pPr.findall(qn("w:pStyle")):
        pPr.remove(existing)
    pStyle = etree.Element(qn("w:pStyle"))
    pStyle.set(qn("w:val"), sid)
    pPr.insert(0, pStyle)


def style_id(p_el) -> str | None:
    pPr = p_el.find(qn("w:pPr"))
    if pPr is None:
        return None
    s = pPr.find(qn("w:pStyle"))
    return s.get(qn("w:val")) if s is not None else None


def get_text(p_el) -> str:
    return "".join(t.text or "" for t in p_el.iter(qn("w:t")))


def set_outline(p_el, val: int) -> None:
    """Set a direct <w:outlineLvl w:val="val"/> in the paragraph's pPr."""
    pPr = p_el.find(qn("w:pPr"))
    if pPr is None:
        pPr = etree.Element(qn("w:pPr"))
        p_el.insert(0, pPr)
    for existing in pPr.findall(qn("w:outlineLvl")):
        pPr.remove(existing)
    ol = etree.SubElement(pPr, qn("w:outlineLvl"))
    ol.set(qn("w:val"), str(val))


def classify(text: str, in_front_matter: bool) -> int:
    t = text.strip()
    if not t:
        return 9
    if t == "Table of Contents":
        return 9
    if in_front_matter:
        return 0  # front-matter section heading -> flush level 1
    if t.startswith("CHAPTER "):
        return 0
    if t == "References":
        return 0
    if SUBSECTION_RE.match(t):
        return 2
    if SECTION_RE.match(t):
        return 1
    return 1  # chapter title or unnumbered section heading -> level 2


def main() -> None:
    doc = Document(str(THESIS))
    paras = doc.paragraphs

    # body starts at the first 944 "CHAPTER I"
    body_start = None
    for i, p in enumerate(paras):
        if style_id(p._p) == "944" and get_text(p._p).strip() == "CHAPTER I":
            body_start = i
            break
    if body_start is None:
        raise SystemExit("Could not locate body start (CHAPTER I, style 944)")

    # ---- consistency fix: Ch II subsections Heading 2 -> Heading 3 ----------
    restyled = 0
    for p in paras:
        if style_id(p._p) == H2_STYLE and SUBSECTION_RE.match(get_text(p._p).strip()):
            set_style(p._p, H3_STYLE)
            restyled += 1
    print(f"Restyled {restyled} X.Y.Z subsections Heading 2 -> Heading 3 (consistency)\n")

    counts: Counter[int] = Counter()
    samples: dict[int, list[str]] = {0: [], 1: [], 2: [], 9: []}
    for i, p in enumerate(paras):
        if style_id(p._p) not in HEADING_STYLES:
            continue
        text = get_text(p._p)
        val = classify(text, in_front_matter=(i < body_start))
        set_outline(p._p, val)
        counts[val] += 1
        if text.strip() and len(samples[val]) < 6:
            samples[val].append(text.strip()[:42])

    doc.save(str(THESIS))

    label = {0: "L1 (flush)", 1: "L2", 2: "L3", 9: "excluded"}
    print("OUTLINE LEVELS APPLIED (heading paragraphs):")
    for val in (0, 1, 2, 9):
        print(f"  val={val} {label[val]:>11}: {counts[val]:>3} paragraphs")
        for s in samples[val]:
            print(f"             e.g. {s!r}")
    print(f"\nSaved {THESIS}")


if __name__ == "__main__":
    main()
