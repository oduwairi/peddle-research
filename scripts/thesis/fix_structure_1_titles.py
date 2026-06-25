"""Fix chapter-structure defects flagged in the professor's review (point #2).

Scope of THIS script (text + style edits only; outline levels and the TOC are
handled by the sibling scripts run afterwards):

  D4  Rename Chapter IV's title "Result and discussion" -> "Results".
      (Discussion is its own chapter now, so the bundled title is stale.)
  D2  Insert the missing Chapter V title heading: "Discussion".
  D3  Insert the missing Chapter VI title heading: "Conclusion and
      Recommendations".
  D5  Make Chapter I consistent with the rest of the thesis:
        * restyle its title "1. Introduction" (currently Body style) to the
          Heading 2 style used by every other chapter title, text -> "Introduction";
        * number its six sections 1.1 .. 1.6 to match the TOC and Chapters II-VI.

Locating is by VISIBLE TEXT + STYLE, scoped to the relevant chapter region --
never by hardcoded paragraph index (indices shift as paragraphs are inserted).
Idempotent: re-running is a no-op once the fixes are in place.

Run from repo root:
    uv run python scripts/thesis/fix_structure_1_titles.py
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

THESIS = Path("docs/research/THESIS.docx")

H1_CHAPTER = "944"  # "Heading 1" -- the "CHAPTER N" paragraphs
H2_TITLE = "945"  # "Heading 2" -- chapter titles + section headings
SPACER = "943"  # "Body Text" -- empty spacer between CHAPTER N and its title


def style_id(p_el) -> str | None:
    pPr = p_el.find(qn("w:pPr"))
    if pPr is None:
        return None
    s = pPr.find(qn("w:pStyle"))
    return s.get(qn("w:val")) if s is not None else None


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


def get_text(p_el) -> str:
    return "".join(t.text or "" for t in p_el.iter(qn("w:t")))


def set_text(p_el, new_text: str) -> None:
    """Set the first <w:t> to new_text, clear the rest. Preserves run props."""
    ts = list(p_el.iter(qn("w:t")))
    if not ts:
        # No text run at all -- append one onto the first run, or make a run.
        r = p_el.find(qn("w:r"))
        if r is None:
            r = etree.SubElement(p_el, qn("w:r"))
        t = etree.SubElement(r, qn("w:t"))
        t.set(qn("xml:space"), "preserve")
        t.text = new_text
        return
    ts[0].text = new_text
    ts[0].set(qn("xml:space"), "preserve")
    for extra in ts[1:]:
        extra.text = ""


def find_body_chapter(paras, roman: str):
    """Return the body 'CHAPTER <roman>' paragraph element (style 944)."""
    target = f"CHAPTER {roman}"
    for p in paras:
        if style_id(p._p) == H1_CHAPTER and get_text(p._p).strip() == target:
            return p._p
    return None


def main() -> None:
    doc = Document(str(THESIS))
    paras = doc.paragraphs
    log: list[str] = []

    # ---- Locate body chapter anchors (style 944, by exact text) -------------
    ch1 = find_body_chapter(paras, "I")
    ch2 = find_body_chapter(paras, "II")
    ch5 = find_body_chapter(paras, "V")
    ch6 = find_body_chapter(paras, "VI")
    if ch1 is None or ch2 is None or ch5 is None or ch6 is None:
        raise SystemExit("Could not locate one of CHAPTER I/II/V/VI body headings")

    body = doc.element.body
    children = list(body)
    idx_ch1 = children.index(ch1)
    idx_ch2 = children.index(ch2)

    # ---- D5a: restyle Chapter I title "1. Introduction" -> "Introduction" ---
    # ---- D5b: number Chapter I sections 1.1 .. 1.6 --------------------------
    ch1_section_titles = [
        "Background",
        "Problem Statement",
        "Research Questions",
        "Contributions",
        "Paper Organization",
        "Survey Methodology",
    ]
    section_num = 0
    for el in children[idx_ch1 + 1 : idx_ch2]:
        if el.tag != qn("w:p"):
            continue
        txt = get_text(el).strip()
        sid = style_id(el)
        # the chapter title sits in the title slot, currently Body style (948)
        if txt in ("1. Introduction", "Introduction"):
            if sid != H2_TITLE or txt != "Introduction":
                set_style(el, H2_TITLE)
                set_text(el, "Introduction")
                log.append("D5  Chapter I title -> 'Introduction' (Heading 2)")
            continue
        # number the six known sections in order
        for n, base in enumerate(ch1_section_titles, start=1):
            numbered = f"1.{n} {base}"
            if txt == base:
                set_text(el, numbered)
                log.append(f"D5  Chapter I section '{base}' -> '{numbered}'")
                section_num = n
                break
            if txt == numbered:
                section_num = n  # already done
                break

    # ---- D4: Chapter IV title "Result and discussion" -> "Results" ----------
    for p in paras:
        if style_id(p._p) == H2_TITLE and get_text(p._p).strip() == "Result and discussion":
            set_text(p._p, "Results")
            log.append("D4  Chapter IV title 'Result and discussion' -> 'Results'")
            break

    # ---- D2 / D3: insert missing Chapter V / VI title headings --------------
    # Clone the safe inline pattern from Chapter III: an empty Body-Text spacer
    # (943) followed by the Heading-2 title (945). Both verified drawing-free.
    methodology_title = None
    spacer_tmpl = None
    for p in paras:
        if style_id(p._p) == H2_TITLE and get_text(p._p).strip() == "Methodology":
            methodology_title = p._p
            prev = methodology_title.getprevious()
            if prev is not None and prev.tag == qn("w:p") and style_id(prev) == SPACER:
                spacer_tmpl = prev
            break
    if methodology_title is None:
        raise SystemExit("Could not find a safe Heading-2 title template ('Methodology')")
    assert methodology_title.find(".//" + qn("w:drawing")) is None, "title template has a drawing!"

    def title_already_present(chapter_el, title_text: str) -> bool:
        nxt = chapter_el.getnext()
        while nxt is not None and nxt.tag == qn("w:p"):
            t = get_text(nxt).strip()
            if t == title_text:
                return True
            if t:  # first non-empty paragraph after CHAPTER -- stop
                return False
            nxt = nxt.getnext()
        return False

    def insert_chapter_title(chapter_el, title_text: str) -> None:
        if title_already_present(chapter_el, title_text):
            log.append(f"     (chapter title '{title_text}' already present -- skipped)")
            return
        title_el = deepcopy(methodology_title)
        set_text(title_el, title_text)
        if spacer_tmpl is not None:
            spacer_el = deepcopy(spacer_tmpl)
            # ensure spacer carries no text
            for t in spacer_el.iter(qn("w:t")):
                t.text = ""
            chapter_el.addnext(spacer_el)
            spacer_el.addnext(title_el)
        else:
            chapter_el.addnext(title_el)
        log.append(f"D    inserted chapter title heading '{title_text}'")

    insert_chapter_title(ch5, "Discussion")
    insert_chapter_title(ch6, "Conclusion and Recommendations")

    # ---- save + report ------------------------------------------------------
    doc.save(str(THESIS))
    print("CHANGES:")
    for line in log:
        print("  " + line)
    if not log:
        print("  (none -- document already in target state)")
    print(f"\nSaved {THESIS}")


if __name__ == "__main__":
    main()
