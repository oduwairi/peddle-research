"""Move the related-work comparison table from Chapter II to the Discussion chapter.

User instruction (2026-06-15): "For now move the comparison table to discussion
chapter." The comparison table is Table 2.6 ("Comparison of this work against
related domain-specialized and marketing language-model systems") — the 16-row
capability matrix added for reviewer #11. It currently closes Chapter II under
§2.9, as caption + table + note + an interpretive paragraph ("The reviewed works
are split into two main categories…").

This script relocates that block (caption + table + spacer + note + interpretive
paragraph) to the end of §5.4 "Domain Specialization Beats Frontier Scale" in
Chapter V (Discussion) — §5.4's body argues precisely the small-FT-vs-frontier
case the table demonstrates — and renumbers it to **Table 5.1** (Chapter V has no
tables yet). To read natively in its new chapter the caption is restyled from
"Body" (1006, em-dash) to "Body Text" (1001, colon) to match Figure 5.1 and the
Chapter IV table captions.

Side edits:
  • Ch II survey wrap-up cross-ref "(Tables 2.3 and 2.6)" -> "(Table 2.3)"
    (2.6 has left the chapter; the survey paragraph stays as the lit-review close).
  • §5.4 body gains a minimal "(Table 5.1)" parenthetical callout (reviewer #5
    pattern — no new sentence).

Bookmark Tbl_2_6 is renamed Tbl_5_1. No PAGEREF field references it (the List of
Tables is a stale hand-built table that does not yet list 2.6), so nothing breaks;
the List of Tables still needs the usual manual OnlyOffice refresh.

Idempotent: anchors located by bookmark / unique visible text; re-running after a
successful move is a no-op (the Tbl_2_6 anchor is gone, Tbl_5_1 already in Ch V).

Run:  uv run python scripts/thesis/move_2_6_to_discussion.py [--dry-run]
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

DOCX = Path("docs/research/THESIS.docx")
LOCK = Path("docs/research/.~lock.THESIS.docx#")

OLD_BM = "Tbl_2_6"
NEW_BM = "Tbl_5_1"
NOTE_PREFIX = "Note."
INTERP_PREFIX = "The reviewed works are split into two main categories"
SURVEY_PREFIX = "This survey has examined the research landscape"
DEST_HEADING = "5.5 Why Backtranslation Over Other Fine-Tuning Methods"
S54_BODY_PREFIX = "The main premise of this thesis is to test whether a small"
STYLE_BODYTEXT = "1001"  # "Body Text" — Chapter III–V caption/body style


def ptext(el) -> str:
    return "".join(t.text or "" for t in el.findall(".//" + qn("w:t")))


def pstyle(p_el):
    pr = p_el.find(qn("w:pPr"))
    if pr is not None:
        s = pr.find(qn("w:pStyle"))
        if s is not None:
            return s.get(qn("w:val"))
    return None


def set_pstyle(p_el, val: str) -> None:
    pr = p_el.find(qn("w:pPr"))
    if pr is None:
        pr = p_el.makeelement(qn("w:pPr"), {})
        p_el.insert(0, pr)
    s = pr.find(qn("w:pStyle"))
    if s is None:
        s = pr.makeelement(qn("w:pStyle"), {})
        pr.insert(0, s)
    s.set(qn("w:val"), val)


def find_caption_by_bookmark(children, name: str):
    for c in children:
        if c.tag.split("}")[-1] != "p":
            continue
        for bm in c.findall(qn("w:bookmarkStart")):
            if bm.get(qn("w:name")) == name:
                return c
    return None


def find_para(children, prefix: str, style: str | None = None):
    for c in children:
        if c.tag.split("}")[-1] != "p":
            continue
        if ptext(c).strip().startswith(prefix) and (style is None or pstyle(c) == style):
            return c
    return None


def renumber_caption(cap) -> None:
    """Table 2.6 — … (Body, em-dash)  ->  Table 5.1: … (renumber + colon)."""
    for r in cap.findall(qn("w:r")):
        t = r.find(qn("w:t"))
        if t is not None and t.text:
            if t.text.strip().startswith("Table 2.6"):
                t.text = "Table 5.1"
            elif t.text.strip() == "—":
                t.text = ": "
        instr = r.find(qn("w:instrText"))
        if instr is not None and instr.text and "TC " in instr.text:
            instr.text = instr.text.replace("Table 2.6 — ", "Table 5.1: ").replace(
                "Table 2.6", "Table 5.1"
            )
    for bm in cap.findall(qn("w:bookmarkStart")):
        if bm.get(qn("w:name")) == OLD_BM:
            bm.set(qn("w:name"), NEW_BM)
    set_pstyle(cap, STYLE_BODYTEXT)


def fix_survey_crossref(survey) -> str:
    """(Tables 2.3 and 2.6) -> (Table 2.3) — reverses the split-run callout."""
    runs = survey.findall(qn("w:r"))
    hits = []
    for r in runs:
        t = r.find(qn("w:t"))
        if t is None or not t.text:
            continue
        if t.text.endswith("(Tables 2"):
            t.text = t.text[: -len("(Tables 2")] + "(Table 2"
            hits.append("open")
        elif t.text == ".3 and 2":
            t.text = ".3)"
            hits.append("mid")
        elif t.text.startswith(".6)"):
            t.text = t.text[len(".6)"):]
            hits.append("tail")
    return "++ Ch II cross-ref: (Tables 2.3 and 2.6) -> (Table 2.3)" if len(hits) == 3 \
        else f"!! cross-ref partial match {hits} — verify [524] by hand"


def add_callout(body_p) -> str:
    if "(Table 5.1)" in ptext(body_p):
        return "== §5.4 callout already present"
    runs = body_p.findall(qn("w:r"))
    for r in reversed(runs):
        t = r.find(qn("w:t"))
        if t is not None and t.text and t.text.rstrip().endswith("."):
            stripped = t.text.rstrip()
            tail = t.text[len(stripped):]
            t.text = stripped[:-1] + " (Table 5.1)." + tail
            return "++ §5.4 callout: appended (Table 5.1)"
    return "!! §5.4 callout anchor not found — add (Table 5.1) by hand"


def main() -> int:
    dry = "--dry-run" in sys.argv
    if not DOCX.exists():
        print(f"ERROR: {DOCX} not found", file=sys.stderr)
        return 1
    if LOCK.exists():
        print("ERROR: THESIS.docx is open in an editor (lock present). Close it.", file=sys.stderr)
        return 1

    doc = Document(str(DOCX))
    body = doc.element.body
    children = list(body.iterchildren())

    cap = find_caption_by_bookmark(children, OLD_BM)
    if cap is None:
        if find_caption_by_bookmark(children, NEW_BM) is not None:
            print("== Already moved: bookmark Tbl_5_1 present, Tbl_2_6 absent. No-op.")
            return 0
        print("ERROR: caption bookmark Tbl_2_6 not found (and Tbl_5_1 absent).", file=sys.stderr)
        return 2

    # Collect the contiguous block: caption -> table -> spacer -> note -> interpretive.
    block = [cap]
    el = cap
    saw_table = saw_note = saw_interp = False
    for _ in range(6):
        el = el.getnext()
        if el is None:
            break
        tag = el.tag.split("}")[-1]
        block.append(el)
        if tag == "tbl":
            saw_table = True
        elif tag == "p":
            txt = ptext(el).strip()
            if txt.startswith(NOTE_PREFIX):
                saw_note = True
            elif txt.startswith(INTERP_PREFIX):
                saw_interp = True
                break
    if not (saw_table and saw_note and saw_interp):
        print(f"ERROR: unexpected block shape (table={saw_table} note={saw_note} "
              f"interp={saw_interp}); aborting to avoid moving the wrong nodes.", file=sys.stderr)
        return 3

    dest = find_para(children, DEST_HEADING, style="1003")
    if dest is None:
        print(f"ERROR: destination heading {DEST_HEADING!r} not found", file=sys.stderr)
        return 4
    survey = find_para(children, SURVEY_PREFIX)
    s54_body = find_para(children, S54_BODY_PREFIX)

    block_desc = []
    for e in block:
        tg = e.tag.split("}")[-1]
        block_desc.append("TABLE" if tg == "tbl" else (ptext(e).strip()[:42] or "(empty)"))

    if dry:
        print("=== DRY RUN — move Table 2.6 -> Table 5.1 (Discussion §5.4) ===\n")
        print("Block to move (in order):")
        for d in block_desc:
            print(f"   - {d}")
        print(f"\nDestination: insert before §5.5 heading {DEST_HEADING!r}")
        print(f"Renumber   : Table 2.6 — …  ->  Table 5.1: …  (style 1006 'Body' -> 1001 'Body Text')")
        print(f"Bookmark   : {OLD_BM} -> {NEW_BM}")
        print("Ch II edit : (Tables 2.3 and 2.6) -> (Table 2.3)")
        print("§5.4 edit  : append (Table 5.1) callout")
        return 0

    # 1. renumber + restyle caption
    renumber_caption(cap)
    # restyle the spacer + note (1006 'Body') to match Chapter V (1001 'Body Text')
    for e in block:
        if e.tag.split("}")[-1] == "p" and pstyle(e) == "1006":
            set_pstyle(e, STYLE_BODYTEXT)

    # 2. move block: addprevious relocates each element before the §5.5 heading, in order
    for e in block:
        dest.addprevious(e)

    print("INSERTIONS / MOVES:")
    print(f"  ++ Moved {len(block)} elements to end of §5.4 (before §5.5):")
    for d in block_desc:
        print(f"       - {d}")
    print(f"  ++ Renumbered caption: Table 2.6 — …  ->  Table 5.1: …  (Body -> Body Text)")
    print(f"  ++ Bookmark {OLD_BM} -> {NEW_BM}")

    # 3. fix Ch II cross-ref
    if survey is not None:
        print("  " + fix_survey_crossref(survey))
    else:
        print("  !! survey paragraph not found — Ch II cross-ref left unchanged")

    # 4. §5.4 callout
    if s54_body is not None:
        print("  " + add_callout(s54_body))
    else:
        print("  !! §5.4 body not found — callout skipped")

    doc.save(str(DOCX))
    print(f"\nSaved {DOCX}")
    print("\nFollow-up (author, OnlyOffice): Ctrl+A -> F9 to refresh the TOC / "
          "List of Tables (Table 5.1 now in Ch V; Table 2.6 no longer in Ch II).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
