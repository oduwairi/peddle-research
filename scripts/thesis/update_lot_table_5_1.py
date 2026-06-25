"""Update the hardcoded List of Tables after Table 2.6 -> Table 5.1 was moved.

The main Table of Contents is a live Word field (TOC \\o "1-3" …) and was not
affected by relocating a table — no heading changed. The List of Tables, however,
is a hand-built **static** table (body[335]); it still carried the stale entry

    Table 2.6: Comparison of this work against related … systems   0

in the 2.x group. After move_2_6_to_discussion.py the table is Table 5.1 in
Chapter V, so this script renumbers that List-of-Tables row to "Table 5.1:" and
relocates it to the end of the list (after Table 4.9) to keep numeric order. The
caption text and the (placeholder) page cell are left untouched — every row's
page number is still "0" pending a real paginated pass, which this script does
not change.

Idempotent: if the "Table 2.6" row is gone and a "Table 5.1" row is already last,
it is a no-op.

Run:  uv run python scripts/thesis/update_lot_table_5_1.py [--dry-run]
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

DOCX = Path("docs/research/THESIS.docx")
LOCK = Path("docs/research/.~lock.THESIS.docx#")


def ptext(el) -> str:
    return "".join(t.text or "" for t in el.findall(".//" + qn("w:t")))


def find_lot_table(children):
    for i, c in enumerate(children):
        if c.tag.split("}")[-1] == "p" and ptext(c).strip() == "List of Tables":
            j = i + 1
            while j < len(children) and children[j].tag.split("}")[-1] != "tbl":
                j += 1
            if j < len(children):
                return children[j]
    return None


def first_cell_text(tr) -> str:
    tc = tr.find(qn("w:tc"))
    return ptext(tc).strip() if tc is not None else ""


def main() -> int:
    dry = "--dry-run" in sys.argv
    if not DOCX.exists():
        print(f"ERROR: {DOCX} not found", file=sys.stderr)
        return 1
    if LOCK.exists():
        print("ERROR: THESIS.docx is open in an editor (lock present). Close it.", file=sys.stderr)
        return 1

    doc = Document(str(DOCX))
    children = list(doc.element.body.iterchildren())
    tbl = find_lot_table(children)
    if tbl is None:
        print("ERROR: List of Tables table not found", file=sys.stderr)
        return 2

    rows = tbl.findall(qn("w:tr"))
    row_2_6 = next((r for r in rows if first_cell_text(r).startswith("Table 2.6")), None)
    row_5_1 = next((r for r in rows if first_cell_text(r).startswith("Table 5.1")), None)

    if row_2_6 is None:
        if row_5_1 is not None and rows[-1] is row_5_1:
            print("== Already updated: 'Table 5.1' row present and last; no 'Table 2.6' row. No-op.")
            return 0
        print("ERROR: no 'Table 2.6' row to update (and 'Table 5.1' not already last).", file=sys.stderr)
        return 3

    if dry:
        print("=== DRY RUN — List of Tables ===")
        print(f"  Found stale row: {first_cell_text(row_2_6)[:70]!r}")
        print("  -> renumber 'Table 2.6:' -> 'Table 5.1:'")
        print(f"  -> move from current position to LAST row (after {first_cell_text(rows[-1])[:30]!r})")
        print("  Page cell left as-is (all rows are placeholder '0').")
        return 0

    # 1. renumber the label run "Table 2.6: " -> "Table 5.1: "
    tc0 = row_2_6.find(qn("w:tc"))
    done = False
    for t in tc0.findall(".//" + qn("w:t")):
        if t.text and "Table 2.6" in t.text:
            t.text = t.text.replace("Table 2.6", "Table 5.1")
            done = True
            break
    if not done:
        print("ERROR: 'Table 2.6' label run not found in cell — aborting.", file=sys.stderr)
        return 4

    # 2. move the row to the end (Table 5.1 sorts after Table 4.9)
    last = rows[-1]
    if last is not row_2_6:
        last.addnext(row_2_6)

    doc.save(str(DOCX))
    print("UPDATED List of Tables:")
    print("  ++ 'Table 2.6: …' -> 'Table 5.1: …' (renumbered)")
    print("  ++ moved to last row (after Table 4.9)")
    print(f"\nSaved {DOCX}")
    print("Note: page numbers in the List of Tables are still placeholder '0' for "
          "every row — they need a real paginated pass (out of scope here).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
