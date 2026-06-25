"""Reviewer round 2026-06-09 — issue #5 one-line critical touches (§2.6, §2.8).

These two sections are already critical, but the reviewer named them explicitly,
so a single critical sentence is appended exactly where the reviewer looked.
Appended inside the section intro paragraph -> no heading/TOC change.

Run:  uv run python scripts/thesis/apply_issue5_touches.py --dry-run
      uv run python scripts/thesis/apply_issue5_touches.py

Per docs/research/THESIS_EDITING.md. Idempotent; anchor-located; aborts on fail.
"""

from __future__ import annotations

import sys

from docx import Document

DOCX = "docs/research/THESIS.docx"


def replace_in_paragraph(p, old: str, new: str) -> bool:
    runs = p.runs
    texts = [r.text for r in runs]
    full = "".join(texts)
    idx = full.find(old)
    if idx == -1:
        return False
    start, end = idx, idx + len(old)
    pos = 0
    spans = []
    for t in texts:
        spans.append((pos, pos + len(t)))
        pos += len(t)
    out = list(texts)
    first = True
    for k, (lo, hi) in enumerate(spans):
        if hi <= start or lo >= end:
            continue
        a = max(start, lo) - lo
        b = min(end, hi) - lo
        if first:
            out[k] = texts[k][:a] + new + texts[k][b:]
            first = False
        else:
            out[k] = texts[k][:a] + texts[k][b:]
    for r, t in zip(runs, out):
        if r.text != t:
            r.text = t
    return True


_SKIP_STYLES = ("toc", "table of figures", "table of contents", "table of tables")


def find_para(paras, anchor: str):
    for p in paras:
        style = (p.style.name if p.style else "").lower()
        if any(s in style for s in _SKIP_STYLES):
            continue
        if anchor in p.text:
            return p
    return None


_T26 = (
    " However, none of these methods were validated specifically for marketing "
    "creative quality, motivating the grounded evaluation framework developed in "
    "this research."
)
_T28 = (
    " Yet prior work largely names these risks without offering concrete "
    "mitigations for a generative ad system, which this research instead builds "
    "into its design."
)

# (label, anchor, old, new, done_marker)
EDITS = [
    (
        "§2.6 touch",
        "With the architectural components of a marketing agent",
        "evaluate the quality of a generative marketing LLM as in this research.",
        "evaluate the quality of a generative marketing LLM as in this research."
        + _T26,
        "validated specifically for marketing creative quality",
    ),
    (
        "§2.8 touch",
        "As for all contemporary AI systems, ethical and legal concerns",
        "amplifying the potential for both benefit and harm at scale.",
        "amplifying the potential for both benefit and harm at scale." + _T28,
        "without offering concrete mitigations for a generative ad system",
    ),
]


def main() -> int:
    dry = "--dry-run" in sys.argv
    doc = Document(DOCX)
    paras = doc.paragraphs

    applied = skipped = failed = 0
    print(f"{'DRY-RUN ' if dry else ''}EDITS on {DOCX}\n" + "-" * 60)
    for label, anchor, old, new, marker in EDITS:
        p = find_para(paras, anchor)
        if p is None:
            print(f"  FAIL  {label}: anchor not found -> {anchor!r}")
            failed += 1
            continue
        if marker in p.text:
            print(f"  skip  {label}: already applied")
            skipped += 1
            continue
        full = "".join(r.text for r in p.runs)
        if old not in full:
            print(f"  FAIL  {label}: old text not found in paragraph")
            print(f"        looked for: {old!r}")
            failed += 1
            continue
        if not dry:
            if not replace_in_paragraph(p, old, new):
                print(f"  FAIL  {label}: replacement did not fire")
                failed += 1
                continue
        print(f"  OK    {label}")
        applied += 1

    print("-" * 60)
    print(f"applied={applied} skipped={skipped} failed={failed}")
    if failed:
        print("ABORT: not saving (some edits failed).")
        return 1
    if dry:
        print("DRY-RUN: not saving.")
        return 0
    doc.save(DOCX)
    print(f"saved {DOCX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
