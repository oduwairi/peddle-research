"""Reviewer-feedback #6 fixes for THESIS.docx (Results / Discussion / Conclusion).

Three surgical, idempotent edits — no new author-voice prose beyond the one
approved triangulation sentence (#3):

  #1  Standardize Config A's model name to GPT-5.5 (the single-shot frontier
      baseline) in the four result-section spots that mislabel it as
      "gpt-5.4-mini" (which is actually the *agent-pipeline orchestrator*).
        - §4.2 intro, §4.1.2 Config A, Fig 4.1 caption, Fig 4.4.1 caption.
  #2  Temper §5.4's "wins on every dimension" overstatement so it matches the
      author's own §4.2.4 (C trails A on the engagement-velocity head).
  #3  Add an explicit cross-arm triangulation note to §6.1's RQ1 answer so the
      "beats frontier" claim does not rest on the learned scorer alone.

Run:  uv run python scripts/thesis/apply_reviewer6_fixes.py
      uv run python scripts/thesis/apply_reviewer6_fixes.py --dry-run

Locate-by-anchor (never hardcoded indices); idempotent (skips an edit whose
result is already present). Per docs/research/THESIS_EDITING.md.
"""

from __future__ import annotations

import sys

from docx import Document

DOCX = "docs/research/THESIS.docx"


def replace_in_paragraph(p, old: str, new: str) -> bool:
    """Replace the first occurrence of `old` with `new` across a paragraph's
    runs, preserving run formatting. Returns True if a replacement happened.

    Works on the concatenation of run texts, so it is robust to however Word
    split the visible text across <w:r> elements. The full `new` lands in the
    first overlapping run; the old-substring tail is cleared from the rest.
    """
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
    """First body paragraph containing `anchor`, skipping TOC / List-of-Figures
    / List-of-Tables field-cache entries (which repeat caption text)."""
    for p in paras:
        style = (p.style.name if p.style else "").lower()
        if any(s in style for s in _SKIP_STYLES):
            continue
        if anchor in p.text:
            return p
    return None


# (label, anchor, old, new, done_marker)
EDITS = [
    # ---- #1 Config A model name -> GPT-5.5 ---------------------------------
    (
        "#1 §4.2 intro",
        "Following the methodology, this chapter",
        "the frontier model (gpt-5.4-mini)",
        "the frontier model (GPT-5.5)",
        "the frontier model (GPT-5.5)",
    ),
    (
        "#1 §4.1.2 Config A",
        "The evaluation pipeline uses five main configurations",
        "which is a GPT-5.4 model (gpt-5.4-mini) that is prompted",
        "which is a GPT-5.5 model that is prompted",
        "which is a GPT-5.5 model that is prompted",
    ),
    (
        "#1 Fig 4.1 caption",
        "Figure 4.1: Composite mean per configuration",
        "A = gpt-5.4-mini frontier",
        "A = GPT-5.5 frontier",
        "A = GPT-5.5 frontier",
    ),
    (
        "#1 Fig 4.4.1 caption",
        "Figure 4.4.1: Per-ad overlap with the real winning ad",
        "whereas A (gpt-5.4-mini) and B",
        "whereas A (GPT-5.5) and B",
        "whereas A (GPT-5.5) and B",
    ),
    # ---- #2 §5.4 temper "every dimension" ----------------------------------
    (
        "#2a §5.4 'nearly every'",
        "The main premise of this thesis is to test whether a small",
        "wins on every dimension our evaluation",
        "wins on nearly every dimension our evaluation",
        "wins on nearly every dimension our evaluation",
    ),
    (
        "#2b §5.4 velocity exception",
        "The main premise of this thesis is to test whether a small",
        "our evaluation measures (the full per-platform",
        "our evaluation measures (the lone exception is the engagement-velocity "
        "head; the full per-platform",
        "the lone exception is the engagement-velocity head",
    ),
    # ---- #3 §6.1 triangulation cross-ref -----------------------------------
    (
        "#3 §6.1 triangulation",
        "The aim of this thesis was to build a small fine-tuned model",
        "outperform a large model in a narrow domain task. RQ2 asked",
        "outperform a large model in a narrow domain task. This ranking is not "
        "an artefact of a single metric: the MAUVE arm also places A lowest "
        "(§4.3) and the reference-overlap arm ranks C > A > B on all five "
        "metrics (§4.4), though all three remain automatic proxies rather "
        "than human or live-deployment measures. RQ2 asked",
        "This ranking is not an artefact of a single metric",
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
            ok = replace_in_paragraph(p, old, new)
            if not ok:
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
