"""Reviewer round 2026-06-09 — issue #3 (abrupt transitions) for THESIS.docx.

Appends one linking sentence to the END of the final body paragraph of three
subsections whose boundaries had no bridge to the next section. The other two
boundaries the reviewer's note implies (2.1.3->2.2, 2.2.3->2.3) already carry a
transition sentence and are left untouched.

Each new sentence mirrors the author's own existing transition formula
("With <X> ..., the following section examines <Y>" — cf. the end of §2.2.3 and
§2.6) so register stays consistent. No citations added; appending inside an
existing paragraph means NO heading/TOC/LoF/LoT change.

  T1 §2.3.2 -> §2.4 (datasets)
  T2 §2.4.2 -> §2.5 (agent architectures)
  T3 §2.5.2 -> §2.6 (evaluation)

Run:  uv run python scripts/thesis/apply_ch2_transitions.py --dry-run
      uv run python scripts/thesis/apply_ch2_transitions.py

Per docs/research/THESIS_EDITING.md. Idempotent; anchor-located; aborts without
saving on any failure.
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


_T1 = (
    " With the application of LLMs in marketing established, the following "
    "section examines the availability of suitable training data for fine-tuning "
    "such models."
)
_T2 = (
    " With dataset construction covered, the following section examines the "
    "agentic architectures that drive these models at inference."
)
_T3 = (
    " As these agentic systems introduce multiple reasoning steps and tool calls, "
    "the following section examines how the quality of their output can be "
    "evaluated."
)

# (label, anchor, old_tail, new_tail, done_marker)
EDITS = [
    (
        "T1 §2.3.2 -> §2.4",
        "Additional uncertainty emerges as conflicting results",
        "while Section 2.8 findings indicate AI disclosure reduces trust.",
        "while Section 2.8 findings indicate AI disclosure reduces trust." + _T1,
        "the availability of suitable training data for fine-tuning such models.",
    ),
    (
        "T2 §2.4.2 -> §2.5",
        "Our proposed system draws from these approaches and aims to deduce",
        "in place of unavailable conversion labels are detailed in §3.3.1.",
        "in place of unavailable conversion labels are detailed in §3.3.1." + _T2,
        "agentic architectures that drive these models at inference.",
    ),
    (
        "T3 §2.5.2 -> §2.6",
        "Surveys have also been conducted Wang et al. (2025)",
        "that allow users to design coordinate and deploy agents in different "
        "domain applications.",
        "that allow users to design coordinate and deploy agents in different "
        "domain applications." + _T3,
        "how the quality of their output can be evaluated.",
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
            print(f"  FAIL  {label}: old tail not found in paragraph")
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
