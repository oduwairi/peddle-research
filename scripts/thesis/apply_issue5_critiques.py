"""Reviewer round 2026-06-09 — issue #5 (weak critical analysis) for THESIS.docx.

Inserts the author's voiced critique paragraphs (Phase-5 polished: voice-to-text
and typo fixes only — verbs, voice, and clause order preserved per the
AI-detector discipline) into the four subsections that were purely descriptive:

  §2.2.2 Live Web vs Static RAG     -> appended to the subsection body
  §2.5.1 Single-Agent Frameworks    -> appended to the subsection body
  §2.5.2 Multi-Agent Systems        -> inserted BEFORE the existing transition
  §2.7.1 Instruction-Tuning         -> appended to the subsection body

§2.2.1 already carries a gap line; §2.6 and §2.8 are already critical (the
reviewer named them but they are among the strongest) — left for the optional
one-line touches handled separately. No new citations (all named works already
in the bibliography). Appending inside existing paragraphs -> no heading/TOC
change.

Run:  uv run python scripts/thesis/apply_issue5_critiques.py --dry-run
      uv run python scripts/thesis/apply_issue5_critiques.py

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


_C222 = (
    " FreshLLMs show that live web search helps mitigate out-of-date information "
    "in QA. This is especially important for ad copy since market trends are "
    "constantly evolving and competitors are constantly changing; a static "
    "database will be out of sync quickly in this scenario. This provides the "
    "basis on which the thesis aims to use live web search RAG as part of the "
    "live market grounding needed for good campaign generation."
)
_C251 = (
    " Single-agent frameworks such as ReAct, Toolformer, WebGPT, or HuggingGPT "
    "are all general-purpose models for general agentic capabilities. The key gap "
    "for this thesis is a marketing-specialized agent tailored around "
    "ad-generation workflows, one that also uses an orchestrator and writer "
    "architecture to maximize efficiency."
)
_C252 = (
    "Multi-agent architectures are proficient at complex multi-step reasoning and "
    "multiple-responsibility workflows. However, these are costly, have high "
    "latency, and are overkill for a single-deliverable task such as delivering "
    "an ad campaign. This thesis balances this approach with an efficient agent "
    "design focused on marketing workflows such as research, competitor analysis, "
    "and drafting campaigns."
)
_C271 = (
    " FLAN, T0, and FLAN-T5 frameworks are used to improve zero-shot "
    "generalization with diverse tasks. However, these frameworks require the "
    "brief to already exist, which is not the case for our collected ad corpus. "
    "This is exactly the point at which the thesis pivots to using "
    "backtranslation (Section 2.7.3)."
)

# (label, anchor, old, new, done_marker)
EDITS = [
    (
        "§2.2.2 critique",
        "Vu et al. (2024) introduced FreshLLMs",
        "degrading LLM performance by pulling its responses back to a stale "
        "outdated dataset.",
        "degrading LLM performance by pulling its responses back to a stale "
        "outdated dataset." + _C222,
        "the live market grounding needed for good campaign generation.",
    ),
    (
        "§2.5.1 critique",
        "Yao et al. (2023) introduced ReAct",
        "models that are trained or fine-tuned for specific agentic tool-calling "
        "tasks.",
        "models that are trained or fine-tuned for specific agentic tool-calling "
        "tasks." + _C251,
        "tailored around ad-generation workflows",
    ),
    (
        "§2.5.2 critique (before transition)",
        "Surveys have also been conducted Wang et al. (2025)",
        "different domain applications. As these agentic systems introduce",
        "different domain applications. " + _C252 + " As these agentic systems "
        "introduce",
        "overkill for a single-deliverable task such as delivering an ad campaign",
    ),
    (
        "§2.7.1 critique",
        "A foundational paper by Wei et al. (2022b)",
        "Longpre et al. (2023) surveyed the FLAN methodology providing explicit "
        "examples and guidelines for effective multitask tuning.",
        "Longpre et al. (2023) surveyed the FLAN methodology providing explicit "
        "examples and guidelines for effective multitask tuning." + _C271,
        "the thesis pivots to using backtranslation",
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
