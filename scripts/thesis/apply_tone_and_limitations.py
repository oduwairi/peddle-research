"""Reviewer round 2026-06-09 — issues #4 (tone) + #6 (limitations) for THESIS.docx.

Surgical, idempotent, anchor-located edits (no paragraph indices). Two groups:

  #4  Temper promotional language to a hedged academic register and scope
      single-arm claims to their arm, where the reviewer flagged overstatement:
        - Abstract: "beats ... by a considerable margin" -> arm-scoped.
        - §4.2.1: "most important takeaway" / "promising finding" -> hedged.
        - §4.2.2: append the missing learned-scorer-vs-MAUVE caveat.
        - §4.3.2: "most promising result" / "shows ... extensive" / "best
          configuration" -> hedged + arm-scoped.
        - §4.4.2: add the §4.4.3 validity caveat inline (was placed after).
        - §4.5.2: "most critical takeaway" -> "key takeaway".
        - §4.6.1: "most important fact" -> "Notably".
        - §6.1: make the RQ1 headline precise (it is true *for RQ1* across all
          three arms; the divergence is RQ2 only) instead of a blanket "yes
          across every evaluation arm".
  #6  Acknowledge the reviewer's live methodological gaps as limitations:
        - §4.6.3: n=94 paired ablation is modest; single seed (42); single RAG
          configuration.
        - §6.3: cross-reference the same (single agent/RAG config, one seed,
          94-brief set) alongside the existing hyperparameter/base-model lines.
        - §6.4: future-work line — larger/multi-seed ablation + RAG-strategy
          sweep.

Run:  uv run python scripts/thesis/apply_tone_and_limitations.py --dry-run
      uv run python scripts/thesis/apply_tone_and_limitations.py

Per docs/research/THESIS_EDITING.md. Idempotent (skips an edit whose result is
already present); aborts without saving if any edit fails to locate its text.
"""

from __future__ import annotations

import sys

from docx import Document

DOCX = "docs/research/THESIS.docx"


def replace_in_paragraph(p, old: str, new: str) -> bool:
    """Replace the first occurrence of `old` with `new` across a paragraph's
    runs, preserving run formatting. Robust to however Word split the visible
    text across <w:r> elements. (Same helper as apply_reviewer6_fixes.py.)"""
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
    """First body paragraph containing `anchor`, skipping TOC / LoF / LoT
    field-cache entries."""
    for p in paras:
        style = (p.style.name if p.style else "").lower()
        if any(s in style for s in _SKIP_STYLES):
            continue
        if anchor in p.text:
            return p
    return None


# (label, anchor, old, new, done_marker)
EDITS = [
    # ===================== #4 TONE ==========================================
    # ---- Abstract: scope the MAUVE win to its arm -------------------------
    (
        "#4 Abstract MAUVE margin",
        "With the rapid development of AI technology and domain-specialized agents",
        "Here again, our agent-wrapped fine-tuned model beats all other "
        "configurations by a considerable margin.",
        "Here again, our agent-wrapped fine-tuned model scores highest among "
        "the non-GOLD configurations on this arm.",
        "scores highest among the non-GOLD configurations on this arm",
    ),
    # ---- §4.2.1 hedge superlatives ----------------------------------------
    (
        "#4 §4.2.1 takeaway",
        "This section reports the findings of the first evaluation arm",
        "The most important takeaway is that our Draper model hits about 95.2%",
        "A key result is that our Draper model hits about 95.2%",
        "A key result is that our Draper model hits about 95.2%",
    ),
    (
        "#4 §4.2.1 promising",
        "This section reports the findings of the first evaluation arm",
        "This immediately reveals a promising finding that our fine-tuned model "
        "has picked up patterns",
        "This indicates that our fine-tuned model has picked up patterns",
        "This indicates that our fine-tuned model has picked up patterns",
    ),
    # ---- §4.2.2 add the missing arm-disagreement caveat -------------------
    (
        "#4 §4.2.2 caveat",
        "B_pipe and C_pipe are the agent-integrated versions of the flat models",
        "The best single-shot model is ahead of the corresponding agentic model "
        "(C_pipe = 0.607) by 0.044 composite.",
        "The best single-shot model is ahead of the corresponding agentic model "
        "(C_pipe = 0.607) by 0.044 composite. This per-ad decrease is specific to "
        "the learned-scorer arm; the MAUVE arm (§4.3) shows the opposite "
        "direction for these same configurations, and the two are reconciled in "
        "§4.6.",
        "This per-ad decrease is specific to the learned-scorer arm",
    ),
    # ---- §4.3.2 hedge + arm-scope -----------------------------------------
    (
        "#4 §4.3.2 promising",
        "The final score rankings are as follows",
        "The most promising result is that our fine-tuned model raises the bar",
        "A notable result is that our fine-tuned model raises the bar",
        "A notable result is that our fine-tuned model raises the bar",
    ),
    (
        "#4 §4.3.2 extensive",
        "The final score rankings are as follows",
        "alone, which shows our fine-tuning has extensive effect on the style",
        "alone, which suggests our fine-tuning has a substantial effect on the "
        "style",
        "which suggests our fine-tuning has a substantial effect on the style",
    ),
    (
        "#4 §4.3.2 best config",
        "The final score rankings are as follows",
        "The best configuration is our fine-tuned model and agent-wrapped "
        "workflow, reaching about 91% of the GOLD ceiling",
        "On this arm, the strongest configuration is our fine-tuned model with "
        "the agent-wrapped workflow, reaching about 91% of the GOLD ceiling",
        "On this arm, the strongest configuration is our fine-tuned model",
    ),
    # ---- §4.4.2 inline validity caveat ------------------------------------
    (
        "#4 §4.4.2 validity caveat",
        "ranking C > A > B on every metric",
        "The headline result of the calculated metrics shows that C leads all "
        "five gold-reference metrics.",
        "The main result of the calculated metrics shows that C leads all five "
        "gold-reference metrics, although §4.4.3 shows these overlap metrics "
        "have limited real-world validity.",
        "although §4.4.3 shows these overlap metrics have limited real-world "
        "validity",
    ),
    # ---- §4.5.2 hedge -----------------------------------------------------
    (
        "#4 §4.5.2 critical takeaway",
        "For paired contrast, we simply cannot just subtract cell scores",
        "the most critical takeaway is that the fine-tuning lifts up the score",
        "the key takeaway is that the fine-tuning lifts up the score",
        "the key takeaway is that the fine-tuning lifts up the score",
    ),
    # ---- §4.6.1 hedge -----------------------------------------------------
    (
        "#4 §4.6.1 important fact",
        "We recap our evaluation procedures",
        "The most important fact is that both evaluation arms position GOLD at "
        "the top",
        "Notably, both evaluation arms position GOLD at the top",
        "Notably, both evaluation arms position GOLD at the top",
    ),
    # ---- §6.1 precise RQ1 headline ----------------------------------------
    (
        "#4 §6.1 RQ1 precision",
        "The core research question this thesis has asked",
        "The presented results have shown that the answer is indeed yes across "
        "every evaluation arm we designed.",
        "The presented results indicate that the answer is yes for this domain "
        "comparison: the fine-tuned writer ranks above the frontier model on all "
        "three evaluation arms (the arms diverge only on the separate question "
        "of the agent and RAG effect, RQ2).",
        "the answer is yes for this domain comparison",
    ),
    # ===================== #6 LIMITATIONS ===================================
    # ---- §4.6.3 ablation size / seed / single config ----------------------
    (
        "#6 §4.6.3 ablation limits",
        "Having discussed the interpretation of the results",
        "However, this has great cost implications and is deferred beyond the "
        "scope of this thesis.",
        "However, this has great cost implications and is deferred beyond the "
        "scope of this thesis. A further limitation is that the 2×2 ablation "
        "rests on a modest paired sample (n=94 briefs that survive in all four "
        "cells), so its contrasts are best read as indicative rather than "
        "definitive; all reported numbers also come from a single training run "
        "at one random seed (seed=42), and the agent arm evaluates a single RAG "
        "configuration rather than a range of retrieval strategies.",
        "the 2×2 ablation rests on a modest paired sample",
    ),
    # ---- §6.3 cross-reference the same gaps -------------------------------
    (
        "#6 §6.3 ablation/seed/config",
        "In this thesis, several limitations have been encountered",
        "however significantly increasing the cost and time of developing the "
        "project.",
        "however significantly increasing the cost and time of developing the "
        "project. The empirical basis for the agent and fine-tuning effects is "
        "likewise limited to a single agent and RAG configuration, one random "
        "seed, and a 94-brief paired ablation set (§4.6.3), so these effect "
        "sizes should be treated as indicative.",
        "limited to a single agent and RAG configuration, one random seed",
    ),
    # ---- §6.4 future-work remedy ------------------------------------------
    (
        "#6 §6.4 future ablation",
        "This work opens the door for many opportunities",
        "since responses will be much larger and more complex to design.",
        "since responses will be much larger and more complex to design. A "
        "complementary direction is to strengthen the empirical basis itself: "
        "larger and multi-seed ablation runs, together with a comparison of "
        "alternative RAG strategies (for example different retrieval and "
        "re-ranking schemes), would test how robust the reported agent and "
        "fine-tuning effects are, since the present study evaluates a single "
        "configuration at one seed over a 94-brief paired set.",
        "larger and multi-seed ablation runs",
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
