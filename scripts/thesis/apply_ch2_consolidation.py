"""Reviewer round 2026-06-09 — issue #3 (repetition) safe consolidations in Ch II.

Conservative, deletion-led consolidations only: collapse consecutive sentences
that state the SAME finding twice, preserving every citation and any unique
caveat. Broader thematic consolidation (the "fast-changing domain" / "small
models beat large" refrains spread across subsections) is left to the author's
Chapter II prose pass — blunt deletion there would drop citations or create the
abrupt transitions the reviewer also flags.

Edits:
  #3a §2.3.1 — Reisenbichler (2025) "match or exceed human-written ads in CTR
      and conversion" is stated in two consecutive sentences; collapse to one,
      keeping the finding AND the "limited to short-form search ads" caveat.
  #3b §2.1.3 — the Belcak (2025) and Aralimatti (2025) sentences are back-to-back
      restatements of the same claim (small fine-tuned models are competitive /
      more suitable in domain-specific agentic settings); collapse to one
      sentence, keeping BOTH citations and every distinct clause.
  #3c §2.5.2 — the citation "(Han et al., 2024; Xi et al., 2023)" appears twice in
      one sentence (the first occurrence also missing its open paren); drop the
      malformed duplicate, keep the well-formed trailing citation. Pure dedup.

Run:  uv run python scripts/thesis/apply_ch2_consolidation.py --dry-run
      uv run python scripts/thesis/apply_ch2_consolidation.py

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


# (label, anchor, old, new, done_marker)
EDITS = [
    (
        "#3a §2.3.1 Reisenbichler dedup",
        "Reisenbichler et al. (2022) provided early evidence",
        "can match or exceed human-written ads in click-through and conversion "
        "rates. These results demonstrate that LLM-based generation in marketing "
        "is empirically validated, with Reisenbichler et al. (2025) reporting "
        "that LLM-generated ads matched or exceeded human-written ads in "
        "click-through and conversion rates despite being limited in scope to "
        "short-form search ads on specific platforms.",
        "can match or exceed human-written ads in click-through and conversion "
        "rates, though this evidence is limited in scope to short-form search "
        "ads on specific platforms.",
        "though this evidence is limited in scope to short-form search ads",
    ),
    (
        "#3b §2.1.3 Belcak/Aralimatti dedup",
        "The main question whether a smaller fine-tuned model",
        "Following these works, a lot of research has maintained that small large "
        "language models used in domain-specific agentic applications are not only "
        "appropriate but very often more suitable and economically viable than "
        "general-purpose models (Belcak et al., 2025). Consequently, recent studies "
        "further demonstrate that small fine-tuned models achieve competitive or "
        "superior performance to larger models in domain-specific agentic settings "
        "(Aralimatti et al., 2025).",
        "Following these works, a lot of research has maintained that small large "
        "language models used in domain-specific agentic applications are not only "
        "appropriate but very often more suitable and economically viable than "
        "general-purpose models, achieving competitive or superior performance to "
        "larger models (Belcak et al., 2025; Aralimatti et al., 2025).",
        "achieving competitive or superior performance to "
        "larger models (Belcak et al., 2025; Aralimatti et al., 2025)",
    ),
    (
        "#3c §2.5.2 duplicate citation",
        "Surveys have also been conducted Wang et al. (2025) and Xi et al. (2023)",
        "have also received wide attention Han et al., 2024; Xi et al., 2023) "
        "demonstrating that multi-agent collaboration can outperform single-agent "
        "systems on complex tasks (Han et al., 2024; Xi et al., 2023).",
        "have also received wide attention, demonstrating that multi-agent "
        "collaboration can outperform single-agent systems on complex tasks "
        "(Han et al., 2024; Xi et al., 2023).",
        "received wide attention, demonstrating that multi-agent",
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
