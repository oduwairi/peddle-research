"""Upgrade the three deployment-screenshot callouts from bare parentheticals to
small pointer sentences (reviewer #10 follow-up: the screenshots need an explicit
in-text mention, not just "(Figure X)").

Replaces, in place:
  §3.5.5 "...the base model (Figure 3.4)."  -> "...the base model. Figure 3.4 shows ..."
  §3.6.3 "...freeform loop (Figure 3.6)."   -> "...freeform loop. Figure 3.6 shows ..."
  §3.6.5 "...campaign card) (Figure 3.7)."  -> "...campaign card). Figure 3.7 shows ..."

The agent-architecture *diagram* (Fig 3.5) keeps its existing parenthetical — it
is not a screenshot. Idempotent (skips a paragraph that already has its
sentence). Anchored by visible text, never by index.

Run:
    uv run python scripts/thesis/add_deployment_figure_sentences.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from docx import Document

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
QT = f"{{{W}}}t"

DOC = Path("docs/research/THESIS.docx")
LOCK = Path("docs/research/.~lock.THESIS.docx#")
BACKUP = Path("docs/research/THESIS.docx.pre-deploy-sentences.bak")

# (anchor prefix, old substring, new substring, idempotency sentinel)
EDITS = [
    (
        "After training completes, the LoRA adapter stays separate",
        " (Figure 3.4).",
        ". Figure 3.4 shows this serverless endpoint serving live inference requests.",
        "Figure 3.4 shows",
    ),
    (
        "When the orchestrator and writer model are set",
        " (Figure 3.6).",
        ". Figure 3.6 shows a logged execution trace of one such run.",
        "Figure 3.6 shows",
    ),
    (
        "Communication between the orchestrator and the writer",
        " (Figure 3.7).",
        ". Figure 3.7 shows one such emitted campaign in the deployed interface.",
        "Figure 3.7 shows",
    ),
]


def visible_texts(p_elem):
    return p_elem.findall(f".//{QT}")


def find_para(doc, prefix):
    for p in doc.paragraphs:
        if p.text.strip().startswith(prefix):
            return p
    return None


def main() -> None:
    if LOCK.exists():
        sys.exit(f"REFUSING: {LOCK} exists — close OnlyOffice/Word first.")
    if not BACKUP.exists():
        shutil.copy2(DOC, BACKUP)
        print(f"backup -> {BACKUP}")

    doc = Document(str(DOC))
    for prefix, old, new, sentinel in EDITS:
        p = find_para(doc, prefix)
        if p is None:
            raise RuntimeError(f"anchor not found: {prefix!r}")
        full = "".join(t.text or "" for t in visible_texts(p._element))
        if sentinel in full:
            print(f"skip (already done): {sentinel!r}")
            continue
        if old not in full:
            raise RuntimeError(f"target {old!r} not found in {prefix!r}")
        done = False
        for t in visible_texts(p._element):
            if t.text and old in t.text:
                t.text = t.text.replace(old, new, 1)
                done = True
                break
        if not done:
            raise RuntimeError(f"target spans runs (unexpected): {old!r}")
        print(f"added sentence: {sentinel!r}")
    doc.save(str(DOC))
    print("DONE")


if __name__ == "__main__":
    main()
