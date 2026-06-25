"""C.3 — make the ablation/sensitivity design explicit in sec 3.8.2.

Two targeted, run-level edits to the existing sec 3.8.2 body paragraph (the
comparison design is already described; this just names the core ablation and
adds the per-platform sensitivity point). No paragraph is created, so the
paragraph's existing formatting is preserved. Idempotent.
"""

from __future__ import annotations

import docx

DOC = "docs/research/THESIS.docx"

EDIT_A = (
    "to see the effect of fine-tuning on the base model.",
    "to see the effect of fine-tuning on the base model; this controlled "
    "comparison, holding the base architecture fixed so that only the trained "
    "weights differ, is the study's core ablation.",
)
EDIT_B = (
    "the other models are compared to.",
    "the other models are compared to. Each arm is additionally reported per "
    "platform, as a sensitivity check on whether an effect holds across "
    "channels or concentrates in a few.",
)


def replace_in_run(p, old, new):
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    raise SystemExit(f"anchor not found in a single run: {old!r}")


def main() -> None:
    d = docx.Document(DOC)
    p = next(
        (p for p in d.paragraphs
         if "held-out test split of the construction dataset serves as" in p.text),
        None,
    )
    if p is None:
        raise SystemExit("§3.8.2 body paragraph not found")

    did = []
    if "core ablation" not in p.text:
        replace_in_run(p, *EDIT_A)
        did.append("A: named the core ablation + control")
    if "sensitivity check" not in p.text:
        replace_in_run(p, *EDIT_B)
        did.append("B: added per-platform sensitivity sentence")

    if not did:
        print("already applied; nothing to do")
        return

    d.save(DOC)
    for x in did:
        print("EDIT", x)


if __name__ == "__main__":
    main()
