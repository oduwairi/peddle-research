"""Dump full text of the in-text Efron citations and the existing reference
entries that overlap the 7 Phase-8 cites, so we can (a) confirm the 4 present
entries match our in-text forms and (b) decide how to reconcile Efron.
"""

from __future__ import annotations

import docx

DOC = "docs/research/THESIS.docx"


def main() -> None:
    d = docx.Document(DOC)
    paras = d.paragraphs

    print("=== IN-TEXT paragraphs containing 'Efron' ===")
    for i, p in enumerate(paras):
        if "Efron" in p.text and p.style.name == "Body Text":
            print(f"\n[{i}] ({p.style.name}):\n{p.text}")

    print("\n\n=== EXISTING REFERENCE ENTRIES (full text) ===")
    needles = [
        "Papineni, K.",
        "Lin, C.",
        "Zhang, T., Kishore",
        "Matias, J. N.",
        "Efron, B. (1979)",
        "Banerjee",   # METEOR — expect none
        "Popovi",     # chrF — expect none
    ]
    for n in needles:
        hit = next((p for p in paras if n in p.text), None)
        if hit is None:
            print(f"\n[{n!r}] -> NOT PRESENT")
        else:
            idx = paras.index(hit)
            print(f"\n[{n!r}] @ [{idx}] style={hit.style.name!r}:\n{hit.text}")


if __name__ == "__main__":
    main()
