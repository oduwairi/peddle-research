"""Scan the WHOLE doc for textual '4.4' / '4.5' / '4.6' references so the
renumber catches every cross-ref, not just the headings/captions/callouts I
already mapped. Read-only."""
from __future__ import annotations

import re

import docx

DOC = "docs/research/THESIS.docx"
PAT = re.compile(r"4\.[456](\.\d)?")


def main() -> None:
    d = docx.Document(DOC)
    for i, p in enumerate(d.paragraphs):
        t = p.text.strip()
        if PAT.search(t):
            kind = "HEAD" if p.style.name.startswith("Heading") else (
                "CAP" if t.startswith(("Figure", "Table")) else "body")
            # show the matched contexts
            hits = [m.group(0) for m in PAT.finditer(t)]
            print(f"[{i}] {kind:4s} {sorted(set(hits))} :: {t[:90]}")


if __name__ == "__main__":
    main()
