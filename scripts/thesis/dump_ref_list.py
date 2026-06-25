"""List the body References entries (after the in-body 'References' heading,
before 'Appendix'), with paragraph index + first author token, so we can find
the exact alphabetical slots for Banerjee (B) and Popović (P)."""

from __future__ import annotations

import docx

DOC = "docs/research/THESIS.docx"


def main() -> None:
    d = docx.Document(DOC)
    paras = d.paragraphs

    # in-body References heading = a non-toc paragraph whose text == 'References'
    start = None
    for i, p in enumerate(paras):
        if p.text.strip() == "References" and p.style.name != "toc 2":
            start = i
            print(f"REFERENCES heading @ [{i}] style={p.style.name!r}")
            break
    if start is None:
        raise SystemExit("in-body References heading not found")

    for i in range(start + 1, len(paras)):
        p = paras[i]
        t = p.text.strip()
        if not t:
            continue
        if t.upper().startswith("APPENDIX") or p.style.name.startswith("Heading 1"):
            print(f"--- end @ [{i}] {p.style.name!r}: {t[:40]!r}")
            break
        print(f"[{i}] {t[:60]}")


if __name__ == "__main__":
    main()
