"""Read-only inspection #2 for the Ch. III methodology edit.

Resolves: table caption above/below convention, what follows APPENDIX,
available styles, exact List-of-Tables text, and full text of 3.4.3 / 3.4.5.
"""

from __future__ import annotations

import docx
from docx.oxml.ns import qn

DOC = "docs/research/THESIS.docx"


def main() -> None:
    d = docx.Document(DOC)
    paras = d.paragraphs
    body = d.element.body

    print("=== BODY ELEMENT ORDER around the eval table (find tbl vs caption) ===")
    # Walk top-level body children; print tag + (text or table marker) near the end.
    children = list(body)
    # locate caption 'Table 3.1' paragraph element and the table element indices
    for idx, ch in enumerate(children):
        tag = ch.tag.split('}')[-1]
        if tag == 'p':
            txt = ''.join(t.text or '' for t in ch.findall('.//' + qn('w:t')))
            if 'Table 3.1' in txt or 'Comparison of the two evaluation' in txt or txt.strip() in ('CHAPTER IV', 'Results'):
                print(f"child[{idx}] <p> :: {txt[:80]}")
        elif tag == 'tbl':
            # first cell text
            first = ch.find('.//' + qn('w:t'))
            print(f"child[{idx}] <TBL> first-t={first.text if first is not None else None!r}")

    print("\n=== PARAGRAPH STYLES AVAILABLE ===")
    for s in d.styles:
        try:
            print(f"  {s.type} :: {s.name}")
        except Exception:  # noqa: BLE001
            pass

    print("\n=== LIST OF TABLES region (paras 313..320) ===")
    for i in range(312, 321):
        if i < len(paras):
            print(f"{i} | {paras[i].style.name} | {paras[i].text!r}")

    print("\n=== APPENDIX region (last paras from APPENDIX heading) ===")
    ap = None
    for i, p in enumerate(paras):
        if p.text.strip() == 'APPENDIX':
            ap = i
            break
    if ap is not None:
        for i in range(ap, min(ap + 25, len(paras))):
            print(f"{i} | {paras[i].style.name} | {paras[i].text[:160]!r}")
    print(f"(total paragraphs in doc: {len(paras)})")

    print("\n=== FULL TEXT: 3.4.3 body (535) and 3.4.5 body (539) ===")
    for i in (535, 539, 546):
        if i < len(paras):
            print(f"\n--- para {i} ({paras[i].style.name}) ---")
            print(paras[i].text)

    print("\n=== References heading + first/last entries (678..690) ===")
    for i in range(676, 692):
        if i < len(paras):
            print(f"{i} | {paras[i].style.name} | {paras[i].text[:120]!r}")


if __name__ == "__main__":
    main()
