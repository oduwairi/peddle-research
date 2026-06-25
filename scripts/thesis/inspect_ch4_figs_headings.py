"""Dump Ch4 headings + figure captions with field detection, to plan the
§4.4 insertion + renumber. Read-only."""

from __future__ import annotations

import docx
from docx.oxml.ns import qn

DOC = "docs/research/THESIS.docx"


def field_kinds(p):
    """Return any SEQ/TC/REF field instrText found in the paragraph."""
    kinds = []
    for it in p._p.iter(qn("w:instrText")):
        t = (it.text or "").strip()
        if t:
            kinds.append(t)
    # fldSimple too
    for fs in p._p.iter(qn("w:fldSimple")):
        instr = fs.get(qn("w:instr"))
        if instr:
            kinds.append("fldSimple:" + instr.strip())
    return kinds


def main() -> None:
    d = docx.Document(DOC)
    paras = d.paragraphs

    # bound Ch4
    start = next(i for i, p in enumerate(paras)
                 if p.text.strip().upper() == "CHAPTER IV")
    end = len(paras)
    for i in range(start + 1, len(paras)):
        if paras[i].text.strip().upper() in ("CHAPTER V", "CHAPTER VI", "REFERENCES"):
            end = i
            break
    print(f"Ch4 paragraphs [{start}, {end})\n")

    for i in range(start, end):
        p = paras[i]
        t = p.text.strip()
        sn = p.style.name
        is_head = sn.startswith("Heading")
        is_cap = t.startswith("Figure") or t.startswith("Table")
        if is_head or is_cap:
            fk = field_kinds(p)
            tag = "HEAD" if is_head else "CAP "
            print(f"[{i}] {tag} {sn!r}: {t[:95]}")
            if fk:
                print(f"        FIELDS: {fk}")

    print("\n=== 'two arms' / arm-count sentences ===")
    for i in range(start, end):
        tl = paras[i].text.lower()
        if ("two evaluation arm" in tl or "two arms" in tl
                or "two main evaluation" in tl):
            print(f"[{i}] {paras[i].text.strip()[:200]}")

    print("\n=== in-text figure callouts (non-caption 'Figure 4.x') ===")
    for i in range(start, end):
        t = paras[i].text.strip()
        if "Figure 4" in t and not t.startswith("Figure"):
            # show snippet around the callout
            j = t.find("Figure 4")
            print(f"[{i}] …{t[max(0,j-40):j+30]}…")


if __name__ == "__main__":
    main()
