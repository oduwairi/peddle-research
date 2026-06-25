"""Verify the Phase-8 citation edits landed on disk: the two new entries sit in
the right alphabetical slot with formatting matching their neighbours, and the
C.1 bootstrap cite now reads (Efron, 1979)."""

from __future__ import annotations

import docx

DOC = "docs/research/THESIS.docx"


def fmt(p):
    pf = p.paragraph_format
    fr = p.runs[0].font if p.runs else None
    return (f"style={p.style.name!r} left={pf.left_indent} "
            f"first={pf.first_line_indent} spacing={pf.line_spacing} "
            f"font={(fr.name, fr.size) if fr else None}")


def main() -> None:
    d = docx.Document(DOC)
    paras = d.paragraphs

    print("=== alphabetical context around the two inserts ===")
    for prefix in ("Bai, J.", "Banerjee, S.", "Belcak, P.",
                   "Pillutla, K.", "Popović, M.", "Qin, Y."):
        p = next((q for q in paras if q.text.strip().startswith(prefix)), None)
        idx = paras.index(p) if p is not None else None
        marker = "  <-- NEW" if prefix in ("Banerjee, S.", "Popović, M.") else ""
        print(f"[{idx}] {prefix}{marker}")

    print("\n=== new entries (full text + formatting vs neighbour) ===")
    for prefix, neighbour in (("Banerjee, S.", "Belcak, P."),
                              ("Popović, M.", "Qin, Y.")):
        new = next(q for q in paras if q.text.strip().startswith(prefix))
        nb = next(q for q in paras if q.text.strip().startswith(neighbour))
        print(f"\n{prefix}\n  {new.text}")
        print(f"  NEW : {fmt(new)}")
        print(f"  NBR : {fmt(nb)}")
        print(f"  MATCH: {fmt(new) == fmt(nb)}")

    print("\n=== Efron alignment in C.1 ===")
    c1 = next(p for p in paras
              if "we need more than one signal to cross-reference" in p.text)
    tail = c1.text[c1.text.rfind("bootstrap"):]
    print(f"  …{tail}")
    print(f"  has (Efron, 1979): {'Efron, 1979' in c1.text}")
    print(f"  has 1993         : {'1993' in c1.text}")

    # alphabetical sanity: collect the run of entries and confirm sorted-ish
    print("\n=== local order check ===")
    band = [q.text.strip()[:30] for q in paras
            if q.text.strip().startswith(("Bai", "Banerjee", "Belcak"))]
    print("  B-band:", band)
    band2 = [q.text.strip()[:30] for q in paras
             if q.text.strip().startswith(("Pillutla", "Popović", "Qin"))]
    print("  P-band:", band2)


if __name__ == "__main__":
    main()
