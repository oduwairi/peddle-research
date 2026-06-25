"""Read-only inspection of the References section of THESIS.docx.

Dumps the heading paragraph(s), the style used for reference entries, and the
full list of entries (with paragraph index + style) so we can interleave the
7 new Phase-8 citations alphabetically and match formatting. Also greps the
whole doc for the 7 author surnames to detect any pre-existing entry.
"""

from __future__ import annotations

import docx

DOC = "docs/research/THESIS.docx"

SURNAMES = [
    "Papineni", "Popović", "Popovic", "Lin", "Banerjee", "Lavie",
    "Zhang", "Matias", "Efron", "Tibshirani",
]


def main() -> None:
    d = docx.Document(DOC)
    paras = d.paragraphs

    # locate REFERENCES heading
    ref_idx = None
    for i, p in enumerate(paras):
        t = p.text.strip().upper()
        if t in ("REFERENCES", "REFERENCE", "BIBLIOGRAPHY") or (
            t.startswith("REFERENCES") and len(t) < 20
        ):
            ref_idx = i
            print(f"[{i}] HEADING style={p.style.name!r}: {p.text.strip()!r}")
            break
    if ref_idx is None:
        raise SystemExit("REFERENCES heading not found")

    # find APPENDIX (end boundary)
    end_idx = len(paras)
    for i in range(ref_idx + 1, len(paras)):
        if paras[i].text.strip().upper().startswith("APPENDIX"):
            end_idx = i
            print(f"[{i}] APPENDIX boundary style={paras[i].style.name!r}: "
                  f"{paras[i].text.strip()!r}")
            break

    print(f"\n--- {end_idx - ref_idx - 1} paragraphs between REFERENCES and "
          f"APPENDIX ---\n")
    for i in range(ref_idx + 1, end_idx):
        p = paras[i]
        txt = p.text.strip()
        if not txt:
            continue
        pf = p.paragraph_format
        print(f"[{i}] style={p.style.name!r} indent={pf.first_line_indent} "
              f"hang={pf.left_indent} spacing={pf.line_spacing}")
        print(f"      {txt[:160]}")

    print("\n--- surname grep across whole doc ---")
    for s in SURNAMES:
        hits = [i for i, p in enumerate(paras) if s in p.text]
        if hits:
            for i in hits:
                print(f"  {s!r} @ [{i}]: {paras[i].text.strip()[:120]}")
        else:
            print(f"  {s!r}: (none)")


if __name__ == "__main__":
    main()
