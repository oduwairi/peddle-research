"""Dump the image-paragraph XML (so the new figures match placement) + figure
PNG pixel dims + per-config reference_scores row counts. Read-only."""
from __future__ import annotations

from pathlib import Path

import docx
import polars as pl
from lxml import etree
from PIL import Image

DOC = "docs/research/THESIS.docx"


def main() -> None:
    d = docx.Document(DOC)
    paras = d.paragraphs
    for i in (657, 659):  # image paragraphs preceding captions 4.3.1 / 4.3.2
        p = paras[i]
        has_img = p._p.find(f".//{{{p._p.nsmap['w']}}}drawing") is not None
        print(f"\n===== [{i}] style={p.style.name!r} has_drawing={has_img} align={p.alignment} =====")
        # print pPr only (drawing XML is huge)
        ppr = p._p.find(f"{{{p._p.nsmap['w']}}}pPr")
        if ppr is not None:
            print(etree.tostring(ppr, pretty_print=True).decode())

    print("\n=== figure PNG pixel dims ===")
    for f in ("fig-4-4-1-overlap-by-config.png", "fig-4-4-2-upworthy-grounding.png"):
        path = Path("docs/research/figures") / f
        if path.exists():
            w, h = Image.open(path).size
            print(f"  {f}: {w}x{h}px  aspect={w/h:.2f}")
        else:
            print(f"  {f}: MISSING")

    print("\n=== per-config reference_scores row counts ===")
    for c in ("A", "B", "C", "GOLD"):
        df = pl.read_parquet(f"data/eval/reference_scores/{c}.parquet")
        print(f"  {c}: n={len(df)}  cols={[x for x in df.columns if 'gold' in x][:6]}")


if __name__ == "__main__":
    main()
