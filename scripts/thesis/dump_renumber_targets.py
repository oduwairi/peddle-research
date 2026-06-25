"""Confirm renumber targets are safe: existing figure display width (EMU),
and whether 'Figure 4.4.x' tokens sit intact in single runs. Read-only."""
from __future__ import annotations

import docx
from docx.oxml.ns import qn

DOC = "docs/research/THESIS.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def wq(t: str) -> str:
    return f"{{{W}}}{t}"


def main() -> None:
    d = docx.Document(DOC)
    paras = d.paragraphs

    # --- existing figure display width from [659] drawing extent ---
    img = paras[659]._p
    for ext in img.iter(qn("wp:extent")):
        print(f"[659] wp:extent cx={ext.get('cx')} cy={ext.get('cy')}")
    for ext in img.iter(qn("a:ext")):
        print(f"[659] a:ext     cx={ext.get('cx')} cy={ext.get('cy')}")

    print("\n=== ablation callout/caption run breakdown ===")
    for i in (663, 665, 671, 673):
        p = paras[i]
        ts = [t.text for t in p._p.iter(wq("t")) if t.text]
        instr = [it.text for it in p._p.iter(wq("instrText")) if it.text]
        bms = [bs.get(wq("name")) for bs in p._p.iter(wq("bookmarkStart"))]
        print(f"\n[{i}] {p.text.strip()[:60]!r}")
        for j, t in enumerate(ts):
            mark = "  <-- token" if ("Figure 4.4" in t) else ""
            print(f"   t[{j}]: {t[:70]!r}{mark}")
        for j, t in enumerate(instr):
            mark = "  <-- TC token" if ("Figure 4.4" in t) else ""
            print(f"   instr[{j}]: {t[:70]!r}{mark}")
        if bms:
            print(f"   bookmarks: {bms}")


if __name__ == "__main__":
    main()
