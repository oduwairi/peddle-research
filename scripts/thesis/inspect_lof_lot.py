"""Inspect List of Tables / List of Figures in both the reference template and
THESIS.docx so we can match the reference's exact structure/format.

Read-only. Dumps, for each doc, every paragraph in the LoT/LoF region with:
  index, style name, alignment, leading/trailing-space flags, tab-stop leaders,
  whether the paragraph carries a TOC/field code, and the text.

Run: uv run python scripts/thesis/inspect_lof_lot.py
"""

from __future__ import annotations

from docx import Document
from docx.oxml.ns import qn

REF = "docs/research/SURE NEU THESIS FORMAT abduallah inal PDF 12.docx"
OURS = "docs/research/THESIS.docx"


def field_kind(p) -> str:
    """Return a short tag describing any field machinery in the paragraph."""
    tags = []
    for r in p._p.iter():
        if r.tag == qn("w:fldChar"):
            tags.append("fldChar:" + (r.get(qn("w:fldCharType")) or "?"))
        elif r.tag == qn("w:instrText"):
            tags.append("instr:" + (r.text or "").strip()[:40])
        elif r.tag == qn("w:fldSimple"):
            tags.append("fldSimple:" + (r.get(qn("w:instr")) or "")[:40])
    return " | ".join(tags)


def tabstops(p) -> str:
    pPr = p._p.find(qn("w:pPr"))
    if pPr is None:
        return ""
    tabs = pPr.find(qn("w:tabs"))
    if tabs is None:
        return ""
    out = []
    for t in tabs.findall(qn("w:tab")):
        out.append(
            f"{t.get(qn('w:val'))}@{t.get(qn('w:pos'))}"
            f"/{t.get(qn('w:leader')) or 'none'}"
        )
    return ", ".join(out)


def dump(path: str, label: str) -> None:
    doc = Document(path)
    paras = doc.paragraphs
    print("\n" + "=" * 78)
    print(f"{label}: {path}")
    print("=" * 78)

    # Find LoT / LoF heading anchors (case-insensitive contains).
    regions = []
    for i, p in enumerate(paras):
        t = p.text.strip().lower()
        if t in ("list of tables", "list of figures", "list of abbreviations",
                 "table of contents"):
            regions.append((i, p.text.strip()))
    print("Anchors found:", regions)

    if not regions:
        return

    start = min(i for i, _ in regions)
    # End: a bit past the last anchor or first CHAPTER heading after it.
    end = len(paras)
    last_anchor = max(i for i, _ in regions)
    for i in range(last_anchor + 1, len(paras)):
        if paras[i].text.strip().upper().startswith("CHAPTER"):
            end = i + 1
            break
        if i - last_anchor > 60:
            end = i
            break

    for i in range(max(0, start - 1), min(end, len(paras))):
        p = paras[i]
        style = p.style.name if p.style else "?"
        align = str(p.alignment) if p.alignment is not None else "None"
        txt = p.text
        lead = "LEAD_SP" if txt[:1] == " " else ""
        fk = field_kind(p)
        ts = tabstops(p)
        flags = " ".join(x for x in (lead, fk, ("TABS[" + ts + "]") if ts else "") if x)
        shown = txt if len(txt) <= 90 else txt[:87] + "..."
        print(f"[{i:4}] ({style:>16} | {align:>14}) {flags}")
        print(f"       {shown!r}")


if __name__ == "__main__":
    dump(REF, "REFERENCE TEMPLATE")
    dump(OURS, "OURS (THESIS.docx)")
