"""Read-only inspection of the thesis' List of Figures / List of Tables and the
figure/table caption paragraphs, so we can convert the lists to live TOC fields
without disturbing caption text or the manual chapter.figure numbering.

Prints:
  * the heading paragraphs (Table of Contents / List of Figures / List of Tables)
  * every paragraph in the LoF and LoT regions (current hardcoded entries)
  * every caption paragraph in the body (text + style_id + style name)
  * whether any field codes already exist in the LoF/LoT regions
"""
from __future__ import annotations

import re

from docx import Document
from docx.oxml.ns import qn

THESIS = "docs/research/THESIS.docx"

doc = Document(THESIS)
paras = doc.paragraphs


def style_of(p):
    sid = p.style.style_id if p.style else None
    sname = p.style.name if p.style else None
    return sid, sname


def has_field(p):
    el = p._element
    return bool(
        el.findall(".//" + qn("w:fldChar"))
        or el.findall(".//" + qn("w:instrText"))
        or el.findall(".//" + qn("w:fldSimple"))
    )


# --- locate the three front-matter headings -------------------------------
heading_re = re.compile(r"^(table of contents|list of figures|list of tables)$", re.I)
markers = []
for i, p in enumerate(paras):
    t = p.text.strip()
    if heading_re.match(t):
        sid, sname = style_of(p)
        markers.append((i, t, sid, sname))

print("=== FRONT-MATTER HEADINGS ===")
for i, t, sid, sname in markers:
    print(f"  [{i:3}] '{t}'  style_id={sid} name={sname!r}")

# region boundaries: LoF runs from its heading to LoT heading; LoT from its
# heading to the next clear section/heading.
def region(name):
    start = next((i for i, t, _, _ in markers if t.lower() == name), None)
    if start is None:
        return None, None
    # end = next marker after start, else +40 paras
    later = [i for i, *_ in markers if i > start]
    end = min(later) if later else start + 40
    return start, end


for region_name in ("list of figures", "list of tables"):
    s, e = region(region_name)
    print(f"\n=== REGION: {region_name}  paras [{s}..{e}) ===")
    if s is None:
        print("  NOT FOUND")
        continue
    for i in range(s, min(e, len(paras))):
        p = paras[i]
        sid, sname = style_of(p)
        fld = "  <FIELD>" if has_field(p) else ""
        txt = p.text.strip().replace("\n", " ")
        print(f"  [{i:3}] sid={sid} name={sname!r}{fld}  | {txt[:90]!r}")

# --- caption paragraphs in the body ---------------------------------------
cap_re = re.compile(r"^(figure|table)\s+\d", re.I)
print("\n=== CAPTION PARAGRAPHS (body) ===")
fig_styles, tbl_styles = {}, {}
for i, p in enumerate(paras):
    t = p.text.strip()
    if not cap_re.match(t):
        continue
    sid, sname = style_of(p)
    kind = t.split()[0].lower()
    bucket = fig_styles if kind == "figure" else tbl_styles
    bucket[sid] = bucket.get(sid, 0) + 1
    print(f"  [{i:3}] sid={sid} name={sname!r}  | {t[:80]!r}")

print("\n=== CAPTION STYLE TALLY ===")
print(f"  figure caption styles: {fig_styles}")
print(f"  table  caption styles: {tbl_styles}")
