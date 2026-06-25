"""Validate the LoF/LoT PAGEREF targets against the bookmarks actually defined in
the document body. Any PAGEREF whose target bookmark is missing would render as
'Error! Bookmark not defined.' after a field refresh — we must catch those before
marking the fields dirty.
"""
from __future__ import annotations

import re

from docx import Document
from docx.oxml.ns import qn

THESIS = "docs/research/THESIS.docx"
doc = Document(THESIS)
paras = doc.paragraphs

PAGEREF_RE = re.compile(r"PAGEREF\s+(\S+)\s")


def pageref_targets(p):
    out = []
    for it in p._element.findall(".//" + qn("w:instrText")):
        m = PAGEREF_RE.search(it.text or "")
        if m:
            out.append(m.group(1))
    return out


# --- LoF/LoT regions -------------------------------------------------------
lof = list(range(294, 308))  # Figure entries
lot = list(range(309, 316))  # Table entries

referenced = []  # (para_idx, entry_text, target)
for i in lof + lot:
    p = paras[i]
    for tgt in pageref_targets(p):
        referenced.append((i, p.text.strip()[:55], tgt))

# --- all bookmarks defined in the doc body --------------------------------
defined = set()
body = doc.element.body
for bm in body.iter(qn("w:bookmarkStart")):
    name = bm.get(qn("w:name"))
    if name:
        defined.add(name)

print(f"bookmarks defined in body: {len(defined)}")
print(f"PAGEREF references in LoF/LoT: {len(referenced)}\n")

missing = []
for i, txt, tgt in referenced:
    ok = tgt in defined
    flag = "OK " if ok else "!! MISSING"
    if not ok:
        missing.append((i, txt, tgt))
    print(f"  [{flag}] [{i:3}] {tgt:12} | {txt!r}")

print(f"\n=== MISSING TARGETS: {len(missing)} ===")
for i, txt, tgt in missing:
    print(f"  [{i}] {tgt} | {txt!r}")

# --- which body bookmarks look like figure/table anchors are unused? ------
anchor_re = re.compile(r"^(Fig|Tbl)_", re.I)
fig_tbl_bm = sorted(b for b in defined if anchor_re.match(b))
used = {t for _, _, t in referenced}
print(f"\n=== Fig/Tbl bookmarks defined but NOT referenced by any list entry ===")
for b in fig_tbl_bm:
    if b not in used:
        print(f"  {b}")

# --- TC field text audit (catch copy-paste errors like Table 3.1) ---------
print("\n=== TC FIELD TEXT (body captions) — check for mismatched text ===")
TC_RE = re.compile(r'TC\s+"(.*?)"\s+\\f\s+(\w)', re.S)
for i, p in enumerate(paras):
    instr = "".join(
        (it.text or "") for it in p._element.findall(".//" + qn("w:instrText"))
    )
    m = TC_RE.search(instr)
    if m:
        text, ident = m.group(1), m.group(2)
        visible = p.text.strip()[:50]
        # flag if the visible caption's leading "Figure/Table X.Y" prefix
        # doesn't match the TC text's prefix
        vis_pref = visible.split(":")[0].split("—")[0].strip()
        tc_pref = text.split(":")[0].split("—")[0].strip()
        flag = "" if vis_pref and tc_pref.startswith(vis_pref) else "  <<< PREFIX MISMATCH"
        print(f"  [{i:3}] \\f {ident} | visible={vis_pref!r} tc={tc_pref!r}{flag}")
