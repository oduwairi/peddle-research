r"""Make the List of Figures / List of Tables refresh automatically and accurately.

The lists are hand-built paragraphs (style 934) of the form
    <short title>  <tab>  { PAGEREF <bookmark> \h }<cached "1">
pointing at bookmarks placed at each body caption. The page numbers are frozen at
"1" because the PAGEREF fields were never refreshed, and several targets/bookmarks
are broken. This repairs the cross-reference graph and arms it for auto-refresh:

  1. body Table 3.2 [590]: bookmark mislabelled Tbl_3_1 (dup of Table 3.1) -> Tbl_3_2 (fresh id)
  2. body Figure 2.7 [467]: add the missing Fig_2_7 bookmark
  3. LoT entry [315] (Table 3.2): PAGEREF Tbl_3_1 -> Tbl_3_2
  4. LoT entry [314] (Table 3.1): plain-text "1" -> real PAGEREF Tbl_3_1 field
  5. TC-field text fixes at [550]/[590] (latent copy-paste bugs; inert today)
  6. mark every LoF/LoT PAGEREF begin dirty + set <w:updateFields/> in settings.xml

No paragraph is added or removed, so paragraph indices stay stable. Entry text and
the manual chapter.figure numbering are left byte-identical.
"""
from __future__ import annotations

import os
import sys

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

THESIS = "docs/research/THESIS.docx"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

if os.path.exists("docs/research/.~lock.THESIS.docx#"):
    sys.exit("ABORT: editor lock present.")

doc = Document(THESIS)
paras = doc.paragraphs


# ---- helpers --------------------------------------------------------------
def max_bookmark_id() -> int:
    mx = 0
    for bm in doc.element.body.iter(qn("w:bookmarkStart")):
        try:
            mx = max(mx, int(bm.get(qn("w:id"))))
        except (TypeError, ValueError):
            pass
    return mx


_next = max_bookmark_id()


def fresh_id() -> str:
    global _next
    _next += 1
    return str(_next)


def make_pageref_runs(target: str, cached: str = "1"):
    runs = []
    rb = OxmlElement("w:r")
    fb = OxmlElement("w:fldChar")
    fb.set(qn("w:fldCharType"), "begin")
    fb.set(qn("w:dirty"), "1")
    rb.append(fb)
    runs.append(rb)

    ri = OxmlElement("w:r")
    it = OxmlElement("w:instrText")
    it.set(XML_SPACE, "preserve")
    it.text = f" PAGEREF {target} \\h "
    ri.append(it)
    runs.append(ri)

    rs = OxmlElement("w:r")
    fs = OxmlElement("w:fldChar")
    fs.set(qn("w:fldCharType"), "separate")
    rs.append(fs)
    runs.append(rs)

    rr = OxmlElement("w:r")
    tt = OxmlElement("w:t")
    tt.set(XML_SPACE, "preserve")
    tt.text = cached
    rr.append(tt)
    runs.append(rr)

    re_ = OxmlElement("w:r")
    fe = OxmlElement("w:fldChar")
    fe.set(qn("w:fldCharType"), "end")
    re_.append(fe)
    runs.append(re_)
    return runs


def set_instr(p, find_sub: str, new_text: str) -> bool:
    for it in p._element.findall(".//" + qn("w:instrText")):
        if find_sub in (it.text or ""):
            it.text = new_text
            return True
    return False


# ---- 1. body Table 3.2 [590]: relabel bookmark Tbl_3_1 -> Tbl_3_2 ---------
p590 = paras[590]._element
old_id = None
for bs in p590.findall(qn("w:bookmarkStart")):
    if bs.get(qn("w:name")) == "Tbl_3_1":
        old_id = bs.get(qn("w:id"))
        nid = fresh_id()
        bs.set(qn("w:name"), "Tbl_3_2")
        bs.set(qn("w:id"), nid)
        for be in p590.findall(qn("w:bookmarkEnd")):
            if be.get(qn("w:id")) == old_id:
                be.set(qn("w:id"), nid)
        print(f"[590] bookmark Tbl_3_1(id={old_id}) -> Tbl_3_2(id={nid})")
        break
else:
    print("[590] WARN: Tbl_3_1 bookmarkStart not found")

# ---- 5. TC text fixes (inert but correct) ---------------------------------
if set_instr(paras[550], 'TC "Table 3.1: Comparison',
             ' TC "Table 3.1: Fine-tuning configuration." \\f T \\l 1 '):
    print('[550] TC text -> "Table 3.1: Fine-tuning configuration."')
if set_instr(paras[590], 'TC "Table 3.1: Comparison',
             ' TC "Table 3.2: Comparison of the two evaluation methods." \\f T \\l 1 '):
    print('[590] TC text -> "Table 3.2: Comparison of the two evaluation methods."')

# ---- 2. body Figure 2.7 [467]: add missing Fig_2_7 bookmark ---------------
p467 = paras[467]._element
have = any(bs.get(qn("w:name")) == "Fig_2_7" for bs in p467.findall(qn("w:bookmarkStart")))
if not have:
    nid = fresh_id()
    bs = OxmlElement("w:bookmarkStart")
    bs.set(qn("w:id"), nid)
    bs.set(qn("w:name"), "Fig_2_7")
    be = OxmlElement("w:bookmarkEnd")
    be.set(qn("w:id"), nid)
    pPr = p467.find(qn("w:pPr"))
    pPr.addnext(be)   # insert after pPr ...
    pPr.addnext(bs)   # ... then bs before be  => order: pPr, bs, be, runs
    print(f"[467] added bookmark Fig_2_7 (id={nid})")
else:
    print("[467] Fig_2_7 already present")

# ---- 3. LoT entry [315] (Table 3.2): retarget PAGEREF ---------------------
if set_instr(paras[315], "PAGEREF Tbl_3_1", " PAGEREF Tbl_3_2 \\h "):
    print("[315] PAGEREF Tbl_3_1 -> Tbl_3_2")

# ---- 4. LoT entry [314] (Table 3.1): plain "1" -> PAGEREF field -----------
p314 = paras[314]._element
plain = None
for r in p314.findall(qn("w:r")):
    t = r.find(qn("w:t"))
    if t is not None and (t.text or "").strip() == "1" and r.find(qn("w:fldChar")) is None:
        plain = r
        break
if plain is not None:
    for nr in make_pageref_runs("Tbl_3_1"):
        plain.addprevious(nr)
    plain.getparent().remove(plain)
    print("[314] plain '1' -> PAGEREF Tbl_3_1 field")
else:
    print("[314] WARN: plain '1' run not found")

# ---- 6a. mark every LoF/LoT PAGEREF begin dirty ---------------------------
dirtied = 0
for i in list(range(294, 308)) + list(range(309, 316)):
    for fb in paras[i]._element.findall(".//" + qn("w:fldChar")):
        if fb.get(qn("w:fldCharType")) == "begin":
            fb.set(qn("w:dirty"), "1")
            dirtied += 1
print(f"marked {dirtied} PAGEREF begins dirty")

# ---- 6b. settings.xml: updateFields on open -------------------------------
settings = doc.settings.element
if settings.find(qn("w:updateFields")) is None:
    uf = OxmlElement("w:updateFields")
    uf.set(qn("w:val"), "true")
    after = {
        "compat", "rsids", "clrSchemeMapping", "decimalSymbol", "listSeparator",
        "themeFontLang", "characterSpacingControl", "defaultTabStop", "shapeDefaults",
    }
    anchor = next(
        (c for c in settings if c.tag.split("}")[-1] in after), None
    )
    if anchor is not None:
        anchor.addprevious(uf)
    else:
        settings.append(uf)
    print("settings.xml: <w:updateFields w:val='true'/> added")
else:
    print("settings.xml: updateFields already present")

doc.save(THESIS)
print("\nSAVED.")
