"""Compare the figures/tables that actually appear in the body against what the
List of Figures / List of Tables enumerate. Distinguishes a likely *caption*
(line begins with 'Figure N.M'/'Table N.M') from an in-text *reference*
('... Figure N.M shows ...')."""
from __future__ import annotations

import re
from collections import defaultdict

from docx import Document

THESIS = "docs/research/THESIS.docx"
doc = Document(THESIS)
paras = doc.paragraphs

NUM = re.compile(r"\b(Figure|Table)\s+(\d+(?:\.\d+)+)", re.I)
CAPTION_START = re.compile(r"^(Figure|Table)\s+(\d+(?:\.\d+)+)\s*[:—\-]", re.I)

# list regions (entries are pStyle 934)
LOF = range(294, 308)
LOT = range(309, 316)
listed_fig, listed_tbl = set(), set()
for i in LOF:
    m = NUM.search(paras[i].text)
    if m:
        listed_fig.add(m.group(2))
for i in LOT:
    m = NUM.search(paras[i].text)
    if m:
        listed_tbl.add(m.group(2))

# body scan (exclude the list + TOC regions 160..316)
captions = defaultdict(list)   # (kind,num) -> [para idx] where it starts a caption
mentions = defaultdict(list)   # (kind,num) -> [para idx] any mention
for i, p in enumerate(paras):
    if 160 <= i <= 316:
        continue
    t = p.text.strip()
    for m in NUM.finditer(t):
        kind, num = m.group(1).capitalize(), m.group(2)
        mentions[(kind, num)].append(i)
    cm = CAPTION_START.match(t)
    if cm:
        captions[(cm.group(1).capitalize(), cm.group(2))].append(i)

def report(kind, listed):
    print(f"\n=== {kind.upper()}S ===")
    nums = sorted({n for (k, n) in mentions if k == kind},
                  key=lambda s: [int(x) for x in s.split(".")])
    for n in nums:
        cap = captions.get((kind, n), [])
        ment = mentions.get((kind, n), [])
        in_list = "LISTED" if n in listed else "**NOT LISTED**"
        cap_txt = ""
        if cap:
            cap_txt = " | caption@%d: %r" % (cap[0], paras[cap[0]].text.strip()[:60])
        print(f"  {kind} {n:8} {in_list:14} caption_paras={cap or '—'} mentions={ment}{cap_txt}")
    # listed but no body caption?
    orphan = sorted(listed - {n for (k, n) in mentions if k == kind})
    if orphan:
        print(f"  listed but NOT found in body: {orphan}")

report("Figure", listed_fig)
report("Table", listed_tbl)
print(f"\nLoF lists {len(listed_fig)} figures: {sorted(listed_fig)}")
print(f"LoT lists {len(listed_tbl)} tables:  {sorted(listed_tbl)}")
