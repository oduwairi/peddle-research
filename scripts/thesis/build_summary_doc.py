"""Build the summary (Phase 1) thesis by CLONING THESIS.docx and DELETING content.

Every kept paragraph/table/heading keeps its ORIGINAL formatting (cover VML, heading
styles, Body Text style, section breaks). The only mutations are: (1) trimming the
Abstract body to ~240 words, (2) stripping dangling (Figure/Table/Algorithm) callouts
that point at dropped artifacts (removes tokens only — never rewords), (3) replacing
the academic bibliography with the advisor-required GitHub + dataset links.

Output: docs/research/THESIS_SUMMARY.docx
"""

import copy
import re
import shutil
from docx import Document
from docx.oxml.ns import qn

SRC = "docs/research/THESIS.docx"
OUT = "docs/research/THESIS_SUMMARY.docx"

shutil.copyfile(SRC, OUT)
doc = Document(OUT)
body = doc.element.body
blocks = list(body.iterchildren())


def ptext(el):
    if el.tag != qn("w:p"):
        return ""
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def has_sectpr(el):
    return el.tag == qn("w:p") and el.find(qn("w:pPr") + "/" + qn("w:sectPr")) is not None


def is_tbl(el):
    return el.tag == qn("w:tbl")


def style(el):
    pPr = el.find(qn("w:pPr"))
    if pPr is None:
        return ""
    s = pPr.find(qn("w:pStyle"))
    return s.get(qn("w:val")) if s is not None else ""


# ---- locate key block indices by content (no hardcoded indices) ----
def find_idx(pred, start=0):
    for i in range(start, len(blocks)):
        if pred(blocks[i]):
            return i
    return -1


approval_idx = find_idx(lambda e: ptext(e).strip() == "Approval")          # first dropped front-matter
abstract_idx = find_idx(lambda e: ptext(e).strip() == "Abstract", approval_idx)
ch1_idx = find_idx(lambda e: ptext(e).strip() == "CHAPTER I")
ch2_idx = find_idx(lambda e: ptext(e).strip() == "CHAPTER II")
refs_idx = find_idx(lambda e: ptext(e).strip() == "References", ch1_idx)

# Introduction ends at the "Thesis Structure" subsection (a full-thesis roadmap that does
# not belong in a summary). Keep Chapter I only up to that heading.
ts_idx = find_idx(lambda e: ptext(e).strip() == "Thesis Structure", ch1_idx)
ch1_end = ts_idx if 0 < ts_idx < ch2_idx else ch2_idx

# the section break paragraph immediately before CHAPTER I (switches body to arabic page nums)
sect_break_idx = max(i for i in range(len(blocks)) if has_sectpr(blocks[i]) and i < ch1_idx)

# Abstract block run: heading + title/degree/pages/body/keywords (through the Keywords line)
kw_idx = find_idx(lambda e: ptext(e).strip().startswith("Keywords:"), abstract_idx)

CHAP_TITLES = {f"CHAPTER {n}" for n in ["II", "III", "IV", "V", "VI"]}
CHAP_NAMES = {"Literature Review", "Methodology", "Results", "Discussion",
              "Conclusion and Recommendations"}

# data tables kept as real grids (caption prefix + the following <w:tbl> [+ a "Note." para]).
# Their (Table x.x) cross-refs are preserved; all other table refs are stripped as dangling.
WANT_TABLES = ("Table 4.1:", "Table 5.1:")
KEPT_TABLE_IDS = ("4.1", "5.1")

ANCHORS = [
    # Chapter II  (opener 416 "This review is organized..." dropped per advisor review;
    # open instead with the substantive small-models finding that sets up the "However")
    "The main question whether a smaller fine-tuned model in the range of 7B-13B",
    "However, existing models have focused on domains such as medicine",
    "The literature demonstrates that RAG + fine-tuning architectures",
    "The Upworthy Research Archive, published by Matias",
    "Forward-synthesis methods covered so far",
    "With all works reviewed in this survey across different domains",
    "This survey has examined the research landscape underpinning",
    # Chapter III
    "The proposed system is mainly developed over three main phases",
    "In order to produce a fine-tuned model capable of generating actual high-performing",
    "These three signals are then combined into a single composite score",
    "Following data collection, the final AdFlex corpus obtained consists of 55,000",
    "The setup phase includes shrinking the base model weights into a four-bit",
    "When the orchestrator and writer model are set, the agent workflow runs as a freeform",
    "For a full-scope evaluation of the model performance",
    # Chapter IV  (Results + Limitations — headline story; dense 717 ablation dropped)
    "This section reports the findings of the first evaluation arm",
    "The final score rankings are as follows: GOLD at 0.462",
    "Having discussed the interpretation of the results, it is important to outline",
    # Chapter V  (Discussion insights; 748/753 + Table 5.1 form the Comparative Analysis section)
    "Without revisiting the evaluation methodology in detail",
    "Two base configurations can be wrapped by the agent workflow",
    "The main premise of this thesis is to test whether a small",
    "The reviewed works are split into two main categories",
    # Chapter VI
    "The core research question this thesis has asked is whether a specialized",
    "This work opens the door for many opportunities for future research",
]


def is_anchor(txt):
    return any(txt.startswith(a) for a in ANCHORS)


# ---- build the keep-set (element identities) ----
keep = set()

# cover: everything up to (not incl) Approval
for i in range(0, approval_idx):
    keep.add(id(blocks[i]))

# abstract heading + its lines
for i in range(abstract_idx, kw_idx + 1):
    keep.add(id(blocks[i]))

# body section break
keep.add(id(blocks[sect_break_idx]))

# Chapter I as the Introduction (title -> just before "Thesis Structure").
# These are sections, not chapters: drop the "CHAPTER I" label (style 1002) and keep only the
# section-name heading ("Introduction", style 1003). Continuous prose: also drop the inner
# subheadings (Background, Problem Statement, Research Questions, Contributions).
seen_chapter_name = False
for i in range(ch1_idx, ch1_end):
    el = blocks[i]
    st = style(el)
    if st == "1002":
        continue                  # drop the "CHAPTER I" label
    if st == "1003":
        if not seen_chapter_name:
            seen_chapter_name = True
            keep.add(id(el))      # keep the section-name heading ("Introduction")
        continue                  # drop every inner subheading
    if not seen_chapter_name and not ptext(el).strip():
        continue                  # drop blank paragraph(s) before the section heading
    keep.add(id(el))

# Chapters II..VI up to References
i = ch2_idx
while i < refs_idx:
    el = blocks[i]
    txt = ptext(el).strip()
    if txt in CHAP_TITLES:
        i += 1
        continue                  # drop the "CHAPTER II..VI" labels (these are sections)
    if txt in CHAP_NAMES:
        keep.add(id(el))
    elif el.tag == qn("w:p") and is_anchor(txt):
        keep.add(id(el))
    elif any(txt.startswith(p) for p in WANT_TABLES):
        # keep the caption, the following table, and a following "Note." paragraph
        keep.add(id(el))
        j = i + 1
        while j < refs_idx and (is_tbl(blocks[j]) or ptext(blocks[j]).strip().startswith("Note.")
                                or ptext(blocks[j]).strip() == ""):
            if is_tbl(blocks[j]) or ptext(blocks[j]).strip().startswith("Note."):
                keep.add(id(blocks[j]))
            if ptext(blocks[j]).strip().startswith("Note."):
                break
            if not is_tbl(blocks[j]) and ptext(blocks[j]).strip() == "":
                j += 1
                continue
            j += 1
    i += 1

# References heading (entries replaced below)
keep.add(id(blocks[refs_idx]))

# ---- delete everything not kept (never the body-level sectPr) ----
for el in blocks:
    if el.tag == qn("w:sectPr"):
        continue  # body-level section properties — always keep
    if id(el) not in keep:
        el.getparent().remove(el)


# ---- (1) trim Abstract body to ~240 verbatim words ----
ab_el = blocks[kw_idx - 1]  # the body paragraph (right before Keywords)
full_ab = ptext(ab_el).strip()
sents = re.split(r"(?<=[.])\s+", full_ab)
abstract = " ".join(sents[k] for k in [0, 1, 2, 3, 4, 7, 13, 14, 18])
runs = ab_el.findall(qn("w:r"))
if runs:
    # keep first run's rPr, set its text to the trimmed abstract, clear the rest
    for t in runs[0].findall(qn("w:t")):
        runs[0].remove(t)
    new_t = runs[0].makeelement(qn("w:t"), {qn("xml:space"): "preserve"})
    new_t.text = abstract
    runs[0].append(new_t)
    for r in runs[1:]:
        ab_el.remove(r)


# ---- (1b) update abstract page count (full thesis = 113 pages; summary is shorter) ----
SUMMARY_PAGES = 16  # estimate (no renderer here); confirm against Word's status bar
for el in body.iterchildren():
    if el.tag == qn("w:p") and "113 pages" in ptext(el):
        for t in el.iter(qn("w:t")):
            if t.text and "113" in t.text:
                t.text = t.text.replace("113", str(SUMMARY_PAGES))


# ---- (2) strip dangling cross-references in kept body paragraphs ----
# Token-only deletion (never rewords): drops pointers to figures/tables/sections that the
# summary removes, plus whole forward-transition sentences ("the following section examines
# ..."). Protected: refs to KEPT tables (4.1, 5.1) and real citations (Author, 2024).
_tbl_keep = "|".join(t.replace(".", r"\.") for t in KEPT_TABLE_IDS)   # e.g. "4\.1|5\.1"
# parenthetical pointers: (Section 2.4), (Figure 3.1), (Algorithm B.1), (see Section 2.7.2)
PARENS_REF = re.compile(r"\s*\((?:see\s+)?(?:Sections?|Figures?|Fig\.?|Algorithms?)[^)]*\)")
# parenthetical table pointers, but keep refs to the tables that are in the summary
PARENS_TBL = re.compile(rf"\s*\((?:see\s+)?Tables?(?!\s*(?:{_tbl_keep}))[^)]*\)")
# §-style section pointers: (§3.2–§3.4), (§3.5)
PARENS_SEC = re.compile(r"\s*\(§[^)]*\)")
# refs embedded inside a citation paren: "(Liu, 2025; discussed further in Section 2.3)"
EMBED_SEC = re.compile(r";\s*(?:discussed|see|described)[^;)]*?\bSections?\s*[0-9.]+")
EMBED_PSEC = re.compile(r";\s*[^;)]*?§[0-9][^)]*(?=\))")
# whole sentences that are figure/table pointers ("Figure 3.6 shows ...", "Tabulates the values
# plotted in Figure 4.1."). The (?<!\() guard leaves parenthetical refs to the paren patterns
# above, so a content sentence ending in "(Table 4.1)" is NOT removed wholesale; refs to kept
# tables (4.1, 5.1) are also preserved. The id (e.g. "4.1") has a dot, matched explicitly.
FIG_SENT = re.compile(r"(?:(?<=\.)\s+)?[^.]*?(?<!\()\bFigure\s*[0-9][0-9.]*[^.]*\.")
TBL_SENT = re.compile(rf"(?:(?<=\.)\s+)?[^.]*?(?<!\()\bTables?\s*(?!(?:{_tbl_keep}))[0-9][0-9.]*[^.]*\.")
# whole forward-transition sentences ("the following section examines ...")
TRANSITION = re.compile(
    r"(?:(?<=\.)\s+)?[^.]*?"
    r"(?:following section|next section|following chapter|"
    r"remainder of this chapter|rest of this chapter)[^.]*\.",
    re.IGNORECASE)
STRIP_PATTERNS = (PARENS_REF, PARENS_TBL, PARENS_SEC, EMBED_SEC, EMBED_PSEC,
                  FIG_SENT, TBL_SENT, TRANSITION)


def strip_callouts(p):
    ts = [t for r in p.findall(qn("w:r")) for t in r.findall(qn("w:t"))]
    if not ts:
        return
    full = "".join(t.text or "" for t in ts)
    if not any(rgx.search(full) for rgx in STRIP_PATTERNS):
        return
    delete = [False] * len(full)
    for rgx in STRIP_PATTERNS:
        for m in rgx.finditer(full):
            for k in range(*m.span()):
                delete[k] = True
    idx = 0
    for t in ts:
        s = t.text or ""
        n = len(s)
        t.text = "".join(ch for k, ch in zip(range(idx, idx + n), s) if not delete[k])
        idx += n


for el in body.iterchildren():
    txt = ptext(el).strip()
    if el.tag == qn("w:p") and txt:
        strip_callouts(el)   # safe on table captions: kept-table ids are protected in the regexes


# ---- (2b) move the Limitations paragraph to close the section (after the last discussion para) ----
def find_para(prefix):
    for el in body.iterchildren():
        if el.tag == qn("w:p") and ptext(el).strip().startswith(prefix):
            return el
    return None


lim_el = find_para("Having discussed the interpretation of the results")
disc_el = find_para("Two base configurations can be wrapped by the agent workflow")
if lim_el is not None and disc_el is not None:
    body.remove(lim_el)
    disc_el.addnext(lim_el)   # Results -> Discussion -> Limitations


# ---- (2c) split out the Comparative Analysis section (advisor structure #7) ----
# The trailing Discussion material (the "main premise" recap + Table 5.1 + the two-category
# positioning of this work against prior related work) is its own deliverable in the advisor's
# outline. Give it its own section heading, cloned from an existing section-name heading so the
# formatting is byte-identical, and insert it just before the "main premise" paragraph (748).
def find_heading(text):
    for el in body.iterchildren():
        if el.tag == qn("w:p") and style(el) == "1003" and ptext(el).strip() == text:
            return el
    return None


def set_heading_text(el, text):
    """Retext a heading paragraph in place, preserving the first run's formatting."""
    runs = el.findall(qn("w:r"))
    if not runs:
        return
    first = runs[0]
    for t in first.findall(qn("w:t")):
        first.remove(t)
    t = first.makeelement(qn("w:t"), {qn("xml:space"): "preserve"})
    t.text = text
    first.append(t)
    for r in runs[1:]:
        el.remove(r)


disc_heading = find_heading("Discussion")
compare_el = find_para("The main premise of this thesis is to test whether a small")
if disc_heading is not None and compare_el is not None:
    ca_heading = copy.deepcopy(disc_heading)
    set_heading_text(ca_heading, "Comparative Analysis of Related Work")
    compare_el.addprevious(ca_heading)


# ---- (2d) rename the final section heading to match the advisor's outline (#8) ----
# The summary keeps only 6.1 Conclusion + 6.4 Future Work (Contributions dropped; Limitations
# live in the Discussion section), so "Conclusion and Future Work" is the accurate label.
concl = find_heading("Conclusion and Recommendations")
if concl is not None:
    set_heading_text(concl, "Conclusion and Future Work")


# ---- (3) replace bibliography with advisor-required GitHub + dataset links ----
refs_el = None
for el in body.iterchildren():
    if el.tag == qn("w:p") and ptext(el).strip() == "References":
        refs_el = el
        break

# template Body-Text paragraph cloned from the (trimmed) abstract body paragraph
template = copy.deepcopy(ab_el)
# strip down template runs to a single clean run we can retext
_tr = template.findall(qn("w:r"))
for r in _tr[1:]:
    template.remove(r)


def _set(pPr, tag, attrs):
    """Get-or-create a direct child of pPr and set its attributes."""
    el = pPr.find(qn(tag))
    if el is None:
        el = pPr.makeelement(qn(tag), {})
        pPr.append(el)
    for k, v in attrs.items():
        if v is None:
            if qn(k) in el.attrib:
                del el.attrib[qn(k)]
        else:
            el.set(qn(k), v)
    return el


def make_para(text, sub=False):
    """A reference-list paragraph: left-aligned, single-spaced, hanging indent.
    The abstract template is justified + double-spaced + first-line indent, which stretches the
    long URLs; override those so the list reads cleanly. sub=True nests dataset entries deeper."""
    p = copy.deepcopy(template)
    pPr = p.find(qn("w:pPr"))
    _set(pPr, "w:jc", {"w:val": "left"})                          # no URL-stretching justification
    _set(pPr, "w:spacing", {"w:line": "240", "w:lineRule": "auto", "w:after": "120"})  # single + small gap
    _set(pPr, "w:ind", {"w:firstLine": None,                      # drop the 0.5" first-line indent
                        "w:left": "1080" if sub else "360",
                        "w:hanging": "360"})                       # wrapped URLs align under the text
    r = p.findall(qn("w:r"))[0]
    for t in r.findall(qn("w:t")):
        r.remove(t)
    t = r.makeelement(qn("w:t"), {qn("xml:space"): "preserve"})
    t.text = text
    r.append(t)
    return p


# (text, is_dataset_sub_entry) — sub entries are nested under "2. Datasets:" with a dash bullet
ref_lines = [
    ("1. Project repository (GitHub): https://github.com/oduwairi/peddle", False),
    ("2. Datasets:", False),
    ("– AdFlex (primary source — multi-platform ad-intelligence API; ~55,000 ads across "
     "Facebook, TikTok, X, Pinterest, and Reddit): https://adflex.io  (API docs: https://doc.adflex.io)", True),
    ("– Upworthy Research Archive (Matias et al., 2021, Nature Scientific Data; "
     "DOI 10.17605/OSF.IO/JD64P): https://osf.io/jd64p/", True),
    ("– Internet Research Agency (IRA) Facebook Ads dataset: https://github.com/umd-mith/irads  "
     "(source release: https://democrats-intelligence.house.gov/social-media-content/"
     "social-media-advertisements.htm)", True),
    ("– Meta Ad Library: https://www.facebook.com/ads/library/", True),
    ("– Google Ads Transparency Center & Political Ads: https://adstransparency.google.com/  "
     "(political ads: https://transparencyreport.google.com/political-ads/)", True),
    ("– TikTok Ad Library: https://library.tiktok.com/ads", True),
    ("– BigSpy: https://bigspy.com/", True),
]
anchor = refs_el
for line, sub in ref_lines:
    p = make_para(line, sub=sub)
    anchor.addnext(p)
    anchor = p

doc.save(OUT)
print("Saved", OUT)
print("abstract words:", len(abstract.split()))
