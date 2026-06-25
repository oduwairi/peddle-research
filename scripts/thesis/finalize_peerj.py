"""Finalize literature-review-peerj.docx for PeerJ submission.

Idempotent. Run repeatedly without breakage.

What it does (matches the approved plan
~/.claude/plans/ok-before-submission-lets-sequential-puddle.md):

A1. Strip 7 PeerJ template guidance callout paragraphs ("remove this box
    before submitting!"). The callouts live as top-level <w:p> elements
    whose drawings contain text-boxes — they don't appear in
    `doc.paragraphs` text but are still in the XML.

A2. Insert 5 required declaration sections between Acknowledgements and
    References: Funding, Competing Interests, Author Contributions,
    Data Availability, AI Use Declaration. Each = bold heading + body
    paragraph, modelled on the existing "Conclusions" / "Acknowledgements"
    paragraph templates (same `normal0` style, just bold vs non-bold runs).

A3. Fix the Chung et al. 2022 first reference: truncate the 16-author list
    to 3 authors + "et al." and surface the year right after the names.

B1. Normalize a small set of inconsistent in-text citation patterns that
    spell author initials before truncating ("Liu, H., Tahmasbi et al.")
    or use ampersand-with-3-authors ("Bhat, Browne & Bingemann, 2025") —
    rewrite to consistent "(First et al., Year)" Harvard form.

B2. Reconcile "+90 references" in Survey Methodology with the actual 89.

C2. Replace the Acknowledgements placeholder with a supervisor thank-you.

D1. Set core.xml metadata (creator, lastModifiedBy, title).
"""
from __future__ import annotations

import re
import sys
from copy import deepcopy
from pathlib import Path

from docx import Document

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
QN_P = f"{{{W}}}p"
QN_T = f"{{{W}}}t"
QN_R = f"{{{W}}}r"


PEERJ_DOCX = Path("docs/research/literature-review-peerj.docx")


# ---------- A2. Declaration content ---------------------------------------

DECLARATIONS: list[tuple[str, str]] = [
    (
        "Funding",
        "The author received no specific funding for this work.",
    ),
    (
        "Competing Interests",
        "The author declares there are no competing interests.",
    ),
    (
        "Author Contributions",
        "Osama Duwairi conceived and designed the review, performed the "
        "literature search and screening, analyzed and synthesized the "
        "literature, and drafted and revised the manuscript.",
    ),
    (
        "Data Availability",
        "This is a literature review of previously published works; no new "
        "primary data were generated. The search strategy, inclusion and "
        "exclusion criteria, and full reference list are reported in the "
        "manuscript. The representative prompts used during the AI-assisted "
        "literature search are provided in Supplementary Material S1 "
        "(`literature-review-peerj.supp.S1.md`).",
    ),
    (
        "AI Use Declaration",
        "The literature search was conducted with the assistance of Claude "
        "Opus 4.6 (Anthropic, 2025) in deep-research mode; representative "
        "prompts are reproduced in Supplementary Material S1. Every "
        "candidate reference returned by the AI-assisted search was "
        "independently retrieved, read, and verified by the author before "
        "inclusion. Portions of the manuscript prose were drafted, "
        "paraphrased, or copy-edited with the assistance of large language "
        "models (Claude Opus 4.6 and Claude Opus 4.7). All AI-generated "
        "text was reviewed, verified for factual accuracy, and revised by "
        "the author, who takes full responsibility for the content, the "
        "accuracy of the cited literature, and the conclusions drawn. No "
        "AI tool is listed as an author; no part of the reference list "
        "was generated without human verification of each entry.",
    ),
]


# C2. New Acknowledgements text.
NEW_ACK_TEXT = (
    "The author thanks Prof. Dr. Fadi Al-Turjman (Department of Artificial "
    "Intelligence Engineering, Near East University) for supervisory "
    "guidance and constructive feedback throughout this work."
)


# ---------- helpers --------------------------------------------------------


def get_top_paragraphs(body):
    return list(body.iterchildren(QN_P))


def para_text(p) -> str:
    return "".join((t.text or "") for t in p.findall(f".//{QN_T}"))


def set_para_text(p, new_text: str) -> None:
    """Replace the paragraph's textual content while preserving the first
    run's properties (rPr). Drops extra runs and tables/drawings inside the
    paragraph."""
    runs = p.findall(f".//{QN_R}")
    if not runs:
        raise RuntimeError("paragraph has no <w:r> to write into")
    # Keep the first run; remove the rest.
    first = runs[0]
    for r in runs[1:]:
        r.getparent().remove(r)
    # Wipe all <w:t> children of the first run, leaving one with new_text.
    ts = first.findall(f".//{QN_T}")
    if not ts:
        raise RuntimeError("first run has no <w:t> to write into")
    ts[0].text = new_text
    for t in ts[1:]:
        t.getparent().remove(t)


# ---------- A1: strip guidance callouts -----------------------------------


GUIDANCE_NEEDLE = "remove this box before submitting"


def strip_guidance_callouts(body) -> int:
    """Remove every top-level paragraph whose descendants include the
    'remove this box before submitting' marker. Idempotent: returns 0 if
    none remain."""
    removed = 0
    for p in list(body.iterchildren(QN_P)):
        text = para_text(p).lower()
        if GUIDANCE_NEEDLE in text:
            body.remove(p)
            removed += 1
    return removed


# ---------- A2: insert declarations ---------------------------------------


def find_para_by_text(body, predicate):
    for p in body.iterchildren(QN_P):
        if predicate(para_text(p)):
            return p
    return None


def insert_declarations(body) -> int:
    """Insert the 5 declarations between Acknowledgements (and its body) and
    References. Idempotent: skips any heading already present."""

    refs_heading = find_para_by_text(body, lambda t: t.strip() == "References")
    if refs_heading is None:
        raise RuntimeError("References heading not found")

    # Templates for heading and body — clone from Conclusions / its first body para.
    conc_heading = find_para_by_text(body, lambda t: t.strip() == "Conclusions")
    if conc_heading is None:
        raise RuntimeError("Conclusions heading not found")
    # The body paragraph immediately after Conclusions:
    body_tpl = None
    seen_conc = False
    for p in body.iterchildren(QN_P):
        if p is conc_heading:
            seen_conc = True
            continue
        if seen_conc and para_text(p).strip():
            body_tpl = p
            break
    if body_tpl is None:
        raise RuntimeError("could not find a body template paragraph after Conclusions")

    heading_tpl = conc_heading

    existing = {para_text(p).strip() for p in body.iterchildren(QN_P)}
    inserted = 0

    # Insert just before References. addprevious() inserts in-order.
    for title, text in DECLARATIONS:
        if title in existing:
            continue
        h = deepcopy(heading_tpl)
        set_para_text(h, title)
        b = deepcopy(body_tpl)
        set_para_text(b, text)
        refs_heading.addprevious(h)
        refs_heading.addprevious(b)
        inserted += 2

    return inserted


# ---------- A3: fix Chung first reference ---------------------------------


CHUNG_NEW = (
    "Chung, H. W., Hou, L., Longpre, S., et al. (2022). Scaling instruction"
    "‑finetuned language models. Journal of Machine Learning Research, "
    "25(202), 1–53. https://jmlr.org/papers/v25/23-0870.html"
)


def fix_chung_reference(body) -> bool:
    """Rewrite the Chung 2022 entry to truncated 3-author + et al. form."""
    for p in body.iterchildren(QN_P):
        text = para_text(p)
        if text.startswith("Chung, H. W."):
            if "et al." in text and text.count(",") < 10:
                return False  # already fixed
            set_para_text(p, CHUNG_NEW)
            return True
    return False


# ---------- B1: normalize stray citation patterns -------------------------


# Cases observed by the audit. Each = (regex, replacement).
#
# Strategy:
#  1. Surname-then-initial-then-another-surname-then-et-al — drop the
#     initial and the second-surname clutter. Covers (Liu, H., Tahmasbi
#     et al., 2025) and narrative "Liu, H., Tahmasbi et al. (2025)".
#  2. Three-author ampersand form "(A, B & C, YYYY)" → "(A et al., YYYY)".
#     Preserves true 2-author "(A & B, YYYY)" — only the 3-comma-and-amp
#     shape is rewritten.
CITATION_NORMALIZATIONS: list[tuple[str, str]] = [
    # 1a. Parenthetical: "(Surname, X., Other et al., YYYY)" → "(Surname et al., YYYY)"
    (
        r"\(([A-Z][A-Za-z\-]+),\s+[A-Z]\.,\s+[A-Z][A-Za-z\-]+\s+et al\.,\s+(\d{4}[a-z]?)\)",
        r"(\1 et al., \2)",
    ),
    # 1b. Narrative: "Surname, X., Other et al. (YYYY)" → "Surname et al. (YYYY)"
    (
        r"\b([A-Z][A-Za-z\-]+),\s+[A-Z]\.,\s+[A-Z][A-Za-z\-]+\s+et al\.\s+\((\d{4}[a-z]?)\)",
        r"\1 et al. (\2)",
    ),
    # 2. Three-or-more-author ampersand form, "Surname1, Surname2(, SurnameN)+ & Last, YYYY"
    #    → "Surname1 et al., YYYY". Matches anywhere (handles paired cites
    #    inside one (... ; ...) as well). Leaves true 2-author "A & B, YYYY"
    #    alone (no leading comma-separated middle authors).
    (
        r"\b([A-Z][A-Za-z\-]+)(?:,\s+[A-Z][A-Za-z\-]+){1,4}\s+&\s+[A-Z][A-Za-z\-]+,\s+(\d{4}[a-z]?)",
        r"\1 et al., \2",
    ),
]


def normalize_citations(body) -> int:
    """Sweep top-level paragraphs; apply citation regex normalizations.
    Returns count of paragraphs touched."""
    touched = 0
    for p in body.iterchildren(QN_P):
        ts = p.findall(f".//{QN_T}")
        if not ts:
            continue
        # Reassemble text across runs, apply, then split back.
        # Because each citation we care about lives inside a single <w:t>,
        # we can apply per-<w:t> safely.
        for t in ts:
            if not t.text:
                continue
            new = t.text
            for pat, rep in CITATION_NORMALIZATIONS:
                new = re.sub(pat, rep, new)
            if new != t.text:
                t.text = new
                touched += 1
    return touched


# ---------- B2: reconcile +90 → 89 ref count -------------------------------


def reconcile_ref_count(body) -> bool:
    """In Survey Methodology, change '+90 references' to '89 references' to
    match the actual bibliography (89 entries)."""
    for p in body.iterchildren(QN_P):
        for t in p.findall(f".//{QN_T}"):
            if t.text and "+90 references" in t.text:
                t.text = t.text.replace("+90 references", "89 references")
                return True
            if t.text and "+90  references" in t.text:  # double-space variant
                t.text = t.text.replace("+90  references", "89 references")
                return True
    return False


# ---------- C2: rewrite Acknowledgements ----------------------------------


def rewrite_acknowledgements(body) -> bool:
    """Replace the 'The author has not declared any acknowledgements.'
    placeholder with the supervisor thank-you."""
    ack_heading = find_para_by_text(body, lambda t: t.strip() == "Acknowledgements")
    if ack_heading is None:
        return False
    # Find the next non-empty paragraph after Acknowledgements.
    seen = False
    for p in body.iterchildren(QN_P):
        if p is ack_heading:
            seen = True
            continue
        if not seen:
            continue
        text = para_text(p).strip()
        if not text:
            continue
        # Stop at the next heading.
        if text in {"Funding", "References"} or text in {h for h, _ in DECLARATIONS}:
            return False
        if text == NEW_ACK_TEXT:
            return False  # already done
        set_para_text(p, NEW_ACK_TEXT)
        return True
    return False


# ---------- D1: core metadata --------------------------------------------


def update_core_metadata(doc) -> None:
    cp = doc.core_properties
    cp.author = "Osama Duwairi"
    cp.last_modified_by = "Osama Duwairi"
    cp.title = (
        "Domain Specialized Marketing Agent for High Performance Campaign "
        "Generation: Fine-Tuning and RAG-Based Approach"
    )


# ---------- driver --------------------------------------------------------


def main() -> int:
    if not PEERJ_DOCX.exists():
        print(f"ERROR: {PEERJ_DOCX} not found", file=sys.stderr)
        return 1

    doc = Document(str(PEERJ_DOCX))
    body = doc.element.body

    print(f"=== Finalizing {PEERJ_DOCX} ===")

    n_removed = strip_guidance_callouts(body)
    print(f"A1. Removed {n_removed} guidance callout paragraph(s).")

    n_inserted = insert_declarations(body)
    print(f"A2. Inserted {n_inserted // 2} declaration section(s) (heading + body).")

    chung_fixed = fix_chung_reference(body)
    print(f"A3. Chung first-reference fix applied: {chung_fixed}")

    n_normalized = normalize_citations(body)
    print(f"B1. Normalized {n_normalized} citation token(s).")

    refs_fixed = reconcile_ref_count(body)
    print(f"B2. Reconciled +90→89 in Survey Methodology: {refs_fixed}")

    ack_rewritten = rewrite_acknowledgements(body)
    print(f"C2. Rewrote Acknowledgements: {ack_rewritten}")

    update_core_metadata(doc)
    print("D1. Updated core.xml metadata (creator, lastModifiedBy, title).")

    doc.save(str(PEERJ_DOCX))
    print(f"=== Saved {PEERJ_DOCX} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
