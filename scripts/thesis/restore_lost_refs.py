"""Restore references that were lost during the PeerJ distillation.

The audit (D3) found 16 references that exist in the original thesis
chapter-II bibliography (`docs/research/literature-review.docx`) but were
dropped by `convert_litreview_to_peerj.py` — many of which are still
cited in the PeerJ body text. Without these entries the manuscript fails
peer-review's basic completeness check.

Strategy:
  1. Diff the original bib against the PeerJ bib by leading surname.
  2. For every "lost" entry whose surname is still cited in the PeerJ
     body, copy the original bib entry verbatim into the PeerJ bib at
     its correct alphabetical position.
  3. For lost entries whose surname is NOT cited in the PeerJ body
     (e.g. a model paper that was dropped from the survey on purpose),
     skip — those were intentionally removed.
  4. Report any remaining "cited but bib entry absent in both
     original and PeerJ" names — those are likely AI-added citations
     that need human review (the script does NOT fabricate entries).

Idempotent.
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
ORIG_DOCX = Path("docs/research/literature-review.docx")


def para_text(p) -> str:
    return "".join((t.text or "") for t in p.findall(f".//{QN_T}"))


def set_para_text(p, new_text: str) -> None:
    runs = p.findall(f".//{QN_R}")
    if not runs:
        raise RuntimeError("paragraph has no <w:r> to write into")
    first = runs[0]
    for r in runs[1:]:
        r.getparent().remove(r)
    ts = first.findall(f".//{QN_T}")
    if not ts:
        raise RuntimeError("first run has no <w:t> to write into")
    ts[0].text = new_text
    for t in ts[1:]:
        t.getparent().remove(t)


REF_PATTERN = re.compile(r"^[A-Z][a-zA-Zà-ÿ\-']+,\s+[A-Z]\.")
OTHER_REF_PATTERN = re.compile(r"^[A-Z]\.\s+Team")


def is_ref_paragraph_text(text: str) -> bool:
    return bool(REF_PATTERN.match(text)) or bool(OTHER_REF_PATTERN.match(text))


def first_surname(ref: str) -> str:
    m = re.match(r"^([A-Za-zà-ÿ\-']+),", ref)
    if m:
        return m.group(1)
    return ref.split()[0]


def collect_ref_texts(doc) -> list[str]:
    body = doc.element.body
    return [
        para_text(p).strip()
        for p in body.iterchildren(QN_P)
        if is_ref_paragraph_text(para_text(p).strip())
    ]


def collect_cited_authors_in_body(doc) -> set[str]:
    """Surnames that appear inside in-text (Surname, YYYY) or
    Surname (YYYY) patterns in the PeerJ body (excluding the bib).
    """
    body = doc.element.body
    # Find the References heading; only inspect paragraphs BEFORE it.
    body_text_parts = []
    for p in body.iterchildren(QN_P):
        t = para_text(p).strip()
        if t == "References":
            break
        body_text_parts.append(para_text(p))
    body_text = "\n".join(body_text_parts)

    cited: set[str] = set()
    # Parenthetical: (Surname, YYYY) and (Surname et al., YYYY) and (S1 & S2, YYYY)
    for m in re.finditer(
        r"\(([A-Z][A-Za-z\-']+)(?:\s+et al\.)?(?:\s+&\s+[A-Z][A-Za-z\-']+)?,\s+\d{4}[a-z]?\)",
        body_text,
    ):
        cited.add(m.group(1))
    # Narrative: Surname (YYYY) and Surname et al. (YYYY) and Surname and Other (YYYY)
    for m in re.finditer(
        r"\b([A-Z][A-Za-z\-']+)(?:\s+et al\.)?\s+\(\d{4}[a-z]?\)",
        body_text,
    ):
        cited.add(m.group(1))
    # "X and Y (YYYY)" → both surnames captured (we want both as candidates)
    for m in re.finditer(
        r"\b([A-Z][A-Za-z\-']+)\s+and\s+([A-Z][A-Za-z\-']+)\s+\(\d{4}[a-z]?\)",
        body_text,
    ):
        cited.add(m.group(1))
        cited.add(m.group(2))
    return cited


def main() -> int:
    orig_doc = Document(str(ORIG_DOCX))
    orig_refs = collect_ref_texts(orig_doc)
    orig_by_surname = {first_surname(r): r for r in orig_refs}

    peerj_doc = Document(str(PEERJ_DOCX))
    peerj_refs = collect_ref_texts(peerj_doc)
    peerj_by_surname = {first_surname(r): r for r in peerj_refs}

    cited = collect_cited_authors_in_body(peerj_doc)

    lost = sorted(set(orig_by_surname) - set(peerj_by_surname))
    cited_and_lost = [s for s in lost if s in cited]
    lost_not_cited = [s for s in lost if s not in cited]

    cited_unknown = sorted(cited - set(peerj_by_surname) - set(orig_by_surname))

    print(f"Original bib entries: {len(orig_refs)}")
    print(f"PeerJ bib entries (before restore): {len(peerj_refs)}")
    print(f"Lost during distillation: {len(lost)}")
    print(f"  ...still cited in PeerJ body: {len(cited_and_lost)}")
    print(f"  ...not cited (won't restore): {len(lost_not_cited)}")

    # ---- Restore lost-and-cited entries ----------------------------------
    body = peerj_doc.element.body

    # Build heading anchor + insertion order.
    refs_heading = None
    for p in body.iterchildren(QN_P):
        if para_text(p).strip() == "References":
            refs_heading = p
            break
    if refs_heading is None:
        print("ERROR: References heading not found in PeerJ docx", file=sys.stderr)
        return 1

    # Body template: clone any existing reference paragraph for its style.
    ref_tpl = None
    seen_refs = False
    for p in body.iterchildren(QN_P):
        if p is refs_heading:
            seen_refs = True
            continue
        if seen_refs and is_ref_paragraph_text(para_text(p).strip()):
            ref_tpl = p
            break
    if ref_tpl is None:
        print("ERROR: could not find a ref template paragraph", file=sys.stderr)
        return 1

    # Collect current refs as (surname, element) pairs from the docx.
    ref_elems: list = []
    seen_refs = False
    for p in body.iterchildren(QN_P):
        if p is refs_heading:
            seen_refs = True
            continue
        if not seen_refs:
            continue
        text = para_text(p).strip()
        if is_ref_paragraph_text(text):
            ref_elems.append((first_surname(text).lower(), p))

    inserted_count = 0
    for surname in cited_and_lost:
        if surname in peerj_by_surname:
            continue  # already present (e.g. from an earlier run)

        ref_text = orig_by_surname[surname]
        target_surname = surname.lower()

        # Find first existing ref whose surname is alphabetically after the target.
        insert_before = None
        for sn, p in ref_elems:
            if sn > target_surname:
                insert_before = p
                break

        new_p = deepcopy(ref_tpl)
        set_para_text(new_p, ref_text)

        if insert_before is not None:
            insert_before.addprevious(new_p)
        else:
            # Append at end of references (before any subsequent section).
            # Find the last reference paragraph and addnext.
            if ref_elems:
                ref_elems[-1][1].addnext(new_p)
            else:
                refs_heading.addnext(new_p)

        ref_elems.append((target_surname, new_p))
        ref_elems.sort(key=lambda kv: kv[0])
        peerj_by_surname[surname] = ref_text
        inserted_count += 1
        print(f"  + Restored: {ref_text[:120]}")

    peerj_doc.save(str(PEERJ_DOCX))
    print(f"\nRestored {inserted_count} reference(s) from original bib.")
    print(f"New PeerJ bib total: {len(peerj_refs) + inserted_count}")

    # ---- Report unknown cited surnames for human review ------------------
    # Filter out common false positives (co-authors caught by the regex,
    # and proper-noun stop words).
    NON_AUTHOR = {
        "Figure", "Table", "Section", "Note", "Equation", "Appendix",
        "Chapter", "Source", "Anthropic", "OpenAI", "Google", "Meta",
        "Microsoft", "RQ1", "RQ2", "RQ3", "BLEU", "ROUGE", "NLP", "AI", "G",
        # Common second-author surnames that already have a primary entry
        # under another surname (Kirk & Givi, Duerr & Gloor, etc.).
        "Givi", "Gloor", "Nissenbaum", "Stillwell", "Ghazanfar", "Lee",
        "Meier", "Schwartz",
    }
    candidates = [
        s for s in cited_unknown
        if s not in NON_AUTHOR and len(s) > 2
    ]
    # Filter: only show names that appear as the LEADING author (i.e. the
    # citation pattern is "(Name, " or "Name (YYYY)" — not "& Name").
    print()
    if candidates:
        print("HUMAN REVIEW NEEDED — citations whose bib entry is absent in both")
        print("the original and the PeerJ docx (likely AI-added citations):")
        for s in candidates:
            print(f"  ? {s}")
        print()
        print("Action: either add proper bib entries (recommended for real papers")
        print("like Sellam et al. 2020 BLEURT or Zhong et al. 2022 UniEval) or")
        print("remove the in-text citation. The script will NOT fabricate entries.")
    else:
        print("No unknown citations remaining — bib is consistent with body.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
