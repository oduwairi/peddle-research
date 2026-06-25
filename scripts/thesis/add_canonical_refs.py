"""Add canonical bibliography entries for three well-known evaluation-metric
papers that the body cites but the original bib was missing: COMET (Rei),
BLEURT (Sellam), and UniEval (Zhong).

These are canonical, widely-cited papers with stable publication metadata
verifiable from any reputable source (ACL Anthology / Semantic Scholar).
The script does NOT fabricate the entries — they reflect published facts.

A fourth cited-but-missing name (Verma, in "Mishra, Verma et al. (2020) used
this approach to train pairwise rankers") could not be matched to a canonical
real paper and is left for human review — the in-text citation may have been
introduced in error during AI-assisted drafting.

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


CANONICAL_REFS = [
    # COMET — Rei, Stewart, Farinha & Lavie, EMNLP 2020.
    (
        "Rei",
        "Rei, R., Stewart, C., Farinha, A. C., & Lavie, A. (2020). COMET: A "
        "neural framework for MT evaluation. In Proceedings of the 2020 "
        "Conference on Empirical Methods in Natural Language Processing "
        "(EMNLP), 2685–2702. https://aclanthology.org/2020.emnlp-main.213/",
    ),
    # BLEURT — Sellam, Das & Parikh, ACL 2020.
    (
        "Sellam",
        "Sellam, T., Das, D., & Parikh, A. P. (2020). BLEURT: Learning "
        "robust metrics for text generation. In Proceedings of the 58th "
        "Annual Meeting of the Association for Computational Linguistics "
        "(ACL), 7881–7892. https://aclanthology.org/2020.acl-main.704/",
    ),
    # UniEval — Zhong et al., EMNLP 2022.
    (
        "Zhong",
        "Zhong, M., Liu, Y., Yin, D., Mao, Y., Jiao, Y., Liu, P., Zhu, C., "
        "Ji, H., & Han, J. (2022). Towards a unified multi-dimensional "
        "evaluator for text generation. In Proceedings of the 2022 "
        "Conference on Empirical Methods in Natural Language Processing "
        "(EMNLP), 2023–2038. https://aclanthology.org/2022.emnlp-main.131/",
    ),
]


# ---------- helpers --------------------------------------------------------


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


def first_surname(ref: str) -> str:
    m = re.match(r"^([A-Za-zà-ÿ\-']+),", ref)
    if m:
        return m.group(1)
    return ref.split()[0]


def main() -> int:
    doc = Document(str(PEERJ_DOCX))
    body = doc.element.body

    # Locate References heading.
    refs_heading = None
    for p in body.iterchildren(QN_P):
        if para_text(p).strip() == "References":
            refs_heading = p
            break
    if refs_heading is None:
        print("ERROR: References heading not found", file=sys.stderr)
        return 1

    # Build current ref list with elements.
    ref_elems: list = []
    seen = False
    ref_tpl = None
    for p in body.iterchildren(QN_P):
        if p is refs_heading:
            seen = True
            continue
        if not seen:
            continue
        text = para_text(p).strip()
        if REF_PATTERN.match(text):
            ref_elems.append((first_surname(text).lower(), p))
            if ref_tpl is None:
                ref_tpl = p

    if ref_tpl is None:
        print("ERROR: no existing ref to clone from", file=sys.stderr)
        return 1

    present_surnames = {sn for sn, _ in ref_elems}
    inserted = 0
    for surname, ref_text in CANONICAL_REFS:
        if surname.lower() in present_surnames:
            print(f"  skip {surname}: already present")
            continue
        target = surname.lower()
        insert_before = None
        for sn, p in ref_elems:
            if sn > target:
                insert_before = p
                break
        new_p = deepcopy(ref_tpl)
        set_para_text(new_p, ref_text)
        if insert_before is not None:
            insert_before.addprevious(new_p)
        else:
            ref_elems[-1][1].addnext(new_p)
        ref_elems.append((target, new_p))
        ref_elems.sort(key=lambda kv: kv[0])
        present_surnames.add(target)
        inserted += 1
        print(f"  + Added: {ref_text[:120]}")

    doc.save(str(PEERJ_DOCX))
    print(f"\nAdded {inserted} canonical reference(s).")
    print()
    print("NOTE: 'Mishra, Verma et al. (2020)' in the body could not be matched "
          "to a canonical paper. The citation may have been introduced in "
          "error during AI-assisted drafting — recommend either removing the "
          "in-text citation or replacing it with a verified primary source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
