"""Phase-8 citation pass for the §3.8 reference-overlap inserts.

Of the 7 in-text cites introduced in C.1/C.2, four already have References
entries with forms matching the in-text use (Papineni 2002, Lin 2004,
Zhang 2020/BERTScore, Matias 2021). The thesis also already cites the bootstrap
as (Efron, 1979) in two places (§3.7, §3.8) with a matching entry. This script:

  1. Aligns the lone out-of-convention bootstrap cite in the C.1 paragraph
     [(Efron & Tibshirani, 1993) -> (Efron, 1979)] so the whole thesis uses one
     bootstrap citation.
  2. Adds the two genuinely-missing References entries, alphabetically:
       - Banerjee, S., & Lavie, A. (2005)  [METEOR]   before "Belcak, ..."
       - Popović, M. (2015)                [chrF]     before "Qin, ..."
     Both verified against the ACL Anthology primary sources (W05-0909,
     W15-3049). Formatting (hanging indent / spacing / run font) is transplanted
     from the adjacent reference entry, since a new paragraph inherits only the
     style, not the thesis's direct paragraph formatting.

Index-free anchors, idempotent. References list is alphabetical by first-author
surname. No live field is touched, so no refresh is required for this edit.
"""

from __future__ import annotations

import copy

import docx
from docx.oxml.ns import qn

DOC = "docs/research/THESIS.docx"

# U+2011 non-breaking hyphen to match the thesis's existing entry style
# (e.g. "W.‑J.", "C.‑Y.", "ACL‑04").
NBH = "‑"

METEOR = (
    "Banerjee, S., & Lavie, A. (2005). METEOR: An automatic metric for MT "
    "evaluation with improved correlation with human judgments. In Proceedings "
    "of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for "
    "Machine Translation and/or Summarization (pp. 65–72). Association for "
    "Computational Linguistics."
)
CHRF = (
    "Popović, M. (2015). chrF: Character n" + NBH + "gram F" + NBH + "score "
    "for automatic MT evaluation. In Proceedings of the Tenth Workshop on "
    "Statistical Machine Translation (pp. 392–395). Association for "
    "Computational Linguistics. https://doi.org/10.18653/v1/W15-3049"
)


def first_text_run_rpr(p_el):
    for r in p_el.findall(qn("w:r")):
        if r.find(qn("w:t")) is not None:
            return r.find(qn("w:rPr"))
    return None


def make_like(anchor, text):
    """Create a new paragraph styled+formatted like `anchor`, carrying `text`."""
    new_p = copy.deepcopy(anchor._p)
    # strip everything but keep pPr; drop all runs / bookmarks / fields
    for child in list(new_p):
        if child.tag != qn("w:pPr"):
            new_p.remove(child)
    # build a single clean run with the anchor's first-text-run rPr
    run = new_p.makeelement(qn("w:r"), {})
    rpr = first_text_run_rpr(anchor._p)
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    t = run.makeelement(qn("w:t"), {})
    t.set(qn("xml:space"), "preserve")
    t.text = text
    run.append(t)
    new_p.append(run)
    return new_p


def main() -> None:
    d = docx.Document(DOC)
    paras = d.paragraphs

    def find_prefix(prefix, style=None):
        return next(
            (p for p in paras
             if p.text.strip().startswith(prefix)
             and (style is None or p.style.name == style)),
            None,
        )

    done = []

    # --- 1. align the bootstrap cite in the C.1 paragraph -------------------
    c1 = next((p for p in paras
               if "we need more than one signal to cross-reference" in p.text),
              None)
    if c1 is None:
        raise SystemExit("C.1 paragraph not found")
    if "Efron & Tibshirani, 1993" in c1.text:
        hit = False
        for r in c1.runs:
            if "Efron & Tibshirani, 1993" in r.text:
                r.text = r.text.replace("Efron & Tibshirani, 1993", "Efron, 1979")
                hit = True
                break
        if not hit:
            raise SystemExit("'Efron & Tibshirani, 1993' spans multiple runs; "
                             "fix manually")
        done.append("aligned bootstrap cite -> (Efron, 1979)")
    elif "Efron, 1979" in c1.text:
        pass  # already aligned
    else:
        raise SystemExit("no bootstrap cite found in C.1 paragraph")

    # --- 2a. METEOR (Banerjee & Lavie, 2005) before Belcak ------------------
    if find_prefix("Banerjee, S., & Lavie") is None:
        anchor = find_prefix("Belcak, P.", style="Body Text")
        if anchor is None:
            raise SystemExit("anchor 'Belcak, P.' not found")
        anchor._p.addprevious(make_like(anchor, METEOR))
        done.append("inserted Banerjee & Lavie (2005) [METEOR] before Belcak")
    else:
        done.append("Banerjee & Lavie (2005) already present")

    # --- 2b. chrF (Popović, 2015) before Qin --------------------------------
    if find_prefix("Popović, M. (2015)") is None:
        anchor = find_prefix("Qin, Y., Hu, S.", style="Body Text")
        if anchor is None:
            raise SystemExit("anchor 'Qin, Y., Hu, S.' not found")
        anchor._p.addprevious(make_like(anchor, CHRF))
        done.append("inserted Popović (2015) [chrF] before Qin")
    else:
        done.append("Popović (2015) already present")

    if not any(s.startswith(("aligned", "inserted")) for s in done):
        print("nothing to do:")
        for s in done:
            print("  -", s)
        return

    d.save(DOC)
    for s in done:
        print("•", s)


if __name__ == "__main__":
    main()
