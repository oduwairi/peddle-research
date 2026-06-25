"""Strengthen §6.1 Conclusion (reviewer feedback #8: weak conclusion).

Folds two author-written, assistant-polished blocks into §6.1:
  1. a synthesis opener BEFORE the existing RQ-by-RQ paragraph, and
  2. a practical-implications paragraph AFTER it (before §6.2 Contributions).

The existing RQ paragraph is left untouched. No new headings -> no TOC refresh.

Pattern follows scripts/thesis conventions:
  - python-docx + lxml, never byte-slice the XML.
  - Anchor by visible text, never a hardcoded paragraph index.
  - Clone the existing Body-Text paragraph's <w:p> (deep copy) so the new
    paragraphs inherit identical style + direct pPr/rPr; set the first run's
    text and drop the rest.
  - Idempotent: skip a block that is already present.
"""

from __future__ import annotations

import copy
import sys

import docx
from docx.oxml.ns import qn

DOC_PATH = "docs/research/THESIS.docx"

HEADING_TEXT = "6.1 Conclusion"
BODY_PREFIX = "The aim of this thesis was to build"

XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

OPENER = (
    "The core research question this thesis has asked is whether a specialized "
    "model in the 7–9B range can beat a large general frontier model on "
    "domain-specific vertical tasks. The presented results have shown that the "
    "answer is indeed yes across every evaluation arm we designed. We see the "
    "fine-tuned 8B writer has scored closer to the real-ad ceiling across almost "
    "every evaluation method and metric compared to the frontier model. This "
    "gives the key takeaway that a small model specializing in marketing and "
    "advertising generation can indeed be fine-tuned to surpass much larger "
    "models on the marketing domain, a narrow vertical."
)

IMPLICATIONS = (
    "It is important to note that a small fine-tuned model is much cheaper to run "
    "at scale than a frontier model’s API. The entire model can run inference "
    "on one mid-range GPU, so a small team or company can own and serve it. The "
    "absence of labeled marketing data can be partially filled by filling out "
    "labels from public engagement signals using the proxy scorer, which is "
    "validated against the ground-truth set. The trained scorer (a "
    "DeBERTa-v3-base regressor) obtained in this thesis is a standalone model "
    "that can provide a performance score on any ad without needing a human or "
    "judge model, using cheap CPU inference. We can see that augmenting "
    "fine-tuning with RAG can sometimes have diminishing effects, where the "
    "output can be pulled back towards a generic, non-fine-tuned style."
)


def make_para_from_template(template_p, text: str):
    """Deep-copy a Body-Text <w:p>, keep one run, set its text."""
    new_p = copy.deepcopy(template_p)
    runs = new_p.findall(qn("w:r"))
    if not runs:
        run = new_p.makeelement(qn("w:r"), {})
        new_p.append(run)
        runs = [run]
    first = runs[0]
    for extra in runs[1:]:
        new_p.remove(extra)
    # keep a single <w:t> on the surviving run
    texts = first.findall(qn("w:t"))
    if not texts:
        t = first.makeelement(qn("w:t"), {})
        first.append(t)
        texts = [t]
    for extra in texts[1:]:
        first.remove(extra)
    t = texts[0]
    t.set(XML_SPACE, "preserve")
    t.text = text
    return new_p


def main() -> int:
    doc = docx.Document(DOC_PATH)
    paras = doc.paragraphs

    # locate §6.1 heading
    heading_idx = next(
        (i for i, p in enumerate(paras) if p.text.strip() == HEADING_TEXT), None
    )
    if heading_idx is None:
        print(f"ERROR: heading '{HEADING_TEXT}' not found", file=sys.stderr)
        return 1

    # the RQ paragraph is the first one after the heading that starts with the
    # known prefix (robust to the opener already sitting between heading + body)
    body_para = next(
        (
            p
            for p in paras[heading_idx + 1 :]
            if p.text.strip().startswith(BODY_PREFIX)
        ),
        None,
    )
    if body_para is None:
        print("ERROR: §6.1 RQ body anchor not found", file=sys.stderr)
        return 1

    existing = {p.text.strip() for p in paras}
    opener_present = any(s.startswith(OPENER[:48]) for s in existing)
    impl_present = any(s.startswith(IMPLICATIONS[:48]) for s in existing)

    body_p = body_para._p
    actions = []

    if not opener_present:
        body_p.addprevious(make_para_from_template(body_p, OPENER))
        actions.append("INSERT opener  -> before §6.1 RQ paragraph (after heading)")
    else:
        actions.append("SKIP   opener  -> already present")

    if not impl_present:
        body_p.addnext(make_para_from_template(body_p, IMPLICATIONS))
        actions.append("INSERT implications -> after §6.1 RQ paragraph")
    else:
        actions.append("SKIP   implications -> already present")

    doc.save(DOC_PATH)

    print("INSERTIONS:")
    for a in actions:
        print("  " + a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
