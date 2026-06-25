"""Insert the author's cross-validation / split paragraph into §3.5.1.

Phase-5-polished prose (author-written, assistant-polished). Appended as a new
Body-Text paragraph at the end of §3.5.1, cloned from the existing §3.5.1 body
paragraph for identical style. Idempotent: skips if already present.
"""

from __future__ import annotations

import copy
import sys

from docx import Document
from docx.oxml.ns import qn

DOC = "docs/research/THESIS.docx"
W_R, W_T, W_RPR, W_PPR = qn("w:r"), qn("w:t"), qn("w:rPr"), qn("w:pPr")
XML_SPACE = qn("xml:space")

PARA = (
    "The dataset was split by a single stratified hold-out split instead of a "
    "k-fold cross-validation, the main justification being the training cost, "
    "since using a k-fold strategy multiplies the training time and cost "
    "significantly. Instead, we split the training examples 2,442 / 215 / 215, "
    "stratified by platform: the validation set is used during training for "
    "evaluation, and the test set is held out completely and unseen by the "
    "model for evaluation after training. The split is done on a deterministic "
    "basis keyed on source-ad ID, so that the split is reproducible and not by "
    "random chance. The split strictly ensures that the same brief never "
    "appears in any two splits at the same time, to prevent cross-contamination. "
    "During training, the evaluation loss is calculated every fifty steps over "
    "the 215 validation set; this number is expected to decay as training goes "
    "on, with an early-stopping mechanism to stop training if the evaluation "
    "loss fails to improve two consecutive times."
)
MARKER = "single stratified hold-out split instead of a k-fold"


def clone_body(template_el, text):
    new = copy.deepcopy(template_el)
    donor = None
    for r in new.findall(W_R):
        if donor is None and r.find(W_RPR) is not None:
            donor = copy.deepcopy(r.find(W_RPR))
        new.remove(r)
    run = new.makeelement(W_R, {})
    if donor is not None:
        run.append(donor)
    t = run.makeelement(W_T, {XML_SPACE: "preserve"})
    t.text = text
    run.append(t)
    new.append(run)
    return new


def main() -> None:
    d = Document(DOC)
    ps = d.paragraphs
    if any(MARKER in p.text for p in ps):
        print("ABORTED (no save): paragraph already present")
        return

    head = nxt = None
    for i, p in enumerate(ps):
        t = p.text.strip()
        if "\t" in t:
            continue
        if t.startswith("3.5.1 Dataset and Splits"):
            head = i
        elif head is not None and t.startswith("3.5.2"):
            nxt = i
            break
    assert head is not None and nxt is not None, "could not bound §3.5.1"

    # last §3.5.1 body paragraph = the one right before §3.5.2
    body = None
    for p in ps[head + 1 : nxt]:
        if p.text.strip() and not p.style.name.startswith("Heading"):
            body = p._element
    assert body is not None, "no §3.5.1 body paragraph to clone"

    body.addnext(clone_body(body, PARA))
    d.save(DOC)
    print(f"INSERTED §3.5.1 cross-validation paragraph ({len(PARA)} chars) after the existing body para.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"ABORTED (no save): {e}", file=sys.stderr)
        sys.exit(1)
