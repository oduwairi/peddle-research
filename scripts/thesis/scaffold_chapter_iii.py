"""Scaffold Chapter III (Methodology).

Replaces the template author's brain-tumour CH3 content with our 8-subsection
methodology skeleton. Each subsection gets a numbered Heading 2 + a placeholder
body paragraph. The chapter intro paragraph is preserved structurally.

Conventions (see docs/research/THESIS_EDITING.md):
- Clone Heading 2 from §2.7.3 (plain inline, no <w:drawing>), never from
  original-template headings.
- Clone Body Text from the existing CH3 intro paragraph (style 943).
- Delete `<w:p>` and `<w:tbl>` between "Methodology" and "CHAPTER IV";
  skip any paragraph containing `<w:sectPr>` (pagination guard).
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from docx import Document

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
QN_P = f"{{{W_NS}}}p"
QN_TBL = f"{{{W_NS}}}tbl"
QN_T = f"{{{W_NS}}}t"
QN_SECTPR = f"{{{W_NS}}}sectPr"
QN_DRAWING = f"{{{W_NS}}}drawing"


INTRO = (
    "[Chapter intro — placeholder. To be drafted: framing of the methodology "
    "around the three co-designed pieces (corpus pipeline, fine-tuned writer, "
    "inference-time agent), and the pivot from multi-format reasoning to "
    "instruction backtranslation introduced in §2.7.3.]"
)

SUBSECTIONS: list[tuple[str, str]] = [
    (
        "3.1 Proposed System",
        "[Section body — placeholder. To be drafted: end-to-end pipeline overview "
        "(Collection → Scoring → Construction → Training → Inference), the "
        "two-role agent topology that separates orchestrator from writer, and the "
        "deployment constraints that motivate §3.2–§3.8.]",
    ),
    (
        "3.2 Data Acquisition",
        "[Section body — placeholder. To be drafted: AdFlex sweep-based collection "
        "(SweepPlanner, SweepExecutor, cursor checkpointing), supplementary scrapers "
        "for Meta, Google, TikTok, and Reddit, and normalization to the RawAd "
        "Pydantic schema.]",
    ),
    (
        "3.3 Engagement-Based Scoring",
        "[Section body — placeholder. To be drafted: scorer evolution from v1 "
        "(hand-tuned CompositeScorer) through v2 (Snorkel weak-supervision with 8 "
        "labelling functions) to v3 (HybridScorer with per-platform Kaplan–Meier "
        "survival curves); shared TierAssigner and percentile-based tier "
        "thresholds.]",
    ),
    (
        "3.4 Training-Data Construction via Backtranslation",
        "[Section body — placeholder. To be drafted: Humpback-style reverse "
        "engineering of plausible briefs from high-tier real ads, the copywriting "
        "rubric and ingestion check, the shared quality-filter / DICE / source-"
        "selector orchestrators, and the format-registry dispatch.]",
    ),
    (
        "3.5 Fine-Tuning",
        "[Section body — placeholder. To be drafted: QLoRA fine-tuning of a small "
        "open base model with Unsloth and TRL, hyperparameter choices, the merge "
        "step, and vLLM serving on a single L4 via Modal.]",
    ),
    (
        "3.6 Agent Architecture",
        "[Section body — placeholder. To be drafted: the orchestrator/writer split "
        "(general LLM versus fine-tuned Draper), the freeform agent loop, the "
        "tool surface (draft_campaign, ask_draper, emit_campaign, score_copy, plus "
        "research and visual tools), and the provenance and requirements gates "
        "guarding emission.]",
    ),
    (
        "3.7 Scoring Predictor",
        "[Section body — placeholder. To be drafted: a text-only DeBERTa-v3-base "
        "regressor trained on the v3-scored corpus with four heads (composite, "
        "survivability, engagement volume, engagement velocity), isotonic "
        "calibration, and CPU serving on Modal as both a live frontend feature "
        "and an absolute eval arm.]",
    ),
    (
        "3.8 Evaluation Methodology",
        "[Section body — placeholder. To be drafted: a three-arm evaluation — "
        "pairwise tournament, reference comparison against held-out gold ads, and "
        "learned-scorer absolute scoring — with the normalize → judge → aggregate "
        "pipeline and the three-judge panel (Claude Sonnet 4.6, Gemini 2.5 Flash, "
        "GPT-5.4-mini).]",
    ),
]


def find_paragraph(doc, predicate, label):
    for p in doc.paragraphs:
        if predicate(p):
            return p._element
    raise RuntimeError(f"{label} not found")


def set_paragraph_text(p_elem, new_text):
    """Set the first <w:t> in the paragraph to new_text; clear the rest.

    Preserves <w:rPr> and run structure so the heading/body style survives.
    """
    ts = p_elem.findall(f".//{QN_T}")
    if not ts:
        raise RuntimeError("paragraph template has no <w:t> to write into")
    ts[0].text = new_text
    for t in ts[1:]:
        t.text = ""


def main():
    path = Path("docs/research/THESIS.docx")
    doc = Document(str(path))

    # Anchors
    meth = find_paragraph(
        doc,
        lambda p: p.text.strip() == "Methodology"
        and (p.style.style_id if p.style else "") == "945",
        "Methodology heading (style 945)",
    )
    ch4 = find_paragraph(
        doc,
        lambda p: p.text.strip().upper() == "CHAPTER IV",
        "CHAPTER IV heading",
    )

    # Templates (deep-copied later)
    h2_tpl = find_paragraph(
        doc,
        lambda p: p.text.strip().startswith("2.7.3")
        and (p.style.style_id if p.style else "") == "945",
        "§2.7.3 heading template",
    )
    if h2_tpl.find(f".//{QN_DRAWING}") is not None:
        raise RuntimeError("§2.7.3 template paragraph unexpectedly contains <w:drawing>")
    body_tpl = find_paragraph(
        doc,
        lambda p: p.text.strip().startswith("The methodology of the research")
        and (p.style.style_id if p.style else "") == "943",
        "CH3 intro Body Text template",
    )

    # Range to wipe
    body = meth.getparent()
    children = list(body)
    i_meth = children.index(meth)
    i_ch4 = children.index(ch4)

    to_delete = []
    skipped_sectpr = 0
    for c in children[i_meth + 1 : i_ch4]:
        if c.tag == QN_P:
            if c.find(f".//{QN_SECTPR}") is not None:
                skipped_sectpr += 1
                continue
            to_delete.append(c)
        elif c.tag == QN_TBL:
            to_delete.append(c)

    print(f"DELETING: {len(to_delete)} elements between Methodology and CHAPTER IV")
    print(f"  (skipped {skipped_sectpr} paragraph(s) holding <w:sectPr> for pagination)")

    for c in to_delete:
        body.remove(c)

    # Insertion
    inserted = []
    insert_after = meth

    intro_p = deepcopy(body_tpl)
    set_paragraph_text(intro_p, INTRO)
    insert_after.addnext(intro_p)
    insert_after = intro_p
    inserted.append(("intro", INTRO[:80]))

    for title, body_text in SUBSECTIONS:
        h = deepcopy(h2_tpl)
        set_paragraph_text(h, title)
        insert_after.addnext(h)
        insert_after = h

        b = deepcopy(body_tpl)
        set_paragraph_text(b, body_text)
        insert_after.addnext(b)
        insert_after = b

        inserted.append((title, body_text[:80]))

    doc.save(str(path))

    print("\nINSERTIONS:")
    for title, snippet in inserted:
        print(f"  {title!r}")
        print(f"    → {snippet}…")


if __name__ == "__main__":
    main()
