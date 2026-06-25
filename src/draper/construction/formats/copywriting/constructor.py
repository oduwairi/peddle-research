"""Format 2 — Copywriting (backtranslation mode).

Skill: tactical copy production. Inverted pipeline (Humpback — Li et al.,
ICLR'24). Each training example is built from a real high-performing ad:

  • Teacher sees the real ad + persona + evol + difficulty dice.
  • Teacher reverse-engineers a plausible brief a copywriter for this
    persona could have received before writing this ad.
  • The assistant_response is the ad itself, reformatted cleanly into
    single H/B/CTA shape with a short craft rationale. Never invented.
  • Student learns: (plausible brief) → (real high-performing copy).

Advantages over forward-mode (teacher invents new copy from a brief):
  • Training target is real copy, not mid-tier teacher-generated copy.
  • Student never sees "0.70 performance score" in the user prompt.
  • Teacher's task is simpler (reasoning, not creative) so budget models
    don't cap the quality ceiling.

Verbatim fidelity is enforced downstream: ingestion rejects examples
where the assistant_response doesn't substantially overlap the source ad.
"""

from __future__ import annotations

from draper.construction.base_constructor import BaseConstructor
from draper.construction.formats.copywriting.ingestion import field_is_english
from draper.construction.schemas import TaskFormat
from draper.scoring.schemas import ScoredAd


def _ad_facts(ad: ScoredAd) -> str:
    """Render the source ad as an unlabeled prose block.

    The teacher sees the advertiser and the ad copy itself — but no
    field labels ("headline:", "body:"), no internal vertical enum
    ("fashion_beauty", "saas_software"), and no platform. Field names
    are ad-platform metadata the student never sees at inference;
    verticals are noisy labels from our own classifier. Platform is a
    scraping-source artifact (which AdFlex endpoint returned the ad),
    not a creative attribute. All three leaked into teacher outputs
    when they appeared in the grounding, so we drop them. Creative
    format and landing-page URL are also omitted: they anchor the
    teacher to surface metadata without improving copy quality.

    Non-English fields (above the langdetect confidence threshold) are
    skipped to avoid foreign-language snippets leaking into grounding.
    """
    copy = ad.ad.ad_copy
    advertiser = ad.ad.advertiser_name or "(unknown advertiser)"

    lead = f"This ad ran for {advertiser}."

    copy_parts: list[str] = []
    for field_text in (copy.headline, copy.body, copy.description):
        if field_text and field_is_english(field_text):
            copy_parts.append(field_text.strip())
    if not copy_parts:
        copy_parts.append("(no usable ad copy fields)")

    return lead + "\n\nThe ad copy reads:\n\n" + "\n\n".join(copy_parts)


class CopywritingConstructor(BaseConstructor):
    """Backtranslation-mode copywriting constructor."""

    # Student-facing system prompt — saved into messages[0] of every
    # training example and seen by the fine-tuned model at inference.
    # Kept to a minimal role tag so it doesn't conflict with whatever
    # system prompt the deployment surface (frontend agent) layers on
    # top, and so behavior is learned from the demonstrated assistant
    # turns rather than from static rules baked into the prompt.
    #
    # Teacher-time rules ("preserve source copy verbatim", "commit to
    # the ad", "ground in observable details", "no field labels",
    # structure-variation rules) live in ``BACKTRANSLATION_STYLE_RULES``
    # in ``draper.construction.bundle`` — that's the meta-prompt the
    # teacher sees, which the student never does.
    SYSTEM_PROMPT = (
        "You are an ad copywriter. When a user describes a product or "
        "campaign, you write ad copy and a short rationale explaining "
        "why the execution works."
    )

    def __init__(self, **kwargs: object) -> None:
        super().__init__(task_format=TaskFormat.COPYWRITING, **kwargs)  # type: ignore[arg-type]

    def format_ads_block(self, source_ads: list[ScoredAd]) -> str:
        if not source_ads:
            return "(no reference ad)"
        return _ad_facts(source_ads[0])
