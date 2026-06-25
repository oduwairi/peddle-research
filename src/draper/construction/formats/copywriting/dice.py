"""Copywriting derived context — no RNG.

Prior architectures conditioned the teacher on independently-sampled
persona + scenario + rationale_depth labels. Pilot review showed this
produced implausible combinations (creator-persona briefing a university
ad, media-planner briefing dairy recipe pins, junior copywriter filing a
finance-justification document) because the labels are sampled without
reference to the source ad.

The model at inference time never sees those labels. The fine-tune was
learning to fit RNG combinations that don't reflect real input
distributions. Removing the labels and letting the teacher derive voice /
shape from the ad itself is closer to how a real copywriter works, and is
the one thing the model can actually do at inference.

What survives: ``source_ad_shape`` (``has_body`` vs ``headline_only``),
provenance only, and ``conversation_register`` — a 3-way axis hash-rolled
from ``ad.ad_id`` that governs the register of *both* sides of the
conversation (a conversational brief gets a conversational response;
structured/imperative likewise). Rationale depth is **not** randomized —
the teacher is always asked for its most thorough analysis regardless of
register; only opener and prose register vary.

The prior ``platform_framing`` axis was removed after the 2026-04 audit:
the corpus's ``platform`` value is a scraping-source artifact (which
AdFlex endpoint returned the ad), not a creative attribute, so
conditioning the teacher on it trained the student on label noise.
Platform-specific formatting is handled at inference via prompt.

The commissioner-inference instructions live on the teacher bundle's
BACKTRANSLATION style block (``bundle.py``), not here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from draper.scoring.schemas import ScoredAd


class SourceAdShape(StrEnum):
    """Structural shape of the source ad — derived from populated fields."""

    HEADLINE_ONLY = "headline_only"
    HAS_BODY = "has_body"


class ConversationRegister(StrEnum):
    """Strongly-enforced register for the *whole* user/assistant turn.

    Three sharply-distinct registers. Each example is assigned exactly
    one at roll time (deterministic from ``ad.ad_id``); the teacher is
    told to write both the brief AND the response in that register, not
    given a menu. This breaks the mode collapse where every provider
    locks to its preferred default ("We're [Brand]…", "Need ad for…",
    "Product:" labels) and the response side defaults to a uniform
    craft-note voice regardless of how the user wrote the brief.
    """

    CONVERSATIONAL = "conversational"
    STRUCTURED = "structured"
    IMPERATIVE = "imperative"


@dataclass(frozen=True)
class CopywritingContext:
    """All copywriting-specific context for one bundle, derived from the ad.

    Two axes, both deterministic from the source ad:
    - ``source_ad_shape``: provenance, derived from which ad fields are
      populated.
    - ``conversation_register``: strongly-enforced, hash-keyed on
      ``ad.ad_id`` so every ad gets a stable assignment and the corpus
      splits ~33/33/33. Governs both sides of the turn.
    """

    source_ad_shape: SourceAdShape
    conversation_register: ConversationRegister

    def as_sidecar(self) -> dict[str, str]:
        """Serialize for the per-bundle sidecar JSON (reproducibility)."""
        return {
            "source_ad_shape": self.source_ad_shape.value,
            "conversation_register": self.conversation_register.value,
        }


def infer_source_ad_shape(ad: ScoredAd) -> SourceAdShape:
    """Derive shape from which ad fields are populated.

    ``HAS_BODY`` when a ``body`` or ``description`` field has real content
    (≥ 10 chars after stripping). Otherwise ``HEADLINE_ONLY``.
    """
    copy = ad.ad.ad_copy
    body = (copy.body or "").strip()
    description = (copy.description or "").strip()
    if len(body) >= 10 or len(description) >= 10:  # noqa: PLR2004
        return SourceAdShape.HAS_BODY
    return SourceAdShape.HEADLINE_ONLY


def infer_conversation_register(ad: ScoredAd) -> ConversationRegister:
    """Deterministic 3-way bucket keyed on ``ad.ad_id``.

    Stable across processes (uses sha256, not Python's randomized
    ``hash``). Source-content-agnostic — only the ad's identity feeds the
    bucket, never its copy or vertical — so registers come out roughly
    evenly distributed across the corpus regardless of what kinds of ads
    are sampled.
    """
    digest = hashlib.sha256(ad.ad.ad_id.encode("utf-8")).digest()
    bucket = digest[0] % 3
    return list(ConversationRegister)[bucket]


def derive_copywriting_context(ad: ScoredAd) -> CopywritingContext:
    """Derive the copywriting context directly from the source ad.

    ``source_ad_shape`` reads off the ad's populated fields.
    ``conversation_register`` is rolled deterministically from
    ``ad.ad_id`` so the same ad always lands in the same register; across
    a corpus, hash distribution gives ~33/33/33.
    """
    return CopywritingContext(
        source_ad_shape=infer_source_ad_shape(ad),
        conversation_register=infer_conversation_register(ad),
    )


# Per-register directive bodies. Each spans BOTH sides of the turn — the
# brief shape AND the response register. Rationale **depth** is constant
# across registers (always the teacher's most thorough analysis); only
# opener and prose register change. Hard invariants that don't depend on
# register (verbatim ad, no field labels in prose, preserve emojis) are
# stated once in BACKTRANSLATION_STYLE_RULES, not duplicated here.
_REGISTER_BODIES: dict[ConversationRegister, str] = {
    ConversationRegister.CONVERSATIONAL: (
        "**Brief:** Casual, natural phrasing. Lead with a greeting, "
        "fragment, or question (\"hey\", \"quick one for you\", "
        "\"wondering if…\"). Lowercase is fine. Incomplete sentences are "
        "fine. Line breaks are fine. No field labels like `Product:`. No "
        "\"Need…\" or \"We're…\" openers. Write the way someone dashes "
        "off a Slack message to a coworker.\n"
        "\n"
        "**Response:** Reply the way you'd answer that Slack message. A "
        "short casual lead-in or none at all (\"oh nice product\", "
        "\"ok so for this one…\", or just go straight in). Lowercase "
        "and fragments are fine in the prose. Then deliver the ad. Then "
        "the rationale, written in the same chatty register — \"the "
        "move here is…\", \"what makes this land is…\", \"notice how "
        "the dash before X does the work of…\". The casual register is "
        "the *prose voice only*; the analysis itself is still your most "
        "thorough breakdown of the ad's craft."
    ),
    ConversationRegister.STRUCTURED: (
        "**Brief:** Field-labeled bullets. Use these exact field "
        "labels, in this order:\n\n"
        "Product: [what it is]\n"
        "Audience: [who buys it]\n"
        "Goal: [what you need from this ad]\n"
        "Facts: [key specs / promo / details]\n\n"
        "Clean, dense, professional. No casual openers, no greetings, "
        "no questions, no narrative paragraphs.\n"
        "\n"
        "**Response:** Match the brief's professional, organized "
        "register. Optionally one tight framing line, then deliver the "
        "ad, then a craft-note rationale: organized prose, paragraphs "
        "or short inline-bolded sub-headings (\"**Hook.**\", "
        "\"**Structure.**\", \"**Word choice.**\" — never `Headline:` / "
        "`Body:` field labels in prose). Formal register, complete "
        "sentences, no casualisms. Cover the analysis exhaustively — "
        "this register doesn't mean *less* analysis, it means the "
        "analysis is presented as a clean professional breakdown."
    ),
    ConversationRegister.IMPERATIVE: (
        "**Brief:** Direct, clipped command. Start with a verb: "
        "\"Need…\", \"Create…\", \"Write…\", \"Draft…\". One or two "
        "tight lines. Fragments. No field labels, no greetings, no "
        "questions, no small talk. Like an executive's curt one-line "
        "ask: \"Need ad for X — facts: Y, Z.\"\n"
        "\n"
        "**Response:** Match the brief's clipped, no-preamble register. "
        "Deliver the ad immediately — no \"Here's…\", no \"Sure, "
        "try…\", no warm-up. Then the rationale: dense, lean prose, "
        "every sentence load-bearing, no throat-clearing or hedging. "
        "Imperative means **no slack in the prose**, not less analysis "
        "— the breakdown of hook, structure, word choice, sequence is "
        "still complete and exhaustive, just delivered without padding."
    ),
}


def render_conversation_register_directive(
    register: ConversationRegister,
) -> str:
    """Render the rolled register as a strong, single-shape directive.

    Imperative wording, no menu. Goes into the teacher prompt as its own
    section so it can't drown inside the BACKTRANSLATION style rules.
    Covers both sides of the turn (brief + response) under one heading.
    """
    body = _REGISTER_BODIES[register]
    return (
        "## Conversation register (REQUIRED)\n"
        "\n"
        f"This whole turn — both the user brief AND the assistant "
        f"response — MUST be written in **{register.value.upper()}** "
        "register. This is non-negotiable; it overrides any structure "
        "guidance in the style rules above. Rationale **depth** is "
        "constant across registers: always provide your most thorough, "
        "ad-grounded analysis. Only the opener and prose register "
        "change; the analysis itself never gets shorter or shallower.\n"
        "\n"
        f"{body}\n"
    )
