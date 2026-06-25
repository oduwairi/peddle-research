"""Copywriting-example bundle builder.

Assembles a single self-contained prompt block the chat agent receives.
The bundle includes (a) the BACKTRANSLATION style rules, (b) the
copywriting context directive (derived from the source ad — no RNG),
(c) the real ad as the gold target, and (d) the required output tag
structure.

The chat agent returns ``<user_prompt>`` + ``<assistant_response>`` tags
in one pass — ``ingest`` parses these into a ``TrainingExample``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from draper.construction.formats.copywriting.dice import CopywritingContext
from draper.construction.formats.registry import get_pipeline
from draper.construction.personas import Persona
from draper.construction.schemas import PromptStyle, TaskFormat
from draper.scoring.schemas import ScoredAd

# Output tags the chat agent must fill in. Kept as constants so ingest
# and bundle builder stay in sync.
USER_PROMPT_OPEN = "<user_prompt>"
USER_PROMPT_CLOSE = "</user_prompt>"
ASSISTANT_RESPONSE_OPEN = "<assistant_response>"
ASSISTANT_RESPONSE_CLOSE = "</assistant_response>"


BACKTRANSLATION_STYLE_RULES = (
    "# Training example — copywriting\n"
    "\n"
    "You are generating one question/answer training pair for "
    "fine-tuning a copywriter model. The model is being trained "
    "to write successful ads (like the one below) naturally, as "
    "part of a normal LLM response.\n"
    "\n"
    "The real, high-performing ad is pasted below. Write a "
    "user/assistant pair such that, if the student model "
    "reproduced this pair at inference, it would deliver this ad "
    "to the user.\n"
    "\n"
    "## **MOST IMPORTANT RULES**\n"
    "\n"
    "**1. Reproduce the source ad word-for-word in the assistant "
    "response.** No paraphrase, translation, or restructuring. "
    "The rationale's specific content varies example to example "
    "(it's about a different ad each time), but the analytic "
    "depth stays constant: always your most thorough analysis.\n"
    "\n"
    "**2. Always emit BOTH `<user_prompt>` and "
    "`<assistant_response>` tags, in that order.** The brief "
    "comes first, the answer second. Skipping either tag makes "
    "the example unusable.\n"
    "\n"
    "## The pair\n"
    "\n"
    "- **User prompt** — a request for help with an ad or "
    "creative, written the way a real person would type it into "
    "an LLM chat. The shape and register of the brief are "
    "assigned per-example in the **Conversation register "
    "(REQUIRED)** section below — follow that directive, do not "
    "pick your own. The content is purely factual about the "
    "product the user is trying to sell — zero creative "
    "knowledge. The user describes what the product is, what it "
    "does, what it contains, what it comes in — the things a "
    "product owner knows about their own product. No tone "
    "guidance, no audience framing, no suggested angles, no "
    "phrasing copied from the ad.\n"
    "\n"
    "- **Assistant response** — the assistant answers with the "
    "ad copy presented as part of the response, the way any "
    "helpful LLM would. Present the copy the way a reader would "
    "see it. The opener and prose register are also governed by "
    "the **Conversation register (REQUIRED)** section below — "
    "the response register matches the brief register for the "
    "example. Deliver the ad, then a rationale grounded in "
    "specific, visible details of the ad (the word, the comma, "
    "the sequence, the line break), not abstract copywriting "
    "principles. **Rationale depth is constant across all "
    "examples and all registers**: always provide your most "
    "thorough, ad-grounded analysis — hook, structure, sequence, "
    "specific word / punctuation / line-break choices, why each "
    "lands, audience read. The conversation register changes the "
    "*prose voice* (casual vs. organized vs. clipped); it never "
    "shortens or shallows the analysis itself. Commit to the ad; "
    "no alternatives. No field labels in prose ('Headline:', "
    "'Body:', 'Description:'). Preserve all source copy "
    "verbatim — emojis, punctuation, and line breaks are "
    "load-bearing."
)


@dataclass
class BundleContext:
    """All the rolled dice + inputs for one copywriting example.

    The scaffolding fields (``persona``, ``seed_idx``, ``seed_text``,
    ``evol_op``, ``difficulty``, ``turn_structure``, ``followup_type``)
    are carried for provenance metadata on :class:`ExampleMetadata` but
    are not rendered into the teacher prompt — backtranslation's voice
    and shape are inferred from the source ad itself.
    """

    task_format: TaskFormat
    style: PromptStyle
    persona: Persona
    seed_idx: int
    seed_text: str
    evol_op: str | None
    source_ads: list[ScoredAd] = field(default_factory=list)
    formatted_ads: str = ""
    response_format: str = ""  # unused for backtranslation
    difficulty: str = "standard"
    turn_structure: str = "single"
    followup_type: str = ""
    provider: str = ""
    # Copywriting-specific derived context (source_ad_shape +
    # conversation_register — the 2026-04 audit dropped platform_framing as
    # scraping-artifact noise).
    copywriting_context: CopywritingContext | None = None


def build_bundle(ctx: BundleContext) -> str:
    """Render the full prompt block for the chat agent."""
    parts: list[str] = []
    parts.append(f"# Training-example generation — task: {ctx.task_format.value}")
    parts.append("")
    parts.append(
        "You are helping generate one high-quality training example for "
        "a marketing-reasoning fine-tune. Follow the instructions below "
        "precisely. Your output MUST use the exact tag structure at the "
        "bottom — nothing else will be saved."
    )
    parts.append("")

    parts.append("## Style rules")
    parts.append(BACKTRANSLATION_STYLE_RULES)
    parts.append("")

    # Per-format axes block (copywriting's ad-derived context directive).
    parts.extend(get_pipeline(ctx.task_format).render_axes_block(ctx))

    parts.append("## Source ad (the gold target — preserve its copy in the response)")
    parts.append(ctx.formatted_ads or "(none provided)")
    parts.append("")

    # Output format.
    parts.append("## Output format")
    parts.append("Return these tags in order. No text before, between, or after.")
    parts.append("")
    parts.append(f"{USER_PROMPT_OPEN}[the brief]{USER_PROMPT_CLOSE}")
    parts.append("")
    parts.append(f"{ASSISTANT_RESPONSE_OPEN}[the response]{ASSISTANT_RESPONSE_CLOSE}")
    parts.append("")

    return "\n".join(parts)


@dataclass
class ParsedBundleOutput:
    """Extracted fields from a chat-agent bundle response."""

    user_prompt: str = ""
    assistant_response: str = ""


def _extract_tag(text: str, open_tag: str, close_tag: str) -> str:
    """Extract the first occurrence of content between open/close tags."""
    start = text.find(open_tag)
    if start < 0:
        return ""
    start += len(open_tag)
    end = text.find(close_tag, start)
    if end < 0:
        return ""
    return text[start:end].strip()


def parse_bundle_output(response_text: str) -> ParsedBundleOutput:
    """Parse the chat agent's tagged response into its fields."""
    return ParsedBundleOutput(
        user_prompt=_extract_tag(response_text, USER_PROMPT_OPEN, USER_PROMPT_CLOSE),
        assistant_response=_extract_tag(
            response_text,
            ASSISTANT_RESPONSE_OPEN,
            ASSISTANT_RESPONSE_CLOSE,
        ),
    )


def make_rng(seed: int, generated_count: int, prompt_index: int) -> random.Random:
    """Deterministic per-example RNG.

    Advancing with ``generated_count`` means resuming a run produces
    fresh rolls rather than repeating. Adding ``prompt_index`` gives
    each prompt in a batch its own stream.
    """
    return random.Random(seed + generated_count * 1000 + prompt_index)
