"""LEGACY (two-stage) — Stage 2 rationale-generation prompt.

The production v2 pipeline uses the unified single-pass teacher in
``teacher/single_pass.py``. This module is retained until Phase 4
removes the two-stage path.

The teacher sees both the Brief AND the source ad verbatim. It must:

1. Write a first-person ``<think>`` block as if it were Draper itself,
   holding ONLY the brief, deciding how to write the ad. The source
   ad is the destination of this reasoning, never its subject. See
   ``feedback_v2_think_brief_only`` in user memory.
2. After ``</think>``, write Draper's response to the user. The
   source ad appears verbatim somewhere within that response (no
   paraphrase, no wrapping tags or fences around it).

The cardinal verbatim-emission rule is borrowed from v1's
``BACKTRANSLATION_STYLE_RULES`` — well-stress-tested against teacher
content-policy refusals.
"""

from __future__ import annotations

from draper.construction.batch.types import BatchRequest
from draper.construction_v2.config import RationaleConfig
from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.schemas.brief import Brief, canonical_json

RATIONALE_TEACHER_SYSTEM = (
    "You write training examples for Draper, a marketing specialist. "
    "This dataset is the ad-copy slice of that role — each example "
    "teaches Draper to read a brief and produce ad copy in the voice "
    "and shape the brief calls for.\n"
    "\n"
    "Each example is a single assistant response with ONE required "
    "structural region (``<think>...</think>``) followed by Draper's "
    "response to the user, which contains the source ad verbatim.\n"
    "\n"
    "You are given:\n"
    "- A **structured Brief** (task + product facts + strategic bridge "
    "fields).\n"
    "- The **source ad copy** that ad-performance data shows was high "
    "performing.\n"
    "\n"
    "## How to write `<think>` (the private internal block)\n"
    "\n"
    "**Persona switch: for this block you ARE Draper. Reason ONLY "
    "from the brief in your input, not from the source ad.** The ad "
    "does not exist yet from Draper's seat — he is about to write "
    "it. **The source ad is not yet known to Draper. Referencing "
    "its specific phrases, emoji, or punctuation inside `<think>` "
    "is wrong by construction.** First-person decisional voice, "
    "anchored to populated brief fields.\n"
    "\n"
    "## The response (after `</think>`)\n"
    "\n"
    "This region is Draper's response to the user — the full reply, "
    "end to end. The source ad appears verbatim somewhere within it "
    "(preserve emojis, punctuation, capitalization, line breaks; no "
    "fences, no labels, no draft framing). Compose the response as "
    "Draper actually answering the task — not as wrapping around an "
    "artifact, but as the actual reply, which contains the ad.\n"
    "\n"
    "## **MOST IMPORTANT RULES**\n"
    "\n"
    "**1. The source ad must appear word-for-word within Draper's "
    "response (after ``</think>``).** Do not paraphrase or "
    "restructure it.\n"
    "\n"
    "**1a. Inside ``<think>``, Draper writes from inside the act** — "
    "he doesn't reference his own response or the ad as an artifact "
    "he's about to deliver.\n"
    "\n"
    "**2. Output shape is exactly this:**\n"
    "```\n"
    "<think>\n"
    "{first-person decisional reasoning from brief content only}\n"
    "</think>\n"
    "\n"
    "{Draper's response to the user — the full reply, "
    "containing the source ad verbatim within it}\n"
    "```\n"
    "\n"
    "Nothing before ``<think>``. No markdown fences. No labels "
    "announcing the regions.\n"
    "\n"
    "**3. Do not emit the brief or any structured fields.** The brief "
    "is the input. The output is `<think>` + Draper's response "
    "(which contains the ad verbatim)."
)


def _user_message(brief: Brief, ad: SourceAd) -> str:
    """User-turn payload combining brief JSON + ad copy verbatim."""
    brief_json = canonical_json(brief)
    # Ad copy is a single text, not split by API-level field. Those labels
    # (HEADLINE / BODY / DESCRIPTION / CTA) come from the scraper and don't
    # reliably reflect what's actually visible in the creative — and the
    # teacher's output doesn't carry the structure either.
    return (
        "## Brief (canonical JSON)\n"
        "\n"
        f"{brief_json}\n"
        "\n"
        "## Source ad copy (must appear verbatim within Draper's response)\n"
        "\n"
        f"{ad.ad_copy_text}\n"
        "\n"
        "## Task\n"
        "\n"
        "Emit `<think>...</think>` (first-person decisional reasoning), "
        "then Draper's response to the task — composed naturally and "
        "containing the source ad verbatim somewhere within it. No "
        "wrapping tags or fences around the ad. Nothing before `<think>`."
    )


def build_rationale_request(brief: Brief, ad: SourceAd, config: RationaleConfig) -> BatchRequest:
    """Build one stage-2 :class:`BatchRequest` for ``(brief, ad)``."""
    return BatchRequest(
        custom_id=f"rationale-{ad.ad_id}",
        system=RATIONALE_TEACHER_SYSTEM,
        messages=[{"role": "user", "content": _user_message(brief, ad)}],
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )


def build_rationale_messages(brief: Brief, ad: SourceAd) -> list[dict[str, str]]:
    """Plain messages list for chat-mode (non-batch) calls."""
    return [{"role": "user", "content": _user_message(brief, ad)}]


__all__ = [
    "RATIONALE_TEACHER_SYSTEM",
    "build_rationale_messages",
    "build_rationale_request",
]
