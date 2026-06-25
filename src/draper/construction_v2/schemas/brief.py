"""Brief schema, canonical JSON serializer, and static system prompt.

This module is the single source of truth for the v2 brief format. The
frontend's ``serializeBriefForDraper`` (TS) must produce byte-identical
output against the same input — enforced by ``tests/contract/`` fixtures.

Three load-bearing exports:

- :class:`Brief` (with :class:`BriefProduct`, :class:`BriefBridge`) — the
  pydantic models the teacher emits during stage 1 and the frontend
  serializes at inference.
- :func:`canonical_json` — deterministic JSON dump (sorted keys, no
  ``null`` elision, no Unicode escaping, no trailing whitespace).
- :data:`STATIC_SYSTEM_PROMPT` — the system prompt baked into both
  training and inference. Byte-identical or the writer drifts off
  distribution.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Platforms the trained writer is expected to handle. Single source of
# truth — the frontend's serializer must accept the same set verbatim.
SUPPORTED_PLATFORMS: tuple[str, ...] = (
    "meta",
    "tiktok",
    "x",
    "google",
    "pinterest",
    "reddit",
)


class BriefProduct(BaseModel):
    """Product facts the brief author (or extracting teacher) knows.

    Grounding contract: every field must be derivable from the source ad
    text or supplied metadata (advertiser_name, landing_page_url). A
    field whose value the ad does not support MUST be left null/empty —
    blanks are the correct, honest answer. ``tone_signals`` is the only
    required field because voice is always readable from any ad's text.
    """

    # Teacher LLMs frequently invent shortened field names in
    # natural-language prompts (e.g. ``what_it_is`` for ``description``,
    # ``usps`` for ``unique_selling_points``). Accept those variants via
    # ``validation_alias`` so the parser is robust across providers; the
    # canonical schema field name stays the only one serialized out.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str | None = Field(
        default=None,
        description=(
            "Product or service name. Use ``advertiser_name`` only when "
            "it is actually the product name; leave null if the ad does "
            "not reveal a product name."
        ),
    )
    description: str | None = Field(
        default=None,
        description=(
            "One-paragraph description of what the product is and does. "
            "Leave null if the ad/landing-page-url does not support a "
            "factual description."
        ),
        validation_alias=AliasChoices(
            "description", "what_it_is", "what_the_product_is", "summary"
        ),
    )
    category: str | None = Field(
        default=None,
        description=(
            "Broad market category. Leave null if not clearly inferable from the ad text."
        ),
    )
    key_features: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete features explicitly mentioned in the ad copy. "
            "Leave empty if the ad does not enumerate features."
        ),
        validation_alias=AliasChoices("key_features", "features"),
    )
    unique_selling_points: list[str] = Field(
        default_factory=list,
        description=(
            "Differentiators / positioning claims made explicitly in "
            "the ad copy. Leave empty if the ad doesn't make any."
        ),
        validation_alias=AliasChoices("unique_selling_points", "usps", "usp", "selling_points"),
    )
    price_info: str | None = Field(
        default=None,
        description=("Pricing only when stated in the ad copy. Never invent."),
        validation_alias=AliasChoices("price_info", "price", "pricing"),
    )
    tone_signals: list[str] = Field(
        ...,
        description=(
            "Voice cues read off the ad's natural voice (e.g. "
            '``"playful"``, ``"warm-but-direct"``, ``"clipped"``). '
            "REQUIRED, non-empty — this is the inference-time style "
            "anchor; the writer triangulates response shape from it."
        ),
    )
    category_context: str | None = Field(
        default=None,
        description=(
            "One-line landscape framing of the category. Leave null if "
            "not clearly readable from the ad."
        ),
    )
    proof_points: list[str] = Field(
        default_factory=list,
        description=(
            "Testimonials, numbers, or named entities mentioned IN the "
            "ad copy. Leave empty if the ad doesn't cite any."
        ),
    )
    offer: str | None = Field(
        default=None,
        description=("Promo / free trial / discount only when stated in the ad. Never invent."),
    )

    @field_validator("key_features", "unique_selling_points", "proof_points", mode="before")
    @classmethod
    def _coerce_null_list(cls, v: Any) -> Any:
        # Teachers correctly emit null for these fields when the ad doesn't
        # enumerate them (per the grounding contract). Coerce to empty list
        # so a null answer parses identically to default_factory=list.
        if v is None:
            return []
        return v

    @field_validator("category_context", mode="before")
    @classmethod
    def _coerce_category_context(cls, v: Any) -> Any:
        # Some teachers return a keyword list instead of the one-line
        # landscape frame. Join non-empty lists with " · "; empty list → null.
        if isinstance(v, list):
            cleaned = [str(item).strip() for item in v if item]
            return " · ".join(cleaned) if cleaned else None
        return v

    @field_validator("tone_signals")
    @classmethod
    def _tone_signals_non_empty(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if s and s.strip()]
        if not cleaned:
            msg = (
                "BriefProduct.tone_signals must contain at least one "
                "non-empty string; it is the inference-time style anchor."
            )
            raise ValueError(msg)
        return cleaned


class BriefBridge(BaseModel):
    """Strategic facts derived from the source ad.

    Never quote the ad's copy — these are facts ABOUT marketing intent.
    The leak guard enforces a 5-gram non-overlap with the source ad.

    ``angle`` and ``buyer_pain`` are required: any successful ad carries
    a readable creative direction and an addressed friction.
    ``positioning`` and ``target_audience`` are optional — they often
    require product context the ad alone doesn't expose, and leaving
    them null is preferable to fabricating them.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    positioning: str | None = Field(
        default=None,
        description=(
            'One-line strategic frame ("premium alternative to '
            'spreadsheet workflows"). Leave null if not readable from '
            "the ad."
        ),
        validation_alias=AliasChoices("positioning", "positioning_frame", "frame"),
    )
    target_audience: str | None = Field(
        default=None,
        description=(
            'Specific buyer persona ("Series-A marketing leads three '
            'months in"). Leave null if the ad does not imply one.'
        ),
        validation_alias=AliasChoices("target_audience", "audience"),
    )
    angle: str = Field(
        ...,
        description=(
            'Creative-direction label ("problem-aware skeptic", '
            '"aspirational founder identity"). REQUIRED — readable '
            "from any successful ad."
        ),
        validation_alias=AliasChoices("angle", "creative_angle", "creative_direction"),
    )
    buyer_pain: str = Field(
        ...,
        description=(
            'Concrete friction being addressed ("Sundays lost to '
            'payroll spreadsheets"). REQUIRED — the ad addresses '
            "something."
        ),
        validation_alias=AliasChoices("buyer_pain", "pain", "pain_point"),
    )


class Brief(BaseModel):
    """The full v2 brief: task + product facts + strategic bridge + platform.

    ``task`` is a free-form natural-language string naming the work the
    caller is asking for (e.g. ``"Write ad copy for the platform below."``).
    It is the brief's only discriminator between Draper's skills — the
    static system prompt is skill-agnostic, so the student learns the
    task-string → output-shape mapping from training data.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _hoist_top_level_tone_signals(cls, data: Any) -> Any:
        # Some teachers emit `tone_signals` at the top level of the
        # brief instead of nested under `product`. Move it down before
        # field validation runs so the canonical shape parses cleanly
        # without an extra_forbidden error at the top level.
        if isinstance(data, dict) and "tone_signals" in data:
            product = data.get("product")
            if isinstance(product, dict) and not product.get("tone_signals"):
                # Pydantic creates a shallow copy for `mode="before"` validation,
                # but the nested dicts may be shared. Shallow-copy both the top
                # level and product dict to avoid mutating a caller-owned dict
                # when we set the tone_signals.
                data = {**data}
                data["product"] = {**product, "tone_signals": data.pop("tone_signals")}
        return data

    task: str = Field(
        ...,
        description=(
            "Natural-language description of what the caller is asking "
            "Draper to produce. Free-form by design — no enum, no slug. "
            "A human or orchestrator should be able to write it directly."
        ),
    )
    product: BriefProduct
    bridge: BriefBridge
    platform: Literal["meta", "tiktok", "x", "google", "pinterest", "reddit"]


def canonical_dict_json(payload: dict[str, Any]) -> str:
    """Serialize an already-dumped brief dict to canonical JSON.

    Contract (shared by every v2 skill — copywriting and image_brief —
    so the byte form is identical regardless of which brief shape it is):

    - Sorted keys at every level.
    - ``null`` fields preserved (no elision) so the byte form is stable
      across optional-field presence/absence flips.
    - ``ensure_ascii=False`` — UTF-8 source bytes preserved (matches the
      TS ``JSON.stringify`` default behavior).
    - No trailing newline. No extra spacing — ``separators=(",", ":")``.

    Operates on a plain dict (typically ``model.model_dump(mode="json")``)
    so the dataset builder + quality filter can render any skill's brief
    without importing that skill's pydantic model.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_json(brief: Brief) -> str:
    """Serialize a copywriting :class:`Brief` to canonical JSON.

    Thin wrapper over :func:`canonical_dict_json`. Pydantic enum values
    are dumped as their string ``.value``. The frontend's
    ``serializeBriefForDraper`` (TS) must produce byte-identical output.
    Verified by ``tests/contract/``.
    """
    return canonical_dict_json(brief.model_dump(mode="json"))


# Byte-identical training + inference system prompt. Both the training
# data row (`system` role content) and the frontend's `system` argument
# to the writer must use this exact string. Any drift and the writer is
# off-distribution.
#
# Locked at design time; changes here require a coordinated training
# rerun + frontend deploy.
STATIC_SYSTEM_PROMPT = (
    "You are Draper, a senior marketing specialist with deep "
    "operational experience in performance-driven creative work. "
    "You have shipped creative against real spend, in-house and on "
    "the agency side, and your craft is grounded in evidence — you "
    "know the difference between what wins and what dies is rarely "
    "cleverness and almost always the right read of the audience, "
    "the moment, and the platform.\n"
    "\n"
    "You approach each piece of work the way a thoughtful "
    "practitioner does. You read the brief the caller hands you, "
    "honor what it tells you, and work strictly within what the "
    "brief supports — empty fields are not invitations to invent. "
    "The caller on the other side of the wire — founder, marketer, "
    "agent — is a peer, not an audience for performance.\n"
    "\n"
    "Before producing any deliverable, you think the work through "
    "in a ``<think>...</think>`` block: first-person, present-tense "
    "reasoning in the voice of a practitioner at the desk, "
    "narrating your decisions as you make them. Weigh tradeoffs, "
    "discard options you considered, and ground choices in fields "
    "the brief actually populated. The block is hidden from the end "
    "user by convention.\n"
    "\n"
    "After ``</think>``, produce the deliverable the brief calls "
    "for. Voice and length follow from the brief's tone signals and "
    "the nature of the work. Peer-to-peer professional voice "
    "throughout — a copywriter answering a founder in chat. Brief "
    "framing before the work, the work itself, and a short note on "
    "why the choices land are all welcome; or just deliver the work "
    "clean. Skip greetings and apologies."
)


__all__ = [
    "SUPPORTED_PLATFORMS",
    "STATIC_SYSTEM_PROMPT",
    "Brief",
    "BriefBridge",
    "BriefProduct",
    "canonical_dict_json",
    "canonical_json",
]
