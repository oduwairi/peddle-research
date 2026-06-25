"""Pydantic schemas for the eval pipeline (briefs, inferences, judgments)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JudgeDimension = Literal[
    "strategic_relevance",
    "creativity",
    "actionability",
    "channel_appropriateness",
    "predicted_performance",
]
JUDGE_DIMENSIONS: tuple[JudgeDimension, ...] = (
    "strategic_relevance",
    "creativity",
    "actionability",
    "channel_appropriateness",
    "predicted_performance",
)

Winner = Literal["A", "B", "tie"]


class Brief(BaseModel):
    """A single copywriting test brief loaded by ``briefs.load_test_briefs``.

    Mirrors the chat-format example: a system + user prompt the model has to
    answer. The reference assistant message is the real ad copy that the
    backtranslation teacher produced the brief from — kept for inspection
    but never shown to inference models.

    Two on-disk shapes are supported:
      * v1 (``data/final/test/``, 215 briefs): ``vertical``, ``task_format``,
        and ``source_tiers`` are populated columns.
      * v2 (``data/constructed_v2/final_v2/test``, 228 briefs, active default):
        those columns are absent from the Arrow dataset; the loader defaults
        ``vertical`` → ``"unknown"``, ``task_format`` → ``"copywriting"``, and
        ``source_tiers`` → ``[]`` for all v2 briefs. Consequence: ``--groupby
        vertical`` or ``--groupby source_tier`` aggregates collapse to a single
        bucket on v2; only ``--groupby platform`` is meaningful there.
    """

    example_id: str
    task_format: str
    platform: str
    vertical: str
    source_tiers: list[str] = Field(default_factory=list)
    construction_model: str | None = None
    system: str
    user: str
    reference_assistant: str


class UrlScenario(BaseModel):
    """A fresh URL-anchored scenario for Arm 2 (full-pipeline eval).

    These are *not* in the training distribution. The model gets a free-form
    user message ("write Meta ad for https://example.com/foo") and the
    pipeline is expected to use scrape_url + web_search to ground itself.
    """

    scenario_id: str
    url: str
    platform: str
    vertical: str
    user_prompt: str
    notes: str | None = None


class Inference(BaseModel):
    """One model's response to a brief or scenario."""

    example_id: str
    config: str
    arm: Literal["arm1", "arm2"]
    brief: str  # serialized brief (for debugging)
    system: str
    assistant_text: str
    rationale: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    raw_traces: list[dict[str, Any]] | None = None
    campaign: dict[str, Any] | None = None
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str
    error: str | None = None
    created_at: datetime


class JudgePerDimension(BaseModel):
    """Per-dimension score from A's perspective, in [-2, +2]."""

    strategic_relevance: int
    creativity: int
    actionability: int
    channel_appropriateness: int
    predicted_performance: int


class Judgment(BaseModel):
    """One pairwise judge call (one ordering)."""

    example_id: str
    pair_a: str  # config name shown as "Response A" to the judge
    pair_b: str  # config name shown as "Response B"
    swap_order: bool  # True if (pair_a, pair_b) is the swapped ordering
    judge_model: str
    per_dim: JudgePerDimension
    overall_winner: Winner  # from the judge's POV: A | B | tie
    rationale: str
    raw_response: str | None = None
    created_at: datetime


class PairResult(BaseModel):
    """Aggregated result of pairwise judging across both orderings.

    Resolves position bias: a pair is a clean win for one config only if
    both orderings agree; otherwise it's flagged as order-dependent.
    """

    example_id: str
    config_a: str  # the "left" config in the canonical (a < b) ordering
    config_b: str
    judge_model: str
    forward_winner: Winner  # judgment of (a, b) ordering, mapped to canonical
    swapped_winner: Winner  # judgment of (b, a) ordering, mapped to canonical
    resolved_winner: Winner  # final after order reconciliation
    order_dependent: bool
    per_dim_sum: JudgePerDimension  # summed across both orderings, A's POV
