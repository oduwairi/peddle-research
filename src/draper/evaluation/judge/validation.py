"""LLM-as-judge methodology validation against external ground truth.

Asks the production judge model to pick the better of two Upworthy headline
variants and compares the prediction against the real A/B-test winner
(highest CTR). Reports accuracy with a binomial p-value and Wilson 95% CI.

This is methodology validation, not output validation — the question is
"do we trust the judge enough to read its verdicts on Draper's outputs?"
If a judge can't beat 50%/chance on real A/B winners, its preference between
two ad-copy responses is noise.

Position bias is controlled the same way as production: each pair is shown
in both orderings; an example only counts as "correctly predicted" if both
orderings agree on the actual winner.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from anthropic.types import ToolUseBlock
from google.genai import types as gtypes

from ..upworthy_loader import UpworthyVariant
from .clients import (
    ANTHROPIC_MAX_TOKENS,
    GEMINI_MAX_OUTPUT_TOKENS,
    OPENAI_MAX_TOKENS,
    anthropic_client,
    gemini_client,
    gemini_compat_schema,
    is_claude,
    is_gemini,
    openai_client,
)

logger = logging.getLogger(__name__)


HEADLINE_JUDGE_SYSTEM = """\
You are evaluating two news/feature headlines that ran as A/B test variants
on the same article. Your job: predict which headline got more clicks.

Score on click-likelihood only — concrete hooks, curiosity gaps, specific
numbers, emotional resonance. Do NOT reward verbosity, padding, or generic
clickbait. Pick "A", "B", or "tie" if they're genuinely indistinguishable.

Output strict JSON matching the schema."""


HEADLINE_JUDGE_USER_TEMPLATE = """\
# Headline A
{headline_a}

# Headline B
{headline_b}

Which headline got more clicks? Output strict JSON."""


HEADLINE_JUDGE_SCHEMA: dict[str, Any] = {
    "name": "headline_judgment",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "winner": {"type": "string", "enum": ["A", "B", "tie"]},
            "rationale": {"type": "string"},
        },
        "required": ["winner", "rationale"],
    },
    "strict": True,
}


@dataclass
class HeadlinePrediction:
    """One judge prediction for one pair, with both orderings reconciled."""

    winner_text: str  # the actual winner's headline (ground truth)
    loser_text: str
    forward_pick: str  # "A" / "B" / "tie" with winner=A
    swapped_pick: str  # "A" / "B" / "tie" with winner=B
    resolved: str  # "winner", "loser", or "tie" (after reconciling orderings)
    order_dependent: bool


@dataclass
class JudgeValidationResult:
    """Aggregated validation result for one judge model on one stream."""

    judge_model: str
    source: str
    n_pairs: int
    n_predicted_winner: int
    n_predicted_loser: int
    n_ties: int
    n_order_dependent: int
    accuracy: float  # n_predicted_winner / n_decisive
    accuracy_ci: tuple[float, float]
    binomial_p_value: float
    predictions: list[HeadlinePrediction] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "judge_model": self.judge_model,
            "source": self.source,
            "n_pairs": self.n_pairs,
            "n_predicted_winner": self.n_predicted_winner,
            "n_predicted_loser": self.n_predicted_loser,
            "n_ties": self.n_ties,
            "n_order_dependent": self.n_order_dependent,
            "accuracy": round(self.accuracy, 4),
            "accuracy_ci": [
                round(self.accuracy_ci[0], 4),
                round(self.accuracy_ci[1], 4),
            ],
            "binomial_p_value": self.binomial_p_value,
        }


async def _call_openai_headline_judge(*, model: str, headline_a: str, headline_b: str) -> str:
    # Headline validation uses a smaller token budget than pairwise judging:
    # the output schema is just {winner, rationale} (2 fields vs 7), so
    # OPENAI_MAX_TOKENS (512) is already generous — we use it for consistency.
    client = openai_client()
    user = HEADLINE_JUDGE_USER_TEMPLATE.format(
        headline_a=headline_a or "(empty)",
        headline_b=headline_b or "(empty)",
    )
    resp = await client.chat.completions.create(  # type: ignore[call-overload]
        model=model,
        messages=[
            {"role": "system", "content": HEADLINE_JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": HEADLINE_JUDGE_SCHEMA},
        temperature=0.0,
        max_completion_tokens=OPENAI_MAX_TOKENS,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = cast(dict[str, Any], json.loads(raw))
    except json.JSONDecodeError:
        return "tie"
    pick = str(data.get("winner", "tie"))
    return pick if pick in ("A", "B", "tie") else "tie"


async def _call_gemini_headline_judge(*, model: str, headline_a: str, headline_b: str) -> str:
    client = gemini_client()
    user = HEADLINE_JUDGE_USER_TEMPLATE.format(
        headline_a=headline_a or "(empty)",
        headline_b=headline_b or "(empty)",
    )
    cfg = gtypes.GenerateContentConfig(
        temperature=0.0,
        system_instruction=HEADLINE_JUDGE_SYSTEM,
        response_mime_type="application/json",
        response_schema=gemini_compat_schema(HEADLINE_JUDGE_SCHEMA["schema"]),
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
    )
    resp = await client.aio.models.generate_content(model=model, contents=user, config=cfg)
    candidates = resp.candidates or []
    raw = ""
    if candidates and candidates[0].content:
        parts = candidates[0].content.parts or []
        raw = "".join(p.text for p in parts if p.text is not None)
    try:
        data = cast(dict[str, Any], json.loads(raw or "{}"))
    except json.JSONDecodeError:
        # Truncated / malformed output — treat as a tie rather than crash
        # the whole batch. The aggregation already handles ties cleanly.
        return "tie"
    pick = str(data.get("winner", "tie"))
    return pick if pick in ("A", "B", "tie") else "tie"


async def _call_anthropic_headline_judge(*, model: str, headline_a: str, headline_b: str) -> str:
    """Call Anthropic headline judge via tool-use.

    If the model refuses or returns no tool-use block, logs a warning
    and defaults to "tie" so validation continues.
    """
    client = anthropic_client()
    user = HEADLINE_JUDGE_USER_TEMPLATE.format(
        headline_a=headline_a or "(empty)",
        headline_b=headline_b or "(empty)",
    )
    response = await client.messages.create(
        model=model,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        temperature=0.0,
        system=HEADLINE_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        tools=[
            {
                "name": "submit_headline_judgment",
                "description": "Submit which headline got more clicks.",
                "input_schema": cast(dict[str, Any], HEADLINE_JUDGE_SCHEMA["schema"]),
            }
        ],
        tool_choice={"type": "tool", "name": "submit_headline_judgment"},
    )
    data: dict[str, Any] = {}
    found_tool = False
    for block in response.content:
        if isinstance(block, ToolUseBlock):
            data = cast(dict[str, Any], block.input)
            found_tool = True
            break
    if not found_tool:
        logger.warning(
            f"Anthropic headline judge ({model}) returned no tool-use block. Defaulting to 'tie'."
        )
    pick = str(data.get("winner", "tie"))
    return pick if pick in ("A", "B", "tie") else "tie"


async def _predict_headline_pair(
    *,
    judge_model: str,
    headline_a: str,
    headline_b: str,
) -> str:
    if is_gemini(judge_model):
        return await _call_gemini_headline_judge(
            model=judge_model, headline_a=headline_a, headline_b=headline_b
        )
    if is_claude(judge_model):
        return await _call_anthropic_headline_judge(
            model=judge_model, headline_a=headline_a, headline_b=headline_b
        )
    return await _call_openai_headline_judge(
        model=judge_model, headline_a=headline_a, headline_b=headline_b
    )


async def _judge_one_pair(
    *,
    judge_model: str,
    winner: UpworthyVariant,
    loser: UpworthyVariant,
) -> HeadlinePrediction:
    """Run both orderings and reconcile back to (winner, loser) frame."""
    forward = await _predict_headline_pair(
        judge_model=judge_model,
        headline_a=winner.headline,
        headline_b=loser.headline,
    )
    swapped = await _predict_headline_pair(
        judge_model=judge_model,
        headline_a=loser.headline,
        headline_b=winner.headline,
    )

    # In the forward ordering, winner is A; "A" pick means "judge picked the
    # winner". In the swapped ordering, winner is B; "B" pick means "judge
    # picked the winner". Translate both to {winner|loser|tie}.
    def _to_canonical(pick: str, winner_is_a: bool) -> str:
        if pick == "tie":
            return "tie"
        if winner_is_a:
            return "winner" if pick == "A" else "loser"
        return "winner" if pick == "B" else "loser"

    fwd_canon = _to_canonical(forward, winner_is_a=True)
    swp_canon = _to_canonical(swapped, winner_is_a=False)

    if fwd_canon == swp_canon:
        resolved = fwd_canon
        order_dependent = False
    else:
        # Order-dependent: any disagreement counts as a tie for accuracy
        # purposes (we don't get to claim a "win" if the prediction depends
        # on which side we showed first).
        resolved = "tie"
        order_dependent = True

    return HeadlinePrediction(
        winner_text=winner.headline,
        loser_text=loser.headline,
        forward_pick=forward,
        swapped_pick=swapped,
        resolved=resolved,
        order_dependent=order_dependent,
    )


async def validate_judge_on_upworthy_pairs(
    *,
    judge_model: str,
    pairs: Sequence[tuple[UpworthyVariant, UpworthyVariant]],
    max_concurrency: int = 8,
    source: str = "upworthy",
) -> JudgeValidationResult:
    """Ask one judge model to predict A/B winners on Upworthy pairs.

    Each pair is judged in both orderings; only pairs where both orderings
    agree count toward accuracy. Order-dependent pairs are reported in
    ``n_order_dependent`` and excluded from the binomial denominator.

    Args:
        judge_model: e.g. ``"gpt-4o"`` or ``"gemini-2.5-pro"``.
        pairs: list of ``(winner, loser)`` UpworthyVariant tuples.
        max_concurrency: max in-flight judge calls.
        source: label for the ground-truth stream (default ``"upworthy"``).
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def _bounded(winner: UpworthyVariant, loser: UpworthyVariant) -> HeadlinePrediction:
        async with sem:
            return await _judge_one_pair(judge_model=judge_model, winner=winner, loser=loser)

    predictions = await asyncio.gather(*[_bounded(w, loser) for (w, loser) in pairs])

    n_predicted_winner = sum(1 for p in predictions if p.resolved == "winner")
    n_predicted_loser = sum(1 for p in predictions if p.resolved == "loser")
    n_ties = sum(1 for p in predictions if p.resolved == "tie" and not p.order_dependent)
    n_order_dependent = sum(1 for p in predictions if p.order_dependent)

    # Decisive = predictions that aren't ties or order-dependent.
    n_decisive = n_predicted_winner + n_predicted_loser
    accuracy = n_predicted_winner / n_decisive if n_decisive > 0 else 0.0

    # Binomial test against 0.5 chance baseline + Wilson 95% CI. We avoid a
    # hard dependency on the proxy_validation module's helpers by inlining
    # the small computations here.
    binomial_p, ci_low, ci_high = _binomial_and_wilson(n_predicted_winner, n_decisive)

    return JudgeValidationResult(
        judge_model=judge_model,
        source=source,
        n_pairs=len(pairs),
        n_predicted_winner=n_predicted_winner,
        n_predicted_loser=n_predicted_loser,
        n_ties=n_ties,
        n_order_dependent=n_order_dependent,
        accuracy=accuracy,
        accuracy_ci=(ci_low, ci_high),
        binomial_p_value=binomial_p,
        predictions=list(predictions),
    )


def _binomial_and_wilson(successes: int, n: int) -> tuple[float, float, float]:
    """One-sided binomial p-value (greater than 0.5) + Wilson 95% CI."""
    import math

    from scipy import stats as sp_stats

    if n == 0:
        return 1.0, 0.0, 0.0

    try:
        p_val = float(sp_stats.binomtest(successes, n, p=0.5, alternative="greater").pvalue)
    except AttributeError:
        # Fallback for older scipy versions.
        p_val = float(sp_stats.binom_test(successes, n, p=0.5, alternative="greater"))

    z = sp_stats.norm.ppf(0.975)
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return p_val, max(0.0, centre - half), min(1.0, centre + half)
