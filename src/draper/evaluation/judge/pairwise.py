"""Pairwise judge: ask an LLM (OpenAI, Anthropic, or Gemini) to score two responses on 5 dimensions.

Run BOTH orderings to control for position bias. Reconciliation lives in
``reconcile_pair`` — a clean win for one config requires both orderings to
agree (or at least not contradict).

Provider dispatch is model-prefix-based (see ``judge/clients.py``):
  - ``claude-*``  → Anthropic, via tool-use forcing for structured output.
  - ``gemini-*``  → Google Gemini, via ``response_schema`` config.
  - everything else → OpenAI, via strict ``json_schema`` response format.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from anthropic.types import ToolUseBlock
from google.genai import types as gtypes

from ..schemas import (
    Inference,
    JudgePerDimension,
    Judgment,
    PairResult,
    Winner,
)
from .clients import (
    ANTHROPIC_MAX_TOKENS,
    GEMINI_MAX_OUTPUT_TOKENS,
    OPENAI_MAX_TOKENS,
    anthropic_client,
    clip_score,
    gemini_client,
    gemini_compat_schema,
    is_claude,
    is_gemini,
    openai_client,
)
from .normalize import judge_input_text
from .prompts import (
    PAIRWISE_JSON_SCHEMA,
    PAIRWISE_SYSTEM,
    build_pairwise_user_prompt,
)

logger = logging.getLogger(__name__)


async def _call_openai_judge(
    *,
    model: str,
    system: str,
    user: str,
) -> tuple[dict[str, Any], str]:
    # OpenAI strict json_schema mode guarantees valid JSON output so no
    # JSONDecodeError fallback is needed here (unlike the Gemini path where
    # reasoning tokens can cause truncation).
    client = openai_client()
    resp = await client.chat.completions.create(  # type: ignore[call-overload]
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": PAIRWISE_JSON_SCHEMA},
        temperature=0.0,
        max_completion_tokens=OPENAI_MAX_TOKENS,
    )
    raw = resp.choices[0].message.content or "{}"
    return cast(dict[str, Any], json.loads(raw)), raw


async def _call_anthropic_judge(
    *,
    model: str,
    system: str,
    user: str,
) -> tuple[dict[str, Any], str]:
    """Force a structured judgment via tool-use.

    Anthropic's standard pattern for JSON output is a single tool with a
    full input_schema and ``tool_choice={"type": "tool", "name": ...}``;
    the model fills the schema and we parse the tool_use block. This is
    more reliable than asking for raw JSON in the prompt.

    If the model refuses or returns no tool-use block (edge case with safety
    filters or API issues), logs a warning and returns an empty dict that
    will be filled with clip defaults by the caller.
    """
    client = anthropic_client()
    schema = PAIRWISE_JSON_SCHEMA["schema"]
    response = await client.messages.create(
        model=model,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[
            {
                "name": "submit_judgment",
                "description": "Submit your scored pairwise judgment.",
                "input_schema": cast(dict[str, Any], schema),
            }
        ],
        tool_choice={"type": "tool", "name": "submit_judgment"},
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
            f"Anthropic judge ({model}) returned no tool-use block. "
            "This may indicate a refusal or safety filter. "
            "Defaulting to empty judgment (will be filled with clip defaults)."
        )
    return data, json.dumps(data)


async def _call_gemini_judge(
    *,
    model: str,
    system: str,
    user: str,
) -> tuple[dict[str, Any], str]:
    client = gemini_client()
    cfg = gtypes.GenerateContentConfig(
        temperature=0.0,
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=gemini_compat_schema(PAIRWISE_JSON_SCHEMA["schema"]),
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
    )
    resp = await client.aio.models.generate_content(
        model=model,
        contents=user,
        config=cfg,
    )
    candidates = resp.candidates or []
    raw = ""
    if candidates and candidates[0].content:
        parts = candidates[0].content.parts or []
        raw = "".join(p.text for p in parts if p.text is not None)
    return cast(dict[str, Any], json.loads(raw or "{}")), raw


async def judge_pair(
    *,
    example_id: str,
    platform: str,
    vertical: str,
    user_prompt: str,
    a: Inference,
    b: Inference,
    judge_model: str,
    swap: bool = True,
    clean_root: Path | None = None,
) -> list[Judgment]:
    """Run one or two judge calls (forward + swapped) for a single pair.

    Returns a list of Judgment objects, one per ordering. Each Judgment's
    ``pair_a`` / ``pair_b`` reflects what the judge actually saw — caller
    is responsible for reconciling these into a canonical PairResult.

    ``clean_root`` enables the LLM ad-copy extractor: when provided, the
    judge sees pre-extracted plain ad copy from
    ``inferences_clean/<config>/<example_id>.json`` instead of the regex
    fallback ``clean_copy(assistant_text)``. Falls through to the regex
    path on a per-example basis when the cleaned cache is missing.
    """
    judgments: list[Judgment] = []
    orderings: list[tuple[Inference, Inference, bool]] = [(a, b, False)]
    if swap:
        orderings.append((b, a, True))

    for left, right, is_swapped in orderings:
        # Resolve what the judge actually sees: cleaned LLM-extracted text
        # when available (uniform shape across configs), else regex fallback.
        # Raw assistant_text on disk is left untouched for forensics.
        user = build_pairwise_user_prompt(
            platform=platform,
            vertical=vertical,
            user_prompt=user_prompt,
            response_a=judge_input_text(left, clean_root=clean_root),
            response_b=judge_input_text(right, clean_root=clean_root),
        )
        if is_gemini(judge_model):
            data, raw = await _call_gemini_judge(
                model=judge_model, system=PAIRWISE_SYSTEM, user=user
            )
        elif is_claude(judge_model):
            data, raw = await _call_anthropic_judge(
                model=judge_model, system=PAIRWISE_SYSTEM, user=user
            )
        else:
            data, raw = await _call_openai_judge(
                model=judge_model, system=PAIRWISE_SYSTEM, user=user
            )
        per_dim = JudgePerDimension(
            strategic_relevance=clip_score(data.get("strategic_relevance", 0)),
            creativity=clip_score(data.get("creativity", 0)),
            actionability=clip_score(data.get("actionability", 0)),
            channel_appropriateness=clip_score(data.get("channel_appropriateness", 0)),
            predicted_performance=clip_score(data.get("predicted_performance", 0)),
        )
        winner_raw = data.get("overall_winner", "tie")
        winner: Winner = winner_raw if winner_raw in ("A", "B", "tie") else "tie"
        judgments.append(
            Judgment(
                example_id=example_id,
                pair_a=left.config,
                pair_b=right.config,
                swap_order=is_swapped,
                judge_model=judge_model,
                per_dim=per_dim,
                overall_winner=winner,
                rationale=str(data.get("rationale", "")),
                raw_response=raw,
                created_at=datetime.now(UTC),
            )
        )
    return judgments


def reconcile_pair(
    *,
    example_id: str,
    config_a: str,
    config_b: str,
    judgments: list[Judgment],
    judge_model: str,
) -> PairResult:
    """Combine forward + swapped judgments into a PairResult in canonical (a,b) form.

    The judge always called the inputs "Response A" / "Response B"; in the
    swapped ordering A==config_b. We map each judgment's winner back to the
    canonical (config_a, config_b) frame, then reconcile.

    Tie-break rule:
      - If both orderings agree (after un-swapping), that's the winner.
      - If they disagree → ``order_dependent=True``; use the per-dim sum
        across both orderings (in canonical frame) to break the tie. If
        that sum is exactly zero, declare ``tie``.
    """
    forward = next((j for j in judgments if not j.swap_order), None)
    swapped = next((j for j in judgments if j.swap_order), None)
    if forward is None:
        raise ValueError(f"Missing forward judgment for {example_id}")

    forward_canonical = _winner_to_canonical(forward, config_a, config_b)
    if swapped is None:
        # Single-ordering path (cross-val or budget-constrained): take it as-is.
        return PairResult(
            example_id=example_id,
            config_a=config_a,
            config_b=config_b,
            judge_model=judge_model,
            forward_winner=forward_canonical,
            swapped_winner="tie",
            resolved_winner=forward_canonical,
            order_dependent=False,
            per_dim_sum=_canonical_per_dim(forward, config_a),
        )

    swapped_canonical = _winner_to_canonical(swapped, config_a, config_b)

    if forward_canonical == swapped_canonical:
        resolved: Winner = forward_canonical
        order_dependent = False
    else:
        # Order-dependent — break by summed per-dim scores in canonical frame.
        canonical_sum = _sum_canonical_per_dim([forward, swapped], config_a)
        total = (
            canonical_sum.strategic_relevance
            + canonical_sum.creativity
            + canonical_sum.actionability
            + canonical_sum.channel_appropriateness
            + canonical_sum.predicted_performance
        )
        if total > 0:
            resolved = "A"
        elif total < 0:
            resolved = "B"
        else:
            resolved = "tie"
        order_dependent = True

    per_dim_sum = _sum_canonical_per_dim([forward, swapped], config_a)
    return PairResult(
        example_id=example_id,
        config_a=config_a,
        config_b=config_b,
        judge_model=judge_model,
        forward_winner=forward_canonical,
        swapped_winner=swapped_canonical,
        resolved_winner=resolved,
        order_dependent=order_dependent,
        per_dim_sum=per_dim_sum,
    )


def _winner_to_canonical(j: Judgment, config_a: str, config_b: str) -> Winner:
    """Map a judgment's 'A'/'B' verdict back to canonical (config_a, config_b).

    In the forward ordering, j.pair_a == config_a, so 'A' → 'A'.
    In the swapped ordering, j.pair_a == config_b, so the judge's 'A'
    verdict is actually a win for config_b → canonical 'B'.
    """
    if j.overall_winner == "tie":
        return "tie"
    if j.pair_a == config_a:
        return j.overall_winner
    # Swapped: judge's A is canonical B and vice versa.
    return "B" if j.overall_winner == "A" else "A"


def _canonical_per_dim(j: Judgment, config_a: str) -> JudgePerDimension:
    """Return per-dim scores in canonical (A=config_a) frame.

    The judge always scores from its 'A' POV; in the swapped ordering we
    flip the sign so positive = config_a is better.
    """
    sign = 1 if j.pair_a == config_a else -1
    pd = j.per_dim
    return JudgePerDimension(
        strategic_relevance=sign * pd.strategic_relevance,
        creativity=sign * pd.creativity,
        actionability=sign * pd.actionability,
        channel_appropriateness=sign * pd.channel_appropriateness,
        predicted_performance=sign * pd.predicted_performance,
    )


def _sum_canonical_per_dim(judgments: list[Judgment], config_a: str) -> JudgePerDimension:
    s = JudgePerDimension(
        strategic_relevance=0,
        creativity=0,
        actionability=0,
        channel_appropriateness=0,
        predicted_performance=0,
    )
    for j in judgments:
        canonical = _canonical_per_dim(j, config_a)
        s = JudgePerDimension(
            strategic_relevance=s.strategic_relevance + canonical.strategic_relevance,
            creativity=s.creativity + canonical.creativity,
            actionability=s.actionability + canonical.actionability,
            channel_appropriateness=s.channel_appropriateness + canonical.channel_appropriateness,
            predicted_performance=s.predicted_performance + canonical.predicted_performance,
        )
    return s
