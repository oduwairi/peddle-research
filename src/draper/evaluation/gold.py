"""GOLD reference config — the real winning ad as just another contestant.

The held-out test split carries the original ad copy in
``Brief.reference_assistant``. For tournament eval we held it back from
both inference and judging; for **reference eval** we promote it to a
synthetic config named ``GOLD`` and pair it against any model's output.

GOLD is not a runner: it has no on-disk inferences. The judge driver
synthesizes a one-off ``Inference`` from the brief at judge time. Win-rate
vs GOLD is the headline arm-1 metric — does the model match a real ad
that already worked in the wild?
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from .schemas import Brief, Inference

GOLD_CONFIG: str = "GOLD"
"""Sentinel config name used in pair tuples and on-disk pair directory names.

The v1 split uses bare ``GOLD``; v2 (and future splits) use ``GOLD_v2``,
``GOLD_v3``, … The :func:`is_gold` check accepts both the bare sentinel
and any ``GOLD_*`` suffix, so per-split GOLD parquets can sit side-by-side
in the flat ``learned_scores/`` and ``judgments/`` caches without colliding.
"""


def is_gold(config_name: str) -> bool:
    """True for the bare sentinel or any ``GOLD_<suffix>`` split variant."""
    return config_name == GOLD_CONFIG or config_name.startswith(GOLD_CONFIG + "_")


def gold_inference_from_brief(brief: Brief, config_name: str = GOLD_CONFIG) -> Inference:
    """Synthesize an Inference whose ``assistant_text`` is the real ad.

    All non-essential fields default to zero/empty: GOLD didn't go through
    a runner, so latency and token counts aren't meaningful. ``config_name``
    defaults to ``GOLD`` but accepts split-specific variants like ``GOLD_v2``.
    """
    return Inference(
        example_id=brief.example_id,
        config=config_name,
        arm="arm1",
        brief=brief.user,
        system=brief.system,
        assistant_text=brief.reference_assistant,
        rationale=None,
        tool_calls=[],
        raw_traces=None,
        campaign=None,
        latency_ms=0,
        input_tokens=0,
        output_tokens=0,
        model_id=config_name,
        error=None,
        created_at=datetime.now(UTC),
    )


def gold_inferences_from_briefs(
    briefs: Iterable[Brief], config_name: str = GOLD_CONFIG
) -> dict[str, Inference]:
    """Return ``{example_id: Inference}`` for every brief, keyed for judge driver."""
    return {b.example_id: gold_inference_from_brief(b, config_name) for b in briefs}
