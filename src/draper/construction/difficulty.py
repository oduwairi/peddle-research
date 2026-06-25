"""Difficulty stratification for training-example composition.

Four difficulty tiers are rolled per example:

- ``standard`` (60%): clean default composition
- ``sparse`` (20%): reduced ad count / weaker signal
- ``conflicting`` (15%): mixed-tier or contradictory evidence
- ``multi_constraint`` (5%): persona carries an extra real-world constraint

Difficulty is applied as a post-processing transform on the batch
returned by ``SourceSelector``, plus a label carried in the bundle so
the chat agent's response reflects the messier input appropriately.
"""

from __future__ import annotations

import random

from draper.construction.schemas import TaskFormat
from draper.scoring.schemas import ScoredAd

STANDARD = "standard"
SPARSE = "sparse"
CONFLICTING = "conflicting"
MULTI_CONSTRAINT = "multi_constraint"

DIFFICULTY_RATIOS: dict[str, float] = {
    STANDARD: 0.60,
    SPARSE: 0.20,
    CONFLICTING: 0.15,
    MULTI_CONSTRAINT: 0.05,
}

def _compute_sparse_disallowed_formats() -> tuple[TaskFormat, ...]:
    """Task formats whose pipelines disallow sparse difficulty rolls."""
    from draper.construction.formats.registry import get_pipeline

    return tuple(fmt for fmt in TaskFormat if get_pipeline(fmt).sparse_disallowed)


def __getattr__(name: str) -> object:
    if name == "_SPARSE_DISALLOWED_FORMATS":
        return _compute_sparse_disallowed_formats()
    raise AttributeError(name)


def _sparse_disallowed(task_format: TaskFormat) -> bool:
    """True when sparse difficulty should be remapped to standard for a format.

    Read from :class:`FormatPipeline.sparse_disallowed` so each format
    owns its own rule. Diagnostic / Copywriting / Optimization set this
    flag: single-ad formats have nothing to shrink, and Optimization's
    teacher prompt assumes both pair ads are present.
    """
    from draper.construction.formats.registry import get_pipeline

    return get_pipeline(task_format).sparse_disallowed

DIFFICULTY_DIRECTIVES: dict[str, str] = {
    STANDARD: (
        "Standard input conditions — produce the expected quality response without special caveats."
    ),
    SPARSE: (
        "LIMITED DATA. The grounding input is intentionally minimal. "
        "Acknowledge uncertainty in the response. Avoid over-specific "
        "claims. Frame recommendations as directional rather than "
        "definitive. Say 'there isn't enough signal here to...' where "
        "appropriate instead of hallucinating confidence."
    ),
    CONFLICTING: (
        "CONFLICTING SIGNALS. The grounding contains mixed-tier or "
        "contradictory evidence. Identify the contradictions explicitly "
        "in the response. Articulate tradeoffs rather than hiding them. "
        "A strong answer here acknowledges the tension and picks a "
        "direction with clear reasoning, not a false-consensus answer."
    ),
    MULTI_CONSTRAINT: (
        "MULTI-CONSTRAINT. The persona faces an additional real-world "
        "constraint. Weave ONE realistic constraint into the user "
        "prompt you produce (e.g., strict budget cap, tight timeline, "
        "legal/compliance limit, team-size restriction, legacy tech). "
        "The response MUST address this constraint directly, not ignore it."
    ),
}


def sample_difficulty(
    rng: random.Random,
    task_format: TaskFormat,
    difficulty_ratios: dict[str, float] | None = None,
) -> str:
    """Roll a difficulty tier from the configured distribution.

    For formats whose pipeline sets ``sparse_disallowed=True`` (diagnostic,
    copywriting, optimization), rolls that land on ``sparse`` are
    remapped to ``standard``. This preserves the RNG stream position
    while reallocating the 20% sparse budget to the default tier.
    """
    ratios = difficulty_ratios if difficulty_ratios is not None else DIFFICULTY_RATIOS
    r = rng.random()
    cumulative = 0.0
    for tier, weight in ratios.items():
        cumulative += weight
        if r < cumulative:
            if tier == SPARSE and _sparse_disallowed(task_format):
                return STANDARD
            return tier
    return STANDARD  # fallback for float rounding


def directive_for(difficulty: str) -> str:
    """Return the bundle-directive text for a difficulty tier."""
    return DIFFICULTY_DIRECTIVES.get(difficulty, DIFFICULTY_DIRECTIVES[STANDARD])


def apply_difficulty(
    batch: list[ScoredAd],
    difficulty: str,
    task_format: TaskFormat,
    rng: random.Random,
) -> list[ScoredAd]:
    """Reshape a source-ad batch according to the difficulty tier.

    ``standard`` and ``multi_constraint`` return the batch unchanged —
    multi-constraint lives in the bundle directive. ``sparse`` reduces
    ad count; ``conflicting`` shuffles multi-ad batches so tiers don't
    cluster by order.
    """
    if not batch or difficulty in (STANDARD, MULTI_CONSTRAINT):
        return batch

    if difficulty == SPARSE:
        return _apply_sparse(batch, task_format)

    if difficulty == CONFLICTING:
        return _apply_conflicting(batch, task_format, rng)

    return batch


def _apply_sparse(batch: list[ScoredAd], task_format: TaskFormat) -> list[ScoredAd]:
    """Shrink the batch to a minimal signal per format.

    Diagnostic, copywriting, and optimization are excluded upstream in
    :func:`sample_difficulty` (via ``FormatPipeline.sparse_disallowed``),
    so only multi-ad formats (positioning, channel_format_fit) reach
    this branch in practice. The guards below remain as defensive
    no-ops in case a caller passes ``SPARSE`` explicitly.
    """
    if _sparse_disallowed(task_format):
        return batch
    # Multi-ad formats: trim to half the cluster (min 2).
    return batch[: max(2, len(batch) // 2)]


def _apply_conflicting(
    batch: list[ScoredAd],
    task_format: TaskFormat,
    rng: random.Random,
) -> list[ScoredAd]:
    """Introduce a contradictory signal into the batch composition."""
    from draper.construction.formats.registry import get_pipeline

    if len(batch) < 2:  # noqa: PLR2004
        return batch

    if not get_pipeline(task_format).shuffle_on_conflicting:
        # Formats whose ad order is semantically meaningful (optimization:
        # [low, high] pair) opt out — the directive does the work.
        return batch

    shuffled = list(batch)
    rng.shuffle(shuffled)
    return shuffled
