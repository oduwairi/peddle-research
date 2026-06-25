"""Loader + text-builder for the v3 scored-ads corpus.

Reads ``data/scored/v3/scored_ads.parquet`` (the active scoring output, see
``CLAUDE.md`` "Data" section) and produces ``Example`` records ready for
tokenization. The text builder concatenates platform/vertical context with the
ad copy fields actually populated in the AdFlex corpus
(``headline``, ``body``, ``description``); ``ad_copy_cta`` is null in v3 so
it is intentionally omitted.

Engagement-target masking mirrors ``HybridScorer._WEAK_ENGAGEMENT_PLATFORMS``
(``reddit``, ``other``): the v3 weighted-sum already drops those signals for
weak platforms, so the parquet has them as 0.0 (post-redistribution) which
would mislead an MSE loss. The ``Example.target_mask`` flags each head as
trainable / not-trainable, and the model masks the loss accordingly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# Heads we predict. Order is fixed and used as the column index in the model
# output tensor — do not reorder without updating ``model.py``.
HEAD_NAMES: tuple[str, ...] = (
    "composite",
    "survivability",
    "engagement_volume",
    "engagement_velocity",
)

# Mirror of ``HybridScorer._WEAK_ENGAGEMENT_PLATFORMS``. Imported by value
# (not from ``hybrid_scorer``) to avoid a hard dep on the scoring stack at
# training time.
WEAK_ENGAGEMENT_PLATFORMS: frozenset[str] = frozenset({"reddit", "other"})

# Engagement heads that get masked out for weak-engagement platforms.
ENGAGEMENT_HEADS: frozenset[str] = frozenset({"engagement_volume", "engagement_velocity"})

# Parquet columns we depend on. Anything else is ignored.
_TEXT_COLUMNS: tuple[str, ...] = (
    "ad_copy_headline",
    "ad_copy_body",
    "ad_copy_description",
)
_META_COLUMNS: tuple[str, ...] = ("platform", "vertical", "language", "ad_id")
_TARGET_COLUMNS: tuple[str, ...] = (
    "composite_score",
    "signal_survivability",
    "signal_engagement_volume",
    "signal_engagement_velocity",
)
_QUALITY_COLUMN = "training_quality"


@dataclass(frozen=True, slots=True)
class Example:
    """One ad ready for tokenization + loss computation.

    ``target_mask`` is a 4-tuple of bools aligned with :data:`HEAD_NAMES`;
    ``False`` entries should be excluded from the loss for this row.
    Currently only the engagement heads are ever masked (for weak platforms),
    but the field is general so we can mask other heads later without a
    schema change.
    """

    ad_id: str
    text: str
    targets: tuple[float, float, float, float]
    target_mask: tuple[bool, bool, bool, bool]
    sample_weight: float
    platform: str
    vertical: str
    language: str


def build_text(
    *,
    platform: str,
    vertical: str,
    headline: str | None,
    body: str | None,
    description: str | None,
) -> str:
    """Build the model's input string from copy + context fields.

    Format::

        [platform=<p>] [vertical=<v>]
        <headline>
        <body>
        <description>

    Empty / missing copy fields are skipped (no blank lines, no ``"None"``
    leakage). ``platform``/``vertical`` are always emitted as control tokens
    so the model can condition on them.

    Returns the empty string if all copy fields are missing — caller is
    responsible for filtering those rows out via :func:`is_usable_row`.
    """
    parts: list[str] = []
    plat = (platform or "unknown").strip().lower() or "unknown"
    vert = (vertical or "unknown").strip().lower() or "unknown"
    parts.append(f"[platform={plat}] [vertical={vert}]")

    for field in (headline, body, description):
        if field is None:
            continue
        cleaned = field.strip()
        if cleaned:
            parts.append(cleaned)

    if len(parts) == 1:
        # Only the control-token line; no actual copy.
        return ""
    return "\n".join(parts)


def is_usable_row(row: dict[str, object]) -> bool:
    """True if the row has at least one non-empty copy field."""
    for col in _TEXT_COLUMNS:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            return True
    return False


def _target_mask_for_platform(platform: str) -> tuple[bool, bool, bool, bool]:
    """Per-head loss mask for the given platform.

    ``composite`` and ``survivability`` are always trained (their values in
    the parquet are well-defined for every row, including weak platforms).
    ``engagement_volume`` and ``engagement_velocity`` are masked out for
    Reddit / ``other`` because the v3 scorer drops those signals there and
    the on-disk values are post-redistribution, not the platform's true
    engagement percentile.
    """
    plat = (platform or "").strip().lower()
    is_weak = plat in WEAK_ENGAGEMENT_PLATFORMS
    return (
        True,  # composite
        True,  # survivability
        not is_weak,  # engagement_volume
        not is_weak,  # engagement_velocity
    )


def _sample_weight(training_quality: int | None) -> float:
    """Map the 1–5 ``training_quality`` rating to a sample loss weight.

    Quality 1 (broken) → 0.0 — exclude entirely from the loss.
    Quality 2 (clickbait) → 0.5 — keep but downweight.
    Quality 3+ (coherent) → 1.0 — full weight.
    Missing quality (rare) → 1.0 — neutral default.

    These are deliberately coarse; tune if Phase 1 offline metrics suggest
    quality-3 rows are noisier than quality-4.
    """
    if training_quality is None:
        return 1.0
    if training_quality <= 1:
        return 0.0
    if training_quality == 2:
        return 0.5
    return 1.0


def load_corpus(parquet_path: str | Path) -> pl.DataFrame:
    """Load the v3 corpus, keeping only the columns we need.

    Returns a Polars DataFrame; downstream code converts to :class:`Example`
    via :func:`iter_examples`. Materialized rather than lazy because the full
    corpus (~55k rows × ~10 cols after projection) easily fits in memory and
    we want deterministic row ordering for split reproducibility.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Scored corpus not found: {path}")

    keep = list(_TEXT_COLUMNS) + list(_META_COLUMNS) + list(_TARGET_COLUMNS) + [_QUALITY_COLUMN]
    df = pl.read_parquet(path, columns=keep)

    # Filter rows with no usable copy. Polars expression so we don't have to
    # materialize and re-build the frame.
    has_copy = (
        (pl.col("ad_copy_headline").fill_null("").str.strip_chars().str.len_chars() > 0)
        | (pl.col("ad_copy_body").fill_null("").str.strip_chars().str.len_chars() > 0)
        | (pl.col("ad_copy_description").fill_null("").str.strip_chars().str.len_chars() > 0)
    )
    return df.filter(has_copy)


def iter_examples(df: pl.DataFrame) -> Iterable[Example]:
    """Yield :class:`Example` records from a loaded corpus DataFrame."""
    rows = df.to_dicts()
    for row in rows:
        platform = str(row.get("platform") or "unknown")
        vertical = str(row.get("vertical") or "unknown")
        text = build_text(
            platform=platform,
            vertical=vertical,
            headline=_as_str(row.get("ad_copy_headline")),
            body=_as_str(row.get("ad_copy_body")),
            description=_as_str(row.get("ad_copy_description")),
        )
        if not text:
            # Defense in depth — load_corpus already filtered these.
            continue

        composite = _as_float(row.get("composite_score"))
        survivability = _as_float(row.get("signal_survivability"))
        engagement_volume = _as_float(row.get("signal_engagement_volume"))
        engagement_velocity = _as_float(row.get("signal_engagement_velocity"))

        weight = _sample_weight(_as_int(row.get(_QUALITY_COLUMN)))
        if weight == 0.0:
            continue

        mask = _target_mask_for_platform(platform)
        yield Example(
            ad_id=str(row.get("ad_id") or ""),
            text=text,
            targets=(composite, survivability, engagement_volume, engagement_velocity),
            target_mask=mask,
            sample_weight=weight,
            platform=platform,
            vertical=vertical,
            language=str(row.get("language") or "unknown"),
        )


def examples_to_polars(examples: Sequence[Example]) -> pl.DataFrame:
    """Round-trip Examples to a DataFrame for split materialization on disk."""
    return pl.DataFrame(
        {
            "ad_id": [e.ad_id for e in examples],
            "text": [e.text for e in examples],
            "target_composite": [e.targets[0] for e in examples],
            "target_survivability": [e.targets[1] for e in examples],
            "target_engagement_volume": [e.targets[2] for e in examples],
            "target_engagement_velocity": [e.targets[3] for e in examples],
            "mask_composite": [e.target_mask[0] for e in examples],
            "mask_survivability": [e.target_mask[1] for e in examples],
            "mask_engagement_volume": [e.target_mask[2] for e in examples],
            "mask_engagement_velocity": [e.target_mask[3] for e in examples],
            "sample_weight": [e.sample_weight for e in examples],
            "platform": [e.platform for e in examples],
            "vertical": [e.vertical for e in examples],
            "language": [e.language for e in examples],
        }
    )


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _as_float(value: object) -> float:
    """Coerce to float; ``None`` becomes 0.0 (the head's mask handles it).

    Raises ValueError for unparseable non-None values to catch schema corruption
    early rather than silently defaulting to 0.0, which could mask real data issues.
    """
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Cannot coerce target value to float: {value!r} ({type(value).__name__})"
        ) from e


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(value)  # type: ignore[call-overload,no-any-return]
    except (TypeError, ValueError):
        return None
