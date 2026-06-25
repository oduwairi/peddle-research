"""Aggregate PairResult lists into win-rate tables, Elo ratings, bootstrap CIs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ..schemas import Brief, PairResult

SEGMENT_COLUMNS: tuple[str, ...] = ("platform", "vertical", "source_tier")
"""Brief-derived columns available for ``--groupby`` segmentation."""


def pair_results_to_dataframe(
    results: Sequence[PairResult],
    *,
    briefs_by_id: dict[str, Brief] | None = None,
) -> pl.DataFrame:
    """Materialize PairResults into a Polars DataFrame for aggregation.

    When ``briefs_by_id`` is provided, enriches each row with brief metadata
    (``platform``, ``vertical``, ``source_tier``) so callers can ``--groupby``
    on segment. ``source_tier`` is the first entry of ``Brief.source_tiers``
    (verticals usually have a single source ad per brief; multi-source briefs
    take the dominant tier).
    """
    if not results:
        # Return an empty DF with the expected schema so downstream groupby
        # operations don't crash.
        return pl.DataFrame(
            schema={
                "example_id": pl.String,
                "config_a": pl.String,
                "config_b": pl.String,
                "judge_model": pl.String,
                "resolved_winner": pl.String,
                "order_dependent": pl.Boolean,
                "per_dim_total": pl.Int64,
                "platform": pl.String,
                "vertical": pl.String,
                "source_tier": pl.String,
            }
        )
    rows = []
    for r in results:
        brief = briefs_by_id.get(r.example_id) if briefs_by_id else None
        rows.append(
            {
                "example_id": r.example_id,
                "config_a": r.config_a,
                "config_b": r.config_b,
                "judge_model": r.judge_model,
                "resolved_winner": r.resolved_winner,
                "order_dependent": r.order_dependent,
                "per_dim_total": (
                    r.per_dim_sum.strategic_relevance
                    + r.per_dim_sum.creativity
                    + r.per_dim_sum.actionability
                    + r.per_dim_sum.channel_appropriateness
                    + r.per_dim_sum.predicted_performance
                ),
                "platform": brief.platform if brief else None,
                "vertical": brief.vertical if brief else None,
                "source_tier": (brief.source_tiers[0] if (brief and brief.source_tiers) else None),
            }
        )
    return pl.DataFrame(rows)


def win_rates_table(df: pl.DataFrame, *, groupby: Sequence[str] | None = None) -> pl.DataFrame:
    """Aggregate per-pair win rates from a PairResult DataFrame.

    Without ``groupby``, output columns are ``config_a, config_b,
    judge_model, n, wins_a, wins_b, ties, order_dep, win_rate_a,
    win_rate_b, tie_rate``.

    With ``groupby=["platform"]`` (or ``"vertical"``, ``"source_tier"``),
    rows are also broken out per segment value — useful for asking "is the
    FT-vs-base inversion uniform across platforms or concentrated?"
    """
    if df.is_empty():
        schema = {
            "config_a": pl.String,
            "config_b": pl.String,
            "judge_model": pl.String,
            "n": pl.Int64,
            "wins_a": pl.Int64,
            "wins_b": pl.Int64,
            "ties": pl.Int64,
            "order_dep": pl.Int64,
            "win_rate_a": pl.Float64,
            "win_rate_b": pl.Float64,
            "tie_rate": pl.Float64,
        }
        if groupby:
            for col in groupby:
                schema[col] = pl.String
        return pl.DataFrame(schema=schema)
    keys = ["config_a", "config_b", "judge_model"]
    if groupby:
        for col in groupby:
            if col not in df.columns:
                raise ValueError(
                    f"groupby column {col!r} not present in pair-result frame "
                    f"(have {df.columns}); pass briefs_by_id to "
                    "pair_results_to_dataframe to enrich segments."
                )
        keys = list(groupby) + keys
    grouped = df.group_by(keys).agg(
        pl.len().alias("n"),
        (pl.col("resolved_winner") == "A").sum().alias("wins_a"),
        (pl.col("resolved_winner") == "B").sum().alias("wins_b"),
        (pl.col("resolved_winner") == "tie").sum().alias("ties"),
        pl.col("order_dependent").sum().alias("order_dep"),
    )
    return grouped.with_columns(
        (pl.col("wins_a") / pl.col("n")).alias("win_rate_a"),
        (pl.col("wins_b") / pl.col("n")).alias("win_rate_b"),
        (pl.col("ties") / pl.col("n")).alias("tie_rate"),
    ).sort(keys)


def bootstrap_win_rate_ci(
    results: Sequence[PairResult],
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Bootstrap 95% CI on config_a's win rate per (config_a, config_b, judge).

    Resamples with replacement at the example level. Returns a mapping
    ``(config_a, config_b, judge_model) -> (ci_low, ci_high)``.
    """
    if not results:
        return {}
    rng = np.random.default_rng(seed)
    # Group results by (config_a, config_b, judge_model).
    groups: dict[tuple[str, str, str], list[PairResult]] = {}
    for r in results:
        key = (r.config_a, r.config_b, r.judge_model)
        groups.setdefault(key, []).append(r)

    out: dict[tuple[str, str, str], tuple[float, float]] = {}
    for key, group in groups.items():
        n = len(group)
        # Encode wins as 1 (A wins), 0 (tie or B wins) — we want CI on
        # P(A wins). Ties are deliberately not split: a tie is not a win.
        wins = np.array([1 if r.resolved_winner == "A" else 0 for r in group], dtype=np.float64)
        boot = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot[i] = wins[idx].mean()
        ci_low = float(np.percentile(boot, 100 * alpha / 2))
        ci_high = float(np.percentile(boot, 100 * (1 - alpha / 2)))
        out[key] = (ci_low, ci_high)
    return out


def elo_ratings(
    results: Sequence[PairResult],
    *,
    k: float = 32.0,
    initial_rating: float = 1000.0,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Elo ratings across configs from pairwise PairResults.

    Order of pairs is randomized for stability; each result counts once.
    Ties contribute 0.5 to both. Single-judge in single-judge runs; for
    multi-judge ensembles the caller should pre-filter or call per-judge.
    """
    if not results:
        return {}
    rng = np.random.default_rng(seed)
    # Collect unique configs.
    configs: set[str] = set()
    for r in results:
        configs.add(r.config_a)
        configs.add(r.config_b)
    ratings: dict[str, float] = {c: initial_rating for c in configs}

    indices = np.arange(len(results))
    rng.shuffle(indices)
    for idx in indices:
        r = results[int(idx)]
        ra = ratings[r.config_a]
        rb = ratings[r.config_b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea
        if r.resolved_winner == "A":
            sa, sb = 1.0, 0.0
        elif r.resolved_winner == "B":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5
        ratings[r.config_a] = ra + k * (sa - ea)
        ratings[r.config_b] = rb + k * (sb - eb)
    return ratings


def cohen_kappa(
    primary: Sequence[PairResult],
    secondary: Sequence[PairResult],
) -> float:
    """Cohen's kappa for agreement between two judges on the same examples.

    Both sequences must have the same (example_id, config_a, config_b)
    keys. Categories are ``A`` / ``B`` / ``tie``.
    """
    p_map = {(r.example_id, r.config_a, r.config_b): r.resolved_winner for r in primary}
    s_map = {(r.example_id, r.config_a, r.config_b): r.resolved_winner for r in secondary}
    common = set(p_map) & set(s_map)
    if not common:
        return 0.0
    cats = ("A", "B", "tie")
    n = len(common)
    agree = sum(1 for k in common if p_map[k] == s_map[k])
    p_o = agree / n
    p_e = 0.0
    for c in cats:
        p_p = sum(1 for k in common if p_map[k] == c) / n
        p_s = sum(1 for k in common if s_map[k] == c) / n
        p_e += p_p * p_s
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)
