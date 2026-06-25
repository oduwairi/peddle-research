"""Tests for the eval pipeline: schemas, briefs loader, judge reconciliation,
aggregation math. Network-dependent code (judges, runners) is not exercised
here — those are integration-tested by running the CLI against real
endpoints in the eval workflow.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from draper.evaluation.briefs import load_test_briefs, load_url_scenarios
from draper.evaluation.config import EvalConfig
from draper.evaluation.gold import is_gold
from draper.evaluation.judge.aggregation import (
    bootstrap_win_rate_ci,
    cohen_kappa,
    elo_ratings,
    pair_results_to_dataframe,
    win_rates_table,
)
from draper.evaluation.judge.pairwise import _canonical_per_dim, reconcile_pair
from draper.evaluation.schemas import (
    Inference,
    JudgePerDimension,
    Judgment,
    PairResult,
)

# ---- briefs / scenarios loading -----------------------------------------


def test_load_test_briefs_real_split() -> None:
    """The held-out test split exists and yields well-formed Briefs."""
    path = Path("data/final/test")
    if not path.exists():
        pytest.skip("test split not available in this checkout")
    briefs = load_test_briefs(path)
    assert briefs, "expected at least one brief"
    b = briefs[0]
    assert b.example_id
    assert b.task_format == "copywriting"
    assert b.system.startswith("You are an ad copywriter")
    assert b.user
    assert b.reference_assistant
    assert b.platform
    assert b.vertical


def test_load_test_briefs_v2_split() -> None:
    """The v2 test split (dual-shape loader) yields well-formed Briefs.

    v2 rows carry only ``messages`` + ``metadata`` — no top-level
    task_format/vertical/source_tiers columns. The loader must default
    them to "copywriting", "unknown", and [] respectively, and must
    populate ``platform`` from ``metadata``.
    """
    path = Path("data/constructed_v2/final_v2/test")
    if not path.exists():
        pytest.skip("v2 test split not available in this checkout")
    briefs = load_test_briefs(path)
    assert briefs, "expected at least one brief"
    b = briefs[0]
    assert b.example_id
    assert b.task_format == "copywriting"
    assert b.vertical == "unknown"
    assert b.source_tiers == []
    # platform must be a non-empty string (meta/x/pinterest/reddit/tiktok)
    assert b.platform and isinstance(b.platform, str)


def test_load_url_scenarios_returns_empty_when_missing(tmp_path: Path) -> None:
    p = tmp_path / "doesnotexist.jsonl"
    assert load_url_scenarios(p) == []


def test_load_url_scenarios_real_file() -> None:
    p = Path("data/eval/url_scenarios.jsonl")
    if not p.exists():
        pytest.skip("url_scenarios.jsonl not present")
    scenarios = load_url_scenarios(p)
    assert scenarios
    s = scenarios[0]
    assert s.scenario_id
    assert s.url.startswith("http")
    assert s.platform
    assert s.vertical
    assert s.user_prompt


# ---- config loading ------------------------------------------------------


def test_eval_config_loads() -> None:
    cfg = EvalConfig.from_yaml("configs/eval.yaml")
    assert "A" in cfg.configs
    assert "C" in cfg.configs
    assert cfg.judges.primary
    assert cfg.judges.secondary
    assert cfg.arm1_pairs
    # Every pair must reference a declared config — except for any GOLD sentinel
    # (bare "GOLD" or a split-specific variant like "GOLD_v2"), which is
    # synthesized from briefs at judge time and therefore not in cfg.configs.
    for pair in cfg.arm1_pairs + cfg.arm2_pairs:
        for side in pair:
            assert is_gold(side) or side in cfg.configs


# ---- helpers for synthesizing judgments / inferences ---------------------


def _judgment(
    *,
    example_id: str,
    pair_a: str,
    pair_b: str,
    swap_order: bool,
    winner: str,
    per_dim: tuple[int, int, int, int, int] = (1, 0, 0, 0, 0),
    judge: str = "gpt-4o",
) -> Judgment:
    return Judgment(
        example_id=example_id,
        pair_a=pair_a,
        pair_b=pair_b,
        swap_order=swap_order,
        judge_model=judge,
        per_dim=JudgePerDimension(
            strategic_relevance=per_dim[0],
            creativity=per_dim[1],
            actionability=per_dim[2],
            channel_appropriateness=per_dim[3],
            predicted_performance=per_dim[4],
        ),
        overall_winner=winner,  # type: ignore[arg-type]
        rationale="t",
        created_at=datetime.now(UTC),
    )


# ---- judge reconciliation ------------------------------------------------


def test_reconcile_both_orderings_agree_on_a() -> None:
    forward = _judgment(
        example_id="ex1", pair_a="A", pair_b="C", swap_order=False, winner="A"
    )
    swapped = _judgment(
        example_id="ex1", pair_a="C", pair_b="A", swap_order=True, winner="B"
    )
    pr = reconcile_pair(
        example_id="ex1",
        config_a="A",
        config_b="C",
        judgments=[forward, swapped],
        judge_model="gpt-4o",
    )
    assert pr.forward_winner == "A"
    assert pr.swapped_winner == "A"
    assert pr.resolved_winner == "A"
    assert pr.order_dependent is False


def test_reconcile_orderings_disagree_breaks_by_per_dim_sum() -> None:
    # Forward says A wins (per-dim +2 for A); swapped says B wins.
    forward = _judgment(
        example_id="ex2",
        pair_a="A",
        pair_b="C",
        swap_order=False,
        winner="A",
        per_dim=(2, 0, 0, 0, 0),
    )
    # In swapped frame the judge sees C as "A"; if it picks B (=A), then
    # canonical winner is A too — but to construct disagreement we say it
    # picks "A" (=C), with per_dim +1 (=C is +1 better, so canonical -1 for A).
    swapped = _judgment(
        example_id="ex2",
        pair_a="C",
        pair_b="A",
        swap_order=True,
        winner="A",
        per_dim=(1, 0, 0, 0, 0),
    )
    pr = reconcile_pair(
        example_id="ex2",
        config_a="A",
        config_b="C",
        judgments=[forward, swapped],
        judge_model="gpt-4o",
    )
    # Canonical: forward says +2 for A; swapped says -1 for A.
    # Total per-dim sum across both = +1 → resolved=A, order_dependent.
    assert pr.forward_winner == "A"
    assert pr.swapped_winner == "C" or pr.swapped_winner == "B"
    # config_a is "A", so canonical 'A wins' means our config_a (A) wins.
    assert pr.resolved_winner == "A"
    assert pr.order_dependent is True


def test_reconcile_order_dependent_zero_sum_is_tie() -> None:
    forward = _judgment(
        example_id="ex3",
        pair_a="A",
        pair_b="C",
        swap_order=False,
        winner="A",
        per_dim=(1, 0, 0, 0, 0),
    )
    swapped = _judgment(
        example_id="ex3",
        pair_a="C",
        pair_b="A",
        swap_order=True,
        winner="A",
        per_dim=(1, 0, 0, 0, 0),
    )
    pr = reconcile_pair(
        example_id="ex3",
        config_a="A",
        config_b="C",
        judgments=[forward, swapped],
        judge_model="gpt-4o",
    )
    # Forward +1 for A (canonical A=A), swapped flips → -1 for A; sum 0.
    assert pr.order_dependent is True
    assert pr.resolved_winner == "tie"


def test_reconcile_single_ordering() -> None:
    forward = _judgment(
        example_id="ex4", pair_a="A", pair_b="C", swap_order=False, winner="B"
    )
    pr = reconcile_pair(
        example_id="ex4",
        config_a="A",
        config_b="C",
        judgments=[forward],
        judge_model="gpt-4o",
    )
    assert pr.resolved_winner == "B"
    assert pr.order_dependent is False


def test_canonical_per_dim_swap_flips_signs() -> None:
    forward = _judgment(
        example_id="ex", pair_a="A", pair_b="C", swap_order=False, per_dim=(2, 1, 0, 0, 0),
        winner="A",
    )
    swapped = _judgment(
        example_id="ex", pair_a="C", pair_b="A", swap_order=True, per_dim=(2, 1, 0, 0, 0),
        winner="A",
    )
    can_fwd = _canonical_per_dim(forward, "A")
    can_swp = _canonical_per_dim(swapped, "A")
    assert can_fwd.strategic_relevance == 2
    assert can_swp.strategic_relevance == -2
    assert can_swp.creativity == -1


# ---- aggregation math ----------------------------------------------------


def _pair_result(
    example_id: str,
    config_a: str,
    config_b: str,
    winner: str,
    judge: str = "gpt-4o",
    order_dep: bool = False,
) -> PairResult:
    z = JudgePerDimension(
        strategic_relevance=0,
        creativity=0,
        actionability=0,
        channel_appropriateness=0,
        predicted_performance=0,
    )
    return PairResult(
        example_id=example_id,
        config_a=config_a,
        config_b=config_b,
        judge_model=judge,
        forward_winner=winner,  # type: ignore[arg-type]
        swapped_winner=winner,  # type: ignore[arg-type]
        resolved_winner=winner,  # type: ignore[arg-type]
        order_dependent=order_dep,
        per_dim_sum=z,
    )


def test_win_rates_table_basic() -> None:
    results = [
        _pair_result("e1", "A", "C", "A"),
        _pair_result("e2", "A", "C", "A"),
        _pair_result("e3", "A", "C", "B"),
        _pair_result("e4", "A", "C", "tie"),
    ]
    df = pair_results_to_dataframe(results)
    summary = win_rates_table(df)
    row = summary.row(0, named=True)
    assert row["n"] == 4
    assert row["wins_a"] == 2
    assert row["wins_b"] == 1
    assert row["ties"] == 1
    assert row["win_rate_a"] == pytest.approx(0.5)
    assert row["tie_rate"] == pytest.approx(0.25)


def test_pair_results_dataframe_enriches_with_briefs() -> None:
    from draper.evaluation.schemas import Brief

    briefs = {
        "e1": Brief(
            example_id="e1",
            task_format="copywriting",
            platform="facebook",
            vertical="cpg",
            source_tiers=["high"],
            system="s",
            user="u",
            reference_assistant="gold copy",
        ),
        "e2": Brief(
            example_id="e2",
            task_format="copywriting",
            platform="tiktok",
            vertical="cpg",
            source_tiers=["medium"],
            system="s",
            user="u",
            reference_assistant="gold copy",
        ),
    }
    results = [
        _pair_result("e1", "A", "C", "A"),
        _pair_result("e2", "A", "C", "B"),
    ]
    df = pair_results_to_dataframe(results, briefs_by_id=briefs)
    assert df.row(by_predicate=pl.col("example_id") == "e1", named=True)["platform"] == "facebook"
    assert df.row(by_predicate=pl.col("example_id") == "e2", named=True)["source_tier"] == "medium"


def test_win_rates_table_groupby_by_platform() -> None:
    from draper.evaluation.schemas import Brief

    briefs = {
        "e1": Brief(
            example_id="e1", task_format="copywriting", platform="facebook",
            vertical="cpg", source_tiers=["high"], system="", user="",
            reference_assistant="g",
        ),
        "e2": Brief(
            example_id="e2", task_format="copywriting", platform="facebook",
            vertical="cpg", source_tiers=["high"], system="", user="",
            reference_assistant="g",
        ),
        "e3": Brief(
            example_id="e3", task_format="copywriting", platform="tiktok",
            vertical="cpg", source_tiers=["high"], system="", user="",
            reference_assistant="g",
        ),
    }
    results = [
        _pair_result("e1", "A", "C", "A"),
        _pair_result("e2", "A", "C", "A"),
        _pair_result("e3", "A", "C", "B"),
    ]
    df = pair_results_to_dataframe(results, briefs_by_id=briefs)
    seg = win_rates_table(df, groupby=["platform"])
    fb_row = seg.row(by_predicate=pl.col("platform") == "facebook", named=True)
    tt_row = seg.row(by_predicate=pl.col("platform") == "tiktok", named=True)
    assert fb_row["n"] == 2
    assert fb_row["wins_a"] == 2
    assert tt_row["n"] == 1
    assert tt_row["wins_b"] == 1


def test_win_rates_table_groupby_missing_column_raises() -> None:
    """If callers ask for a segment that wasn't enriched, raise loudly."""
    results = [_pair_result("e1", "A", "C", "A")]
    # No briefs_by_id — segment cols are None but exist in schema.
    df = pair_results_to_dataframe(results)
    with pytest.raises(ValueError, match="not present"):
        win_rates_table(df.drop("platform"), groupby=["platform"])


def test_win_rates_table_empty() -> None:
    df = pair_results_to_dataframe([])
    summary = win_rates_table(df)
    assert isinstance(summary, pl.DataFrame)
    assert summary.is_empty()


def test_bootstrap_ci_returns_valid_interval() -> None:
    results = [_pair_result(f"e{i}", "A", "C", "A") for i in range(80)]
    results += [_pair_result(f"f{i}", "A", "C", "B") for i in range(20)]
    cis = bootstrap_win_rate_ci(results, n_bootstrap=200, seed=0)
    (lo, hi) = cis[("A", "C", "gpt-4o")]
    # Empirical win rate is 0.8 → CI should sit around it.
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    assert lo <= hi
    assert lo < 0.8 < hi


def test_bootstrap_ci_empty() -> None:
    assert bootstrap_win_rate_ci([], n_bootstrap=10) == {}


def test_elo_ratings_winner_higher_than_loser() -> None:
    # A consistently beats C → A's Elo should rise above C's.
    results = [_pair_result(f"e{i}", "A", "C", "A") for i in range(50)]
    elos = elo_ratings(results, k=32.0, seed=0)
    assert elos["A"] > elos["C"]
    assert elos["A"] > 1000.0
    assert elos["C"] < 1000.0


def test_elo_ratings_with_ties() -> None:
    results = [_pair_result(f"e{i}", "A", "C", "tie") for i in range(20)]
    elos = elo_ratings(results, k=32.0, seed=0)
    # Ties leave ratings at initial.
    assert elos["A"] == pytest.approx(1000.0)
    assert elos["C"] == pytest.approx(1000.0)


def test_cohen_kappa_perfect_agreement() -> None:
    primary = [_pair_result("e1", "A", "C", "A"), _pair_result("e2", "A", "C", "B")]
    secondary = [_pair_result("e1", "A", "C", "A"), _pair_result("e2", "A", "C", "B")]
    assert cohen_kappa(primary, secondary) == pytest.approx(1.0)


def test_cohen_kappa_no_agreement() -> None:
    primary = [_pair_result("e1", "A", "C", "A"), _pair_result("e2", "A", "C", "A")]
    secondary = [_pair_result("e1", "A", "C", "B"), _pair_result("e2", "A", "C", "B")]
    # Both judges always pick a single but different category — kappa is 0
    # by definition (no expected-by-chance agreement either).
    k = cohen_kappa(primary, secondary)
    assert k <= 0.0


# ---- inference schema validation ----------------------------------------


def test_inference_serialization_roundtrip() -> None:
    inf = Inference(
        example_id="e",
        config="C",
        arm="arm1",
        brief="hi",
        system="sys",
        assistant_text="copy",
        latency_ms=100,
        model_id="x",
        created_at=datetime.now(UTC),
    )
    payload = inf.model_dump_json()
    parsed = Inference.model_validate_json(payload)
    assert parsed.example_id == "e"
    assert parsed.config == "C"
    assert parsed.arm == "arm1"
