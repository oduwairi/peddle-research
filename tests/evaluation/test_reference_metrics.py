"""Unit tests for the reference-overlap eval arm.

The dep-free path (ROUGE-L) is always exercised. sacrebleu / nltk / bert-score
are optional: tests that need them are guarded with ``importorskip`` so a CI
environment without the ``[refmetrics]`` extra still passes. ``score_configs``
runs with metric backends monkeypatched to None so it exercises the orchestration
and parquet shape without pulling any heavy library.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from draper.evaluation import reference_metrics as rm
from draper.evaluation.mauve_reference import ALL_KEY

# --- metric cores --------------------------------------------------------


def test_rouge_l_identical_is_one() -> None:
    assert rm._rouge_l_score("buy our running shoes now", ["buy our running shoes now"]) == 1.0


def test_rouge_l_multi_ref_takes_max() -> None:
    hyp = "fast free shipping today"
    refs = ["completely unrelated text", "fast free shipping today only"]
    # Max over refs => dominated by the near-identical second reference.
    assert rm._rouge_l_score(hyp, refs) > rm._rouge_l_score(hyp, [refs[0]])


def test_rouge_l_empty_inputs_are_zero() -> None:
    assert rm._rouge_l_score("", ["something"]) == 0.0
    assert rm._rouge_l_score("something", []) == 0.0
    assert rm._rouge_l_score("something", ["  "]) == 0.0


def test_bleu_identical_high_when_available() -> None:
    pytest.importorskip("sacrebleu")
    score = rm._bleu_score("buy our running shoes now today", ["buy our running shoes now today"])
    assert score is not None and score > 0.9  # normalized to [0, 1]


def test_metric_soft_fails_to_none_without_dep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rm, "_sacrebleu", lambda: None)
    monkeypatch.setattr(rm, "_meteor_fn", lambda: None)
    monkeypatch.setattr(rm, "_bertscorer", lambda _m: None)
    assert rm._bleu_score("a b c", ["a b c"]) is None
    assert rm._chrf_score("a b c", ["a b c"]) is None
    assert rm._meteor_score("a b c", ["a b c"]) is None
    assert rm._bertscore_score("a b c", ["a b c"], "roberta-large") is None
    # ROUGE-L is dep-free, so it must still produce a number.
    assert rm._rouge_l_score("a b c", ["a b c"]) == 1.0


# --- multi-ref selection -------------------------------------------------


def test_select_multi_refs_excludes_gold_self_match() -> None:
    corpus = {
        ALL_KEY: ["the gold winning ad", "ref one", "ref two", "ref three"],
        "facebook": ["the gold winning ad", "ref one", "ref two", "ref three"],
    }
    refs = rm.select_multi_refs(
        gold_text="The Gold  Winning Ad",  # different whitespace/case → same hash
        platform="facebook",
        reference_corpus_by_platform=corpus,
        k=5,
    )
    assert "the gold winning ad" not in refs
    assert len(refs) == 3


def test_select_multi_refs_falls_back_to_all_when_platform_too_small() -> None:
    corpus = {
        ALL_KEY: ["a", "b", "c", "d", "e", "f"],
        "tiktok": ["a"],  # only 1 < k → use ALL
    }
    refs = rm.select_multi_refs(
        gold_text="zzz",
        platform="tiktok",
        reference_corpus_by_platform=corpus,
        k=5,
    )
    assert len(refs) == 5  # drawn from the ALL pool


def test_select_multi_refs_caps_at_k_and_is_deterministic() -> None:
    corpus = {ALL_KEY: [f"ref-{i}" for i in range(20)], "facebook": [f"ref-{i}" for i in range(20)]}
    refs = rm.select_multi_refs(
        gold_text="gold", platform="facebook", reference_corpus_by_platform=corpus, k=4
    )
    assert refs == ["ref-0", "ref-1", "ref-2", "ref-3"]  # first-k, corpus order


# --- compute_row ---------------------------------------------------------


def test_compute_row_soft_fail_keeps_rouge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rm, "_sacrebleu", lambda: None)
    monkeypatch.setattr(rm, "_meteor_fn", lambda: None)
    row = rm.compute_row(
        config="C_v2",
        example_id="ex1",
        platform="facebook",
        generation="our product is great",
        gold_text="our product is great",
        multi_refs=["a totally different ad"],
        used_clean=True,
        enable_bertscore=False,
        bertscore_model="roberta-large",
        embedding_model_label="",
    )
    assert row.bleu_gold is None  # backend absent
    assert row.bertscore_gold is None  # disabled
    assert row.rouge_l_gold == 1.0  # dep-free, identical
    assert row.gold_overlap_excess is not None
    assert row.gold_overlap_excess > 0.0  # gold echo > broad-style overlap
    assert row.n_multi_refs == 1


# --- score_configs + summarize -------------------------------------------


def _write_clean(clean_dir: Path, config: str, example_id: str, text: str) -> None:
    d = clean_dir / config
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{example_id}.json").write_text(
        '{"example_id":"' + example_id + '","config":"' + config + '",'
        '"assistant_text_clean":"' + text + '",'
        '"extractor_model":"test","extracted_at":"2026-01-01T00:00:00Z",'
        '"raw_text_sha256":"deadbeef"}'
    )


def test_score_configs_writes_per_config_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No heavy backends — keep the test fast and dep-free (only ROUGE-L populated).
    monkeypatch.setattr(rm, "_sacrebleu", lambda: None)
    monkeypatch.setattr(rm, "_meteor_fn", lambda: None)
    monkeypatch.setattr(rm, "_embedder", lambda: None)  # first-k selection
    from draper.evaluation.schemas import Brief

    briefs = {
        "ex1": Brief(
            example_id="ex1",
            task_format="copywriting",
            platform="meta",  # aliases to facebook
            vertical="health",
            source_tiers=["high"],
            construction_model=None,
            system="",
            user="",
            reference_assistant="real-ad-1",
        ),
        "ex2": Brief(
            example_id="ex2",
            task_format="copywriting",
            platform="tiktok",
            vertical="health",
            source_tiers=["high"],
            construction_model=None,
            system="",
            user="",
            reference_assistant="real-ad-2",
        ),
    }
    clean_dir = tmp_path / "inferences_clean"
    raw_dir = tmp_path / "inferences"
    for cfg in ("A_v2",):
        _write_clean(clean_dir, cfg, "ex1", "candidate copy one")
        _write_clean(clean_dir, cfg, "ex2", "candidate copy two")

    reference = {
        ALL_KEY: ["ref-a", "ref-b", "ref-c"],
        "facebook": ["ref-a", "ref-b"],
        "tiktok": ["ref-c"],
    }
    gold_texts = {"ex1": "real winning ad one", "ex2": "real winning ad two"}

    written = rm.score_configs(
        briefs_by_id=briefs,
        reference_corpus_by_platform=reference,
        gold_texts_by_id=gold_texts,
        configs=["A_v2"],
        inferences_clean_dir=clean_dir,
        inferences_raw_dir=raw_dir,
        out_dir=tmp_path / "reference_scores",
        platforms=["facebook", "tiktok"],
        k_multi=2,
        enable_bertscore=False,
    )
    assert "A_v2" in written
    df = pl.read_parquet(written["A_v2"])
    assert df.height == 2  # one row per example
    assert set(df["platform"].to_list()) == {"facebook", "tiktok"}  # meta aliased
    for col in (
        "config",
        "example_id",
        "platform",
        "used_clean",
        "rouge_l_gold",
        "rouge_l_multi",
        "bleu_gold",
        "bertscore_gold",
        "n_multi_refs",
        "gold_overlap_excess",
        "created_at",
    ):
        assert col in df.columns, f"missing column {col}"
    # Float columns stay Float64 even when entirely null (bertscore disabled).
    assert df.schema["bertscore_gold"] == pl.Float64


def test_summarize_groups_and_is_null_safe(tmp_path: Path) -> None:
    out_dir = tmp_path / "reference_scores"
    out_dir.mkdir()
    for cfg, base in (("A_v2", 0.2), ("C_v2", 0.6)):
        rows = []
        for plat in ("facebook", "tiktok"):
            row: dict[str, object] = {
                "config": cfg,
                "example_id": f"{cfg}-{plat}",
                "platform": plat,
                "used_clean": True,
                "bleu_gold": base,
                "chrf_gold": base,
                "rouge_l_gold": base,
                "meteor_gold": base,
                "bertscore_gold": None,  # entirely null column
                "bleu_multi": base,
                "chrf_multi": base,
                "rouge_l_multi": base - 0.05,
                "meteor_multi": base,
                "bertscore_multi": None,
                "n_multi_refs": 5,
                "gold_overlap_excess": 0.05,
                "embedding_model": "all-MiniLM-L6-v2",
                "created_at": "2026-01-01T00:00:00Z",
            }
            rows.append(row)
        overrides = {c: pl.Float64 for c in rm._METRIC_COLUMNS}
        pl.DataFrame(rows, schema_overrides=overrides).write_parquet(out_dir / f"{cfg}.parquet")

    headline = rm.summarize(out_dir=out_dir, configs=["A_v2", "C_v2"], by=None)
    assert set(headline["config"].to_list()) == {"A_v2", "C_v2"}
    assert "rouge_l_gold_mean" in headline.columns
    # Null metric column aggregates to null without raising.
    assert headline.filter(pl.col("config") == "A_v2")["bertscore_gold_mean"].item() is None

    by_platform = rm.summarize(out_dir=out_dir, configs=["A_v2", "C_v2"], by=["platform"])
    assert by_platform.height == 4  # 2 configs × 2 platforms


# --- Upworthy grounding --------------------------------------------------


class _FakeVariant:
    def __init__(self, test_id: str, headline: str) -> None:
        self.test_id = test_id
        self.headline = headline
        self.excerpt = ""


def test_validate_on_upworthy_winner_closer_to_other_winners() -> None:
    # Winners share a vocabulary ("save money fast"); losers are off-topic.
    # Each variant scored vs OTHER tests' winners → winners should win.
    pairs = [
        (_FakeVariant("t1", "save money fast today"), _FakeVariant("t1", "purple balloon")),
        (_FakeVariant("t2", "save money fast now"), _FakeVariant("t2", "random phrase")),
        (_FakeVariant("t3", "save money fast deal"), _FakeVariant("t3", "nothing alike")),
    ]
    results = rm.validate_on_upworthy(pairs=pairs, metrics=["rouge_l"], enable_bertscore=False)
    assert "rouge_l" in results
    res = results["rouge_l"]
    assert res.n_pairs == 3
    assert res.accuracy > 0.5  # similarity-to-known-winners has signal


def test_validate_on_upworthy_excludes_bertscore_when_disabled() -> None:
    pairs = [(_FakeVariant("t1", "a b c"), _FakeVariant("t1", "x y z"))]
    results = rm.validate_on_upworthy(
        pairs=pairs, metrics=["rouge_l", "bertscore"], enable_bertscore=False
    )
    assert "bertscore" not in results
    assert "rouge_l" in results
