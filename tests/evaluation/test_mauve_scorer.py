"""Unit tests for the MAUVE eval arm.

Tests use synthetic v3 parquet rows and a mocked ``mauve.compute_mauve``
so they run on CPU in <2s without pulling GPT-2-large weights.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from draper.evaluation.mauve_reference import (
    ALL_KEY,
    ContaminationError,
    load_reference_corpus,
)


def _write_synthetic_v3(path: Path, rows: list[dict[str, object]]) -> None:
    pl.DataFrame(rows).write_parquet(path)


def _row(
    *,
    headline: str,
    body: str = "",
    description: str = "",
    platform: str = "facebook",
    tier: str = "high",
) -> dict[str, object]:
    return {
        "ad_copy_headline": headline,
        "ad_copy_body": body,
        "ad_copy_description": description,
        "platform": platform,
        "tier": tier,
    }


def test_load_reference_corpus_filters_by_tier_and_platform(tmp_path: Path) -> None:
    pq = tmp_path / "v3.parquet"
    _write_synthetic_v3(
        pq,
        [
            _row(headline="hi-tier-fb-a", platform="facebook", tier="high"),
            _row(headline="hi-tier-fb-b", platform="facebook", tier="high"),
            _row(headline="hi-tier-tt-a", platform="tiktok", tier="high"),
            _row(headline="mid-tier-fb", platform="facebook", tier="medium"),
            _row(headline="low-tier-fb", platform="facebook", tier="low"),
        ],
    )

    out = load_reference_corpus(
        parquet_path=pq,
        tier="high",
        platforms=["facebook", "tiktok"],
    )

    assert ALL_KEY in out
    assert len(out[ALL_KEY]) == 3  # 3 high-tier rows
    assert len(out["facebook"]) == 2
    assert len(out["tiktok"]) == 1
    # Medium / low rows must not leak in.
    assert not any("mid-tier" in b for b in out[ALL_KEY])
    assert not any("low-tier" in b for b in out[ALL_KEY])


def test_load_reference_corpus_dedupes_identical_blobs(tmp_path: Path) -> None:
    pq = tmp_path / "v3.parquet"
    _write_synthetic_v3(
        pq,
        [
            _row(headline="same", body="copy", platform="facebook"),
            _row(headline="same", body="copy", platform="facebook"),
            _row(headline="other", body="copy", platform="facebook"),
        ],
    )
    out = load_reference_corpus(parquet_path=pq, tier="high", platforms=["facebook"])
    assert len(out[ALL_KEY]) == 2  # one duplicate dropped


def test_load_reference_corpus_contamination_strict_aborts(tmp_path: Path) -> None:
    pq = tmp_path / "v3.parquet"
    _write_synthetic_v3(
        pq,
        [
            _row(headline="Buy our amazing product NOW", platform="facebook"),
            _row(headline="other ad", platform="facebook"),
        ],
    )

    # Same text, different whitespace/casing → hash should still collide.
    held_out = ["buy our amazing  product now"]

    with pytest.raises(ContaminationError) as exc_info:
        load_reference_corpus(
            parquet_path=pq,
            tier="high",
            platforms=["facebook"],
            held_out_texts=held_out,
            contamination_strict=True,
        )
    assert "contamination" in str(exc_info.value).lower()


def test_load_reference_corpus_filters_held_out_by_default(tmp_path: Path) -> None:
    """Default behavior: held-out overlap is filtered out, not aborted."""
    pq = tmp_path / "v3.parquet"
    _write_synthetic_v3(
        pq,
        [
            _row(headline="held out ad", platform="facebook"),
            _row(headline="kept ad 1", platform="facebook"),
            _row(headline="kept ad 2", platform="tiktok"),
        ],
    )
    out = load_reference_corpus(
        parquet_path=pq,
        tier="high",
        platforms=["facebook", "tiktok"],
        held_out_texts=["held out ad"],
    )
    # 1 of 3 filtered out, 2 remain.
    assert len(out[ALL_KEY]) == 2
    assert len(out["facebook"]) == 1
    assert "held out ad" not in out[ALL_KEY]


def test_load_reference_corpus_total_overlap_still_aborts(tmp_path: Path) -> None:
    """If the held-out set covers every reference row, abort — nothing to compare."""
    pq = tmp_path / "v3.parquet"
    _write_synthetic_v3(
        pq,
        [
            _row(headline="ad-a", platform="facebook"),
            _row(headline="ad-b", platform="tiktok"),
        ],
    )
    with pytest.raises(ContaminationError) as exc_info:
        load_reference_corpus(
            parquet_path=pq,
            tier="high",
            platforms=["facebook", "tiktok"],
            held_out_texts=["ad-a", "ad-b"],
        )
    assert "nothing left" in str(exc_info.value).lower()


def test_load_reference_corpus_no_overlap_passes(tmp_path: Path) -> None:
    pq = tmp_path / "v3.parquet"
    _write_synthetic_v3(
        pq,
        [
            _row(headline="real ad copy", platform="facebook"),
        ],
    )
    out = load_reference_corpus(
        parquet_path=pq,
        tier="high",
        platforms=["facebook"],
        held_out_texts=["entirely different held-out copy"],
    )
    assert len(out[ALL_KEY]) == 1


def test_load_reference_corpus_round_trips_through_cache(tmp_path: Path) -> None:
    pq = tmp_path / "v3.parquet"
    cache = tmp_path / "cache"
    _write_synthetic_v3(
        pq,
        [
            _row(headline="a", platform="facebook"),
            _row(headline="b", platform="tiktok"),
        ],
    )
    first = load_reference_corpus(
        parquet_path=pq,
        tier="high",
        platforms=["facebook", "tiktok"],
        cache_dir=cache,
    )
    assert (cache / "ALL.parquet").exists()
    assert (cache / "facebook.parquet").exists()

    # Now point at a non-existent parquet and rely on cache.
    second = load_reference_corpus(
        parquet_path=tmp_path / "does-not-exist.parquet",
        tier="high",
        platforms=["facebook", "tiktok"],
        cache_dir=cache,
    )
    assert second == first


def _install_fake_mauve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake ``mauve`` and ``mauve.compute_mauve`` module.

    Lets us exercise ``score_configs`` without loading GPT-2-large.
    """

    fake_mauve = types.ModuleType("mauve")

    class _Result:
        def __init__(self, score: float) -> None:
            self.mauve = score

    def fake_compute_mauve(
        p_features=None,  # type: ignore[no-untyped-def]
        q_features=None,
        **_kwargs,
    ):
        n_p = 0 if p_features is None else len(p_features)
        n_q = 0 if q_features is None else len(q_features)
        # Deterministic synthetic score that depends on shape, so we get
        # a different number per slice.
        return _Result(0.5 + 0.001 * (n_p + n_q))

    fake_mauve.compute_mauve = fake_compute_mauve  # type: ignore[attr-defined]

    fake_compute_mauve_mod = types.ModuleType("mauve.compute_mauve")

    def fake_get_features(
        features=None,  # type: ignore[no-untyped-def]
        tokenized_texts=None,
        texts=None,
        featurize_model_name=None,
        max_len=None,
        device_id=None,
        name=None,
        batch_size=None,
        verbose=False,
        use_float64=False,
    ):
        n = len(texts or [])
        # Deterministic 8-d "embedding": index, length, hash mix. Lets us
        # confirm slices line up without invoking torch.
        rng = np.random.default_rng(abs(hash(name)) % 2**32)
        return rng.standard_normal(size=(n, 8))

    fake_compute_mauve_mod.get_features_from_input = fake_get_features  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mauve", fake_mauve)
    monkeypatch.setitem(sys.modules, "mauve.compute_mauve", fake_compute_mauve_mod)


def test_score_configs_writes_per_config_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mauve(monkeypatch)
    from draper.evaluation.mauve_scorer import score_configs
    from draper.evaluation.schemas import Brief

    # Two synthetic briefs across two platforms.
    briefs = {
        "ex1": Brief(
            example_id="ex1",
            task_format="copywriting",
            platform="facebook",
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

    # Materialize cleaned inferences on disk. GOLD reads from the same
    # cache as A — the rationale-stripped clean copy, not Brief.reference_assistant.
    clean_dir = tmp_path / "inferences_clean"
    raw_dir = tmp_path / "inferences"
    for cfg in ["A", "GOLD"]:
        d = clean_dir / cfg
        d.mkdir(parents=True)
        for ex_id in ["ex1", "ex2"]:
            (d / f"{ex_id}.json").write_text(
                '{"example_id":"' + ex_id + '","config":"' + cfg + '",'
                '"assistant_text_clean":"' + cfg + "-text-" + ex_id + '",'
                '"extractor_model":"test","extracted_at":"2026-01-01T00:00:00Z",'
                '"raw_text_sha256":"deadbeef"}'
            )

    reference = {
        ALL_KEY: ["ref-1", "ref-2", "ref-3"],
        "facebook": ["ref-1", "ref-2"],
        "tiktok": ["ref-3"],
    }

    written = score_configs(
        briefs_by_id=briefs,
        reference_corpus_by_platform=reference,
        configs=["A", "GOLD"],
        inferences_clean_dir=clean_dir,
        inferences_raw_dir=raw_dir,
        out_dir=tmp_path / "mauve_scores",
        platforms=["facebook", "tiktok"],
        embedding_model="fake-model",
        bootstrap_n=3,
        seed=42,
    )

    assert "A" in written
    assert "GOLD" in written
    a_df = pl.read_parquet(written["A"])
    gold_df = pl.read_parquet(written["GOLD"])

    # Per config: one row for ALL + one per platform = 3 rows.
    assert a_df.height == 3
    assert set(a_df["platform"].to_list()) == {"ALL", "facebook", "tiktok"}
    assert set(gold_df["platform"].to_list()) == {"ALL", "facebook", "tiktok"}

    # Schema sanity.
    for col in (
        "config",
        "platform",
        "mauve",
        "ci_low",
        "ci_high",
        "n_gen",
        "n_ref",
        "embedding_model",
        "bootstrap_n",
        "seed",
        "created_at",
    ):
        assert col in a_df.columns, f"missing column {col}"


def test_alias_platform_meta_maps_to_facebook() -> None:
    from draper.evaluation.mauve_scorer import _alias_platform

    assert _alias_platform("meta") == "facebook"


def test_alias_platform_x_maps_to_twitter() -> None:
    from draper.evaluation.mauve_scorer import _alias_platform

    assert _alias_platform("x") == "twitter"


def test_alias_platform_passthrough_for_canonical_names() -> None:
    from draper.evaluation.mauve_scorer import _alias_platform

    assert _alias_platform("pinterest") == "pinterest"
    assert _alias_platform("tiktok") == "tiktok"
    assert _alias_platform("facebook") == "facebook"
    assert _alias_platform("reddit") == "reddit"


def test_summarize_groups_by_platform(tmp_path: Path) -> None:
    # Hand-build per-config parquets to exercise summarize in isolation.
    from draper.evaluation.mauve_scorer import summarize

    out_dir = tmp_path / "mauve_scores"
    out_dir.mkdir()
    for cfg, score in (("A", 0.4), ("B", 0.7)):
        rows = [
            {
                "config": cfg,
                "platform": plat,
                "mauve": score + 0.01,
                "ci_low": score - 0.02,
                "ci_high": score + 0.02,
                "n_gen": 100,
                "n_ref": 1000,
                "n_dropped": 0,
                "embedding_model": "fake",
                "bootstrap_n": 10,
                "seed": 42,
                "created_at": "2026-01-01T00:00:00Z",
            }
            for plat in (ALL_KEY, "facebook", "tiktok")
        ]
        pl.DataFrame(rows).write_parquet(out_dir / f"{cfg}.parquet")

    headline = summarize(out_dir=out_dir, configs=["A", "B"], by=None)
    assert set(headline["config"].to_list()) == {"A", "B"}
    assert "mauve_mean" in headline.columns

    by_platform = summarize(out_dir=out_dir, configs=["A", "B"], by=["platform"])
    assert by_platform.height == 6  # 2 configs × 3 platforms
