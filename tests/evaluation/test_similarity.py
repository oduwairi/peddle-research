"""Tests for similarity-to-gold diagnostics. Cosine is soft-deps so tests
use rouge_l_f1 as the deterministic check; cosine is exercised through the
``similarity_to_gold`` shape contract.
"""

from __future__ import annotations

from draper.evaluation.judge.similarity import (
    cosine_similarity,
    rouge_l_f1,
    similarity_to_gold,
)


def test_rouge_l_identical() -> None:
    assert rouge_l_f1("Glow All Day Primer.", "Glow All Day Primer.") == 1.0


def test_rouge_l_empty() -> None:
    assert rouge_l_f1("", "anything") == 0.0
    assert rouge_l_f1("anything", "") == 0.0
    assert rouge_l_f1("", "") == 0.0


def test_rouge_l_disjoint() -> None:
    assert rouge_l_f1("Apple banana cherry", "xenon yacht zebra") == 0.0


def test_rouge_l_partial_overlap_is_in_unit_interval() -> None:
    score = rouge_l_f1(
        "Glow All Day Primer with no filter needed.",
        "Get a flawless finish all day with our primer.",
    )
    assert 0.0 < score < 1.0


def test_rouge_l_case_insensitive() -> None:
    a = "GLOW ALL DAY"
    b = "glow all day"
    assert rouge_l_f1(a, b) == 1.0


def test_rouge_l_punctuation_insensitive() -> None:
    a = "Glow, all day!"
    b = "glow all day"
    assert rouge_l_f1(a, b) == 1.0


def test_similarity_to_gold_shape() -> None:
    out = similarity_to_gold("Try our primer.", "Try Before You Buy.")
    assert "rouge_l_f1" in out
    assert "cosine_to_gold" in out
    assert isinstance(out["rouge_l_f1"], float)
    # cosine_to_gold may be None if sentence-transformers isn't installed
    # or the model fails to load; treat as soft.
    assert out["cosine_to_gold"] is None or isinstance(out["cosine_to_gold"], float)


def test_cosine_similarity_returns_none_or_float() -> None:
    """Soft contract: should never raise — returns None on missing dep."""
    val = cosine_similarity("hello world", "greeting world")
    assert val is None or isinstance(val, float)
