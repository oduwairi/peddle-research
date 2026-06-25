"""Tests for the difficulty stratification module."""

from __future__ import annotations

import random

from draper.construction.difficulty import (
    CONFLICTING,
    DIFFICULTY_RATIOS,
    MULTI_CONSTRAINT,
    SPARSE,
    STANDARD,
    apply_difficulty,
    directive_for,
    sample_difficulty,
)
from draper.construction.schemas import TaskFormat
from draper.scoring.schemas import ScoredAd
from draper.scraping.schemas import AdCopy, AdSource, CreativeFormat, Platform, RawAd


def _make_ad(ad_id: str, tier: str = "high") -> ScoredAd:
    return ScoredAd(
        ad=RawAd(
            ad_id=ad_id,
            source=AdSource.ADFLEX,
            platform=Platform.FACEBOOK,
            creative_format=CreativeFormat.IMAGE,
            vertical="facebook:broad",
            ad_copy=AdCopy(headline=f"ad {ad_id}", body="body", cta="cta"),
            advertiser_name="TestBrand",
        ),
        composite_score=0.8,
        signal_scores={},
        tier_probs={},
        tier=tier,
        scoring_version="v3",
    )


class TestDistribution:
    def test_ratios_sum_to_one(self) -> None:
        assert abs(sum(DIFFICULTY_RATIOS.values()) - 1.0) < 1e-9

    def test_copywriting_never_sees_sparse(self) -> None:
        """Copywriting is sparse-disallowed (single-ad format); rolls land on
        standard so the 20% budget is spent on meaningful variations."""
        rng = random.Random(42)
        n = 10000
        counts: dict[str, int] = dict.fromkeys(DIFFICULTY_RATIOS, 0)
        for _ in range(n):
            counts[sample_difficulty(rng, TaskFormat.COPYWRITING)] += 1
        assert counts[SPARSE] == 0
        # Standard absorbs the 20% that would have gone to sparse.
        observed_standard = counts[STANDARD] / n
        assert 0.78 < observed_standard < 0.82, (
            f"standard should be ~0.80, got {observed_standard:.3f}"
        )


class TestDirective:
    def test_each_tier_has_directive(self) -> None:
        for tier in DIFFICULTY_RATIOS:
            text = directive_for(tier)
            assert text
            assert len(text) > 20

    def test_directives_are_distinct(self) -> None:
        texts = {directive_for(t) for t in DIFFICULTY_RATIOS}
        assert len(texts) == len(DIFFICULTY_RATIOS)

    def test_unknown_tier_falls_back_to_standard(self) -> None:
        assert directive_for("bogus") == directive_for(STANDARD)


class TestApplyDifficulty:
    def test_standard_unchanged(self) -> None:
        batch = [_make_ad(str(i)) for i in range(5)]
        rng = random.Random(0)
        assert apply_difficulty(batch, STANDARD, TaskFormat.COPYWRITING, rng) == batch

    def test_multi_constraint_unchanged(self) -> None:
        batch = [_make_ad(str(i)) for i in range(5)]
        rng = random.Random(0)
        assert (
            apply_difficulty(batch, MULTI_CONSTRAINT, TaskFormat.COPYWRITING, rng) == batch
        )

    def test_sparse_on_copywriting_is_defensive_no_op(self) -> None:
        batch = [_make_ad("x")]
        rng = random.Random(0)
        result = apply_difficulty(batch, SPARSE, TaskFormat.COPYWRITING, rng)
        assert len(result) == 1

    def test_conflicting_on_copywriting_single_ad_unchanged(self) -> None:
        batch = [_make_ad("x")]
        rng = random.Random(0)
        # Copywriting is single-ad: conflicting has nothing to shuffle.
        result = apply_difficulty(batch, CONFLICTING, TaskFormat.COPYWRITING, rng)
        assert result == batch

    def test_empty_batch_returns_empty(self) -> None:
        rng = random.Random(0)
        assert apply_difficulty([], SPARSE, TaskFormat.COPYWRITING, rng) == []
