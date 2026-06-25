"""Load Upworthy Research Archive for creative-feature validation.

The Upworthy archive (Matias et al., 2021, Nature Scientific Data) contains
randomized A/B tests of news headlines + thumbnails on the Upworthy platform.
Each test has multiple variants, each with real impressions and clicks.

NOTE: This dataset CANNOT directly validate the AdFlex composite scorer
because the scorer expects engagement signals (likes/comments/shares) as
inputs, while Upworthy provides clicks/impressions as the LABEL. Using it
both ways would be circular.

Instead, this loader supports a *complementary* validation hypothesis:
"creative text features predict A/B winners with above-chance accuracy."
This validates the broader thesis premise (creative content matters for
ad performance) without claiming to validate the specific AdFlex scorer.

For each test_id, we identify the winning variant by highest CTR with a
chi-squared significance check, and return paired (winner, loser) records
for downstream pairwise validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from scipy import stats as sp_stats

logger = logging.getLogger("draper")


@dataclass
class UpworthyVariant:
    """A single A/B test variant (one headline + image package)."""

    test_id: str
    headline: str
    excerpt: str
    lede: str
    impressions: int
    clicks: int
    eyecatcher_id: str
    is_winner: bool

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions > 0 else 0.0


@dataclass
class UpworthyTest:
    """An A/B test consisting of multiple competing variants."""

    test_id: str
    variants: list[UpworthyVariant]
    chi2_p_value: float
    has_significant_winner: bool

    @property
    def winner(self) -> UpworthyVariant | None:
        return next((v for v in self.variants if v.is_winner), None)

    @property
    def losers(self) -> list[UpworthyVariant]:
        return [v for v in self.variants if not v.is_winner]


class UpworthyLoader:
    """Load and pair Upworthy A/B test variants for validation."""

    def load(self, path: str | Path) -> list[UpworthyTest]:
        """Load Upworthy CSV and group into tests with winner identification.

        Filters out tests where:
        - Fewer than 2 variants
        - Total impressions across variants is < 100 (too few for stable CTR)
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Upworthy CSV not found: %s", path)
            return []

        df = pl.read_csv(path, infer_schema_length=10_000)
        logger.info("Loaded %d Upworthy variants from %s", len(df), path)

        # Group by test
        tests: list[UpworthyTest] = []
        for test_id, group in df.group_by("clickability_test_id"):
            # Polars group_by returns the key as a tuple
            tid = test_id[0] if isinstance(test_id, tuple) else test_id
            if not isinstance(tid, str):
                continue

            variants_df = group
            if len(variants_df) < 2:
                continue

            total_imp = int(variants_df["impressions"].sum() or 0)
            if total_imp < 100:
                continue

            # Build variants
            raw_variants: list[dict[str, Any]] = []
            for row in variants_df.iter_rows(named=True):
                raw_variants.append(dict(row))

            # Identify winner via chi-squared test on contingency table:
            # rows = variants, columns = [clicks, non_clicks]
            try:
                contingency = [
                    [
                        int(v.get("clicks", 0) or 0),
                        max(0, int(v.get("impressions", 0) or 0) - int(v.get("clicks", 0) or 0)),
                    ]
                    for v in raw_variants
                ]
                _, p_val, _, _ = sp_stats.chi2_contingency(contingency)
            except (ValueError, ZeroDivisionError):
                p_val = 1.0

            # Pick winner = highest CTR
            ctrs = [
                (
                    int(v.get("clicks", 0) or 0) / int(v.get("impressions", 1) or 1)
                    if int(v.get("impressions", 0) or 0) > 0
                    else 0.0
                )
                for v in raw_variants
            ]
            winner_idx = ctrs.index(max(ctrs))

            variants: list[UpworthyVariant] = []
            for i, v in enumerate(raw_variants):
                variants.append(
                    UpworthyVariant(
                        test_id=tid,
                        headline=str(v.get("headline", "") or ""),
                        excerpt=str(v.get("excerpt", "") or ""),
                        lede=str(v.get("lede", "") or ""),
                        impressions=int(v.get("impressions", 0) or 0),
                        clicks=int(v.get("clicks", 0) or 0),
                        eyecatcher_id=str(v.get("eyecatcher_id", "") or ""),
                        is_winner=(i == winner_idx),
                    )
                )

            tests.append(
                UpworthyTest(
                    test_id=tid,
                    variants=variants,
                    chi2_p_value=float(p_val),
                    has_significant_winner=p_val < 0.05,
                )
            )

        logger.info(
            "Built %d Upworthy tests (%d with significant winners)",
            len(tests),
            sum(1 for t in tests if t.has_significant_winner),
        )
        return tests

    @staticmethod
    def to_pairs(
        tests: list[UpworthyTest],
        only_significant: bool = True,
    ) -> list[tuple[UpworthyVariant, UpworthyVariant]]:
        """Convert tests to (winner, loser) pairs for pairwise validation.

        For each test, pairs the winner against each loser.
        """
        pairs: list[tuple[UpworthyVariant, UpworthyVariant]] = []
        for test in tests:
            if only_significant and not test.has_significant_winner:
                continue
            winner = test.winner
            if winner is None:
                continue
            for loser in test.losers:
                pairs.append((winner, loser))
        return pairs
