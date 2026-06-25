"""Train/val/test splits for the scoring-predictor corpus.

Three deliberate strategies — see plan ``Phase 1 — splits.py`` row:

* ``random``: 80 / 10 / 10 stratified by platform. The in-distribution
  baseline; this is what the random-split Spearman gate measures.
* ``heldout-platform``: train on Facebook + TikTok + Twitter + Pinterest,
  validate on a slice of training, test on Reddit. Calibration probe for the
  weak-engagement-platform behavior.
* ``heldout-vertical``: train on every Facebook vertical except
  ``facebook:ecommerce_tech`` (the 2nd-largest FB vertical and a clear
  semantic cluster), test on it. Generalization probe within a single
  platform without losing data scale.

All splits are materialized to disk (parquet) so training and offline eval
read identical rows and the splits are reproducible across machines.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from draper.scoring_predictor.data import Example, examples_to_polars

SplitName = Literal["random", "heldout-platform", "heldout-vertical"]
SPLIT_NAMES: tuple[SplitName, ...] = ("random", "heldout-platform", "heldout-vertical")

# Held-out partitioning targets. Hard-coded because they're load-bearing for
# the validation gates in the plan — accidental drift would silently change
# what the gate measures.
HELDOUT_PLATFORM = "reddit"
HELDOUT_VERTICAL = "facebook:ecommerce_tech"

# Train-base platforms for ``heldout-platform`` (everything except Reddit and
# the meaningless ``other`` bucket).
HELDOUT_PLATFORM_TRAIN_BASE: frozenset[str] = frozenset(
    {"facebook", "tiktok", "twitter", "pinterest"}
)


@dataclass(frozen=True, slots=True)
class Split:
    """One materialized split."""

    name: SplitName
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame

    def write(self, root: Path) -> None:
        out = root / self.name
        out.mkdir(parents=True, exist_ok=True)
        self.train.write_parquet(out / "train.parquet")
        self.val.write_parquet(out / "val.parquet")
        self.test.write_parquet(out / "test.parquet")


def _hash_to_unit(key: str, *, salt: str) -> float:
    """Deterministic [0, 1) hash for stable example assignment.

    Using a hash rather than ``random.shuffle`` means reruns assign the same
    ad to the same split even if the corpus order changes.
    """
    h = hashlib.blake2b(f"{salt}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / (1 << 64)


def _stratified_random(
    df: pl.DataFrame,
    *,
    train_frac: float,
    val_frac: float,
    salt: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Stratify by platform: every platform contributes the same train/val/test ratio."""
    train_parts: list[pl.DataFrame] = []
    val_parts: list[pl.DataFrame] = []
    test_parts: list[pl.DataFrame] = []

    for plat, group in df.group_by("platform"):
        # ``plat`` is a tuple because group_by returns the key-tuple even for
        # a single-column grouping.
        plat_str = plat[0] if isinstance(plat, tuple) else plat
        # ad_id is already a string; don't double-convert it.
        units = [
            _hash_to_unit(ad_id, salt=f"{salt}:{plat_str}")
            for ad_id in group["ad_id"].to_list()
        ]
        with_unit = group.with_columns(pl.Series("_unit", units))
        train_parts.append(with_unit.filter(pl.col("_unit") < train_frac).drop("_unit"))
        val_parts.append(
            with_unit.filter(
                (pl.col("_unit") >= train_frac) & (pl.col("_unit") < train_frac + val_frac)
            ).drop("_unit")
        )
        test_parts.append(
            with_unit.filter(pl.col("_unit") >= train_frac + val_frac).drop("_unit")
        )

    return (pl.concat(train_parts), pl.concat(val_parts), pl.concat(test_parts))


def make_random_split(
    df: pl.DataFrame, *, train_frac: float = 0.8, val_frac: float = 0.1, salt: str = "v1"
) -> Split:
    """80 / 10 / 10 stratified by platform."""
    train, val, test = _stratified_random(df, train_frac=train_frac, val_frac=val_frac, salt=salt)
    return Split(name="random", train=train, val=val, test=test)


def make_heldout_platform_split(
    df: pl.DataFrame,
    *,
    heldout: str = HELDOUT_PLATFORM,
    train_base: frozenset[str] = HELDOUT_PLATFORM_TRAIN_BASE,
    val_frac_of_train: float = 0.1,
    salt: str = "v1",
) -> Split:
    """Train on ``train_base`` platforms, validate on a slice, test on ``heldout``.

    Reddit is the test set; ``other`` is excluded entirely (4 rows, not
    informative). Validation comes from the train-base platforms because we
    need a held-in distribution to early-stop on without contaminating the
    held-out test signal.
    """
    train_pool = df.filter(pl.col("platform").is_in(list(train_base)))
    test = df.filter(pl.col("platform") == heldout)

    units = [
        _hash_to_unit(ad_id, salt=f"{salt}:heldout-platform")
        for ad_id in train_pool["ad_id"].to_list()
    ]
    train_pool = train_pool.with_columns(pl.Series("_unit", units))
    val = train_pool.filter(pl.col("_unit") < val_frac_of_train).drop("_unit")
    train = train_pool.filter(pl.col("_unit") >= val_frac_of_train).drop("_unit")
    return Split(name="heldout-platform", train=train, val=val, test=test)


def make_heldout_vertical_split(
    df: pl.DataFrame,
    *,
    heldout: str = HELDOUT_VERTICAL,
    val_frac_of_train: float = 0.1,
    salt: str = "v1",
) -> Split:
    """Train on every vertical except ``heldout``, test on ``heldout``."""
    train_pool = df.filter(pl.col("vertical") != heldout)
    test = df.filter(pl.col("vertical") == heldout)

    units = [
        _hash_to_unit(ad_id, salt=f"{salt}:heldout-vertical")
        for ad_id in train_pool["ad_id"].to_list()
    ]
    train_pool = train_pool.with_columns(pl.Series("_unit", units))
    val = train_pool.filter(pl.col("_unit") < val_frac_of_train).drop("_unit")
    train = train_pool.filter(pl.col("_unit") >= val_frac_of_train).drop("_unit")
    return Split(name="heldout-vertical", train=train, val=val, test=test)


def make_all_splits(examples: Sequence[Example]) -> list[Split]:
    """Build all three splits from a single Examples corpus."""
    df = examples_to_polars(examples)
    return [
        make_random_split(df),
        make_heldout_platform_split(df),
        make_heldout_vertical_split(df),
    ]


def load_split(root: Path, name: SplitName) -> Split:
    """Read a previously-materialized split from disk."""
    base = Path(root) / name
    return Split(
        name=name,
        train=pl.read_parquet(base / "train.parquet"),
        val=pl.read_parquet(base / "val.parquet"),
        test=pl.read_parquet(base / "test.parquet"),
    )
