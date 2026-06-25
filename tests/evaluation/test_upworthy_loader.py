"""Tests for the Upworthy loader."""

from __future__ import annotations

import csv
from pathlib import Path

from draper.evaluation.upworthy_loader import UpworthyLoader


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_row(
    test_id: str,
    headline: str,
    impressions: int,
    clicks: int,
    eyecatcher: str = "img1",
) -> dict[str, object]:
    return {
        "": "0",
        "created_at": "2014-11-20 06:43:16.005",
        "updated_at": "2016-04-02 16:33:38.062",
        "clickability_test_id": test_id,
        "excerpt": "test excerpt",
        "headline": headline,
        "lede": "test lede",
        "slug": "test-slug",
        "eyecatcher_id": eyecatcher,
        "impressions": impressions,
        "clicks": clicks,
        "significance": 100.0,
        "first_place": True,
        "winner": True,
        "share_text": "share",
        "square": "",
        "test_week": 201446,
    }


class TestUpworthyLoader:
    def test_loads_basic_test(self, tmp_path: Path) -> None:
        path = tmp_path / "upworthy.csv"
        _write_csv(
            path,
            [
                _make_row("test1", "Headline A is great", 1000, 50),  # CTR 5%
                _make_row("test1", "Headline B is bad", 1000, 10),  # CTR 1%
            ],
        )
        loader = UpworthyLoader()
        tests = loader.load(path)

        assert len(tests) == 1
        t = tests[0]
        assert len(t.variants) == 2
        winner = t.winner
        assert winner is not None
        assert winner.headline == "Headline A is great"
        assert t.has_significant_winner  # 5% vs 1% should be significant

    def test_filters_low_impression_tests(self, tmp_path: Path) -> None:
        path = tmp_path / "upworthy.csv"
        _write_csv(
            path,
            [
                _make_row("test1", "A", 30, 5),
                _make_row("test1", "B", 30, 2),
            ],
        )
        loader = UpworthyLoader()
        tests = loader.load(path)
        assert len(tests) == 0  # < 100 total impressions filtered

    def test_filters_single_variant_tests(self, tmp_path: Path) -> None:
        path = tmp_path / "upworthy.csv"
        _write_csv(
            path,
            [
                _make_row("test1", "Only one variant", 1000, 50),
            ],
        )
        loader = UpworthyLoader()
        tests = loader.load(path)
        assert len(tests) == 0

    def test_to_pairs(self, tmp_path: Path) -> None:
        path = tmp_path / "upworthy.csv"
        _write_csv(
            path,
            [
                _make_row("test1", "A wins", 5000, 250),  # CTR 5%
                _make_row("test1", "B loses", 5000, 50),  # CTR 1%
                _make_row("test1", "C also loses", 5000, 75),  # CTR 1.5%
            ],
        )
        loader = UpworthyLoader()
        tests = loader.load(path)
        pairs = loader.to_pairs(tests, only_significant=True)
        assert len(pairs) == 2  # winner paired with each loser
        for winner, loser in pairs:
            assert winner.is_winner
            assert not loser.is_winner

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        loader = UpworthyLoader()
        tests = loader.load(tmp_path / "nonexistent.csv")
        assert tests == []
