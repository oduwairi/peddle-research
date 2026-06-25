"""Tests for I/O utilities."""

from pathlib import Path

from draper.scraping.schemas import AdSource, RawAd
from draper.utils.io import Checkpoint, append_jsonl, read_jsonl, write_jsonl


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    records = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
    path = tmp_path / "test.jsonl"
    count = write_jsonl(records, path)
    assert count == 2

    loaded = read_jsonl(path)
    assert loaded == records


def test_jsonl_pydantic(tmp_path: Path) -> None:
    ads = [
        RawAd(ad_id="1", source=AdSource.BIGSPY, likes=10),
        RawAd(ad_id="2", source=AdSource.META_LIBRARY, likes=20),
    ]
    path = tmp_path / "ads.jsonl"
    write_jsonl(ads, path)

    loaded = read_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0]["ad_id"] == "1"
    assert loaded[1]["likes"] == 20


def test_append_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "append.jsonl"
    append_jsonl([{"x": 1}], path)
    append_jsonl([{"x": 2}], path)

    loaded = read_jsonl(path)
    assert len(loaded) == 2


def test_checkpoint(tmp_path: Path) -> None:
    cp = Checkpoint(tmp_path / "scrape")
    assert cp.get("page") is None

    cp.update(page=5, count=100)
    assert cp.get("page") == 5

    # Reload from disk
    cp2 = Checkpoint(tmp_path / "scrape")
    assert cp2.get("page") == 5
    assert cp2.get("count") == 100

    cp2.clear()
    cp3 = Checkpoint(tmp_path / "scrape")
    assert cp3.get("page") is None
