"""Source selector: filter + stratified sample."""

from __future__ import annotations

import json
from pathlib import Path

from draper.construction_v2.config import ConstructionV2Config, SelectionConfig
from draper.construction_v2.dataset.source_selector import (
    load_source_ads_by_id,
    select_source_ads,
)


def _scored_row(
    ad_id: str, platform: str, composite: float, *, headline: str = "Buy now."
) -> dict[str, object]:
    return {
        "ad": {
            "ad_id": ad_id,
            "source": "adflex",
            "platform": platform,
            "ad_copy": {
                "headline": headline,
                "body": "Body text long enough to count.",
                "description": "",
                "cta": "",
            },
        },
        "composite_score": composite,
        "signal_scores": {},
        "tier_probs": {},
        "tier": "high",
        "scoring_version": "v3",
    }


def _write_scored_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _make_config(scored_path: Path, output_dir: Path, **selection: object) -> ConstructionV2Config:
    kwargs: dict[str, object] = {
        "scored_ads_path": str(scored_path),
        "target_count": 10,
        "min_composite": 0.5,
        # Test data doesn't include business_vertical_confidence, so disable
        # the gate to allow test ads through. Real data includes it.
        "business_vertical_min_confidence": 0,
        # Use balanced stratification (not allow_unbalanced) so test data
        # with a single platform doesn't hit PlatformConcentration.
        "allow_unbalanced": False,
    }
    kwargs.update(selection)
    return ConstructionV2Config(
        output_dir=str(output_dir),
        final_dir=str(output_dir / "final"),
        audit_dir=str(output_dir / "_audit"),
        selection=SelectionConfig(**kwargs),  # type: ignore[arg-type]
    )


def test_select_filters_by_min_composite(tmp_path: Path) -> None:
    scored = tmp_path / "scored.jsonl"
    _write_scored_jsonl(
        scored,
        [
            _scored_row("a", "facebook", 0.9),
            _scored_row("b", "facebook", 0.3),
            _scored_row("c", "facebook", 0.7),
        ],
    )
    config = _make_config(scored, tmp_path / "out")
    chosen = select_source_ads(config)
    assert {ad.ad_id for ad in chosen} == {"a", "c"}


def test_select_drops_empty_copy(tmp_path: Path) -> None:
    scored = tmp_path / "scored.jsonl"
    rows = [
        _scored_row("good", "facebook", 0.9),
        _scored_row("blank", "facebook", 0.95, headline=""),
    ]
    # Zero out all copy fields on the blank row.
    rows[1]["ad"]["ad_copy"] = {  # type: ignore[index]
        "headline": "",
        "body": "",
        "description": "",
        "cta": "",
    }
    _write_scored_jsonl(scored, rows)
    config = _make_config(scored, tmp_path / "out")
    chosen = select_source_ads(config)
    assert [ad.ad_id for ad in chosen] == ["good"]


def test_select_balanced_stratifies_per_platform(tmp_path: Path) -> None:
    scored = tmp_path / "scored.jsonl"
    rows: list[dict[str, object]] = []
    for i in range(10):
        rows.append(_scored_row(f"fb-{i}", "facebook", 0.95))
    for i in range(2):
        rows.append(_scored_row(f"tt-{i}", "tiktok", 0.6))
    _write_scored_jsonl(scored, rows)
    config = _make_config(scored, tmp_path / "out", target_count=4, allow_unbalanced=False)
    chosen = select_source_ads(config)
    platforms = {ad.platform for ad in chosen}
    assert "facebook" in platforms
    assert "tiktok" in platforms


def test_select_writes_audit_parquet(tmp_path: Path) -> None:
    scored = tmp_path / "scored.jsonl"
    _write_scored_jsonl(scored, [_scored_row("a", "facebook", 0.8)])
    config = _make_config(scored, tmp_path / "out")
    select_source_ads(config)
    assert (tmp_path / "out" / "_audit" / "selection.parquet").exists()


def _image_capable_row(
    ad_id: str,
    *,
    platform: str = "facebook",
    composite: float = 0.9,
    creative_format: str = "image",
    creative_url: str = "https://cdn.adflex.io/a.jpg",
) -> dict[str, object]:
    return {
        "ad": {
            "ad_id": ad_id,
            "source": "adflex",
            "platform": platform,
            "ad_copy": {
                "headline": "Buy now.",
                "body": "Body text long enough to count.",
                "description": "",
                "cta": "",
            },
            "creative_format": creative_format,
            "creative_url": creative_url,
        },
        "composite_score": composite,
        "signal_scores": {},
        "tier_probs": {},
        "tier": "high",
        "scoring_version": "v3",
    }


def test_image_brief_skill_filters_to_image_capable_only(tmp_path: Path) -> None:
    """When skill=image_brief, only image+carousel ads with a URL pass.

    Caption availability is NOT gated here — captioning is a construction
    step that runs after select. Captions don't exist yet at this point.
    """
    scored = tmp_path / "scored.jsonl"
    _write_scored_jsonl(
        scored,
        [
            _image_capable_row("img-1", creative_format="image"),
            _image_capable_row("car-1", creative_format="carousel"),
            _image_capable_row("vid-1", creative_format="video"),
            _image_capable_row("noURL", creative_format="image", creative_url=""),
        ],
    )
    base = _make_config(scored, tmp_path / "out")
    config = base.model_copy(update={"skill": "image_brief"})
    chosen = select_source_ads(config)
    assert {ad.ad_id for ad in chosen} == {"img-1", "car-1"}


def test_load_source_ads_by_id(tmp_path: Path) -> None:
    scored = tmp_path / "scored.jsonl"
    _write_scored_jsonl(
        scored,
        [
            _scored_row("a", "facebook", 0.9),
            _scored_row("b", "facebook", 0.8),
            _scored_row("c", "tiktok", 0.7),
        ],
    )
    config = _make_config(scored, tmp_path / "out")
    out = load_source_ads_by_id(config, ["a", "c"])
    assert set(out) == {"a", "c"}
    assert out["c"].platform == "tiktok"
