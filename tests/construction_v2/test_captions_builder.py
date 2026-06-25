"""Tests for the VLM captioning infrastructure (construction_v2/captions/)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from draper.construction.batch.types import BatchResponse
from draper.construction_v2.captions.builder import (
    CAPTION_TASK_FORMAT,
    CaptionableAd,
    build_caption_request,
    estimate_caption_cost_usd,
    iter_captionable_ads,
    parse_caption_response,
    write_caption_rows,
)

# ---------------------------------------------------------------------------
# iter_captionable_ads
# ---------------------------------------------------------------------------


def _scored_jsonl(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    p = tmp_path / "scored.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


class TestIterCaptionableAds:
    def test_filters_to_image_and_carousel_with_url(self, tmp_path: Path) -> None:
        path = _scored_jsonl(
            tmp_path,
            [
                {
                    "ad": {
                        "ad_id": "a1",
                        "creative_format": "image",
                        "creative_url": "https://x/a.jpg",
                    }
                },
                {
                    "ad": {
                        "ad_id": "a2",
                        "creative_format": "carousel",
                        "creative_url": "https://x/b.jpg",
                    }
                },
                {
                    "ad": {
                        "ad_id": "a3",
                        "creative_format": "video",
                        "creative_url": "https://x/c.mp4",
                    }
                },
                {
                    "ad": {
                        "ad_id": "a4",
                        "creative_format": "other",
                        "creative_url": "https://x/d.mp4",
                    }
                },
                {"ad": {"ad_id": "a5", "creative_format": "image", "creative_url": ""}},
                {"ad": {"ad_id": "a6", "creative_format": "image"}},  # no url at all
            ],
        )
        ids = [a.ad_id for a in iter_captionable_ads(path)]
        assert ids == ["a1", "a2"]

    def test_excludes_given_ids(self, tmp_path: Path) -> None:
        path = _scored_jsonl(
            tmp_path,
            [
                {
                    "ad": {
                        "ad_id": "a1",
                        "creative_format": "image",
                        "creative_url": "https://x/a.jpg",
                    }
                },
                {
                    "ad": {
                        "ad_id": "a2",
                        "creative_format": "image",
                        "creative_url": "https://x/b.jpg",
                    }
                },
                {
                    "ad": {
                        "ad_id": "a3",
                        "creative_format": "image",
                        "creative_url": "https://x/c.jpg",
                    }
                },
            ],
        )
        ids = [a.ad_id for a in iter_captionable_ads(path, exclude_ad_ids={"a2"})]
        assert ids == ["a1", "a3"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list(iter_captionable_ads(tmp_path / "nope.jsonl"))


# ---------------------------------------------------------------------------
# build_caption_request
# ---------------------------------------------------------------------------


class TestBuildCaptionRequest:
    def _ad(self) -> CaptionableAd:
        return CaptionableAd(
            ad_id="42", creative_url="https://cdn.adflex.io/x.jpg", creative_format="image"
        )

    def test_request_shape_strategic_prompt(self) -> None:
        # The default prompt_variant flipped to "literal" (caption-source flip);
        # pin "strategic" explicitly here to keep asserting the strategic prompt.
        req = build_caption_request(
            self._ad(), model="gemini-3.5-flash", prompt_variant="strategic"
        )
        assert req.custom_id == "caption-42"
        assert req.system is None
        assert req.model == "gemini-3.5-flash"
        # Single user message with content list.
        assert len(req.messages) == 1
        content = req.messages[0]["content"]
        assert isinstance(content, list)
        # Image first, then prompt text.
        assert content[0]["type"] == "image_url"
        assert content[0]["url"] == "https://cdn.adflex.io/x.jpg"
        assert content[1]["type"] == "text"
        assert "what is in this image AND why it works" in content[1]["text"]

    def test_literal_prompt_variant(self) -> None:
        req = build_caption_request(self._ad(), model="gpt-4o-mini", prompt_variant="literal")
        assert req.messages[0]["content"][1]["text"].startswith(
            "You are describing an advertising creative image"
        )
        assert "literally visible" in req.messages[0]["content"][1]["text"]

    def test_unknown_prompt_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown prompt_variant"):
            build_caption_request(self._ad(), model="gpt-4o-mini", prompt_variant="bogus")


# ---------------------------------------------------------------------------
# parse_caption_response
# ---------------------------------------------------------------------------


class TestParseCaptionResponse:
    def test_happy_path(self) -> None:
        ad = CaptionableAd(ad_id="42", creative_url="https://x.jpg", creative_format="image")
        resp = BatchResponse(
            custom_id="caption-42",
            content="A wide shot of …",
            input_tokens=800,
            output_tokens=120,
            model="gemini-3.5-flash",
        )
        row = parse_caption_response(
            resp,
            ad=ad,
            captioner_model="gemini-3.5-flash",
            prompt_variant="strategic",
        )
        assert row.ad_id == "42"
        assert row.caption == "A wide shot of …"
        assert row.captioner_model == "gemini-3.5-flash"
        assert row.caption_prompt_version == "strategic"
        assert row.provider_error == ""

    def test_provider_error_propagates(self) -> None:
        ad = CaptionableAd(ad_id="42", creative_url="https://x.jpg", creative_format="image")
        resp = BatchResponse(custom_id="caption-42", content="", error="content_filter")
        row = parse_caption_response(
            resp,
            ad=ad,
            captioner_model="gemini-3.5-flash",
            prompt_variant="strategic",
        )
        assert row.caption == ""
        assert row.provider_error == "content_filter"


# ---------------------------------------------------------------------------
# write_caption_rows
# ---------------------------------------------------------------------------


class TestWriteCaptionRows:
    def _row(self, ad_id: str, caption: str = "x") -> object:
        from draper.construction_v2.captions.builder import CaptionRow

        return CaptionRow(
            ad_id=ad_id,
            creative_url=f"https://x/{ad_id}.jpg",
            creative_format="image",
            caption=caption,
            captioner_model="gemini-3.5-flash",
            caption_prompt_version="strategic",
            captioned_at="2026-05-26T00:00:00+00:00",
            provider_error="",
        )

    def test_creates_parquet_on_first_call(self, tmp_path: Path) -> None:
        out = tmp_path / "captions.parquet"
        rows = [self._row("a1"), self._row("a2")]
        path = write_caption_rows(rows, output_path=out)
        assert path == out
        df = pl.read_parquet(out)
        assert sorted(df["ad_id"].to_list()) == ["a1", "a2"]
        assert set(df.columns) == {
            "ad_id",
            "creative_url",
            "creative_format",
            "caption",
            "captioner_model",
            "caption_prompt_version",
            "captioned_at",
            "provider_error",
        }

    def test_overwrites_existing_ad_ids(self, tmp_path: Path) -> None:
        out = tmp_path / "captions.parquet"
        write_caption_rows([self._row("a1", "old")], output_path=out)
        write_caption_rows([self._row("a1", "new"), self._row("a2", "fresh")], output_path=out)
        df = pl.read_parquet(out).sort("ad_id")
        assert df["ad_id"].to_list() == ["a1", "a2"]
        captions = dict(zip(df["ad_id"].to_list(), df["caption"].to_list(), strict=True))
        assert captions["a1"] == "new"
        assert captions["a2"] == "fresh"


# ---------------------------------------------------------------------------
# estimate_caption_cost_usd
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_zero_ads_is_zero(self) -> None:
        assert (
            estimate_caption_cost_usd(n_ads=0, input_price_per_m=0.30, output_price_per_m=2.50)
            == 0.0
        )

    def test_default_50pct_batch_discount(self) -> None:
        # 1000 ads × (850×0.30 + 120×2.50) / 1M × 0.5 (batch)
        # = 1000 × (255 + 300) / 1e6 × 0.5
        # = 1000 × 0.000555 × 0.5 = 0.2775
        est = estimate_caption_cost_usd(n_ads=1000, input_price_per_m=0.30, output_price_per_m=2.50)
        assert abs(est - 0.2775) < 1e-6


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_task_format_constant_is_stable() -> None:
    # Pin the registry task_format so on-disk paths don't drift silently.
    assert CAPTION_TASK_FORMAT == "vlm_caption_v1"


# ---------------------------------------------------------------------------
# load_captions_lookup + enrich_source_ads_with_captions
# ---------------------------------------------------------------------------


class _FakeSourceAd:
    """Stand-in for SourceAd with the surface enrich_source_ads_with_captions reads."""

    def __init__(self, ad_id: str, raw: dict[str, object] | None = None) -> None:
        self.ad_id = ad_id
        self.raw: dict[str, object] = raw if raw is not None else {}


class TestCaptionsJoin:
    def _row(self, ad_id: str, caption: str = "x", error: str = "") -> object:
        from draper.construction_v2.captions.builder import CaptionRow

        return CaptionRow(
            ad_id=ad_id,
            creative_url=f"https://x/{ad_id}.jpg",
            creative_format="image",
            caption=caption,
            captioner_model="gemini-3.5-flash",
            caption_prompt_version="strategic",
            captioned_at="2026-05-26T00:00:00+00:00",
            provider_error=error,
        )

    def test_lookup_empty_when_file_missing(self, tmp_path: Path) -> None:
        from draper.construction_v2.captions.builder import load_captions_lookup

        assert load_captions_lookup(tmp_path / "nope.parquet") == {}

    def test_lookup_skips_provider_error_rows(self, tmp_path: Path) -> None:
        from draper.construction_v2.captions.builder import (
            load_captions_lookup,
            write_caption_rows,
        )

        out = tmp_path / "captions.parquet"
        write_caption_rows(
            [
                self._row("a1", caption="good"),
                self._row("a2", caption="", error="content_filter"),
                self._row("a3", caption="also good"),
            ],
            output_path=out,
        )
        lookup = load_captions_lookup(out)
        assert lookup == {"a1": "good", "a3": "also good"}

    def test_enrich_attaches_caption_to_raw(self, tmp_path: Path) -> None:
        from draper.construction_v2.captions.builder import (
            enrich_source_ads_with_captions,
            write_caption_rows,
        )
        from draper.construction_v2.teacher.image_brief_single_pass import CAPTION_RAW_KEY

        out = tmp_path / "captions.parquet"
        write_caption_rows([self._row("a1", "the caption")], output_path=out)
        ads = [_FakeSourceAd("a1"), _FakeSourceAd("a2")]
        enriched, missing = enrich_source_ads_with_captions(
            ads, captions_parquet=out, require_caption=False
        )
        # Both ads pass through (require_caption=False).
        assert {a.ad_id for a in enriched} == {"a1", "a2"}
        # a1 picked up the caption; a2 didn't (no caption available).
        a1 = next(a for a in enriched if a.ad_id == "a1")
        a2 = next(a for a in enriched if a.ad_id == "a2")
        assert a1.raw[CAPTION_RAW_KEY] == "the caption"
        assert CAPTION_RAW_KEY not in a2.raw
        assert missing == ["a2"]

    def test_enrich_drops_missing_when_required(self, tmp_path: Path) -> None:
        from draper.construction_v2.captions.builder import (
            enrich_source_ads_with_captions,
            write_caption_rows,
        )

        out = tmp_path / "captions.parquet"
        write_caption_rows([self._row("a1", "cap")], output_path=out)
        enriched, missing = enrich_source_ads_with_captions(
            [_FakeSourceAd("a1"), _FakeSourceAd("a2")],
            captions_parquet=out,
            require_caption=True,
        )
        assert {a.ad_id for a in enriched} == {"a1"}
        assert missing == ["a2"]
