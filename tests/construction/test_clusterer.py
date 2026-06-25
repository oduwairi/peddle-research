"""Tests for the ad clusterer (copywriting-only)."""

from __future__ import annotations

from draper.construction.clusterer import AdClusterer
from draper.construction.schemas import FormatConfig, PromptStyle
from draper.scoring.schemas import ScoredAd
from draper.scraping.schemas import AdCopy, AdSource, Platform, RawAd


def _make_ad(
    ad_id: str,
    advertiser: str = "BrandA",
    platform: Platform = Platform.FACEBOOK,
    vertical: str = "facebook:broad",
    score: float = 0.80,
    tier: str | None = None,
    headline: str = "Test headline",
    body: str = "A test body long enough to pass the copywriting filter.",
) -> ScoredAd:
    derived_tier = tier or ("high" if score >= 0.70 else ("low" if score <= 0.35 else "medium"))  # noqa: PLR2004
    return ScoredAd(
        ad=RawAd(
            ad_id=ad_id,
            source=AdSource.ADFLEX,
            platform=platform,
            advertiser_name=advertiser,
            vertical=vertical,
            business_vertical=vertical,
            business_vertical_confidence=1.0,
            ad_copy=AdCopy(headline=headline, body=body),
        ),
        composite_score=score,
        tier=derived_tier,
    )


def _formats(copywriting_min: float = 0.70) -> dict[str, FormatConfig]:
    return {
        "copywriting": FormatConfig(
            target=100,
            score_min=copywriting_min,
            valid_styles=[PromptStyle.BACKTRANSLATION],
            style_ratios={PromptStyle.BACKTRANSLATION.value: 1.0},
        ),
    }


class TestClusterByAdvertiser:
    def test_groups_by_name(self) -> None:
        ads = [
            _make_ad("a1", advertiser="Brand1"),
            _make_ad("a2", advertiser="Brand1"),
            _make_ad("a3", advertiser="Brand1"),
            _make_ad("a4", advertiser="Brand2"),
        ]
        clusterer = AdClusterer(ads, formats=_formats())
        clusters = clusterer.cluster_by_advertiser(min_size=3)
        assert len(clusters) == 1
        assert clusters[0].advertiser_name == "Brand1"
        assert len(clusters[0].ad_ids) == 3

    def test_min_size_filter(self) -> None:
        ads = [
            _make_ad("a1", advertiser="Brand1"),
            _make_ad("a2", advertiser="Brand1"),
        ]
        clusterer = AdClusterer(ads, formats=_formats())
        clusters = clusterer.cluster_by_advertiser(min_size=3)
        assert len(clusters) == 0


class TestClusterByVertical:
    def test_vertical_floor_filters_small_verticals(self) -> None:
        # Two ads in each vertical but floor is 30 → all filtered out.
        ads = [_make_ad(f"a{i}", vertical="facebook:ecommerce") for i in range(3)]
        clusterer = AdClusterer(ads, formats=_formats())
        clusters = clusterer.cluster_by_vertical()
        assert clusters == []

    def test_vertical_floor_lets_large_verticals_through(self) -> None:
        ads = [_make_ad(f"a{i}", vertical="facebook:ecommerce") for i in range(30)]
        clusterer = AdClusterer(ads, formats=_formats())
        clusters = clusterer.cluster_by_vertical()
        assert len(clusters) == 1
        assert clusters[0].vertical == "facebook:ecommerce"


class TestCopywritingAds:
    def test_requires_score_and_total_copy(self) -> None:
        long_body = (
            "A reasonably long body copy that easily clears the "
            "sixty-character total-copy threshold for copywriting."
        )
        long_headline = "A punchy headline long enough on its own to clear the total threshold"
        ads = [
            _make_ad("a1", score=0.80, body=long_body),  # body alone passes
            _make_ad("a2", score=0.80, headline=long_headline, body=""),  # headline-only passes
            _make_ad("a3", score=0.80, headline="short", body="tiny"),  # too little copy
            _make_ad("a4", score=0.50, body=long_body),  # below score threshold
        ]
        clusterer = AdClusterer(ads, formats=_formats(copywriting_min=0.70))
        result = clusterer.get_copywriting_ads()
        ids = {a.ad.ad_id for a in result}
        assert ids == {"a1", "a2"}


class TestLookup:
    def test_finds_by_id(self) -> None:
        ads = [_make_ad("x1"), _make_ad("x2")]
        clusterer = AdClusterer(ads, formats=_formats())
        assert clusterer.lookup("x1") is not None
        assert clusterer.lookup("nonexistent") is None
