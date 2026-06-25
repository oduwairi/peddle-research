"""Tests for scraping schemas."""

from datetime import date

from draper.scraping.schemas import (
    AdCopy,
    AdSource,
    CreativeFormat,
    Platform,
    RawAd,
)


def test_raw_ad_minimal() -> None:
    ad = RawAd(ad_id="123", source=AdSource.BIGSPY)
    assert ad.ad_id == "123"
    assert ad.source == AdSource.BIGSPY
    assert ad.platform == Platform.OTHER
    assert ad.total_engagement == 0


def test_raw_ad_date_parsing() -> None:
    ad = RawAd(
        ad_id="456",
        source=AdSource.BIGSPY,
        first_seen="2025-01-01",
        last_seen="2025-03-01",
    )
    assert ad.first_seen == date(2025, 1, 1)
    assert ad.last_seen == date(2025, 3, 1)
    assert ad.longevity_days == 59


def test_raw_ad_engagement() -> None:
    ad = RawAd(
        ad_id="789",
        source=AdSource.BIGSPY,
        likes=100,
        comments=20,
        shares=10,
        first_seen="2025-01-01",
        last_seen="2025-01-11",
    )
    assert ad.total_engagement == 130
    assert ad.engagement_velocity == 13.0


def test_raw_ad_null_dates() -> None:
    ad = RawAd(ad_id="000", source=AdSource.META_LIBRARY)
    assert ad.longevity_days is None
    assert ad.engagement_velocity is None


def test_raw_ad_serialization() -> None:
    ad = RawAd(
        ad_id="test",
        source=AdSource.BIGSPY,
        platform=Platform.FACEBOOK,
        ad_copy=AdCopy(headline="Buy now", body="Great product"),
        creative_format=CreativeFormat.IMAGE,
        likes=50,
    )
    data = ad.model_dump()
    assert data["ad_id"] == "test"
    assert data["ad_copy"]["headline"] == "Buy now"

    # Round-trip
    ad2 = RawAd.model_validate(data)
    assert ad2.ad_id == ad.ad_id
    assert ad2.likes == 50
