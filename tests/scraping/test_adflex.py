"""Tests for AdFlex client ad parsing logic."""

from draper.scraping.adflex import AdFlexClient
from draper.scraping.schemas import (
    AdSource,
    CreativeFormat,
    Platform,
)

# Sample ad data matching AdFlex API response structure
SAMPLE_FACEBOOK_AD = {
    "id": 11347479,
    "engagements": {
        "reactions": 10564857,
        "shares": 52,
        "comments": 138599,
        "views": None,
    },
    "display_site_url": "meesho.com",
    "updated_at": "2026-03-16T13:13:53.000000Z",
    "title": "Prices too good to scroll away",
    "active_days": 579,
    "attachments": [
        {
            "id": 239261805,
            "files": {"main": "https://cdn.adflex.io/facebook_ads/attachments/example.jpg"},
            "type": "image",
            "subtitle": "Deep Cleansing Foot Pads",
            "description": "",
        },
        {
            "id": 239261806,
            "files": {"main": "https://cdn.adflex.io/facebook_ads/attachments/example2.jpg"},
            "type": "image",
            "subtitle": "Detox Patch",
            "description": "",
        },
    ],
    "owner": {
        "id": 768850,
        "name": "Meesho",
        "avatar_url": "https://cdn.adflex.io/avatar.jpg",
        "is_verified": True,
    },
    "locations": [{"code": "IN", "label": "India"}],
    "placements": [
        {"code": 1, "label": "Feed"},
        {"code": 2, "label": "Right Column"},
    ],
}

SAMPLE_TIKTOK_AD = {
    "id": 104611,
    "title": "Couldn't do what we do without Shopify",
    "engagements": {
        "likes": 707591,
        "shares": 2017,
        "favorites": 29860,
        "comments": 34,
        "plays": 295607699,
    },
    "duration": 47,
    "seen_counts": 1528,
    "updated_at": "2026-03-25T06:23:48.000000Z",
    "active_days": 267,
    "attachments": [
        {
            "id": 1,
            "files": {"main": "https://cdn.adflex.io/tiktok_ads/video.mp4"},
            "type": "video",
            "subtitle": "",
            "description": "",
        }
    ],
    "owner": {
        "id": 21325,
        "avatar_url": "https://cdn.adflex.io/avatar.webp",
        "name": "Olivia & the Zelons",
    },
    "locations": [{"code": "CA", "label": "Canada"}],
}

SAMPLE_X_AD = {
    "id": 1074592,
    "title": "Shop must-haves for your home",
    "display_url": "amazon.in",
    "engagements": {"reposts": 4387, "replies": 296, "likes": 75529},
    "seen_counts": 28169,
    "updated_at": "2025-08-31T17:02:08.000000Z",
    "active_days": 142,
    "attachments": [
        {
            "id": 1,
            "files": {"main": "https://cdn.adflex.io/x_ads/image.png"},
            "type": "image",
            "subtitle": "Shop Now",
        }
    ],
    "owner": {
        "id": 110615,
        "avatar_url": "https://cdn.adflex.io/avatar.jpg",
        "name": "Amazon",
        "is_blue_verified": 1,
    },
    "locations": [{"code": "IN", "label": "India"}],
    "placements": [{"code": "HOME_TIMELINES", "label": "Home Timelines"}],
}

SAMPLE_PINTEREST_AD = {
    "id": 4561087,
    "display_url": "homedepot.com",
    "engagements": {"saves": 150949, "reactions": 3, "repins": 145854},
    "seen_counts": 11,
    "updated_at": "2026-01-26T00:41:51.000000Z",
    "active_days": 3135,
    "attachments": [
        {
            "id": 1,
            "files": {"main": "https://cdn.adflex.io/pinterest_ads/image.jpg"},
            "type": "image",
            "subtitle": "Personalized Moon Phase 65% OFF!",
            "description": "NOW IT IS POSSIBLE TO KNOW WHAT THE MOON LOOKS LIKE ON A SPECIAL DATE",
        }
    ],
    "owner": {"id": 36, "name": "The Home Depot", "is_verified": 1},
    "locations": [{"code": "WW", "label": "World Wide"}],
    "placements": [{"code": "BROWSE", "label": "Browse"}],
}

SAMPLE_REDDIT_AD = {
    "id": 99999,
    "title": "Check out our product",
    "description": "Amazing product description",
    "display_url": "example.com",
    "engagements": {"upvotes": 6595, "comments": 2299},
    "seen_counts": 100,
    "updated_at": "2026-03-20T00:00:00.000000Z",
    "active_days": 30,
    "attachments": [],
    "owner": {"id": 1, "name": "TestAdvertiser"},
    "locations": [{"code": "US", "label": "United States"}],
    "placements": [],
}


def test_parse_facebook_ad() -> None:
    ad = AdFlexClient._parse_ad(SAMPLE_FACEBOOK_AD, "facebook")

    assert ad.ad_id == "11347479"
    assert ad.source == AdSource.ADFLEX
    assert ad.platform == Platform.FACEBOOK
    assert ad.ad_copy.headline == "Prices too good to scroll away"
    assert ad.ad_copy.body == "Deep Cleansing Foot Pads"
    assert ad.active_days == 579
    assert ad.likes == 10564857
    assert ad.comments == 138599
    assert ad.shares == 52
    assert ad.advertiser_name == "Meesho"
    assert ad.advertiser_id == "768850"
    assert ad.country == ["IN"]
    assert ad.placements == ["Feed", "Right Column"]
    assert ad.landing_page_url == "meesho.com"
    # Multiple attachments → carousel
    assert ad.creative_format == CreativeFormat.CAROUSEL
    # updated_at → last_seen
    assert ad.last_seen is not None


def test_parse_tiktok_ad() -> None:
    ad = AdFlexClient._parse_ad(SAMPLE_TIKTOK_AD, "tiktok")

    assert ad.ad_id == "104611"
    assert ad.platform == Platform.TIKTOK
    assert ad.creative_format == CreativeFormat.VIDEO
    assert ad.likes == 707591
    assert ad.comments == 34
    assert ad.shares == 2017
    assert ad.views == 295607699
    assert ad.reactions == 29860  # favorites
    assert ad.active_days == 267
    assert ad.country == ["CA"]
    assert ad.advertiser_name == "Olivia & the Zelons"


def test_parse_x_ad() -> None:
    ad = AdFlexClient._parse_ad(SAMPLE_X_AD, "x")

    assert ad.ad_id == "1074592"
    assert ad.platform == Platform.TWITTER
    assert ad.likes == 75529
    assert ad.comments == 296  # replies → comments
    assert ad.shares == 4387  # reposts → shares
    assert ad.landing_page_url == "amazon.in"  # display_url fallback
    assert ad.last_seen is not None
    assert ad.placements == ["Home Timelines"]


def test_parse_pinterest_ad() -> None:
    ad = AdFlexClient._parse_ad(SAMPLE_PINTEREST_AD, "pinterest")

    assert ad.ad_id == "4561087"
    assert ad.platform == Platform.PINTEREST
    assert ad.reactions == 150949  # saves → reactions
    assert ad.shares == 145854  # repins → shares
    assert ad.landing_page_url == "homedepot.com"  # display_url fallback
    assert ad.ad_copy.body == "Personalized Moon Phase 65% OFF!"  # subtitle
    # attachment description → ad_copy.description (no top-level description)
    assert ad.ad_copy.description.startswith("NOW IT IS POSSIBLE")
    assert ad.last_seen is not None


def test_parse_reddit_ad() -> None:
    ad = AdFlexClient._parse_ad(SAMPLE_REDDIT_AD, "reddit")

    assert ad.ad_id == "99999"
    assert ad.platform == Platform.REDDIT
    assert ad.comments == 2299
    assert ad.reactions == 6595  # upvotes
    assert ad.creative_format == CreativeFormat.OTHER  # no attachments
    assert ad.country == ["US"]
    assert ad.landing_page_url == "example.com"  # display_url fallback
    # top-level description takes priority over attachment description
    assert ad.ad_copy.description == "Amazing product description"
    assert ad.last_seen is not None


def test_engagement_computed_properties() -> None:
    ad = AdFlexClient._parse_ad(SAMPLE_FACEBOOK_AD, "facebook")

    assert ad.total_engagement > 0
    assert ad.longevity_days == 579
    velocity = ad.engagement_velocity
    assert velocity is not None
    assert velocity > 0


def test_parse_ad_missing_fields() -> None:
    """Parsing should handle missing optional fields gracefully."""
    minimal = {
        "id": 1,
        "title": "Test",
        "engagements": {},
        "attachments": [],
        "owner": {},
        "locations": [],
    }
    ad = AdFlexClient._parse_ad(minimal, "facebook")
    assert ad.ad_id == "1"
    assert ad.likes == 0
    assert ad.comments == 0
    assert ad.country == []
    assert ad.advertiser_name == ""


def test_raw_data_preserved() -> None:
    ad = AdFlexClient._parse_ad(SAMPLE_FACEBOOK_AD, "facebook")
    assert ad.raw_data["id"] == 11347479
    assert "engagements" in ad.raw_data
