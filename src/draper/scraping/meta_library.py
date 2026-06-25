"""Meta Ad Library scraper using Apify actor."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

from apify_client import ApifyClient

from draper.scraping.schemas import (
    AdCopy,
    AdSource,
    CreativeFormat,
    Platform,
    RawAd,
)

logger = logging.getLogger("draper")

# Correct Apify actor for Meta Ad Library
META_AD_LIBRARY_ACTOR = "apify/facebook-ads-scraper"

# EU country codes — ads in these countries include spend/impression/demographic data
EU_COUNTRIES = {
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
    "GB",
}


def _build_ad_library_url(
    search_term: str | None = None,
    country: str = "US",
    ad_type: str = "all",
    active_status: str = "active",
) -> str:
    """Build a Meta Ad Library URL with search filters embedded."""
    base = "https://www.facebook.com/ads/library/"
    params = [
        f"active_status={active_status}",
        f"ad_type={ad_type}",
        f"country={country.upper()}",
        "search_type=keyword_unordered",
    ]
    if search_term:
        params.append(f"q={quote(search_term)}")
    return f"{base}?{'&'.join(params)}"


def _parse_spend_bound(value: Any) -> int | None:
    """Parse a spend/impression bound from Meta's range format."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


class MetaLibraryClient:
    """Scraper for Meta Ad Library via Apify.

    For EU/UK ads, this returns spend ranges, impression ranges,
    demographic distribution, and EU total reach.
    For non-EU ads, only creative content, dates, and basic metadata.
    No engagement metrics (likes/comments/shares) for any ad type.
    """

    def __init__(self, apify_token: str) -> None:
        self._client = ApifyClient(apify_token)

    async def search_ads(
        self,
        search_term: str | None = None,
        country: str = "US",
        ad_type: str = "all",
        max_results: int = 100,
        active_only: bool = True,
    ) -> list[RawAd]:
        """Search Meta Ad Library and return normalized RawAd objects.

        Args:
            search_term: Keyword to search for.
            country: Country code (e.g. "US", "DE", "FR").
            ad_type: Ad category ("all", "political_and_issue_ads", etc).
            max_results: Maximum ads to return.
            active_only: Only return currently active ads.
        """
        is_eu = country.upper() in EU_COUNTRIES

        # Build the Ad Library URL with filters embedded
        ad_library_url = _build_ad_library_url(
            search_term=search_term,
            country=country,
            ad_type=ad_type,
            active_status="active" if active_only else "",
        )

        run_input: dict[str, Any] = {
            "startUrls": [{"url": ad_library_url}],
            "resultsLimit": max_results,
            # Enable detailed scraping for EU ads to get spend/impression/demographic data
            "isDetailsPerAd": is_eu,
        }

        logger.info(
            f"Starting Meta Ad Library scrape: {search_term or 'all'}, "
            f"{country} (EU={is_eu}, details={'on' if is_eu else 'off'})"
        )
        logger.debug(f"Ad Library URL: {ad_library_url}")

        actor = self._client.actor(META_AD_LIBRARY_ACTOR)
        run = actor.call(run_input=run_input)
        if run is None:
            logger.warning("Meta Ad Library actor returned no run data")
            return []
        dataset = self._client.dataset(run["defaultDatasetId"])

        ads: list[RawAd] = []
        for item in dataset.iterate_items():
            try:
                ad = self._parse_ad(item, country=country.upper())
                ads.append(ad)
            except Exception as e:
                logger.warning(f"Failed to parse Meta ad: {e}")
                continue

        logger.info(f"Collected {len(ads)} ads from Meta Ad Library")
        return ads

    @staticmethod
    def _parse_ad(data: dict[str, Any], country: str = "") -> RawAd:
        """Parse Apify Meta Ad Library result into RawAd."""
        # Determine creative format
        snapshot = data.get("snapshot", {}) or {}
        creative_format = CreativeFormat.OTHER
        if data.get("videos") or snapshot.get("videos"):
            creative_format = CreativeFormat.VIDEO
        elif data.get("images") or snapshot.get("images"):
            creative_format = CreativeFormat.IMAGE
        elif data.get("cards") or snapshot.get("cards"):
            creative_format = CreativeFormat.CAROUSEL

        # Extract platform
        platforms = data.get("publisherPlatform", data.get("publisherPlatforms", []))
        if isinstance(platforms, str):
            platforms = [platforms]
        platform = Platform.FACEBOOK
        if platforms:
            first = str(platforms[0]).lower()
            if "instagram" in first:
                platform = Platform.INSTAGRAM

        # Extract ad copy from snapshot or top-level fields
        body_raw = snapshot.get("body", data.get("body", ""))
        if isinstance(body_raw, dict):
            body = body_raw.get("text", body_raw.get("markup", ""))
        else:
            body = str(body_raw) if body_raw else ""

        headline = (
            snapshot.get("title", "") or data.get("title", "") or snapshot.get("link_title", "")
        )
        description = snapshot.get("link_description", "") or data.get("linkDescription", "")
        cta = snapshot.get("cta_text", "") or data.get("ctaText", "")

        # Creative URL
        creative_url = ""
        images = snapshot.get("images", data.get("images", []))
        if images and isinstance(images, list) and len(images) > 0:
            img = images[0]
            if isinstance(img, dict):
                creative_url = str(img.get("url", img.get("original_image_url", "")))
            else:
                creative_url = ""

        # Link URL
        link_url = snapshot.get("link_url", "") or data.get("linkUrl", "")

        # EU transparency fields — spend/impression ranges
        spend = data.get("spend", {}) or {}
        spend_lower = _parse_spend_bound(
            spend.get("lower_bound") if isinstance(spend, dict) else None
        )
        spend_upper = _parse_spend_bound(
            spend.get("upper_bound") if isinstance(spend, dict) else None
        )

        impressions_data = data.get("impressions", {}) or {}
        impression_lower = _parse_spend_bound(
            impressions_data.get("lower_bound") if isinstance(impressions_data, dict) else None
        )
        impression_upper = _parse_spend_bound(
            impressions_data.get("upper_bound") if isinstance(impressions_data, dict) else None
        )

        # EU demographic distribution
        demographics: dict[str, Any] = {}
        demo_dist = data.get("demographicDistribution", data.get("demographic_distribution", []))
        if demo_dist:
            demographics["distribution"] = demo_dist
        eu_reach = data.get("euTotalReach", data.get("eu_total_reach"))
        if eu_reach is not None:
            demographics["eu_total_reach"] = eu_reach
        delivery_by_region = data.get("deliveryByRegion", data.get("delivery_by_region", []))
        if delivery_by_region:
            demographics["delivery_by_region"] = delivery_by_region

        # Country
        country_list: list[str] = []
        if country:
            country_list = [country]
        elif data.get("country"):
            raw_country = data["country"]
            country_list = [raw_country] if isinstance(raw_country, str) else list(raw_country)

        return RawAd(
            ad_id=str(data.get("adArchiveID", data.get("id", ""))),
            source=AdSource.META_LIBRARY,
            platform=platform,
            ad_copy=AdCopy(
                headline=headline or "",
                body=body or "",
                description=description or "",
                cta=cta or "",
            ),
            creative_format=creative_format,
            creative_url=creative_url,
            country=country_list,
            demographics=demographics,
            first_seen=data.get("startDate"),
            last_seen=data.get("endDate"),
            # Meta Ad Library does NOT provide engagement metrics
            likes=0,
            comments=0,
            shares=0,
            # EU spend/impression ranges
            spend_lower=spend_lower,
            spend_upper=spend_upper,
            impression_lower=impression_lower,
            impression_upper=impression_upper,
            advertiser_id=str(data.get("pageID", data.get("pageId", ""))),
            advertiser_name=data.get("pageName", ""),
            landing_page_url=link_url or "",
            raw_data=data,
        )
