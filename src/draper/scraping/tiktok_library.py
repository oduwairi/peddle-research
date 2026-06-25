"""TikTok Ad Library scraper using Apify actor."""

from __future__ import annotations

import logging
from typing import Any

from apify_client import ApifyClient

from draper.scraping.schemas import (
    AdCopy,
    AdSource,
    CreativeFormat,
    Platform,
    RawAd,
)

logger = logging.getLogger("draper")

# Apify actor for TikTok Ad Library
TIKTOK_AD_LIBRARY_ACTOR = "apify/tiktok-ads-library"


class TikTokLibraryClient:
    """Scraper for TikTok Ad Library via Apify.

    TikTok ads are video-first. Impression counts are sometimes available.
    """

    def __init__(self, apify_token: str) -> None:
        self._client = ApifyClient(apify_token)

    async def search_ads(
        self,
        search_term: str | None = None,
        country: str = "US",
        max_results: int = 100,
    ) -> list[RawAd]:
        """Search TikTok Ad Library and return normalized RawAd objects.

        Args:
            search_term: Keyword to search for.
            country: Country code.
            max_results: Maximum ads to return.
        """
        run_input: dict[str, Any] = {
            "country": country,
            "maxItems": max_results,
        }
        if search_term:
            run_input["searchTerms"] = [search_term]

        logger.info(f"Starting TikTok Ad Library scrape: {search_term or 'all'}, {country}")

        actor = self._client.actor(TIKTOK_AD_LIBRARY_ACTOR)
        run = actor.call(run_input=run_input)
        if run is None:
            logger.warning("TikTok Ad Library actor returned no run data")
            return []
        dataset = self._client.dataset(run["defaultDatasetId"])

        ads: list[RawAd] = []
        for item in dataset.iterate_items():
            try:
                ads.append(self._parse_ad(item))
            except Exception as e:
                logger.warning(f"Failed to parse TikTok ad: {e}")
                continue

        logger.info(f"Collected {len(ads)} ads from TikTok Ad Library")
        return ads

    @staticmethod
    def _parse_ad(data: dict[str, Any]) -> RawAd:
        """Parse Apify TikTok Ad Library result into RawAd."""
        # TikTok ads are predominantly video
        creative_format = CreativeFormat.VIDEO
        if data.get("adFormat", "").lower() == "image":
            creative_format = CreativeFormat.IMAGE

        # Extract regions
        regions = data.get("targetRegions", data.get("regions", []))
        countries = [str(r) for r in regions] if regions else []

        return RawAd(
            ad_id=str(data.get("adId", data.get("id", ""))),
            source=AdSource.TIKTOK_LIBRARY,
            platform=Platform.TIKTOK,
            ad_copy=AdCopy(
                headline=data.get("title", ""),
                body=data.get("caption", data.get("text", "")),
                cta=data.get("callToAction", ""),
            ),
            creative_format=creative_format,
            creative_url=data.get("videoUrl", data.get("imageUrl", "")),
            country=countries,
            first_seen=data.get("firstShown", data.get("createTime")),
            last_seen=data.get("lastShown", data.get("updateTime")),
            # TikTok sometimes exposes impression counts
            likes=int(data.get("likes", 0) or 0),
            comments=int(data.get("comments", 0) or 0),
            shares=int(data.get("shares", 0) or 0),
            advertiser_id=str(data.get("advertiserId", "")),
            advertiser_name=data.get("advertiserName", ""),
            landing_page_url=data.get("landingPageUrl", ""),
            raw_data=data,
        )
