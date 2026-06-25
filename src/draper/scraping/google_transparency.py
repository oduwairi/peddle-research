"""Google Ads Transparency Center scraper using SerpApi or Apify."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from draper.scraping.rate_limiter import RateLimiter
from draper.scraping.schemas import (
    AdCopy,
    AdSource,
    CreativeFormat,
    Platform,
    RawAd,
)

logger = logging.getLogger("draper")

SERPAPI_BASE_URL = "https://serpapi.com/search"


class GoogleTransparencyClient:
    """Scraper for Google Ads Transparency Center via SerpApi.

    Key feature: provides `total_days_shown` as a direct longevity signal.
    """

    def __init__(
        self,
        serpapi_key: str,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._api_key = serpapi_key
        self._client = httpx.AsyncClient(timeout=30.0)
        self._limiter = rate_limiter or RateLimiter(requests_per_minute=30, burst_size=5)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> GoogleTransparencyClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def search_ads(
        self,
        advertiser_id: str | None = None,
        query: str | None = None,
        region: str = "US",
        max_results: int = 100,
    ) -> list[RawAd]:
        """Search Google Ads Transparency Center.

        Args:
            advertiser_id: Google advertiser ID to look up.
            query: Text search query.
            region: Region filter.
            max_results: Maximum ads to return.
        """
        params: dict[str, Any] = {
            "engine": "google_ads_transparency_center",
            "api_key": self._api_key,
            "region": region,
        }
        if advertiser_id:
            params["advertiser_id"] = advertiser_id
        if query:
            params["q"] = query

        logger.info(f"Searching Google Transparency: {query or advertiser_id}")

        ads: list[RawAd] = []
        async with self._limiter:
            response = await self._client.get(SERPAPI_BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        for ad_data in data.get("ads", [])[:max_results]:
            try:
                ads.append(self._parse_ad(ad_data))
            except Exception as e:
                logger.warning(f"Failed to parse Google ad: {e}")
                continue

        logger.info(f"Collected {len(ads)} ads from Google Transparency")
        return ads

    @staticmethod
    def _parse_ad(data: dict[str, Any]) -> RawAd:
        """Parse SerpApi Google Transparency result into RawAd."""
        format_str = str(data.get("format", "")).lower()
        creative_format = CreativeFormat.OTHER
        if "video" in format_str:
            creative_format = CreativeFormat.VIDEO
        elif "image" in format_str:
            creative_format = CreativeFormat.IMAGE
        elif "text" in format_str:
            creative_format = CreativeFormat.TEXT

        # Google provides region data as list
        regions = data.get("regions", [])
        countries = [r.get("code", r) if isinstance(r, dict) else str(r) for r in regions]

        return RawAd(
            ad_id=str(data.get("ad_id", data.get("creative_id", ""))),
            source=AdSource.GOOGLE_TRANSPARENCY,
            platform=Platform.GOOGLE,
            ad_copy=AdCopy(
                headline=data.get("title", ""),
                body=data.get("text", ""),
                description=data.get("description", ""),
            ),
            creative_format=creative_format,
            creative_url=data.get("image_url", ""),
            country=countries,
            first_seen=data.get("first_shown"),
            last_seen=data.get("last_shown"),
            # Google Transparency does not provide engagement metrics
            likes=0,
            comments=0,
            shares=0,
            advertiser_id=str(data.get("advertiser_id", "")),
            advertiser_name=data.get("advertiser_name", ""),
            landing_page_url=data.get("destination_url", ""),
            raw_data=data,
        )
