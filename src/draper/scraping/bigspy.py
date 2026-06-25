"""BigSpy API client for ad intelligence scraping."""

from __future__ import annotations

import asyncio
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
from draper.utils.io import Checkpoint

logger = logging.getLogger("draper")

BIGSPY_BASE_URL = "https://api.bigspy.com/api/v1"

# BigSpy platform mappings
_PLATFORM_MAP: dict[str, Platform] = {
    "facebook": Platform.FACEBOOK,
    "instagram": Platform.INSTAGRAM,
    "tiktok": Platform.TIKTOK,
    "youtube": Platform.YOUTUBE,
    "google": Platform.GOOGLE,
    "linkedin": Platform.LINKEDIN,
    "pinterest": Platform.PINTEREST,
    "twitter": Platform.TWITTER,
}

_FORMAT_MAP: dict[str, CreativeFormat] = {
    "image": CreativeFormat.IMAGE,
    "video": CreativeFormat.VIDEO,
    "carousel": CreativeFormat.CAROUSEL,
    "collection": CreativeFormat.COLLECTION,
}


class BigSpyClient:
    """Async client for the BigSpy ad intelligence API.

    Args:
        api_key: BigSpy API key.
        rate_limiter: Optional rate limiter instance. Creates a default if not provided.
    """

    def __init__(
        self,
        api_key: str,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=BIGSPY_BASE_URL,
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )
        self._limiter = rate_limiter or RateLimiter(requests_per_minute=30, burst_size=5)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> BigSpyClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def _request(self, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Make a rate-limited API request with retry logic."""
        async with self._limiter:
            for attempt in range(3):
                try:
                    response = await self._client.request(
                        method,
                        endpoint,
                        **kwargs,
                    )
                    response.raise_for_status()
                    data: dict[str, Any] = response.json()
                    return data
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait)
                        continue
                    raise
                except httpx.RequestError as e:
                    if attempt < 2:
                        logger.warning(f"Request error: {e}, retrying (attempt {attempt + 1})")
                        await asyncio.sleep(1)
                        continue
                    raise
        return {}  # unreachable but satisfies type checker

    async def search_ads(
        self,
        keyword: str | None = None,
        platform: str = "facebook",
        country: str | None = None,
        industry: str | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "last_seen",
        order: str = "desc",
    ) -> dict[str, Any]:
        """Search for ads matching criteria.

        Returns:
            API response with 'data' (list of ads) and pagination info.
        """
        params: dict[str, Any] = {
            "api_key": self._api_key,
            "platform": platform,
            "page": page,
            "page_size": page_size,
            "sort_by": sort_by,
            "order": order,
        }
        if keyword:
            params["keyword"] = keyword
        if country:
            params["country"] = country
        if industry:
            params["industry"] = industry

        return await self._request("GET", "/ad/search", params=params)

    async def get_ad_detail(self, ad_id: str) -> dict[str, Any]:
        """Get detailed information for a specific ad."""
        params = {"api_key": self._api_key, "ad_id": ad_id}
        return await self._request("GET", "/ad/detail", params=params)

    async def get_advertiser_ads(
        self,
        advertiser_id: str,
        platform: str = "facebook",
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """Get all ads from a specific advertiser."""
        params: dict[str, Any] = {
            "api_key": self._api_key,
            "advertiser_id": advertiser_id,
            "platform": platform,
            "page": page,
            "page_size": page_size,
        }
        return await self._request("GET", "/advertiser/ads", params=params)

    async def search_ads_paginated(
        self,
        keyword: str | None = None,
        platform: str = "facebook",
        country: str | None = None,
        industry: str | None = None,
        max_results: int = 500,
        page_size: int = 20,
        checkpoint: Checkpoint | None = None,
    ) -> list[RawAd]:
        """Search with automatic pagination and checkpoint/resume.

        Args:
            keyword: Search keyword.
            platform: Ad platform.
            country: Country filter.
            industry: Industry/vertical filter.
            max_results: Maximum total ads to fetch.
            page_size: Results per page.
            checkpoint: Optional checkpoint for resume support.

        Returns:
            List of RawAd objects.
        """
        start_page = 1
        collected: list[RawAd] = []

        if checkpoint:
            start_page = checkpoint.get("page", 1)
            logger.info(f"Resuming from page {start_page}")

        page = start_page
        while len(collected) < max_results:
            logger.info(f"Fetching page {page} ({len(collected)}/{max_results} collected)")
            response = await self.search_ads(
                keyword=keyword,
                platform=platform,
                country=country,
                industry=industry,
                page=page,
                page_size=page_size,
            )

            ads_data = response.get("data", [])
            if not ads_data:
                logger.info("No more results, stopping pagination")
                break

            for ad_data in ads_data:
                if len(collected) >= max_results:
                    break
                try:
                    raw_ad = self._parse_ad(ad_data)
                    collected.append(raw_ad)
                except Exception as e:
                    logger.warning(f"Failed to parse ad: {e}")
                    continue

            if checkpoint:
                checkpoint.update(page=page + 1, collected=len(collected))

            page += 1

        logger.info(f"Collected {len(collected)} ads total")
        if checkpoint:
            checkpoint.clear()

        return collected

    @staticmethod
    def _parse_ad(data: dict[str, Any]) -> RawAd:
        """Parse a BigSpy API ad object into canonical RawAd schema."""
        platform_str = str(data.get("platform", "")).lower()
        format_str = str(data.get("creative_type", "")).lower()

        return RawAd(
            ad_id=str(data.get("ad_id", data.get("id", ""))),
            source=AdSource.BIGSPY,
            platform=_PLATFORM_MAP.get(platform_str, Platform.OTHER),
            ad_copy=AdCopy(
                headline=data.get("title", ""),
                body=data.get("content", ""),
                description=data.get("description", ""),
                cta=data.get("call_to_action", ""),
            ),
            creative_format=_FORMAT_MAP.get(format_str, CreativeFormat.OTHER),
            creative_url=data.get("creative_url", ""),
            country=[data["country"]] if data.get("country") else [],
            vertical=data.get("industry", ""),
            first_seen=data.get("first_seen_at") or data.get("created_at"),
            last_seen=data.get("last_seen_at") or data.get("updated_at"),
            likes=int(data.get("likes", 0) or 0),
            comments=int(data.get("comments", 0) or 0),
            shares=int(data.get("shares", 0) or 0),
            is_redelivered=bool(data.get("is_redelivered", False)),
            advertiser_id=str(data.get("advertiser_id", data.get("page_id", ""))),
            advertiser_name=data.get("advertiser_name", data.get("page_name", "")),
            advertiser_ad_count=int(data.get("advertiser_total_ads", 0) or 0),
            landing_page_url=data.get("landing_page", ""),
            raw_data=data,
        )
