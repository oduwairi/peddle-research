"""AdFlex API client for ad intelligence scraping."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
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


def _parse_iso_date(value: Any) -> date | None:
    """Parse ISO 8601 datetime strings (with or without TZ) to date.

    AdFlex's detail endpoint returns timestamps like "2023-06-27T16:50:25.000000Z".
    Direct assignment to a `date`-typed Pydantic field bypasses validators,
    so we coerce here before assignment.
    """
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


ADFLEX_BASE_URL = "https://api.adflex.io/api/v1"

# AdFlex platform → canonical Platform mapping
_PLATFORM_MAP: dict[str, Platform] = {
    "facebook": Platform.FACEBOOK,
    "tiktok": Platform.TIKTOK,
    "x": Platform.TWITTER,
    "pinterest": Platform.PINTEREST,
    "reddit": Platform.REDDIT,
}

# Attachment type → CreativeFormat mapping
_FORMAT_MAP: dict[str, CreativeFormat] = {
    "image": CreativeFormat.IMAGE,
    "video": CreativeFormat.VIDEO,
    "carousel": CreativeFormat.CAROUSEL,
}

# Supported platforms for search
SUPPORTED_PLATFORMS = ("facebook", "tiktok", "x", "pinterest", "reddit")


class AdFlexClient:
    """Async client for the AdFlex ad intelligence API.

    Uses POST-based search with cursor pagination (last_hit).
    100 credits per search call, 18 ads per page.

    Args:
        api_key: AdFlex API key.
        rate_limiter: Optional rate limiter instance.
    """

    def __init__(
        self,
        api_key: str,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=ADFLEX_BASE_URL,
            timeout=30.0,
        )
        self._limiter = rate_limiter or RateLimiter(requests_per_minute=30, burst_size=5)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AdFlexClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a rate-limited API request with retry logic."""
        async with self._limiter:
            for attempt in range(3):
                try:
                    response = await self._client.request(
                        method,
                        endpoint,
                        params={"api_key": self._api_key},
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

    async def get_filters(self, platform: str = "facebook") -> dict[str, Any]:
        """Get available search filters for a platform."""
        return await self._request("GET", f"/filters/{platform}/search")

    async def search_ads(
        self,
        platform: str = "facebook",
        keyword: str | None = None,
        orderby: str = "popularity",
        # Select filters (list of integer codes from /filters endpoint)
        countries: list[int] | None = None,
        ecommerce: list[int] | None = None,
        cms: list[int] | None = None,
        funnels: list[int] | None = None,
        affiliate: list[int] | None = None,
        tracker: list[int] | None = None,
        arbitrages: list[int] | None = None,
        interests: list[int] | None = None,
        behaviors: list[int] | None = None,
        call_to_actions: list[int] | None = None,
        owner_category: list[int] | None = None,
        categories: list[int] | None = None,
        ad_format: list[int] | None = None,
        placements: list[int] | None = None,
        gender: list[int] | None = None,
        devices: list[int] | None = None,
        ad_os: list[int] | None = None,
        # Range filters: {api_key: [min, max]} e.g. {"reaction": [0, 1000]}
        ranges: dict[str, list[int]] | None = None,
        # Search field: raw list of search_field dicts for advanced queries
        # e.g. [{"type": "url_chain", "text": "shopify"}]
        # Takes precedence over keyword if both are set.
        search_field: list[dict[str, str]] | None = None,
        # Pagination
        last_hit: int | None = None,
        page: int = 1,
    ) -> dict[str, Any]:
        """Search for ads on a platform.

        Args:
            platform: One of facebook, tiktok, x, pinterest, reddit.
            keyword: Text search query.
            orderby: Sort order (popularity, days_active,
                     seen_counts, updated_at, oldest, most_relevant).
            countries: Country filter codes.
            ecommerce: E-commerce tech stack filter codes (Shopify, etc).
            cms: CMS filter codes (WordPress, etc).
            funnels: Funnel platform codes (ClickFunnels, etc).
            affiliate: Affiliate network codes.
            tracker: Tracker codes (Voluum, etc).
            arbitrages: Arbitrage codes.
            interests: Interest/vertical codes (TikTok, X, Pinterest, Reddit).
            behaviors: Behavior filter codes (Facebook only).
            call_to_actions: CTA type codes.
            owner_category: Advertiser industry codes (X only, 919 codes).
            categories: Content category codes (Pinterest only, 33 codes).
            ad_format: Creative format codes (image, video, carousel).
            placements: Placement codes.
            gender: Gender targeting codes.
            devices: Device filter codes.
            ad_os: OS filter codes.
            ranges: Numeric range filters, e.g. {"reaction": [0, 1000]}.
            last_hit: Cursor for pagination (from previous response).
            page: Page number (use with last_hit for pages > 1).

        Returns:
            API response with data.ads list, data.last_hit,
            and data.has_next_page.
        """
        body: dict[str, Any] = {
            "page": page,
            "orderby": orderby,
        }
        if search_field:
            body["search_field"] = search_field
        elif keyword:
            body["search_field"] = [{"type": "text", "text": keyword}]

        # Map param names to API body keys
        _filter_map: dict[str, Any] = {
            "countries": countries,
            "ecommerce": ecommerce,
            "cms": cms,
            "funnels": funnels,
            "affiliate": affiliate,
            "tracker": tracker,
            "arbitrages": arbitrages,
            "interests": interests,
            "behaviors": behaviors,
            "type_call_to_actions": call_to_actions,
            "owner_category": owner_category,
            "categories": categories,
            "format": ad_format,
            "placements": placements,
            "gender": gender,
            "devices": devices,
            "os": ad_os,
        }
        for api_key, value in _filter_map.items():
            if value:
                body[api_key] = value

        # Range filters go directly into body
        if ranges:
            body.update(ranges)

        if last_hit is not None:
            body["last_hit"] = last_hit

        return await self._request("POST", f"/{platform}/ads/search", json=body)

    async def get_ad_detail(self, ad_id: int, platform: str = "facebook") -> dict[str, Any]:
        """Get detailed information for a specific ad."""
        return await self._request("GET", f"/{platform}/ads/{ad_id}")

    async def search_ads_paginated(
        self,
        platform: str = "facebook",
        max_results: int = 500,
        checkpoint: Checkpoint | None = None,
        **search_kwargs: Any,
    ) -> list[RawAd]:
        """Search with automatic cursor pagination and checkpoint/resume.

        Args:
            platform: Ad platform.
            max_results: Maximum total ads to fetch.
            checkpoint: Optional checkpoint for resume support.
            **search_kwargs: All other params forwarded to search_ads().

        Returns:
            List of RawAd objects.
        """
        page = 1
        last_hit: int | None = None
        collected: list[RawAd] = []

        if checkpoint:
            page = checkpoint.get("page", 1)
            last_hit = checkpoint.get("last_hit")
            logger.info(f"Resuming from page {page}")

        while len(collected) < max_results:
            logger.info(f"Fetching page {page} ({len(collected)}/{max_results} collected)")
            response = await self.search_ads(
                platform=platform,
                last_hit=last_hit,
                page=page,
                **search_kwargs,
            )

            resp_data = response.get("data", {})
            ads_data = resp_data.get("ads", [])
            if not ads_data:
                logger.info("No more results, stopping pagination")
                break

            for ad_data in ads_data:
                if len(collected) >= max_results:
                    break
                try:
                    raw_ad = self._parse_ad(ad_data, platform)
                    collected.append(raw_ad)
                except Exception as e:
                    logger.warning(f"Failed to parse ad: {e}")
                    continue

            # Update cursor for next page
            last_hit = resp_data.get("last_hit")
            has_next = resp_data.get("has_next_page", False)

            if checkpoint:
                checkpoint.update(
                    page=page + 1,
                    last_hit=last_hit,
                    collected=len(collected),
                )

            if not has_next or last_hit is None:
                logger.info("No more pages available")
                break

            page += 1

        logger.info(f"Collected {len(collected)} ads total")
        if checkpoint:
            checkpoint.clear()

        return collected

    @staticmethod
    def _merge_detail(raw_ad: RawAd, detail_resp: dict[str, Any]) -> RawAd:
        """Merge detail endpoint response fields into an existing RawAd.

        Updates fields that the search endpoint leaves empty:
        cta, impressions, first/last seen, demographics, interests, devices.
        """
        detail_data = detail_resp.get("data", detail_resp)
        if isinstance(detail_data, list) and detail_data:
            detail_data = detail_data[0]
        if not isinstance(detail_data, dict):
            return raw_ad

        # Inner "ad" object has the targeting/performance fields
        inner = detail_data.get("ad", {})
        if isinstance(inner, dict):
            if inner.get("cta"):
                raw_ad.ad_copy.cta = str(inner["cta"])
            if inner.get("impressions"):
                raw_ad.impressions = int(inner["impressions"])
            if inner.get("first_seen_at"):
                parsed_first = _parse_iso_date(inner["first_seen_at"])
                if parsed_first is not None:
                    raw_ad.first_seen = parsed_first
            if inner.get("last_seen_at"):
                parsed_last = _parse_iso_date(inner["last_seen_at"])
                if parsed_last is not None:
                    raw_ad.last_seen = parsed_last
            if inner.get("interests"):
                raw_ad.interests = [
                    i.get("name", str(i)) if isinstance(i, dict) else str(i)
                    for i in inner["interests"]
                ]
            if inner.get("devices"):
                raw_ad.devices = [
                    d.get("name", str(d)) if isinstance(d, dict) else str(d)
                    for d in inner["devices"]
                ]

            # Demographics: age range, gender
            demos: dict[str, Any] = dict(raw_ad.demographics)
            if inner.get("ageRange"):
                demos["age_range"] = inner["ageRange"]
            if inner.get("gender"):
                demos["gender"] = inner["gender"]
        else:
            demos = dict(raw_ad.demographics)

        # Domain info
        domain = detail_data.get("domain", {})
        if isinstance(domain, dict) and domain.get("monthly_traffic"):
            demos["domain_traffic"] = domain["monthly_traffic"]

        # Technologies
        techs = detail_data.get("technologies", [])
        if techs:
            demos["technologies"] = techs

        # URL chains
        chains = detail_data.get("url_chains", [])
        if chains:
            demos["url_chains"] = chains

        if demos:
            raw_ad.demographics = demos

        return raw_ad

    @staticmethod
    def _parse_ad(data: dict[str, Any], platform: str = "facebook") -> RawAd:
        """Parse an AdFlex API ad object into canonical RawAd schema."""
        engagements = data.get("engagements", {}) or {}

        # Determine creative format from first attachment
        attachments = data.get("attachments", [])
        creative_format = CreativeFormat.OTHER
        creative_url = ""
        subtitle = ""
        att_description = ""
        if attachments:
            first = attachments[0]
            att_type = str(first.get("type", "")).lower()
            creative_format = _FORMAT_MAP.get(att_type, CreativeFormat.OTHER)
            files = first.get("files", {})
            creative_url = files.get("main", "")
            subtitle = first.get("subtitle", "")
            att_description = first.get("description") or ""
            # Multiple attachments → carousel
            if len(attachments) > 1:
                creative_format = CreativeFormat.CAROUSEL

        # Owner/advertiser info
        owner = data.get("owner", {}) or {}

        # Locations → country codes
        locations = data.get("locations", [])
        country_codes = [loc.get("code", "") for loc in locations if loc.get("code")]

        # Placements
        placement_list = [
            p.get("label", "") for p in (data.get("placements", []) or []) if p.get("label")
        ]

        # Engagement fields vary by platform:
        # Facebook: reactions, shares, comments, views
        # TikTok: likes, shares, favorites, comments, plays
        # X: reposts, replies, likes
        # Pinterest: saves, reactions, repins
        # Reddit: upvotes, comments
        likes = int(engagements.get("reactions", 0) or engagements.get("likes", 0) or 0)
        comments = int(engagements.get("comments", 0) or engagements.get("replies", 0) or 0)
        shares = int(
            engagements.get("shares", 0)
            or engagements.get("reposts", 0)
            or engagements.get("repins", 0)
            or 0
        )
        views = int(engagements.get("views", 0) or engagements.get("plays", 0) or 0)
        reactions = int(
            engagements.get("saves", 0)
            or engagements.get("favorites", 0)
            or engagements.get("upvotes", 0)
            or 0
        )

        return RawAd(
            ad_id=str(data.get("id", "")),
            source=AdSource.ADFLEX,
            platform=_PLATFORM_MAP.get(platform, Platform.OTHER),
            ad_copy=AdCopy(
                headline=data.get("title", ""),
                body=subtitle,
                description=data.get("description") or att_description or "",
                cta="",
            ),
            creative_format=creative_format,
            creative_url=creative_url,
            country=country_codes,
            last_seen=data.get("updated_at"),
            active_days=data.get("active_days"),
            likes=likes,
            comments=comments,
            shares=shares,
            reactions=reactions,
            views=views,
            advertiser_id=str(owner.get("id", "")),
            advertiser_name=owner.get("name", ""),
            landing_page_url=data.get("display_site_url") or data.get("display_url") or "",
            placements=placement_list,
            raw_data=data,
        )
