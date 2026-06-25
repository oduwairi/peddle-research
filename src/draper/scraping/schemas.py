"""Canonical data schemas for scraped ad data and knowledge corpus."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Platform(StrEnum):
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    GOOGLE = "google"
    LINKEDIN = "linkedin"
    PINTEREST = "pinterest"
    TWITTER = "twitter"
    REDDIT = "reddit"
    OTHER = "other"


class CreativeFormat(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    CAROUSEL = "carousel"
    COLLECTION = "collection"
    TEXT = "text"
    OTHER = "other"


class AdSource(StrEnum):
    BIGSPY = "bigspy"
    ADSPY = "adspy"
    ADFLEX = "adflex"
    META_LIBRARY = "meta_library"
    GOOGLE_TRANSPARENCY = "google_transparency"
    TIKTOK_LIBRARY = "tiktok_library"
    LINKEDIN_LIBRARY = "linkedin_library"


class AdCopy(BaseModel):
    """Structured ad copy fields."""

    headline: str = ""
    body: str = ""
    description: str = ""
    cta: str = ""

    @field_validator("headline", "body", "description", "cta", mode="before")
    @classmethod
    def coerce_none_to_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class RawAd(BaseModel):
    """Canonical schema for a scraped advertisement.

    All scrapers normalize their output into this schema.
    Fields are optional where data may not be available from all sources.
    """

    ad_id: str
    source: AdSource
    platform: Platform = Platform.OTHER
    ad_copy: AdCopy = Field(default_factory=AdCopy)
    creative_format: CreativeFormat = CreativeFormat.OTHER
    creative_url: str = ""

    # Targeting
    country: list[str] = Field(default_factory=list)
    demographics: dict[str, Any] = Field(default_factory=dict)
    vertical: str = ""

    # Language of the ad copy (ISO 639-1, e.g. "en", "fr", "ar").
    # Empty string means not yet detected — run scripts/ops/enrich_language.py.
    language: str = ""

    # True business vertical of the ad, assigned by an LLM at enrichment time
    # (NOT the sweep-bucket ``vertical`` field above, which records the search
    # context the ad was found in). Empty string means not yet labelled —
    # run scripts/ops/label_verticals.py.
    business_vertical: str = ""
    business_vertical_confidence: float = 0.0

    # Content-safety label (one of: safe, profanity, adult_sexual,
    # hate_discrimination, violence_graphic, substance_drugs, shock_misleading)
    # assigned by scripts/ops/label_content_safety.py. Empty string means not
    # yet labelled.
    content_safety_label: str = ""
    content_safety_confidence: float = 0.0

    # Training-quality rating (1-5) assigned by label_verticals.py alongside
    # the vertical label: 1 = broken/empty/placeholder, 2 = generic clickbait
    # with no substance, 3 = coherent but generic, 4 = clear product + value
    # prop + voice, 5 = distinctive hook worth learning from. 0 = not yet
    # labelled. The clusterer drops ads below ``training_quality_min``.
    training_quality: int = 0

    # Dates
    first_seen: date | None = None
    last_seen: date | None = None

    # Engagement metrics
    likes: int = 0
    comments: int = 0
    shares: int = 0
    reactions: int = 0
    views: int = 0

    # Performance signals
    active_days: int | None = None
    impressions: int | None = None
    spend_lower: int | None = None
    spend_upper: int | None = None
    impression_lower: int | None = None
    impression_upper: int | None = None

    # Signals
    is_redelivered: bool | None = None

    # Targeting
    interests: list[str] | None = None
    devices: list[str] | None = None
    placements: list[str] = Field(default_factory=list)

    # Advertiser
    advertiser_id: str = ""
    advertiser_name: str = ""
    advertiser_ad_count: int = 0

    # Landing page
    landing_page_url: str = ""

    # Raw response preserved for debugging
    raw_data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("first_seen", "last_seen", mode="before")
    @classmethod
    def parse_dates(cls, v: Any) -> date | None:
        if v is None or v == "":
            return None
        if isinstance(v, date):
            return v
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, str):
            # Try ISO 8601 with fromisoformat (handles 'Z' suffix)
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00")).date()
            except ValueError:
                pass
            # Try common formats
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
                try:
                    return datetime.strptime(v, fmt).date()
                except ValueError:
                    continue
            # Try Unix timestamp as string
            try:
                return datetime.fromtimestamp(int(v)).date()
            except (ValueError, OSError):
                pass
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v).date()
        return None

    @property
    def longevity_days(self) -> int | None:
        """Days the ad has been running."""
        if self.active_days is not None:
            return self.active_days
        if self.first_seen and self.last_seen:
            return (self.last_seen - self.first_seen).days
        return None

    @property
    def total_engagement(self) -> int:
        return self.likes + self.comments + self.shares + self.reactions + self.views

    @property
    def weighted_engagement(self) -> float:
        """Quality-weighted engagement: actions > reactions > passive reach.

        Weights: shares 5x, comments 3x, likes/reactions 1x, views 0.1x.
        Views are kept but heavily discounted — high views still contributes
        positively, but a low-view ad with strong shares/comments beats a
        high-view ad with no interaction.
        """
        return (
            self.shares * 5.0
            + self.comments * 3.0
            + self.likes * 1.0
            + self.reactions * 1.0
            + self.views * 0.1
        )

    @property
    def weighted_engagement_velocity(self) -> float | None:
        """Weighted engagement per day."""
        days = self.longevity_days
        if days and days > 0:
            return self.weighted_engagement / days
        return None

    @property
    def engagement_velocity(self) -> float | None:
        """Engagement per day."""
        days = self.longevity_days
        if days and days > 0:
            return self.total_engagement / days
        return None


class KnowledgeArticle(BaseModel):
    """Structured marketing knowledge extracted from case studies and expert content."""

    url: str
    source_name: str = ""  # e.g. "HubSpot", "Neil Patel"
    title: str = ""
    topic: str = ""  # e.g. "channel selection", "audience targeting"
    channel: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)  # e.g. "AIDA", "PAS"
    metrics: dict[str, Any] = Field(default_factory=dict)  # e.g. {"roas": 3.2, "ctr": 0.05}
    key_insights: list[str] = Field(default_factory=list)
    raw_text: str = ""  # original extracted text
    extraction_model: str = ""  # which LLM did the extraction
