"""ScrapingConfig: Pydantic model for configs/scraping.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RateLimitConfig(BaseModel):
    """Rate limit parameters for a single scraping source."""

    requests_per_minute: int = 30
    burst_size: int = 5


class AdFlexSourceConfig(RateLimitConfig):
    """AdFlex-specific rate limit and pagination config."""

    credits_per_call: int = 100
    ads_per_page: int = 18


class ScrapingTargetsConfig(BaseModel):
    """Collection volume targets."""

    ads_per_vertical: int = 2500
    exploratory_count: int = 500
    exploratory_verticals: list[str] = Field(default_factory=list)


class ScrapingRateLimitsConfig(BaseModel):
    """Rate limits keyed by source name."""

    adflex: AdFlexSourceConfig = Field(default_factory=AdFlexSourceConfig)
    bigspy: RateLimitConfig = Field(default_factory=RateLimitConfig)
    apify: RateLimitConfig = Field(
        default_factory=lambda: RateLimitConfig(requests_per_minute=60, burst_size=10)
    )
    serpapi: RateLimitConfig = Field(default_factory=RateLimitConfig)


class ScrapingConfig(BaseModel):
    """Full scraping configuration loaded from configs/scraping.yaml."""

    verticals: list[str] = Field(default_factory=list)
    rate_limits: ScrapingRateLimitsConfig = Field(default_factory=ScrapingRateLimitsConfig)
    targets: ScrapingTargetsConfig = Field(default_factory=ScrapingTargetsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/scraping.yaml") -> ScrapingConfig:
        """Load scraping config from a YAML file."""
        with Path(path).open() as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        # Filter to only known keys — ignore adflex:, output:, collection: platform blocks
        return cls(**{k: v for k, v in raw.items() if k in cls.model_fields})
