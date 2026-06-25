"""Scoring schemas: configuration models and scored ad output."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from draper.scraping.schemas import RawAd


class Transform(StrEnum):
    LOG = "log"
    LINEAR = "linear"
    BINARY = "binary"


class SignalConfig(BaseModel):
    """Configuration for a single scoring signal."""

    weight: float
    transform: Transform
    threshold_days: int | None = None  # only used by early_death


class TierThresholds(BaseModel):
    """Percentile thresholds for tier assignment.

    Values represent the score percentile cutoff for each tier boundary.
    """

    high: float = 0.80
    medium: float = 0.30
    low: float = 0.0


class SnorkelConfig(BaseModel):
    """Labeling function thresholds for the Snorkel (v2) scorer."""

    high_pct: float = 0.80
    low_pct: float = 0.20
    early_death_days: int = 3
    long_runner_days: int = 90


class HybridV3Config(BaseModel):
    """Signal weights for the v3 hybrid scorer.

    Tuning happens here, not in source — the weights are calibration
    artifacts, not architectural choices.
    """

    survivability: float = 0.50
    engagement_volume: float = 0.25
    engagement_velocity: float = 0.25

    def as_dict(self) -> dict[str, float]:
        return {
            "survivability": self.survivability,
            "engagement_volume": self.engagement_volume,
            "engagement_velocity": self.engagement_velocity,
        }


class ScoringConfig(BaseModel):
    """Full scoring configuration loaded from configs/scoring.yaml."""

    signals: dict[str, SignalConfig]
    tiers: TierThresholds = Field(default_factory=TierThresholds)
    snorkel: SnorkelConfig = Field(default_factory=SnorkelConfig)
    hybrid_v3: HybridV3Config = Field(default_factory=HybridV3Config)

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/scoring.yaml") -> ScoringConfig:
        """Load scoring config from a YAML file."""
        with Path(path).open() as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @property
    def total_weight(self) -> float:
        return sum(s.weight for s in self.signals.values())


class ScoredAd(BaseModel):
    """A RawAd with computed performance scores."""

    ad: RawAd
    composite_score: float = 0.0  # 0.0 to 1.0
    signal_scores: dict[str, float] = Field(default_factory=dict)
    tier_probs: dict[str, float] = Field(default_factory=dict)
    tier: str = "low"  # "high", "medium", "low"
    scoring_version: str = "v1"

    model_config = ConfigDict(arbitrary_types_allowed=True)
