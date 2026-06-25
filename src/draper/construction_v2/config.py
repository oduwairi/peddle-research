"""Construction v2 config: pydantic mirror of configs/construction_v2.yaml.

The unified pipeline runs a single-pass teacher (one LLM call emits
``<brief>`` + ``<think>`` + deliverable). Provider models are resolved
from :class:`ProviderConfig` entries under :attr:`providers`.

:class:`BriefExtractionConfig` and :class:`RationaleConfig` are retained
as optional fields so legacy two-stage YAMLs still load during the
migration. They are scheduled for deletion once Phase 4 lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SelectionConfig(BaseModel):
    """Source-ad selection parameters.

    The full v1-parity filter ladder runs here, BEFORE any teacher call,
    so we never spend tokens distilling content we'd reject downstream.
    """

    model_config = ConfigDict(extra="forbid")

    scored_ads_path: str = "data/scored/v3/scored_ads.jsonl"
    # Production target. Smoke runs override via `--target` or per-config.
    target_count: int = 3000
    min_composite: float = 0.7
    stratify: str = "platform"
    # Allow selection to pass through even if the corpus is dominated by
    # one platform. v3 corpus is overwhelmingly Facebook today; widen the
    # platform mix later by adding more sources.
    allow_unbalanced: bool = True

    # Maximum fraction of the selection allowed to come from a single
    # platform when ``allow_unbalanced`` is True. Defends against silent
    # degenerate single-platform selections from a skewed corpus. Caller
    # may override with ``force_unbalanced=True`` on the CLI.
    max_platform_share: float = 0.85

    # Sum of headline+body+description+cta character counts must meet
    # this floor. Defends against single-token ads with no learnable
    # copy; intentionally low (30) so short headline-only X/Reddit posts
    # still pass.
    min_copy_chars: int = 30

    # English-only language gate. Empty language (undetected, e.g. very
    # short copy) passes through; non-"en" labelled rows are dropped.
    english_only: bool = True

    # Content-safety gate (v1-equivalent semantics): drop only when label
    # is in ``unsafe_labels_to_drop`` AND confidence ≥ floor. Unlabelled
    # (empty string) and "safe" labels are always kept. hate_discrimination
    # and violence_graphic are intentionally excluded — those buckets are
    # dominated by false positives (political ads, movie trailers).
    drop_unsafe: bool = True
    content_safety_min_confidence: float = 0.9
    unsafe_labels_to_drop: list[str] = Field(
        default_factory=lambda: [
            "adult_sexual",
            "profanity",
            "shock_misleading",
            "substance_drugs",
        ]
    )

    # Business-vertical confidence floor. 0.5 drops the clearly-wrong
    # 0.2 tier (e.g. real estate labelled as gaming); 0.5 tier is noisy
    # but usable and preserves pool size.
    business_vertical_min_confidence: float = 0.5

    # Verticals dropped outright (compliance-driven). A general-purpose
    # copywriting model shouldn't be trained to confidently generate
    # gambling hooks, drug claims, cannabis offers, or campaign ads.
    drop_verticals: list[str] = Field(
        default_factory=lambda: [
            "gambling_betting",
            "pharmaceuticals",
            "cannabis_cbd",
            "political_advocacy",
            "religious_faith",
        ]
    )

    # Religious-scripture detector: drop ads whose copy is Quranic verse,
    # Bible verse, or canonical sacred-text quotation. Detector is
    # high-precision and preserves generic faith-themed marketing.
    drop_religious_scripture: bool = True

    # Structural cleanliness check (v1 selector parity): reject obvious
    # scraper artifacts — URLs in headlines, hashtag dumps, headline==body
    # duplicates, wall-of-text headlines. Defends against mangled rows.
    drop_structural_artifacts: bool = True

    # Training-quality floor. The label_verticals labeler rates copy 1-5:
    # 1 broken, 2 pure clickbait / non-ad spec dumps, 3 coherent-generic,
    # 4 clear voice, 5 distinctive. Default 3 drops 1s and 2s. Ads with
    # training_quality == 0 (unlabelled, legacy rows) are kept so the
    # filter is a no-op until the labeler has run. Set to 0 to disable.
    min_training_quality: int = 3

    seed: int = 42

    @field_validator("min_training_quality")
    @classmethod
    def _validate_min_training_quality(cls, v: int) -> int:
        if v < 0 or v > 5:
            msg = f"min_training_quality must be in range [0, 5], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("content_safety_min_confidence")
    @classmethod
    def _validate_content_safety_conf(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            msg = f"content_safety_min_confidence must be in [0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("business_vertical_min_confidence")
    @classmethod
    def _validate_bv_conf(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            msg = f"business_vertical_min_confidence must be in [0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("max_platform_share")
    @classmethod
    def _validate_max_platform_share(cls, v: float) -> float:
        if v <= 0.0 or v > 1.0:
            msg = f"max_platform_share must be in (0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v


class ProviderConfig(BaseModel):
    """Per-provider teacher model + sampling defaults.

    Each entry binds a logical provider key (``anthropic``, ``openai``,
    ``gemini``) to the concrete model string the single-pass teacher
    will use. Model whitelist enforcement lives in
    :func:`draper.construction.batch.factory.validate_batch_model`.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    max_tokens: int = 4000
    temperature: float = 0.4

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int) -> int:
        if v <= 0 or v > 20000:
            msg = f"max_tokens must be in range (0, 20000], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if v < 0.0 or v > 2.0:
            msg = f"temperature must be in range [0.0, 2.0], got {v}"
            raise ValueError(msg)
        return v


class SinglePassConfig(BaseModel):
    """Single-pass teacher: leak-guard + briefs cache location.

    The single-pass teacher emits ``<brief>``, ``<think>``, and the
    deliverable in one shot. We still cache the parsed brief (so
    ``ingest_responses`` can run leak / fidelity / grounding gates) and
    enforce the n-gram leak constraint between brief bridge fields and
    the source ad copy.
    """

    model_config = ConfigDict(extra="forbid")

    forbid_ngram_overlap: int = 5
    briefs_cache_path: str = "data/constructed_v2/copywriting/briefs.jsonl"

    @field_validator("forbid_ngram_overlap")
    @classmethod
    def _validate_forbid_ngram_overlap(cls, v: int) -> int:
        if v < 2 or v > 10:
            msg = f"forbid_ngram_overlap must be in range [2, 10], got {v}"
            raise ValueError(msg)
        return v


class BatchConfig(BaseModel):
    """Batch-API lifecycle thresholds shared across providers."""

    model_config = ConfigDict(extra="forbid")

    # A batch that has not reached a terminal state after this many
    # minutes is auto-cancelled by ``collect`` to defend against the
    # observed 3-day OpenAI ``validating`` hang.
    stuck_timeout_minutes: int = 360
    auto_force_cancel: bool = True
    # If ``provider_errors / request_count`` exceeds this fraction,
    # ``collect`` raises ``PartialFailureThreshold`` so the operator is
    # warned before downstream ingest swallows the gap silently.
    max_partial_error_rate: float = 0.05

    @field_validator("stuck_timeout_minutes")
    @classmethod
    def _validate_stuck_timeout(cls, v: int) -> int:
        if v < 5 or v > 72 * 60:
            msg = f"stuck_timeout_minutes must be in range [5, 4320], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("max_partial_error_rate")
    @classmethod
    def _validate_max_partial_error_rate(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            msg = f"max_partial_error_rate must be in [0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v


class BriefExtractionConfig(BaseModel):
    """LEGACY (two-stage) — kept optional until Phase 4 deletes it."""

    model_config = ConfigDict(extra="forbid")

    model: str = "claude-haiku-4-5"
    max_tokens: int = 3000
    temperature: float = 0.3
    forbid_ngram_overlap: int = 5
    cache_path: str = "data/constructed_v2/briefs.jsonl"

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int) -> int:
        if v <= 0 or v > 20000:
            msg = f"max_tokens must be in range (0, 20000], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if v < 0.0 or v > 2.0:
            msg = f"temperature must be in range [0.0, 2.0], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("forbid_ngram_overlap")
    @classmethod
    def _validate_forbid_ngram_overlap(cls, v: int) -> int:
        if v < 2 or v > 10:
            msg = f"forbid_ngram_overlap must be in range [2, 10], got {v}"
            raise ValueError(msg)
        return v


class RationaleConfig(BaseModel):
    """LEGACY (two-stage) — kept optional until Phase 4 deletes it."""

    model_config = ConfigDict(extra="forbid")

    model: str = "claude-haiku-4-5"
    max_tokens: int = 4000
    temperature: float = 0.4

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int) -> int:
        if v <= 0 or v > 20000:
            msg = f"max_tokens must be in range (0, 20000], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if v < 0.0 or v > 2.0:
            msg = f"temperature must be in range [0.0, 2.0], got {v}"
            raise ValueError(msg)
        return v


class FilterConfig(BaseModel):
    """Quality-filter knobs."""

    model_config = ConfigDict(extra="forbid")

    dedup_similarity_threshold: float = 0.95
    max_tokens: int = 8192
    min_deliverable_chars: int = 40

    @field_validator("dedup_similarity_threshold")
    @classmethod
    def _validate_dedup_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            msg = f"dedup_similarity_threshold must be in range [0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int) -> int:
        if v <= 0 or v > 20000:
            msg = f"max_tokens must be in range (0, 20000], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("min_deliverable_chars")
    @classmethod
    def _validate_min_deliverable_chars(cls, v: int) -> int:
        if v < 10 or v > 1000:
            msg = f"min_deliverable_chars must be in range [10, 1000], got {v}"
            raise ValueError(msg)
        return v


class DatasetConfig(BaseModel):
    """HF DatasetDict assembly parameters."""

    model_config = ConfigDict(extra="forbid")

    train_ratio: float = 0.90
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42


class ConstructionV2Config(BaseModel):
    """Top-level v2 construction config."""

    model_config = ConfigDict(extra="forbid")

    # Which skill this config drives. Determines the subdirectory layout
    # under ``output_dir`` and the ingest gate bundle (see
    # ``draper.construction_v2.ingest.skills``). New skills register
    # themselves on import; ``str`` not ``Literal[...]`` so adding a
    # skill is a code-only change, not a config schema bump.
    skill: str = "copywriting"

    output_dir: str = "data/constructed_v2"
    final_dir: str = "data/final_v2"
    audit_dir: str = "data/constructed_v2/_audit"

    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    # Per-provider teacher model bindings. Single source of truth for
    # which concrete model each ``--provider <p>`` resolves to.
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    single_pass: SinglePassConfig = Field(default_factory=SinglePassConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)

    # LEGACY (two-stage) — None for fresh configs, populated for old YAMLs.
    # Deleted in Phase 4.
    brief_extraction: BriefExtractionConfig | None = None
    rationale: RationaleConfig | None = None

    filter: FilterConfig = Field(default_factory=FilterConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    def provider_config(self, provider: str) -> ProviderConfig:
        """Return the :class:`ProviderConfig` for ``provider``.

        Raises a clear error when the operator picked a provider that
        isn't bound in the config.
        """
        if provider not in self.providers:
            available = sorted(self.providers.keys())
            msg = (
                f"Provider {provider!r} not configured. "
                f"Available: {available}. Add it under `providers:` in "
                f"configs/construction_v2.yaml."
            )
            raise KeyError(msg)
        return self.providers[provider]

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/construction_v2.yaml") -> ConstructionV2Config:
        """Load construction v2 config from YAML."""
        with Path(path).open() as f:
            raw = yaml.safe_load(f)
        data: dict[str, Any] = raw.get("construction_v2", raw)
        return cls(**data)


__all__ = [
    "BatchConfig",
    "BriefExtractionConfig",
    "ConstructionV2Config",
    "DatasetConfig",
    "FilterConfig",
    "ProviderConfig",
    "RationaleConfig",
    "SelectionConfig",
    "SinglePassConfig",
]
