"""Construction schemas: training example models and configuration."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PromptStyle(StrEnum):
    """Prompt style for training examples.

    - ``DATA_GROUNDED``: user prompt contains ad data, teacher sees ad data.
    - ``CONTEXT_DISTILLED``: user prompt is natural (no ads), teacher sees
      ads and abstracts patterns without citing specifics.
    - ``NATURAL``: no ads anywhere. Conversational fluency fallback.
    - ``BACKTRANSLATION``: inverted pipeline (Humpback / Li et al. ICLR'24).
      Teacher sees a real high-performing ad and reverse-engineers the brief
      that would plausibly have produced it. The real ad itself becomes the
      assistant response (reformatted into a consistent headline/body/CTA
      shape, never altered in content). Student learns brief → real copy.
    """

    NATURAL = "natural"
    DATA_GROUNDED = "data_grounded"
    CONTEXT_DISTILLED = "context_distilled"
    BACKTRANSLATION = "backtranslation"


class TaskFormat(StrEnum):
    """Active task format. Collapsed to copywriting-only after the 2026-04
    pivot; see ``archive/`` for the retired positioning / diagnostic /
    optimization / channel-format-fit formats.
    """

    COPYWRITING = "copywriting"


class ChatMessage(BaseModel):
    """A single message in a training conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ExampleMetadata(BaseModel):
    """Provenance metadata attached to every training example."""

    # Tolerate unknown fields so older JSONL rows keep loading.
    model_config = ConfigDict(extra="ignore")

    source_ad_ids: list[str] = Field(default_factory=list)
    source_tiers: list[str] = Field(default_factory=list)
    source_scores: list[float] = Field(default_factory=list)
    platform: str = ""
    vertical: str = ""
    construction_model: str = ""
    prompt_style: PromptStyle = PromptStyle.DATA_GROUNDED
    persona_id: str = ""
    seed_idx: int = -1
    evol_op: str = ""
    difficulty: str = "standard"
    turn_structure: str = "single"
    followup_type: str = ""
    # Copywriting-specific axis derived from the source ad (no RNG):
    # ``source_ad_shape`` is computed from which ad fields are populated.
    # Empty for other formats.
    source_ad_shape: str = ""
    # Copywriting-specific strongly-enforced conversation register
    # (``conversational`` / ``structured`` / ``imperative``). Hash-rolled
    # from ``ad.ad_id`` so registers split ~33/33/33 across the corpus.
    # Governs the register of *both* the user brief and the assistant
    # response. Empty for other formats and for older rows.
    conversation_register: str = ""
    # Provider batch that produced this example. Empty for chat-mode / pilot
    # ingestion (no batch). Carries enough identity to filter reviews to the
    # output of a single batch submission.
    batch_id: str = ""
    construction_timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class TrainingExample(BaseModel):
    """A single instruction-tuning example in chat-message format."""

    example_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    task_format: TaskFormat
    messages: list[ChatMessage]
    metadata: ExampleMetadata = Field(default_factory=ExampleMetadata)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class ClusterInfo(BaseModel):
    """Metadata for a group of related ads."""

    cluster_id: str
    cluster_type: Literal["advertiser", "vertical", "platform_vertical"] = "advertiser"
    advertiser_name: str = ""
    platform: str = ""
    vertical: str = ""
    ad_ids: list[str] = Field(default_factory=list)
    tier_counts: dict[str, int] = Field(default_factory=dict)
    score_stats: dict[str, float] = Field(default_factory=dict)


class AdPair(BaseModel):
    """A high/low ad pair for the optimization format."""

    high_ad_id: str
    low_ad_id: str
    pair_type: Literal["same_advertiser", "same_vertical"] = "same_advertiser"
    advertiser_name: str = ""
    platform: str = ""
    vertical: str = ""
    high_score: float = 0.0
    low_score: float = 0.0
    spread: float = 0.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ScoreBand(BaseModel):
    """Continuous-score filter for a format (Decision 1)."""

    min: float = 0.0
    max: float = 1.0


class FormatClusteringConfig(BaseModel):
    """Per-format clustering parameters."""

    # Tolerate stale entries from old YAML/JSONL (positioning_*, diagnostic_*,
    # optimization_*, channel_fit_*) so the config loads after the pivot.
    model_config = ConfigDict(extra="ignore")

    copywriting_min_copy_chars: int = 60


class ClusteringConfig(BaseModel):
    """Clustering-layer configuration (T10).

    Replaces the hardcoded constants in ``clusterer.py``. All defaults
    match the Decision-1 initial proposals; tune in
    ``configs/construction.yaml`` after the capacity study.
    """

    min_vertical_cluster: int = 30
    max_per_vertical: int = 500
    max_per_advertiser: int = 50
    min_advertiser_cluster: int = 3

    # When True, only ads with language == "en" (or "" — too short to detect)
    # are used for clustering and training data construction.
    english_only: bool = True

    # Content-safety gate. When ``drop_unsafe`` is True, ads whose
    # ``content_safety_label`` appears in ``unsafe_labels_to_drop`` are
    # dropped, provided the label's confidence is
    # >= ``content_safety_min_confidence``. Unlabelled ads (empty label) and
    # "safe" labels are always kept.
    #
    # ``unsafe_labels_to_drop`` intentionally excludes ``hate_discrimination``
    # and ``violence_graphic``: in practice those buckets are dominated by
    # false positives (political ads and news / movie trailers) even at 1.0
    # confidence, so dropping them costs more good data than it's worth.
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

    # Business-vertical gate. Ads with ``business_vertical_confidence`` below
    # this floor are dropped before clustering. Default 0.5 drops only the
    # clearly-wrong 0.2 tier (e.g. real-estate firms labelled as gaming); the
    # 0.5 tier is noisy but mostly usable and preserves pool size.
    business_vertical_min_confidence: float = 0.5

    # Training-quality gate. Ads with ``training_quality`` below this floor
    # are dropped before clustering. The labeler rates copy on a 1-5 scale:
    # 1 broken/empty, 2 pure clickbait, 3 coherent-generic, 4 clear voice,
    # 5 distinctive. Default 3 drops 1s and 2s — teacher models waste their
    # budget distilling broken/clickbait copy into anything useful. Ads with
    # ``training_quality == 0`` (unlabelled, e.g. legacy rows) are kept so
    # the filter is a no-op until the labeler has run.
    training_quality_min: int = 3

    # Verticals to drop outright from the pool regardless of other filters.
    # Compliance-driven exclusions — a general-purpose copywriting model
    # shouldn't be trained to confidently generate gambling hooks, drug
    # claims, cannabis offers, or campaign ads. The copy patterns are also
    # idiosyncratic enough that they don't help craft on other verticals.
    drop_verticals: list[str] = Field(
        default_factory=lambda: [
            "gambling_betting",
            "pharmaceuticals",
            "cannabis_cbd",
            "political_advocacy",
            "religious_faith",
        ]
    )

    # Religious-scripture gate. When True (default), ads whose copy consists
    # of Quranic verses, Bible verses, or other sacred-text quotations are
    # dropped from the clustering pool. Marketing briefs built around
    # scripture copy collapse into scenario/ad mismatches at construction
    # time (the "ad copy" is ancient text, not strategic craft). The
    # detector is keyword-based and high-precision — generic faith-themed
    # marketing (nonprofit appeals, church events, religious retail) is
    # preserved. See ``draper.construction.religious_scripture``.
    drop_religious_scripture: bool = True

    format: FormatClusteringConfig = Field(default_factory=FormatClusteringConfig)


class FormatConfig(BaseModel):
    """Per-format settings (target + style mapping + score band)."""

    target: int
    valid_styles: list[PromptStyle] = Field(
        default_factory=lambda: [
            PromptStyle.DATA_GROUNDED,
            PromptStyle.CONTEXT_DISTILLED,
            PromptStyle.NATURAL,
        ]
    )
    style_ratios: dict[str, float] = Field(
        default_factory=lambda: {
            PromptStyle.DATA_GROUNDED.value: 0.30,
            PromptStyle.CONTEXT_DISTILLED.value: 0.50,
            PromptStyle.NATURAL.value: 0.20,
        }
    )
    score_min: float = 0.0
    score_max: float = 1.0

    @model_validator(mode="after")
    def _check_style_ratios(self) -> FormatConfig:
        """Validate the style ratios match ``valid_styles`` and sum to 1.0."""
        for key in self.style_ratios:
            if key not in {s.value for s in PromptStyle}:
                msg = f"Unknown prompt style in style_ratios: {key!r}"
                raise ValueError(msg)
        total = sum(self.style_ratios.values())
        if abs(total - 1.0) > 1e-6:  # noqa: PLR2004
            msg = f"Format style_ratios must sum to 1.0, got {total:.4f}"
            raise ValueError(msg)
        for style in self.valid_styles:
            if self.style_ratios.get(style.value, 0.0) <= 0.0:
                msg = (
                    f"Style {style.value!r} is listed in valid_styles but has "
                    f"zero ratio — either remove it from valid_styles or "
                    f"give it a positive share."
                )
                raise ValueError(msg)
        for key, ratio in self.style_ratios.items():
            if ratio > 0.0 and PromptStyle(key) not in self.valid_styles:
                msg = (
                    f"Style {key!r} has ratio {ratio} but is not in "
                    f"valid_styles for this format."
                )
                raise ValueError(msg)
        return self


class ApiModeConfig(BaseModel):
    """LLM model settings for API-mode construction."""

    bulk_model: str = "claude-haiku-4-5-20251001"
    quality_model: str = "claude-sonnet-4-20250514"


class ProviderRotationConfig(BaseModel):
    """Target share per teacher provider for distribution diversity."""

    claude_ratio: float = 0.40
    gpt_ratio: float = 0.35
    gemini_ratio: float = 0.25

    def validate_sum(self) -> None:
        total = self.claude_ratio + self.gpt_ratio + self.gemini_ratio
        if abs(total - 1.0) > 1e-6:  # noqa: PLR2004
            msg = (
                f"Provider ratios must sum to 1.0, got {total:.4f} "
                f"(claude={self.claude_ratio}, gpt={self.gpt_ratio}, "
                f"gemini={self.gemini_ratio})"
            )
            raise ValueError(msg)

    def targets(self) -> dict[str, float]:
        return {
            "claude": self.claude_ratio,
            "gpt": self.gpt_ratio,
            "gemini": self.gemini_ratio,
        }


class QualityFilterConfig(BaseModel):
    """Quality filter thresholds."""

    min_response_length: int = 200
    dedup_threshold: float = 0.80
    prompt_dedup_threshold: float = 0.85
    cross_format_source_dedup: bool = True
    quality_sample_pct: float = 0.10
    min_quality_score: float = 3.0
    min_quality_score_by_style: dict[str, float] = Field(
        default_factory=lambda: {
            PromptStyle.DATA_GROUNDED.value: 3.5,
            PromptStyle.CONTEXT_DISTILLED.value: 4.0,
            PromptStyle.NATURAL.value: 3.5,
        }
    )

    def threshold_for(self, style: PromptStyle) -> float:
        return self.min_quality_score_by_style.get(style.value, self.min_quality_score)


class PromptStyleConfig(BaseModel):
    """Fallback three-way style ratio for callers that don't have a format.

    Decision 3 makes style ratios per-format: the format's ``style_ratios``
    wins. This config stays as the default when a format hasn't specified
    its own ratios (shouldn't happen in practice — ``FormatConfig`` has
    explicit defaults).
    """

    data_grounded_ratio: float = 0.30
    context_distilled_ratio: float = 0.50
    natural_ratio: float = 0.20
    seed: int = 42

    def validate_sum(self) -> None:
        total = self.data_grounded_ratio + self.context_distilled_ratio + self.natural_ratio
        if abs(total - 1.0) > 1e-6:  # noqa: PLR2004
            msg = (
                f"Prompt style ratios must sum to 1.0, got {total:.4f} "
                f"(data_grounded={self.data_grounded_ratio}, "
                f"context_distilled={self.context_distilled_ratio}, "
                f"natural={self.natural_ratio})"
            )
            raise ValueError(msg)


class DatasetSplitConfig(BaseModel):
    """Train/val/test split ratios."""

    train_ratio: float = 0.85
    val_ratio: float = 0.075
    test_ratio: float = 0.075


class CompositionConfig(BaseModel):
    """Turn-structure and difficulty distribution for bundle composition."""

    multi_turn_rate: float = 0.18
    # Per-format overrides keyed by ``TaskFormat.value``. Example:
    # ``{"copywriting": 0.0}`` pins copywriting bundles single-turn
    # because backtranslation can't produce a verbatim turn-2 response.
    multi_turn_rate_by_format: dict[str, float] = Field(default_factory=dict)
    followup_ratios: dict[str, float] = Field(
        default_factory=lambda: {
            "constraint_change": 0.50,
            "deep_dive": 0.35,
            "correction": 0.15,
        }
    )
    difficulty_ratios: dict[str, float] = Field(
        default_factory=lambda: {
            "standard": 0.60,
            "sparse": 0.20,
            "conflicting": 0.15,
            "multi_constraint": 0.05,
        }
    )

    def multi_turn_rate_for(self, task_format: TaskFormat) -> float:
        return self.multi_turn_rate_by_format.get(task_format.value, self.multi_turn_rate)

    @model_validator(mode="after")
    def _check_ratios(self) -> CompositionConfig:
        for name, ratios in [
            ("followup_ratios", self.followup_ratios),
            ("difficulty_ratios", self.difficulty_ratios),
        ]:
            total = sum(ratios.values())
            if abs(total - 1.0) > 1e-6:  # noqa: PLR2004
                msg = f"{name} must sum to 1.0, got {total:.4f}"
                raise ValueError(msg)
        return self


class ConstructionConfig(BaseModel):
    """Full construction configuration loaded from configs/construction.yaml."""

    scored_ads_path: str = "data/scored/v3/scored_ads.jsonl"
    output_dir: str = "data/constructed"
    clusters_dir: str = "data/constructed/_clusters"
    final_dir: str = "data/final"

    overgeneration_buffer: float = 1.25

    formats: dict[str, FormatConfig] = Field(default_factory=dict)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    api_mode: ApiModeConfig = Field(default_factory=ApiModeConfig)
    prompt_style: PromptStyleConfig = Field(default_factory=PromptStyleConfig)
    provider_rotation: ProviderRotationConfig = Field(default_factory=ProviderRotationConfig)
    quality_filter: QualityFilterConfig = Field(default_factory=QualityFilterConfig)
    dataset: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)
    composition: CompositionConfig = Field(default_factory=CompositionConfig)

    @model_validator(mode="after")
    def _check_format_keys(self) -> ConstructionConfig:
        """Reject unknown format keys in ``configs/construction.yaml``."""
        for fmt_name in self.formats:
            try:
                TaskFormat(fmt_name)
            except ValueError as exc:
                msg = (
                    f"Unknown format {fmt_name!r} in configs/construction.yaml. "
                    f"Valid: {[f.value for f in TaskFormat]}"
                )
                raise ValueError(msg) from exc
        return self

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/construction.yaml") -> ConstructionConfig:
        """Load construction config from a YAML file."""
        with Path(path).open() as f:
            raw = yaml.safe_load(f)
        data: dict[str, Any] = raw.get("construction", raw)
        cfg = cls(**data)
        cfg.prompt_style.validate_sum()
        cfg.provider_rotation.validate_sum()
        return cfg

    def target_for(self, task_format: TaskFormat) -> int:
        fmt = self.formats.get(task_format.value)
        if fmt is None:
            return 0
        return fmt.target

    def raw_target_for(self, task_format: TaskFormat) -> int:
        return int(round(self.target_for(task_format) * self.overgeneration_buffer))

    def format_config(self, task_format: TaskFormat) -> FormatConfig:
        """Return the ``FormatConfig`` for a task format (with defaults)."""
        fmt = self.formats.get(task_format.value)
        if fmt is not None:
            return fmt
        defaults = _default_valid_styles(task_format)
        return FormatConfig(
            target=0,
            valid_styles=defaults,
            style_ratios=_default_style_ratios(defaults),
        )

    def valid_styles_for(self, task_format: TaskFormat) -> list[PromptStyle]:
        return self.format_config(task_format).valid_styles

    def style_ratios_for(self, task_format: TaskFormat) -> dict[str, float]:
        return self.format_config(task_format).style_ratios


def _default_valid_styles(task_format: TaskFormat) -> list[PromptStyle]:
    # Copywriting is backtranslation-only; no other formats are active.
    return [PromptStyle.BACKTRANSLATION]


def _default_style_ratios(styles: list[PromptStyle]) -> dict[str, float]:
    return {PromptStyle.BACKTRANSLATION.value: 1.0}
