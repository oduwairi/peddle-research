"""Load and validate configs/eval.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from draper.evaluation.paths import EvalPaths, validate_config_name

RunnerType = Literal["openai", "vllm", "frontend"]


class ConfigDef(BaseModel):
    """One ablation config (A / B / C / A_pipe / D / A_pipe@variant)."""

    label: str
    runner: RunnerType
    model: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    base_url_env: str | None = None
    api_key_env: str | None = None
    token_env: str | None = None
    timeout_s: int = 60


class JudgesDef(BaseModel):
    """Judge panel.

    ``primary`` and ``secondary`` are required for backwards compatibility
    with the existing aggregation code (Cohen's kappa is computed between
    them). ``panel`` is an optional broader list of judges to run when
    you want a 3+ judge panel — defaults to ``[primary, secondary]``.
    """

    primary: str
    secondary: str
    panel: list[str] = Field(default_factory=list)

    @property
    def all_judges(self) -> list[str]:
        """Deduplicated union of (primary, secondary, *panel)."""
        seen: list[str] = []
        for j in (self.primary, self.secondary, *self.panel):
            if j not in seen:
                seen.append(j)
        return seen


class MauveConfig(BaseModel):
    """Settings for the MAUVE distribution-matching eval arm.

    All fields have defaults so an absent ``mauve:`` block in
    ``configs/eval.yaml`` still produces a usable config. CLI flags on
    ``scripts/eval.py mauve`` override these values on a per-run basis.
    """

    model_config = ConfigDict(extra="ignore")

    reference_tier: str = "high"
    embedding_model: str = "gpt2-large"
    bootstrap_n: int = 100
    platforms: list[str] = Field(
        default_factory=lambda: ["facebook", "pinterest", "reddit", "tiktok", "twitter"]
    )
    random_seed: int = 42
    batch_size: int = 8
    max_text_length: int = 1024
    v3_parquet: Path = Path("data/scored/v3/scored_ads.parquet")


class ReferenceMetricsConfig(BaseModel):
    """Settings for the reference-overlap eval arm (BLEU/chrF/ROUGE-L/METEOR/BERTScore).

    All fields have defaults so an absent ``reference_metrics:`` block in
    ``configs/eval.yaml`` still produces a usable config. CLI flags on
    ``scripts/eval.py reference-metrics`` override these per-run. The reference
    corpus is the same v3 high-tier pool (and on-disk cache) as the MAUVE arm.
    """

    model_config = ConfigDict(extra="ignore")

    reference_tier: str = "high"
    platforms: list[str] = Field(
        default_factory=lambda: ["facebook", "pinterest", "reddit", "tiktok", "twitter"]
    )
    k_multi: int = 5  # nearest high-tier same-platform refs per brief
    enable_bertscore: bool = True
    bertscore_model: str = "roberta-large"
    random_seed: int = 42
    v3_parquet: Path = Path("data/scored/v3/scored_ads.parquet")


class EvalConfig(BaseModel):
    """Parsed ``configs/eval.yaml``.

    All on-disk paths are derived from ``root`` (default ``data/eval``) via
    :class:`draper.evaluation.paths.EvalPaths` — accessible as ``cfg.paths``.
    The per-attribute path properties (``inferences_dir``, ``judgments_dir``,
    etc.) are aliases for the flat per-config caches kept for backwards
    compatibility with downstream callers. Per-run artifacts (aggregates,
    manifests, batches, diagnostics) live exclusively under
    ``runs/<run_id>/`` and must be reached via ``cfg.paths``.

    Unknown YAML keys are ignored so the schema can evolve without
    breaking config files that carry stale fields (e.g. the legacy
    ``aggregates_dir`` / ``manifests_dir`` entries).
    """

    model_config = ConfigDict(extra="ignore")

    configs: dict[str, ConfigDef]
    arm1_pairs: list[tuple[str, str]] = Field(default_factory=list)
    arm2_pairs: list[tuple[str, str]] = Field(default_factory=list)
    judges: JudgesDef
    cross_val_sample_size: int = 50
    position_swap: bool = True
    bootstrap_n: int = 1000
    bootstrap_seed: int = 42
    elo_k: float = 32.0
    elo_seed: int = 42
    test_split_dir: Path
    url_scenarios_path: Path = Path("data/eval/url_scenarios.jsonl")
    root: Path = Path("data/eval")
    mauve: MauveConfig = Field(default_factory=MauveConfig)
    reference_metrics: ReferenceMetricsConfig = Field(default_factory=ReferenceMetricsConfig)

    @model_validator(mode="after")
    def _validate_config_names(self) -> EvalConfig:
        """Enforce ``<base>[@<variant>]`` naming for every config key.

        Catches typos and stray characters at load time so a malformed name
        can't propagate into on-disk paths (which would then be unparseable
        by ``EvalPaths``).
        """
        for name in self.configs:
            validate_config_name(name)
        return self

    @property
    def paths(self) -> EvalPaths:
        """Canonical resolver for every ``data/eval/...`` path."""
        return EvalPaths(root=self.root)

    # ---- Backwards-compat aliases for flat per-config caches -------------
    # New code should prefer ``cfg.paths.inferences_dir(config)`` etc.
    # These return the *root* of each cache, matching the legacy attribute
    # shape that downstream callers expect.

    @property
    def inferences_dir(self) -> Path:
        return self.paths.inferences_root

    @property
    def inferences_clean_dir(self) -> Path:
        return self.paths.inferences_clean_root

    @property
    def judgments_dir(self) -> Path:
        return self.paths.judgments_root

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw["eval"])
