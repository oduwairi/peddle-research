"""Pydantic config for the scoring-predictor pipeline.

Single source of truth for hyperparameters and paths. Loaded from
``configs/scoring_predictor.yaml`` via :meth:`PredictorConfig.from_yaml`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from draper.scoring_predictor.data import HEAD_NAMES


class HeadWeights(BaseModel):
    """Per-head loss multipliers."""

    composite: float = 1.0
    survivability: float = 1.0
    engagement_volume: float = 1.0
    engagement_velocity: float = 1.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return weights aligned with :data:`HEAD_NAMES`."""
        return (
            self.composite,
            self.survivability,
            self.engagement_volume,
            self.engagement_velocity,
        )


class PredictorConfig(BaseModel):
    """Top-level predictor config."""

    # Inputs / outputs
    corpus_path: Path
    splits_dir: Path
    checkpoint_dir: Path
    default_split: Literal["random", "heldout-platform", "heldout-vertical"] = "random"

    # Model
    model_name: str = "microsoft/deberta-v3-base"
    max_seq_length: int = 320
    dropout: float = 0.1

    # Training
    learning_rate: float = 2.0e-5
    head_learning_rate: float = 1.0e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "linear"
    head_weights: HeadWeights = Field(default_factory=HeadWeights)
    use_quality_weights: bool = True

    # Eval / early stopping
    eval_strategy: str = "steps"
    eval_steps: int = 250
    logging_steps: int = 50
    save_steps: int = 250
    metric_for_best_model: str = "spearman_composite"
    greater_is_better: bool = True
    early_stopping_patience: int = 3
    load_best_model_at_end: bool = True

    # Hardware
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> PredictorConfig:
        """Load from ``configs/scoring_predictor.yaml`` (or any peer file)."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return cls(**payload.get("predictor", {}))

    @property
    def head_names(self) -> tuple[str, ...]:
        """Re-expose :data:`HEAD_NAMES` for callers that import config only."""
        return HEAD_NAMES
