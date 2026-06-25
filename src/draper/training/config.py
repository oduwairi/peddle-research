"""Training configuration: Pydantic mirror of configs/training.yaml."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SmokeConfig(BaseModel):
    """Overrides applied when `train.py smoke` is invoked."""

    model_config = ConfigDict(extra="forbid")

    n_examples: int = 10
    max_steps: int = 2
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    # If True, skip eval entirely (cheaper but doesn't exercise the eval
    # path). Default False — we'd rather catch eval-side bugs at smoke
    # time. Set True if even bs=1 eval OOMs on a smaller GPU.
    skip_eval: bool = False


class TrainingConfig(BaseModel):
    """All hyperparameters and paths for a fine-tuning run."""

    model_config = ConfigDict(extra="forbid")

    # Model
    base_model: str
    smoke_base_model: str
    load_in_4bit: bool = True
    max_length: int = 4096

    # LoRA / PEFT
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    init_lora_weights: bool | str = True
    use_rslora: bool = True
    use_dora: bool = True
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Optimizer / schedule
    learning_rate: float = 2.0e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    optim: str = "adamw_8bit"
    bf16: bool = True
    seed: int = 42

    # SFT-specific
    packing: bool = False
    padding_free: bool = False
    use_liger_kernel: bool = False
    assistant_only_loss: bool = True

    @field_validator("use_liger_kernel")
    @classmethod
    def _block_liger(cls, v: bool) -> bool:
        # Locally reproduced 2026-04-30 (ablation FT run on Qwen2.5-0.5B +
        # xformers + torch 2.10): use_liger_kernel=True collapses train_loss
        # to <0.001 by step 30 regardless of padding_free. Same pattern as
        # cloud Run #001 on Qwen3-8B. Hard-block until we have an upstream
        # fix; override with DRAPER_ALLOW_LIGER=1 if knowingly testing it.
        if v and os.environ.get("DRAPER_ALLOW_LIGER") != "1":
            msg = (
                "use_liger_kernel=True is blocked by Pydantic validator. "
                "Local ablation 2026-04-30 (Qwen2.5-0.5B) showed it collapses "
                "train_loss to ~0 by step 20 with assistant-only-loss masking. "
                "Set DRAPER_ALLOW_LIGER=1 to override (e.g. for an ablation)."
            )
            raise ValueError(msg)
        return v

    # Eval / checkpointing
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    eval_steps: int = 50
    save_strategy: Literal["no", "steps", "epoch"] = "steps"
    save_steps: int = 50
    save_total_limit: int = 5
    logging_steps: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 2

    # Tracking
    report_to: str = "trackio"
    trackio_project: str = "draper-copywriting"

    # Paths
    dataset_dir: str = "data/final"
    output_dir: str = "outputs/qwen3-8b-copywriting"
    merged_dir: str = "outputs/qwen3-8b-copywriting/merged"

    # Smoke
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/training.yaml") -> TrainingConfig:
        """Load training config from YAML."""
        with Path(path).open() as f:
            raw = yaml.safe_load(f)
        data: dict[str, Any] = raw.get("training", raw)
        return cls(**data)

    def run_dir(self, suffix: str | None = None) -> Path:
        """Return a per-run output directory.

        Adds a UTC timestamp + DoRA/rank tag so concurrent runs don't collide.

        DRAPER_RUN_DIR_OVERRIDE env var, when set, short-circuits to that path —
        used to resume training into an existing run dir's checkpoint set.
        """
        if (override := os.environ.get("DRAPER_RUN_DIR_OVERRIDE")):
            return Path(override)
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        tag = f"r{self.lora_r}-dora" if self.use_dora else f"r{self.lora_r}"
        parts = [tag, ts]
        if suffix:
            parts.append(suffix)
        return Path(self.output_dir) / "-".join(parts)
