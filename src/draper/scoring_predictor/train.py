"""Training driver for the 4-head scoring predictor.

Orchestrates the full Phase-1 cycle for a single split:

1. Load the v3 corpus, materialize the requested split to disk if missing.
2. Build the tokenized datasets.
3. Construct the :class:`FourHeadRegressor` and HuggingFace ``Trainer``.
4. Train; the best checkpoint by ``spearman_composite`` on val is restored
   at the end and saved to ``{checkpoint_dir}/{split_name}/``.

The CLI in ``scripts/predict.py`` is the user-facing entry point — this module
exposes :func:`train_predictor` for programmatic use (e.g. from the Modal
training app).
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import (
    EvalPrediction,
)
from transformers.trainer_utils import (
    get_last_checkpoint as _transformers_get_last_checkpoint,
)

from draper.scoring_predictor.config import PredictorConfig
from draper.scoring_predictor.data import HEAD_NAMES, iter_examples, load_corpus
from draper.scoring_predictor.dataset import ScoringCollator, ScoringDataset
from draper.scoring_predictor.metrics import regression_metrics
from draper.scoring_predictor.model import FourHeadRegressor
from draper.scoring_predictor.splits import (
    SPLIT_NAMES,
    Split,
    SplitName,
    load_split,
    make_heldout_platform_split,
    make_heldout_vertical_split,
    make_random_split,
)


def ensure_splits(config: PredictorConfig, *, force: bool = False) -> None:
    """Materialize all three splits from the v3 corpus if they aren't present.

    A split is considered present when ``train.parquet``, ``val.parquet``, and
    ``test.parquet`` all exist under ``{splits_dir}/{name}/``.
    """
    needed = [
        name
        for name in SPLIT_NAMES
        if force
        or not all(
            (config.splits_dir / name / f"{phase}.parquet").exists()
            for phase in ("train", "val", "test")
        )
    ]
    if not needed:
        return

    df = load_corpus(config.corpus_path)
    examples = list(iter_examples(df))
    if not examples:
        raise RuntimeError(f"Loaded 0 usable examples from {config.corpus_path}")

    from draper.scoring_predictor.data import examples_to_polars

    materialized = examples_to_polars(examples)
    config.splits_dir.mkdir(parents=True, exist_ok=True)
    if "random" in needed:
        make_random_split(materialized).write(config.splits_dir)
    if "heldout-platform" in needed:
        make_heldout_platform_split(materialized).write(config.splits_dir)
    if "heldout-vertical" in needed:
        make_heldout_vertical_split(materialized).write(config.splits_dir)


def _build_compute_metrics() -> Any:
    def _compute(eval_pred: EvalPrediction) -> dict[str, float]:
        # ``Trainer`` passes ``predictions`` as either a tensor/np.array (when
        # the model returns a single tensor under "logits") or a tuple (when
        # multiple outputs). We always emit just ``logits`` from forward in
        # eval mode — but defensively unwrap a tuple.
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = eval_pred.label_ids
        if isinstance(labels, tuple) and len(labels) == 2:
            # We packed (targets, target_mask) as label_ids — see below.
            targets, target_mask = labels
        else:
            # Single label tensor — assume all heads valid (unlikely in our
            # pipeline but keep the fallback safe).
            targets = labels
            target_mask = np.ones_like(targets, dtype=bool)
        return regression_metrics(
            predictions=np.asarray(preds, dtype=float),
            targets=np.asarray(targets, dtype=float),
            target_mask=np.asarray(target_mask, dtype=bool),
        )

    return _compute


class _MetricsTrainer(Trainer):
    """``Trainer`` subclass that routes targets+mask into ``compute_metrics``
    and applies a head-vs-backbone learning-rate split.

    Two overrides:

    * ``prediction_step`` — the vanilla ``Trainer`` extracts ``label_ids`` from
      the field named in ``label_names`` (default ``["labels"]``). Our model
      takes ``targets`` and ``target_mask`` as separate kwargs; we package
      both into the returned label tuple so ``compute_metrics`` can mask.

    * ``create_optimizer`` — applies layer-wise learning rates: head params
      train at ``head_learning_rate`` (typically 50x backbone), backbone at
      the base ``learning_rate``. A fresh-init head learning at the same LR
      as a pretrained backbone is the canonical cause of "predict-the-mean"
      regression collapse — AdamW's weight decay drives the head toward zero
      before it picks up signal.
    """

    def __init__(self, *args: Any, head_learning_rate: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._head_learning_rate = head_learning_rate

    def create_optimizer(self, model: Any = None) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        head_params: list[torch.nn.Parameter] = []
        backbone_params: list[torch.nn.Parameter] = []
        assert self.model is not None
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (head_params if name.startswith("head.") else backbone_params).append(param)
        grouped: list[dict[str, Any]] = [
            {
                "params": head_params,
                "lr": self._head_learning_rate,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": backbone_params,
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            grouped,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[Any, Any, Any]:
        has_targets = "targets" in inputs
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.get("loss")
            logits = outputs["logits"]
        if prediction_loss_only or not has_targets:
            return (loss.detach() if loss is not None else None, None, None)

        labels = (
            inputs["targets"].detach().cpu(),
            inputs["target_mask"].detach().cpu(),
        )
        return (loss.detach() if loss is not None else None, logits.detach().cpu(), labels)


class JSONLMetricsCallback(TrainerCallback):
    """Mirrors every ``Trainer.log`` event to a structured JSONL file.

    HF's stdout logging is mangled by tqdm carriage returns when redirected to
    a file (every step rewrites the same line). This callback writes one
    structured JSON object per log/eval event to ``{out_dir}/metrics.jsonl``,
    making it trivial to grep / plot / ingest later. Each line has at least
    ``timestamp``, ``step``, ``epoch``, plus whatever metric keys HF emitted.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        entry: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "step": state.global_step,
            "epoch": state.epoch,
        }
        entry.update(logs)
        self._fp.write(json.dumps(entry) + "\n")
        self._fp.flush()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        with contextlib.suppress(Exception):
            self._fp.close()


def train_predictor(
    config: PredictorConfig,
    *,
    split_name: SplitName | None = None,
) -> Path:
    """Train one model on one split. Returns the checkpoint directory."""
    name: SplitName = split_name if split_name is not None else config.default_split
    ensure_splits(config)
    split = load_split(config.splits_dir, name)
    return _train_on_split(config, split)


def _train_on_split(config: PredictorConfig, split: Split) -> Path:
    out_dir = config.checkpoint_dir / split.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    train_df, val_df = _maybe_drop_zero_weight(split.train), _maybe_drop_zero_weight(split.val)

    train_ds = ScoringDataset(train_df, tokenizer, max_length=config.max_seq_length)
    val_ds = ScoringDataset(val_df, tokenizer, max_length=config.max_seq_length)
    collator = ScoringCollator(tokenizer=tokenizer)

    model = FourHeadRegressor(
        model_name=config.model_name,
        dropout=config.dropout,
        head_weights=config.head_weights.as_tuple(),
    )

    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        report_to=[],  # No tracking integrations by default; trackio etc. opt-in.
        label_names=["targets", "target_mask"],
        remove_unused_columns=False,
        # Disable the tqdm progress bar — when redirected to a log file the
        # carriage returns make grepping eval/loss lines a nightmare. The
        # JSONLMetricsCallback below gives clean per-event records instead.
        disable_tqdm=True,
    )

    trainer = _MetricsTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=_build_compute_metrics(),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience),
            JSONLMetricsCallback(out_dir / "metrics.jsonl"),
        ],
        head_learning_rate=config.head_learning_rate,
    )

    last_ckpt = cast(str | None, _transformers_get_last_checkpoint(str(out_dir)))  # type: ignore[no-untyped-call]
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    best_dir = out_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    (best_dir / "predictor_meta.json").write_text(
        json.dumps(
            {
                "backbone_model_name": config.model_name,
                "max_seq_length": config.max_seq_length,
                "head_names": list(HEAD_NAMES),
            },
            indent=2,
        )
    )

    # Persist run metadata + best metrics so callers don't have to scrape logs.
    run_meta = {
        "split": split.name,
        "model_name": config.model_name,
        "max_seq_length": config.max_seq_length,
        "num_train_epochs": config.num_train_epochs,
        "head_names": list(HEAD_NAMES),
        "metric_for_best_model": config.metric_for_best_model,
        "best_metric": float(trainer.state.best_metric)
        if trainer.state.best_metric is not None
        else None,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_runtime_s": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_samples": int(train_result.metrics.get("train_samples", len(train_ds))),
    }
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2))
    return out_dir / "best"


def _maybe_drop_zero_weight(df: pl.DataFrame) -> pl.DataFrame:
    """Remove rows whose ``sample_weight`` is exactly 0.

    The weight derivation in :func:`draper.scoring_predictor.data._sample_weight`
    already filters quality-1 rows during corpus iteration, but if a caller
    materialized splits using a different rule we don't want to silently
    drag noise back in.
    """
    return df.filter(pl.col("sample_weight") > 0.0)
