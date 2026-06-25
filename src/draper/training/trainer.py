"""QLoRA SFT training loop wrapping Unsloth + TRL SFTTrainer.

Heavy ML imports (``unsloth``, ``trl``, ``torch``) are deferred to method
bodies so this module is importable on a CPU-only laptop for static checks
and dry-run validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from datasets import DatasetDict
from transformers import TrainerCallback

from draper.training.config import TrainingConfig
from draper.training.data_loader import load_dataset_dict, subset_for_smoke
from draper.utils.logging import get_logger

logger = get_logger("draper.training")


@dataclass
class SetupResult:
    """Bundle returned by ``Trainer.setup()`` so callers can inspect state."""

    model: Any
    tokenizer: Any
    dataset: DatasetDict
    sft_trainer: Any
    run_dir: Path


class LossCollapseCallback(TrainerCallback):
    """Abort training if loss collapses to ~0 in the first few steps.

    Run #001 wasted ~50 GPU-min before the collapse was visible at eval.
    This callback raises within seconds of detection so the cloud script's
    auto-shutdown trap fires. Defaults trip if 3 consecutive logged losses
    are below 0.05 within the first 30 steps — the exact pattern of the
    Liger-induced collapse reproduced in local ablation 2026-04-30.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        consecutive: int = 3,
        within_steps: int = 30,
    ) -> None:
        self.threshold = threshold
        self.consecutive = consecutive
        self.within_steps = within_steps
        self._consec_below = 0

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if state.global_step > self.within_steps:
            return
        loss = (logs or {}).get("loss")
        if loss is None:
            return
        if float(loss) < self.threshold:
            self._consec_below += 1
        else:
            self._consec_below = 0
        if self._consec_below >= self.consecutive:
            msg = (
                f"LOSS_COLLAPSE: train_loss < {self.threshold} for "
                f"{self.consecutive} consecutive logs within first "
                f"{self.within_steps} steps (current step={state.global_step}, "
                f"loss={loss}). Aborting — likely use_liger_kernel or similar."
            )
            raise RuntimeError(msg)


class Trainer:
    """Encapsulates one fine-tuning run.

    Construction is cheap; ``setup()`` does the actual model/dataset/trainer
    instantiation. Splitting them lets the smoke ``--dry-run`` path validate
    config + dataset shape without paying for model download.
    """

    def __init__(self, config: TrainingConfig, *, smoke: bool = False) -> None:
        self.config = config
        self.smoke = smoke
        self._setup: SetupResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self, *, dry_run: bool = False) -> SetupResult:
        """Build model + tokenizer + dataset + SFTTrainer.

        With ``dry_run=True``, skips the model load — useful for a CPU-only
        sanity check that the config + dataset are valid.
        """
        ds = self._build_dataset()
        model, tokenizer = (None, None) if dry_run else self._build_model()

        run_dir = self.config.run_dir(suffix="smoke" if self.smoke else None)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_config(run_dir)

        sft_trainer = (
            None
            if dry_run
            else self._build_sft_trainer(model, tokenizer, ds, run_dir)
        )

        self._setup = SetupResult(
            model=model,
            tokenizer=tokenizer,
            dataset=ds,
            sft_trainer=sft_trainer,
            run_dir=run_dir,
        )
        return self._setup

    def train(self, *, resume: bool = False) -> Path:
        """Run training. Returns the path to the final adapter."""
        if self._setup is None:
            self.setup()
        assert self._setup is not None  # noqa: S101 — narrowing for mypy
        if self._setup.sft_trainer is None:
            msg = "Trainer was set up with dry_run=True; cannot call train()"
            raise RuntimeError(msg)

        self._baseline_loss(self._setup.sft_trainer, self._setup.model)

        logger.info("Starting training run in %s", self._setup.run_dir)
        self._setup.sft_trainer.train(resume_from_checkpoint=resume or None)

        final_dir = self._setup.run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        self._setup.sft_trainer.save_model(str(final_dir))
        logger.info("Adapter saved to %s", final_dir)
        return final_dir

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_dataset(self) -> DatasetDict:
        ds = load_dataset_dict(self.config.dataset_dir)
        if self.smoke:
            ds = subset_for_smoke(ds, self.config.smoke.n_examples)
        return ds

    def _build_model(self) -> tuple[Any, Any]:
        from unsloth import FastLanguageModel

        base = (
            self.config.smoke_base_model if self.smoke else self.config.base_model
        )
        logger.info("Loading base model %s (4bit=%s)", base, self.config.load_in_4bit)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base,
            max_seq_length=self.config.max_length,
            load_in_4bit=self.config.load_in_4bit,
            dtype=None,  # auto: bf16 on Ampere+, fp16 elsewhere
        )

        logger.info(
            "Attaching LoRA: r=%d alpha=%d dora=%s rslora=%s init=%r",
            self.config.lora_r,
            self.config.lora_alpha,
            self.config.use_dora,
            self.config.use_rslora,
            self.config.init_lora_weights,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
            use_rslora=self.config.use_rslora,
            use_dora=self.config.use_dora,
            init_lora_weights=self.config.init_lora_weights,
        )
        return model, tokenizer

    def _build_sft_trainer(
        self,
        model: Any,
        tokenizer: Any,
        ds: DatasetDict,
        run_dir: Path,
    ) -> Any:
        from transformers import EarlyStoppingCallback
        from trl import SFTConfig, SFTTrainer  # type: ignore[attr-defined]
        from unsloth.chat_templates import train_on_responses_only

        cfg = self.config
        smoke_overrides = cfg.smoke if self.smoke else None
        per_device_bs = (
            smoke_overrides.per_device_train_batch_size
            if smoke_overrides
            else cfg.per_device_train_batch_size
        )
        per_device_eval_bs = (
            smoke_overrides.per_device_eval_batch_size
            if smoke_overrides
            else cfg.per_device_eval_batch_size
        )
        grad_accum = (
            smoke_overrides.gradient_accumulation_steps
            if smoke_overrides
            else cfg.gradient_accumulation_steps
        )
        eval_strategy: str = (
            "no"
            if (smoke_overrides and smoke_overrides.skip_eval)
            else cfg.eval_strategy
        )
        load_best = cfg.load_best_model_at_end and eval_strategy != "no"

        # Unsloth's monkey-patched SFTTrainer doesn't auto-template the
        # messages column the way vanilla TRL does — we must hand it a
        # formatting_func that returns a *list* of rendered strings (per
        # row or per batch; the patched _prepare_dataset can call either
        # way). Assistant-only loss is enforced post-construction via
        # `train_on_responses_only` (Unsloth's idiom; TRL's
        # `assistant_only_loss=True` flag is a no-op on the Unsloth-patched
        # trainer).
        def _format_messages(example: dict[str, Any]) -> list[str]:
            msgs = example["messages"]
            # Single-row call: msgs is list[dict]. Batched call: msgs is
            # list[list[dict]]. Detect by inspecting the first element.
            if msgs and isinstance(msgs[0], dict):
                return [
                    str(
                        tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=False
                        )
                    )
                ]
            return [
                str(
                    tokenizer.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=False
                    )
                )
                for m in msgs
            ]

        sft_config = SFTConfig(
            output_dir=str(run_dir),
            max_length=cfg.max_length,
            packing=cfg.packing,
            padding_free=cfg.padding_free,
            use_liger_kernel=cfg.use_liger_kernel,
            # Hard-coded False: Unsloth's train_on_responses_only (below) is
            # the masker; setting True here would double-mask. TRL's flag is
            # also a no-op on the Unsloth-patched trainer (see comment above)
            # but we make it explicit so a future edit can't silently flip it.
            assistant_only_loss=False,
            # Optimizer / schedule
            learning_rate=cfg.learning_rate,
            num_train_epochs=cfg.num_train_epochs,
            max_steps=smoke_overrides.max_steps if smoke_overrides else -1,
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=per_device_eval_bs,
            gradient_accumulation_steps=grad_accum,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type,
            weight_decay=cfg.weight_decay,
            optim=cfg.optim,
            bf16=cfg.bf16,
            seed=cfg.seed,
            # Eval / checkpoint
            eval_strategy=eval_strategy,
            eval_steps=cfg.eval_steps,
            save_strategy=cfg.save_strategy,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            logging_steps=cfg.logging_steps,
            load_best_model_at_end=load_best,
            metric_for_best_model=cfg.metric_for_best_model,
            greater_is_better=cfg.greater_is_better,
            # Tracking
            report_to=cfg.report_to,
            run_name=run_dir.name,
            # space_id=None forces Trackio into local-only mode; the TRL
            # default ("trackio-by-trl") triggers a HF Space deploy that
            # requires a logged-in HF token.
            trackio_space_id=None,
        )

        callbacks: list[TrainerCallback] = [LossCollapseCallback()]
        if cfg.early_stopping_patience > 0 and cfg.eval_strategy != "no":
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
            )

        sft_trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            args=sft_config,
            callbacks=callbacks,
            # Unsloth wants list[str]; TRL stubs declare str. Runtime is fine.
            formatting_func=_format_messages,  # type: ignore[arg-type]
        )

        if cfg.assistant_only_loss:
            # ChatML markers (Qwen3 / Qwen2.5). Mask everything that isn't
            # an assistant turn so gradient flows only into the response.
            sft_trainer = train_on_responses_only(
                sft_trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )

        self._assert_label_sanity(sft_trainer)

        return sft_trainer

    def _assert_label_sanity(self, sft_trainer: Any) -> None:
        """Check the masked-label distribution on the first training batch.

        Run #001 collapsed because the labels tensor desynced from the
        loss-compute layout. A 1-second pre-flight check catches this class
        of bug before the optimizer takes its first step.
        """
        batch = next(iter(sft_trainer.get_train_dataloader()))
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        n_total = int(labels.numel())
        n_masked = int((labels == -100).sum().item())
        frac = n_masked / n_total if n_total else 0.0

        if not (0.10 <= frac <= 0.90):
            msg = (
                f"Label sanity FAIL: masked fraction {frac:.2%} outside "
                f"[10%, 90%] (n_masked={n_masked}/{n_total}). Masker broken."
            )
            raise RuntimeError(msg)

        logger.info(
            "Label sanity OK: masked=%.1f%% (%d/%d tokens), batch shape=%s",
            frac * 100,
            n_masked,
            n_total,
            tuple(input_ids.shape),
        )

    def _baseline_loss(self, sft_trainer: Any, model: Any) -> float:
        """Compute frozen-base CE loss on a training batch BEFORE step 1.

        A baseline near zero on an untrained model means the loss compute
        itself is degenerate (Run #001 symptom: model trivially solves the
        task and gradients optimize a wrong objective). Catch at step 0
        instead of after eval at end of epoch 1.
        """
        import torch

        batch = next(iter(sft_trainer.get_train_dataloader()))
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                outputs = model(**batch)
            baseline = float(outputs.loss.item())
        finally:
            if was_training:
                model.train()

        if baseline < 1.5:
            msg = (
                f"Step-0 baseline CE loss {baseline:.4f} is degenerate "
                "(< 1.5). Loss compute is broken (Run #001 class bug). Aborting."
            )
            raise RuntimeError(msg)
        if baseline > 12.0:
            logger.warning(
                "Step-0 baseline CE loss %.2f > 12 — base model may be "
                "poorly initialised or chat template mismatched. Continuing.",
                baseline,
            )
        logger.info("Step-0 baseline CE loss: %.4f (untrained reference)", baseline)
        return baseline

    def _snapshot_config(self, run_dir: Path) -> None:
        """Write the loaded TrainingConfig to the run dir for reproducibility."""
        snapshot = run_dir / "config.snapshot.yaml"
        with snapshot.open("w") as f:
            yaml.safe_dump({"training": self.config.model_dump()}, f, sort_keys=False)
        logger.info("Config snapshot written to %s", snapshot)
