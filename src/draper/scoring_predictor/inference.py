"""Loader + batched scorer for trained scoring-predictor checkpoints.

This is the public surface that Phase 2 (`src/draper/evaluation/learned_scorer.py`)
consumes. It loads a checkpoint produced by :func:`draper.scoring_predictor.train.train_predictor`,
optionally pulls calibrators from a sibling JSON file, and exposes
:meth:`ScoringPredictor.score_text` for single-shot scoring and
:meth:`ScoringPredictor.score_many` for batched scoring.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from draper.scoring_predictor.calibrate import HeadCalibrators
from draper.scoring_predictor.data import HEAD_NAMES, build_text
from draper.scoring_predictor.model import FourHeadRegressor

CALIBRATOR_FILENAME = "calibrators.json"
PREDICTOR_META_FILENAME = "predictor_meta.json"
DEFAULT_BACKBONE = "microsoft/deberta-v3-base"


class ScoringPredictor:
    """Trained 4-head regressor + optional calibrators, ready for inference."""

    def __init__(
        self,
        *,
        model: FourHeadRegressor,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        calibrators: HeadCalibrators | None,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.calibrators = calibrators
        self.device = device

    @torch.no_grad()
    def _forward(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(HEAD_NAMES)), dtype=float)
        enc = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_seq_length,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.model(**enc)
        return np.asarray(outputs["logits"].detach().cpu().numpy(), dtype=np.float32)

    def score_text(
        self,
        *,
        platform: str,
        vertical: str,
        headline: str | None = None,
        body: str | None = None,
        description: str | None = None,
    ) -> dict[str, float]:
        """Score one ad. Returns a dict keyed by head name (calibrated if available)."""
        text = build_text(
            platform=platform,
            vertical=vertical,
            headline=headline,
            body=body,
            description=description,
        )
        if not text:
            # No copy at all — return a uniform-uninformative score so
            # callers don't have to special-case empty inputs. 0.5 mirrors
            # ``HybridScorer._weighted_sum`` no-signal fallback.
            return {name: 0.5 for name in HEAD_NAMES}

        raw = self._forward([text])
        if self.calibrators is not None:
            # Isotonic regressors are fit with y_min=0, y_max=1, so the output
            # is already bounded.
            scored = self.calibrators.transform(raw)[0]
        else:
            # No calibrator (pre-calibration or untuned checkpoint) — clamp
            # the linear-output predictions to [0, 1] before returning. The
            # model itself produces unbounded reals; this gives downstream
            # consumers the same value range whether or not calibration ran.
            scored = np.clip(raw[0], 0.0, 1.0)
        return {name: float(val) for name, val in zip(HEAD_NAMES, scored, strict=True)}

    def score_many(
        self,
        items: Sequence[dict[str, str | None]],
        *,
        batch_size: int = 64,
    ) -> list[dict[str, float]]:
        """Score a list of ad dicts. Each dict needs ``platform``+``vertical`` and
        any subset of ``headline``/``body``/``description`` keys."""
        results: list[dict[str, float]] = []
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            texts: list[str] = []
            empty_idx: set[int] = set()
            for i, item in enumerate(batch):
                text = build_text(
                    platform=str(item.get("platform") or "unknown"),
                    vertical=str(item.get("vertical") or "unknown"),
                    headline=_coerce_optional_str(item.get("headline")),
                    body=_coerce_optional_str(item.get("body")),
                    description=_coerce_optional_str(item.get("description")),
                )
                if not text:
                    empty_idx.add(i)
                    texts.append("")  # placeholder, won't be used
                else:
                    texts.append(text)

            non_empty_texts = [t for i, t in enumerate(texts) if i not in empty_idx]
            raw = self._forward(non_empty_texts)
            if self.calibrators is not None and raw.size > 0:
                scored = self.calibrators.transform(raw)
            elif raw.size > 0:
                # No calibrator — clamp linear outputs to [0, 1].
                scored = np.clip(raw, 0.0, 1.0)
            else:
                scored = raw

            cal_iter = iter(scored.tolist()) if scored.size > 0 else iter([])
            for i in range(len(batch)):
                if i in empty_idx:
                    results.append({name: 0.5 for name in HEAD_NAMES})
                else:
                    row = next(cal_iter)
                    results.append(
                        {name: float(val) for name, val in zip(HEAD_NAMES, row, strict=True)}
                    )
        return results


def load_predictor(
    checkpoint_dir: str | Path,
    *,
    device: str | None = None,
    calibrators_path: str | Path | None = None,
    max_seq_length: int | None = None,
) -> ScoringPredictor:
    """Load a trained predictor from disk.

    Args:
        checkpoint_dir: Directory containing the HF-saved model weights and
            tokenizer (``train_predictor`` writes this under ``{out}/best``).
        device: Torch device string. Defaults to ``cuda`` if available else ``cpu``.
        calibrators_path: Path to the calibrators JSON. Defaults to
            ``{checkpoint_dir}/calibrators.json``. Missing file → no calibration.
        max_seq_length: Override the tokenizer's truncation length. When
            ``None``, reads ``max_seq_length`` from ``predictor_meta.json``
            (training-time value); falls back to 320 only if neither is
            provided. Mismatches between train- and inference-time truncation
            silently change which tokens reach the model — read from meta
            unless you have a specific reason to override.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)

    # Reconstruct the model with random weights, then load the trained state
    # dict. We can't use ``from_pretrained`` because ``FourHeadRegressor`` is
    # not a HF ``PreTrainedModel``; we save its state via ``Trainer.save_model``
    # which writes ``model.safetensors`` to disk but no backbone ``config.json``.
    # The original backbone HF id is recorded in ``predictor_meta.json`` at
    # save time; we use it to instantiate a fresh backbone (its weights are
    # immediately overwritten by the loaded state dict below).
    meta_path = checkpoint_dir / PREDICTOR_META_FILENAME
    meta_max_seq_length: int | None = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        backbone_name = meta.get("backbone_model_name", DEFAULT_BACKBONE)
        meta_max = meta.get("max_seq_length")
        if isinstance(meta_max, int):
            meta_max_seq_length = meta_max
    else:
        backbone_name = DEFAULT_BACKBONE
    model = FourHeadRegressor(model_name=backbone_name)

    state_path = _find_state_path(checkpoint_dir)
    if state_path is not None:
        if state_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(state_path), device="cpu")
        else:
            state = torch.load(str(state_path), map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Unexpected keys (e.g., backbone weights already in ``from_pretrained``)
        # are harmless, but missing keys are a sign of a real problem — the saved
        # checkpoint is incomplete or schema mismatch. Raise to surface the issue.
        if missing:
            raise RuntimeError(
                f"Checkpoint at {state_path} is missing keys "
                f"(incomplete or schema mismatch): {missing}"
            )

    calibrators: HeadCalibrators | None = None
    cal_path = (
        Path(calibrators_path)
        if calibrators_path
        else (checkpoint_dir / CALIBRATOR_FILENAME)
    )
    if cal_path.exists():
        calibrators = HeadCalibrators.load(cal_path)

    if max_seq_length is not None:
        resolved_max_seq_length = max_seq_length
    elif meta_max_seq_length is not None:
        resolved_max_seq_length = meta_max_seq_length
    else:
        resolved_max_seq_length = 320

    return ScoringPredictor(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=resolved_max_seq_length,
        calibrators=calibrators,
        device=resolved_device,
    )


def _find_state_path(checkpoint_dir: Path) -> Path | None:
    """Locate the trained state dict in a HF-saved checkpoint directory.

    Trainer.save_model writes either ``model.safetensors`` (preferred) or
    ``pytorch_model.bin``. Sharded checkpoints are not expected at this scale.
    """
    for fname in ("model.safetensors", "pytorch_model.bin"):
        candidate = checkpoint_dir / fname
        if candidate.exists():
            return candidate
    return None


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
