"""Offline-evaluation harness for the scoring predictor.

Reports the metrics the Phase-1 validation gates (in the plan) check:

* ``spearman_<head>``, ``pearson_<head>``, ``mae_<head>`` — overall.
* Per-platform Spearman on ``composite``.
* Calibration ECE on ``composite`` (post-isotonic, if a calibrators file is
  present in the checkpoint dir).
* Top-tier classification AUC at the 0.80 / 0.30 thresholds — these are the
  v3 ``high`` / ``low`` tier cutoffs from ``configs/scoring.yaml``.

Output is a JSON dict (also pretty-printed with Rich), so callers can pipe it
into a CI gate or read it from a script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from draper.scoring_predictor.calibrate import HeadCalibrators, fit_calibrators
from draper.scoring_predictor.config import PredictorConfig
from draper.scoring_predictor.data import HEAD_NAMES
from draper.scoring_predictor.dataset import ScoringCollator, ScoringDataset
from draper.scoring_predictor.inference import CALIBRATOR_FILENAME, load_predictor
from draper.scoring_predictor.metrics import (
    expected_calibration_error,
    regression_metrics,
    top_tier_auc,
)
from draper.scoring_predictor.splits import SplitName, load_split

# v3 tier thresholds — from configs/scoring.yaml.
HIGH_THRESHOLD = 0.80
LOW_THRESHOLD = 0.30


def evaluate_split(
    *,
    config: PredictorConfig,
    split_name: SplitName,
    checkpoint_dir: Path | None = None,
    fit_calibration: bool = True,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run the model over the test parquet of one split; return metrics dict.

    Args:
        config: Loaded :class:`PredictorConfig`.
        split_name: Which split's test set to evaluate against.
        checkpoint_dir: Defaults to ``{config.checkpoint_dir}/{split_name}/best``.
        fit_calibration: If true and no calibrators file exists, fit one on
            the val set first and persist to ``{checkpoint_dir}/calibrators.json``.
            This makes ``eval-offline`` idempotent — one command produces a
            ready-to-serve checkpoint.
        batch_size: Eval batch size. Memory-bounded on the GPU; 64 is safe
            for DeBERTa-v3-base on a 16 GB card at ``max_seq_length=320``.
    """
    ckpt = checkpoint_dir or (config.checkpoint_dir / split_name / "best")
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained checkpoint at {ckpt}")

    split = load_split(config.splits_dir, split_name)

    if fit_calibration and not (ckpt / CALIBRATOR_FILENAME).exists():
        _fit_and_save_calibrators(ckpt, split.val, config, batch_size=batch_size)

    predictor = load_predictor(ckpt)

    raw = _predict_split(predictor, split.test, config, batch_size=batch_size)
    targets = _targets_array(split.test)
    target_mask = _mask_array(split.test)

    cal_path = ckpt / CALIBRATOR_FILENAME
    calibrated = (
        HeadCalibrators.load(cal_path).transform(raw) if cal_path.exists() else raw
    )

    overall = regression_metrics(calibrated, targets, target_mask)

    # Per-platform Spearman on composite.
    per_platform: dict[str, dict[str, float]] = {}
    platforms = split.test["platform"].to_list()
    for plat in sorted(set(platforms)):
        mask = np.array([p == plat for p in platforms], dtype=bool)
        if mask.sum() < 10:
            continue
        plat_metrics = regression_metrics(
            calibrated[mask], targets[mask], target_mask[mask]
        )
        per_platform[plat] = {
            "n": int(mask.sum()),
            "spearman_composite": plat_metrics["spearman_composite"],
            "pearson_composite": plat_metrics["pearson_composite"],
            "mae_composite": plat_metrics["mae_composite"],
        }

    # Composite-only calibration + AUC. The composite head is the headline
    # metric; sub-head calibration is reported as part of regression_metrics
    # already (Pearson is a coarse proxy for slope, MAE for absolute error).
    composite_pred = calibrated[:, 0].tolist()
    composite_tgt = targets[:, 0].tolist()
    ece_composite = expected_calibration_error(composite_pred, composite_tgt)

    auc_high = top_tier_auc(
        composite_pred, composite_tgt, threshold=HIGH_THRESHOLD, high=True
    )
    auc_low = top_tier_auc(
        composite_pred, composite_tgt, threshold=LOW_THRESHOLD, high=False
    )

    return {
        "split": split_name,
        "checkpoint": str(ckpt),
        "n_test": int(targets.shape[0]),
        "calibrated": cal_path.exists(),
        "metrics": overall,
        "per_platform": per_platform,
        "composite_ece": ece_composite,
        "composite_auc_top_tier": auc_high,
        "composite_auc_bottom_tier": auc_low,
    }


def _fit_and_save_calibrators(
    ckpt: Path,
    val_df: pl.DataFrame,
    config: PredictorConfig,
    *,
    batch_size: int,
) -> None:
    """Fit isotonic calibrators on the val set and save to disk.

    Note: This function is called from ``evaluate_split`` before the predictor
    is loaded for test evaluation, so the early load here is the first access
    (not wasteful). See the note in ``evaluate_split`` at line 71.
    """
    predictor = load_predictor(ckpt)
    raw = _predict_split(predictor, val_df, config, batch_size=batch_size)
    targets = _targets_array(val_df)
    target_mask = _mask_array(val_df)
    cals = fit_calibrators(raw, targets, target_mask)
    cals.save(ckpt / CALIBRATOR_FILENAME)


def _predict_split(
    predictor: Any, df: pl.DataFrame, config: PredictorConfig, *, batch_size: int
) -> np.ndarray:
    """Run the model over a Polars split DataFrame and return raw (uncalibrated)
    predictions as a ``(N, NUM_HEADS)`` array.

    Predictor may be either a :class:`ScoringPredictor` (calibrated path) or
    we may want raw — so we use the underlying tokenizer + model directly.
    """
    tokenizer = predictor.tokenizer
    model = predictor.model
    device = predictor.device

    ds = ScoringDataset(df, tokenizer, max_length=config.max_seq_length, include_targets=False)
    collator = ScoringCollator(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    all_preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_preds.append(out["logits"].detach().cpu().numpy())

    if not all_preds:
        return np.empty((0, len(HEAD_NAMES)), dtype=float)
    return np.concatenate(all_preds, axis=0)


def _targets_array(df: pl.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(df["target_composite"].to_list(), dtype=float),
            np.asarray(df["target_survivability"].to_list(), dtype=float),
            np.asarray(df["target_engagement_volume"].to_list(), dtype=float),
            np.asarray(df["target_engagement_velocity"].to_list(), dtype=float),
        ]
    )


def _mask_array(df: pl.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(df["mask_composite"].to_list(), dtype=bool),
            np.asarray(df["mask_survivability"].to_list(), dtype=bool),
            np.asarray(df["mask_engagement_volume"].to_list(), dtype=bool),
            np.asarray(df["mask_engagement_velocity"].to_list(), dtype=bool),
        ]
    )


def write_report(report: dict[str, Any], path: str | Path) -> None:
    """Persist the offline-eval report as JSON next to the checkpoint."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
