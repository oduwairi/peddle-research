"""Spearman / Pearson / MAE / ECE / AUC metrics for the predictor.

All functions accept plain numpy / list inputs so they can be used both from
the HF ``Trainer.compute_metrics`` callback and from the offline-eval CLI.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from draper.scoring_predictor.data import HEAD_NAMES


def regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, float]:
    """Per-head Spearman, Pearson, MAE — masked.

    Args:
        predictions: ``(N, H)`` float, model output.
        targets: ``(N, H)`` float.
        target_mask: ``(N, H)`` bool — only ``True`` rows enter each head's metric.

    Returns:
        Dict with keys ``spearman_<head>``, ``pearson_<head>``, ``mae_<head>``.
        Heads with fewer than 2 valid rows return ``nan`` for spearman/pearson.
        Heads with constant predictions (no rank variance) return ``0.0`` for
        spearman/pearson — semantically "no correlation," and lets the HF
        ``Trainer`` early-stopping callback compare the metric across evals
        without tripping on NaN.
    """
    out: dict[str, float] = {}
    for h, name in enumerate(HEAD_NAMES):
        mask = target_mask[:, h].astype(bool)
        n_valid = int(mask.sum())
        pred_h = predictions[mask, h]
        tgt_h = targets[mask, h]

        if n_valid < 2:
            out[f"spearman_{name}"] = float("nan")
            out[f"pearson_{name}"] = float("nan")
            out[f"mae_{name}"] = float("nan")
            continue

        spearman = stats.spearmanr(pred_h, tgt_h).statistic
        pearson = stats.pearsonr(pred_h, tgt_h).statistic
        mae = float(np.mean(np.abs(pred_h - tgt_h)))
        # spearmanr / pearsonr return NaN when either input is constant. Treat
        # that as "no correlation" (0.0) for the purposes of metric reporting
        # and early-stopping comparisons.
        sp = float(spearman) if spearman is not None else float("nan")
        pe = float(pearson) if pearson is not None else float("nan")
        out[f"spearman_{name}"] = 0.0 if np.isnan(sp) else sp
        out[f"pearson_{name}"] = 0.0 if np.isnan(pe) else pe
        out[f"mae_{name}"] = mae
    return out


def expected_calibration_error(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Bucketed |mean(pred) − mean(target)| weighted by bucket frequency.

    For a regression-on-[0,1] head this is the standard ECE definition: bucket
    predictions into deciles, take |mean(pred) − mean(target)| per bucket,
    average weighted by bucket size. ECE near 0 = well-calibrated.
    """
    preds = np.asarray(predictions, dtype=float)
    tgts = np.asarray(targets, dtype=float)
    if preds.size == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-inclusive on the last bin so 1.0 doesn't fall outside.
    bin_idx = np.clip(np.digitize(preds, bin_edges[1:-1]), 0, n_bins - 1)

    total = float(preds.size)
    err = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        mean_pred = float(preds[mask].mean())
        mean_tgt = float(tgts[mask].mean())
        weight = float(mask.sum()) / total
        err += weight * abs(mean_pred - mean_tgt)
    return err


def top_tier_auc(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    threshold: float,
    high: bool = True,
) -> float:
    """ROC-AUC for "is this row in the top/bottom tier?".

    Args:
        threshold: Score cutoff defining the positive class. The tier
            assigner uses 0.80 (high) and 0.30 (low) — pass those.
        high: If ``True``, positive class = ``target >= threshold``.
            If ``False``, positive class = ``target <= threshold`` (and
            the prediction is negated so higher pred → more likely positive).

    Returns:
        ROC-AUC in [0, 1]; 0.5 = random; ``nan`` if there's only one class.
    """
    preds = np.asarray(predictions, dtype=float)
    tgts = np.asarray(targets, dtype=float)
    if high:
        labels = (tgts >= threshold).astype(int)
        scores = preds
    else:
        labels = (tgts <= threshold).astype(int)
        scores = -preds
    if labels.sum() in (0, labels.size):
        return float("nan")
    # Mann–Whitney U → AUC equivalence; avoids importing sklearn just for this.
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = pos.size
    n_neg = neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u_stat, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
    return float(u_stat) / (n_pos * n_neg)  # type: ignore[no-any-return]
