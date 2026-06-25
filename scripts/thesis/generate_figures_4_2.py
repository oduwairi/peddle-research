"""Generate §4.2 figures from the learned-scorer eval runs.

Outputs four PNGs into docs/research/figures/:
  - fig-4-2-1-composite-by-config.png    (¶1 + ¶2 headline)
  - fig-4-2-2-composite-by-platform.png  (¶3)
  - fig-4-2-3-per-head-by-config.png     (¶4)
  - fig-4-2-4-predictor-reliability.png  (¶5 — two-panel reliability)

Numbers source:
  - A, B, C, GOLD : data/eval/runs/2026-05-10-learned-may-smoke/
  - B_pipe, C_pipe: data/eval/runs/2026-05-14-clean-pipe/

Bootstrap 95% CIs computed from per-row learned_scores parquets where
available; otherwise the percentile p25/p75 in the summary parquet is
used as a fallback range.

Style: serif body off, sans-serif tick labels, 300 DPI, minimal grid,
no top/right spines. Single accent colour for the spotlighted config
(C), neutral greys for baselines, gold for the ceiling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

FIG_DIR = Path("docs/research/figures")
RUN_SINGLE = Path("data/eval/runs/2026-05-10-learned-may-smoke/aggregates")
RUN_PIPE = Path("data/eval/runs/2026-05-14-clean-pipe/aggregates")
PER_ROW_DIR = Path("data/eval/learned_scores")
EVAL_REPORT = Path("data/scoring_predictor/checkpoints/random/best/eval_report.json")

CONFIG_ORDER = ["A", "B", "B_pipe", "C", "C_pipe", "GOLD"]
CONFIG_LABEL = {
    "A": "A\n(gpt-5.4-mini)",
    "B": "B\n(Qwen3-8B)",
    "B_pipe": "B_pipe\n(B + agent)",
    "C": "C\n(Draper)",
    "C_pipe": "C_pipe\n(C + agent)",
    "GOLD": "GOLD\n(real ads)",
}
CONFIG_COLOR = {
    "A": "#7f7f7f",
    "B": "#7f7f7f",
    "B_pipe": "#bdbdbd",
    "C": "#1f4e79",
    "C_pipe": "#7fa3c4",
    "GOLD": "#c08400",
}
PLATFORM_ORDER = ["facebook", "pinterest", "reddit", "tiktok", "twitter"]
PLATFORM_LABEL = {
    "facebook": "Facebook",
    "pinterest": "Pinterest",
    "reddit": "Reddit",
    "tiktok": "TikTok",
    "twitter": "X (Twitter)",
}
HEAD_ORDER = ["survivability", "engagement_volume", "engagement_velocity"]
HEAD_LABEL = {
    "survivability": "Survivability",
    "engagement_volume": "Engagement\nvolume",
    "engagement_velocity": "Engagement\nvelocity",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#dddddd",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 2000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        means[i] = rng.choice(values, size=n, replace=True).mean()
    lower = float(np.percentile(means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(means, (1 + ci) / 2 * 100))
    return float(values.mean()), lower, upper


def load_per_row_composite(config: str) -> np.ndarray | None:
    path = PER_ROW_DIR / f"{config}.parquet"
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    if "composite" not in df.columns:
        return None
    arr = df["composite"].drop_nulls().to_numpy()
    return arr if len(arr) > 0 else None


def composite_stats(config: str, summary_fallback: pl.DataFrame) -> tuple[float, float, float, int]:
    arr = load_per_row_composite(config)
    if arr is not None:
        mean, lo, hi = bootstrap_ci(arr)
        return mean, lo, hi, len(arr)
    row = summary_fallback.filter(pl.col("config") == config).row(0, named=True)
    mean = float(row["composite_mean"])
    p25 = float(row.get("composite_p25", mean))
    p75 = float(row.get("composite_p75", mean))
    return mean, p25, p75, int(row["n"])


def figure_1_composite_by_config() -> None:
    single = pl.read_parquet(RUN_SINGLE / "learned_scores_summary.parquet")
    pipe = pl.read_parquet(RUN_PIPE / "learned_scores_summary.parquet")
    combined = pl.concat([single, pipe.filter(pl.col("config").is_in(["B_pipe", "C_pipe"]))])

    rows = []
    for cfg in CONFIG_ORDER:
        mean, lo, hi, n = composite_stats(cfg, combined)
        rows.append((cfg, mean, lo, hi, n))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = np.arange(len(rows))
    means = [r[1] for r in rows]
    los = [r[2] for r in rows]
    his = [r[3] for r in rows]
    ns = [r[4] for r in rows]
    colors = [CONFIG_COLOR[r[0]] for r in rows]
    errs = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]

    bars = ax.bar(xs, means, yerr=errs, color=colors, edgecolor="black", linewidth=0.6, capsize=4, error_kw={"linewidth": 0.8})
    for x, m, n in zip(xs, means, ns):
        ax.annotate(f"{m:.3f}\n(n={n})", xy=(x, m), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels([CONFIG_LABEL[r[0]] for r in rows])
    ax.set_ylabel("Composite score (mean, 95% bootstrap CI)")
    ax.set_title("Composite score by configuration on the 215-brief held-out test set")
    ax.set_ylim(0.5, 0.78)
    ax.set_axisbelow(True)

    out = FIG_DIR / "fig-4-2-1-composite-by-config.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"WROTE {out}")


def figure_2_composite_by_platform() -> None:
    df = pl.read_parquet(RUN_SINGLE / "learned_scores_by_platform.parquet")
    configs = ["A", "B", "C", "GOLD"]
    pivot = df.select(["config", "platform", "composite_mean"]).pivot(on="config", index="platform", values="composite_mean").sort("platform")

    platforms = pivot["platform"].to_list()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    xs = np.arange(len(platforms))
    width = 0.2

    for i, cfg in enumerate(configs):
        means = pivot[cfg].to_list()
        offset = (i - 1.5) * width
        ax.bar(xs + offset, means, width, label=cfg, color=CONFIG_COLOR[cfg], edgecolor="black", linewidth=0.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([PLATFORM_LABEL[p] for p in platforms])
    ax.set_ylabel("Composite score (mean)")
    ax.set_title("Composite score by platform and configuration (single-shot configs)")
    ax.set_ylim(0.5, 0.78)
    ax.legend(title="Config", loc="upper left", frameon=False, ncols=4, bbox_to_anchor=(0.0, 1.0))
    ax.set_axisbelow(True)

    out = FIG_DIR / "fig-4-2-2-composite-by-platform.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"WROTE {out}")


def figure_3_per_head_by_config() -> None:
    df = pl.read_parquet(RUN_SINGLE / "learned_scores_summary.parquet")
    configs = ["A", "B", "C", "GOLD"]
    head_cols = [f"{h}_mean" for h in HEAD_ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = np.arange(len(HEAD_ORDER))
    width = 0.2

    for i, cfg in enumerate(configs):
        row = df.filter(pl.col("config") == cfg).row(0, named=True)
        means = [float(row[col]) for col in head_cols]
        offset = (i - 1.5) * width
        ax.bar(xs + offset, means, width, label=cfg, color=CONFIG_COLOR[cfg], edgecolor="black", linewidth=0.5)
        for x, m in zip(xs, means):
            ax.annotate(f"{m:.2f}", xy=(x + offset, m), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color="black")

    ax.set_xticks(xs)
    ax.set_xticklabels([HEAD_LABEL[h] for h in HEAD_ORDER])
    ax.set_ylabel("Head score (mean)")
    ax.set_title("Per-head means by configuration (single-shot configs)")
    ax.set_ylim(0.5, 0.8)
    ax.legend(title="Config", loc="upper right", frameon=False)
    ax.set_axisbelow(True)

    out = FIG_DIR / "fig-4-2-3-per-head-by-config.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"WROTE {out}")


def _ensure_predictions_cache() -> pl.DataFrame:
    """Predict composite + heads on the random test split; cache per-row."""
    cache = Path("data/scoring_predictor/checkpoints/random/best/test_predictions.parquet")
    if cache.exists():
        return pl.read_parquet(cache)

    print("Running predictor inference on 5412-row random test split (cache miss)…")
    from draper.scoring_predictor.config import PredictorConfig
    from draper.scoring_predictor.eval_offline import _predict_split
    from draper.scoring_predictor.inference import load_predictor

    predictor = load_predictor(
        "data/scoring_predictor/checkpoints/random/best", device="cpu"
    )
    test_df = pl.read_parquet("data/scoring_predictor/splits/random/test.parquet")
    config = PredictorConfig.from_yaml("configs/scoring_predictor.yaml")
    raw = _predict_split(predictor, test_df, config, batch_size=64)
    if predictor.calibrators is not None:
        calibrated = predictor.calibrators.transform(raw)
    else:
        calibrated = np.clip(raw, 0.0, 1.0)

    out = test_df.with_columns(
        [
            pl.Series("predicted_composite", calibrated[:, 0]),
            pl.Series("predicted_survivability", calibrated[:, 1]),
            pl.Series("predicted_engagement_volume", calibrated[:, 2]),
            pl.Series("predicted_engagement_velocity", calibrated[:, 3]),
        ]
    )
    out.write_parquet(cache)
    print(f"WROTE prediction cache: {cache}")
    return out


def _roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC (FPR, TPR) and AUC. No sklearn dependency."""
    order = np.argsort(-scores)
    s_sorted = scores[order]
    l_sorted = labels[order]
    pos = int(l_sorted.sum())
    neg = int((1 - l_sorted).sum())
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    tps = np.cumsum(l_sorted)
    fps = np.cumsum(1 - l_sorted)
    tpr = tps / pos
    fpr = fps / neg
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def figure_4_predictor_reliability() -> None:
    df = _ensure_predictions_cache()
    pred = df["predicted_composite"].to_numpy()
    target = df["target_composite"].to_numpy()

    # ---- Left panel: ROC for top-tier and bottom-tier discrimination ----
    top_labels = (target >= 0.80).astype(int)
    bot_labels = (target <= 0.30).astype(int)
    fpr_top, tpr_top, auc_top = _roc_curve(pred, top_labels)
    fpr_bot, tpr_bot, auc_bot = _roc_curve(-pred, bot_labels)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    ax_left.plot(fpr_top, tpr_top, color="#1f4e79", linewidth=1.8, label=f"Top-tier (target ≥ 0.80), AUC = {auc_top:.3f}")
    ax_left.plot(fpr_bot, tpr_bot, color="#c0504d", linewidth=1.8, label=f"Bottom-tier (target ≤ 0.30), AUC = {auc_bot:.3f}")
    ax_left.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--", linewidth=0.8, label="Chance")
    ax_left.set_xlabel("False positive rate")
    ax_left.set_ylabel("True positive rate")
    ax_left.set_title("Top- vs. bottom-tier discrimination (ROC)")
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1.02)
    ax_left.legend(loc="lower right", frameon=False, fontsize=8)
    ax_left.set_aspect("equal", adjustable="box")

    # ---- Right panel: Reliability diagram (10 quantile bins of predictions) ----
    n_bins = 10
    order = np.argsort(pred)
    pred_sorted = pred[order]
    target_sorted = target[order]
    bin_size = len(pred) // n_bins
    bin_pred_mean = np.array([pred_sorted[i * bin_size : (i + 1) * bin_size].mean() for i in range(n_bins)])
    bin_target_mean = np.array([target_sorted[i * bin_size : (i + 1) * bin_size].mean() for i in range(n_bins)])
    bin_counts = np.array([len(pred_sorted[i * bin_size : (i + 1) * bin_size]) for i in range(n_bins)])

    ax_right.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--", linewidth=0.8, label="Perfect calibration")
    ax_right.plot(bin_pred_mean, bin_target_mean, color="#1f4e79", linewidth=1.6, marker="o", markersize=5, label="Predictor")
    for x, y, n in zip(bin_pred_mean, bin_target_mean, bin_counts):
        ax_right.annotate(f"n={n}", xy=(x, y), xytext=(4, -8), textcoords="offset points", fontsize=7, color="#555555")
    ax_right.set_xlabel("Mean predicted composite (decile bin)")
    ax_right.set_ylabel("Mean actual composite (decile bin)")
    ax_right.set_title("Calibration (predicted vs. actual)")
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    ax_right.legend(loc="upper left", frameon=False, fontsize=8)
    ax_right.set_aspect("equal", adjustable="box")

    import json as _json
    report = _json.loads(EVAL_REPORT.read_text())
    fig.suptitle(
        f"Predictor reliability on the held-out random split "
        f"(n={report['n_test']:,}, ECE = {report['composite_ece']:.4f})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = FIG_DIR / "fig-4-2-4-predictor-reliability.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"WROTE {out} (top-tier AUC={auc_top:.3f}, bottom-tier AUC={auc_bot:.3f})")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_1_composite_by_config()
    figure_2_composite_by_platform()
    figure_3_per_head_by_config()
    figure_4_predictor_reliability()


if __name__ == "__main__":
    main()
