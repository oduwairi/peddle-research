"""Generate the §4.4 (Reference-Overlap Metrics) figures.

Distinct from generate_figures_4_4.py (the ablation generator) — this is the
new reference-overlap arm that becomes §4.4 once the ablation shifts to §4.5.

Outputs into docs/research/figures/:
  - fig-4-4-1-overlap-by-config.png   (per-metric RAW per-ad overlap clouds for
                                        A/B/C with mean ± 95% CI and GOLD ceiling)
  - fig-4-4-2-upworthy-grounding.png  (Upworthy A/B decisions per metric split
                                        into correct / wrong / tied)

Each figure carries structure the §4.4 prose numbers cannot:
  - Fig 1 shows the per-ad spread — C wins on the mean but with a far wider
    distribution (some ads near-verbatim, some misses), while A/B cluster
    tightly — not just the point means and CIs.
  - Fig 2 shows the decision mechanism — every metric ties (is undecided) on
    ~a third of pairs, and only METEOR is right more often than wrong among the
    pairs it decides; the bare accuracy hides both facts.

Numbers source:
  - data/eval/reference_scores/{A,B,C,GOLD}.parquet         (per-ad gold + pool overlap)
  - data/eval/validation/refmetrics_{metric}_upworthy.json  (A/B grounding: n_correct, n_ties)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

FIG_DIR = Path("docs/research/figures")
REF_DIR = Path("data/eval/reference_scores")
VAL_DIR = Path("data/eval/validation")

CONFIG_LABEL = {"A": "A (gpt-5.4-mini)", "B": "B (Qwen3-8B)", "C": "C (Draper)"}
CONFIG_COLOR = {"A": "#7f7f7f", "B": "#4d4d4d", "C": "#1f4e79"}

# gold-reference columns + display names, ordered lexical -> char -> embedding.
METRICS = [
    ("bleu_gold", "BLEU"),
    ("rouge_l_gold", "ROUGE-L"),
    ("meteor_gold", "METEOR"),
    ("chrf_gold", "chrF"),
    ("bertscore_gold", "BERTScore"),
]

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


def boot_ci(vals, n_boot=1000, seed=42):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def figure_1_overlap_by_config() -> None:
    """Per-metric small-multiples showing the RAW per-ad overlap distribution
    (one faint point per ad) with the mean +/- 95% CI on top, plus the GOLD
    multi-ref ceiling. The raw cloud carries the real information the bare
    forest hid: C wins on the mean but with a far wider per-ad spread, while
    A/B cluster tightly at low overlap.
    """
    data = {c: pl.read_parquet(REF_DIR / f"{c}.parquet") for c in ("A", "B", "C")}
    gold = pl.read_parquet(REF_DIR / "GOLD.parquet")
    rows = ["C", "A", "B"]  # top to bottom; C highlighted
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(3, 2, figsize=(6.8, 6.6))
    axflat = list(axes.flatten())
    for ax, (col, label) in zip(axflat, METRICS, strict=False):  # 6th axis = legend
        ceil = float(gold[col.replace("_gold", "_multi")].mean())
        ax.axvline(ceil, color="#c08400", linestyle="--", linewidth=1.2, zorder=1)
        for i, cfg in enumerate(rows):
            vals = np.asarray([v for v in data[cfg][col].to_list() if v is not None], float)
            vals = vals[~np.isnan(vals)]
            jit = i + rng.uniform(-0.17, 0.17, size=len(vals))
            ax.scatter(vals, jit, s=5, color=CONFIG_COLOR[cfg], alpha=0.22,
                       edgecolors="none", zorder=2)
            m, lo, hi = boot_ci(vals.tolist())
            ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o",
                        color=CONFIG_COLOR[cfg], ecolor="black",
                        markersize=9 if cfg == "C" else 7.5, markeredgecolor="black",
                        markeredgewidth=0.7, elinewidth=1.5, capsize=3.5, zorder=4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.margins(x=0.04)

    # 6th cell holds the legend.
    legax = axflat[5]
    legax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=CONFIG_COLOR[c],
                   markeredgecolor="black", markeredgewidth=0.5, label=CONFIG_LABEL[c])
        for c in ("C", "A", "B")
    ]
    handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color="#888888",
                              alpha=0.4, markersize=5, label="individual ads"))
    handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color="#444444",
                              markeredgecolor="black", markersize=7, label="mean ± 95% CI"))
    handles.append(plt.Line2D([0], [0], color="#c08400", linestyle="--",
                              label="GOLD ceiling\n(vs winner pool)"))
    legax.legend(handles=handles, loc="center", frameon=False, fontsize=8,
                 title="overlap with the real\nwinning ad (gold)", title_fontsize=8.5)
    fig.suptitle(
        "Per-ad overlap with the real winning ad: spread, mean ± 95% CI, and the GOLD ceiling",
        fontsize=10.5, y=0.996,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / "fig-4-4-1-overlap-by-config.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"WROTE {out}")


def figure_2_upworthy_grounding() -> None:
    """Decision breakdown of the 200 Upworthy A/B tests per metric: correct /
    wrong / tied. Shows the mechanism the bare accuracy hides — every metric is
    undecided (a tie) on roughly a third of pairs, and only METEOR is right more
    often than wrong among the pairs it does decide.
    """
    metrics = [("meteor", "METEOR"), ("chrf", "chrF"),
               ("rouge_l", "ROUGE-L"), ("bleu", "BLEU")]
    recs = []
    for key, label in metrics:
        d = json.loads((VAL_DIR / f"refmetrics_{key}_upworthy.json").read_text())
        npairs, nc, nt = d["n_pairs"], d["n_correct"], d["n_ties"]
        recs.append((label, nc, npairs - nt - nc, nt, d["accuracy"], d["binomial_p_value"]))

    c_correct, c_wrong, c_tie = "#1f4e79", "#c0392b", "#dddddd"
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for i, (_label, nc, nw, nt, acc, p) in enumerate(recs):
        ax.barh(i, nc, color=c_correct, edgecolor="white", height=0.64, zorder=3,
                label="correct" if i == 0 else None)
        ax.barh(i, nw, left=nc, color=c_wrong, edgecolor="white", height=0.64, zorder=3,
                label="wrong" if i == 0 else None)
        ax.barh(i, nt, left=nc + nw, color=c_tie, edgecolor="white", height=0.64,
                hatch="////", zorder=3, label="tied (undecided)" if i == 0 else None)
        sig = p < 0.05
        ax.text(205, i, f"acc {acc:.3f}" + ("  *" if sig else ""), va="center",
                fontsize=8.5, fontweight="bold" if sig else "normal",
                color=c_correct if sig else "#555555")
    ax.set_yticks(range(len(recs)))
    ax.set_yticklabels([r[0] for r in recs])
    ax.set_ylim(len(recs) - 0.5, -0.5)
    ax.set_xlim(0, 248)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Number of Upworthy A/B pairs (of 200), by decision outcome")
    ax.set_title("Can each overlap metric pick the real higher-CTR headline?",
                 loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False,
              ncol=3, fontsize=8.5)
    fig.tight_layout()
    out = FIG_DIR / "fig-4-4-2-upworthy-grounding.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"WROTE {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_1_overlap_by_config()
    figure_2_upworthy_grounding()


if __name__ == "__main__":
    main()
