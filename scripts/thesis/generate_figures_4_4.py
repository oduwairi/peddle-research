"""Generate §4.4 figures from the 2×2 learned-scorer data.

Outputs into docs/research/figures/:
  - fig-4-4-1-paired-contrasts.png   (forest plot of 6 paired contrasts)
  - fig-4-4-2-cell-means.png         (2×2 cell-means grid)

Numbers source:
  - data/eval/learned_scores/{B,C,B_pipe,C_pipe}.parquet  (per-ad composites)
  - Paired analysis: inner-join on example_id across all four cells
    (n=94 paired briefs, matching RQ2_OFFLINE_2x2_RESULTS_2026-05.md)
  - Bootstrap: 1000 resamples, seed=42, 95% CI from 2.5/97.5 percentiles

Reproduces every number reported in §4.4 ¶1 and ¶2.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

FIG_DIR = Path("docs/research/figures")
SCORE_DIR = Path("data/eval/learned_scores")

CELLS = ["B", "C", "B_pipe", "C_pipe"]

# Same order as ¶2 prose: significant first, then non-significant.
CONTRASTS: list[tuple[str, str, str]] = [
    ("C", "B", "FT, no agent"),
    ("C_pipe", "C", "agent on FT"),
    ("C", "B_pipe", "FT vs base+agent"),
    ("C_pipe", "B", "full product vs base"),
    ("B_pipe", "B", "agent on base"),
    ("C_pipe", "B_pipe", "FT effect, both with agent"),
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


def load_paired() -> tuple[pl.DataFrame, dict[str, tuple[float, int]]]:
    """Load per-cell composites, inner-join on example_id, return:
      - paired wide df with B,C,B_pipe,C_pipe columns
      - full per-cell (mean, n) for the cell-means figure
    """
    parts: dict[str, pl.DataFrame] = {}
    full_stats: dict[str, tuple[float, int]] = {}
    for cfg in CELLS:
        df = pl.read_parquet(SCORE_DIR / f"{cfg}.parquet").select(
            ["example_id", "composite"]
        )
        full_stats[cfg] = (float(df["composite"].mean()), len(df))
        parts[cfg] = df.rename({"composite": cfg})

    joined = parts["B"]
    for cfg in CELLS[1:]:
        joined = joined.join(parts[cfg], on="example_id", how="inner")
    return joined, full_stats


def bootstrap_ci(
    diffs: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Match RQ2 doc: seed=42, fresh RNG per contrast, 1000 resamples."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = diffs[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Figure 1 — Forest plot of paired contrasts
# ---------------------------------------------------------------------------


def figure_1_paired_contrasts(paired: pl.DataFrame) -> None:
    n = len(paired)
    arr = paired.select(CELLS).to_numpy()

    rows: list[dict] = []
    for a, b, label in CONTRASTS:
        ai, bi = CELLS.index(a), CELLS.index(b)
        diffs = arr[:, ai] - arr[:, bi]
        m, lo, hi = bootstrap_ci(diffs)
        sig = (lo > 0) or (hi < 0)
        rows.append(
            {"a": a, "b": b, "label": label, "m": m, "lo": lo, "hi": hi, "sig": sig}
        )

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    y_pos = np.arange(len(rows))[::-1]  # first row → top of plot
    for y, row in zip(y_pos, rows):
        color = "#1f4e79" if row["sig"] else "#999999"
        ax.errorbar(
            row["m"],
            y,
            xerr=[[row["m"] - row["lo"]], [row["hi"] - row["m"]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.4,
            elinewidth=1.4,
            zorder=3,
        )
        ax.text(
            0.105,
            y,
            f"{row['m']:+.3f}  [{row['lo']:+.3f}, {row['hi']:+.3f}]",
            fontsize=9,
            va="center",
            ha="left",
            color="#222222" if row["sig"] else "#666666",
            family="DejaVu Sans Mono",
        )

    ax.axvline(0, color="#cc6666", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{r['a']} − {r['b']}  ({r['label']})" for r in rows],
        fontsize=10,
    )
    ax.set_xlim(-0.10, 0.20)
    ax.set_xlabel("Mean paired difference in composite score (95% bootstrap CI)")
    ax.set_title(
        f"Paired contrasts on the 2×2 ablation (n={n} paired briefs, 1000 resamples, seed=42)",
        loc="left",
        fontsize=11,
    )
    sig_proxy = plt.Line2D(
        [], [], color="#1f4e79", marker="o", linestyle="", markersize=7, label="significant (95% CI excludes 0)"
    )
    nsig_proxy = plt.Line2D(
        [], [], color="#999999", marker="o", linestyle="", markersize=7, label="not significant"
    )
    ax.legend(
        handles=[sig_proxy, nsig_proxy],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig-4-4-1-paired-contrasts.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"WROTE {out}")


# ---------------------------------------------------------------------------
# Figure 2 — 2×2 cell-means grid
# ---------------------------------------------------------------------------


def figure_2_cell_means(full_stats: dict[str, tuple[float, int]]) -> None:
    grid = [
        ["B", "B_pipe"],
        ["C", "C_pipe"],
    ]
    row_labels = ["Base Qwen3-8B", "Fine-tuned Draper"]
    col_labels = ["Agent off", "Agent on"]

    means = np.array([[full_stats[cell][0] for cell in row] for row in grid])
    # Narrow vmin/vmax so the four cells differentiate visually.
    vmin, vmax = means.min() - 0.005, means.max() + 0.005

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    im = ax.imshow(means, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(2))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(2))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for r in range(2):
        for c in range(2):
            cell = grid[r][c]
            mean, n = full_stats[cell]
            # Pick text colour by background luminance.
            txt_color = "white" if (mean - vmin) / (vmax - vmin) > 0.55 else "#222222"
            ax.text(
                c, r - 0.22, cell,
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=txt_color,
            )
            ax.text(
                c, r + 0.02, f"{mean:.3f}",
                ha="center", va="center",
                fontsize=20, fontweight="bold", color=txt_color,
            )
            ax.text(
                c, r + 0.24, f"n = {n}",
                ha="center", va="center",
                fontsize=10, color=txt_color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Composite score", fontsize=10)
    cbar.outline.set_visible(False)

    ax.set_title(
        "2×2 cell means on the held-out 215-brief test set",
        loc="left", fontsize=11, pad=10,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig-4-4-2-cell-means.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"WROTE {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paired, full_stats = load_paired()
    print(f"Paired n = {len(paired)}")
    print("Per-cell full-set means:")
    for cfg in CELLS:
        m, n = full_stats[cfg]
        print(f"  {cfg:7s}: mean={m:.4f}  n={n}")
    figure_2_cell_means(full_stats)
    figure_1_paired_contrasts(paired)


if __name__ == "__main__":
    main()
