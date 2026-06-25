"""Generate §4.3 figures from the MAUVE eval data.

Outputs into docs/research/figures/:
  - fig-4-3-1-umap-embeddings.png       (UMAP of GPT-2 Large embeddings)
  - fig-4-3-2-per-platform-forest.png   (forest plot of per-platform CIs)

Neither figure is a redraw of numbers stated in §4.3 prose:
  - Fig 1 directly visualises the embedding cloud overlap that MAUVE
    integrates — the picture behind the corpus-level number.
  - Fig 2 shows per-platform CI structure that the prose deliberately
    omitted (after pruning the per-platform paragraph).

Numbers source:
  - data/eval/inferences_clean/{config}/*.json  (clean per-ad texts)
  - data/eval/mauve_ref/ALL.parquet             (8,931 v3 high-tier reference ads)
  - data/eval/runs/2026-05-22-mauve-with-pipes/aggregates/
      mauve_scores_by_platform.parquet          (CIs)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

FIG_DIR = Path("docs/research/figures")
RUN_DIR = Path("data/eval/runs/2026-05-22-mauve-with-pipes/aggregates")
INFER_DIR = Path("data/eval/inferences_clean")
REF_PATH = Path("data/eval/mauve_ref/ALL.parquet")

CONFIG_ORDER = ["A", "B", "B_pipe", "C", "C_pipe", "GOLD"]
CONFIG_LABEL = {
    "A": "A (gpt-5.4-mini)",
    "B": "B (Qwen3-8B)",
    "B_pipe": "B_pipe (B + agent)",
    "C": "C (Draper)",
    "C_pipe": "C_pipe (C + agent)",
    "GOLD": "GOLD (real ads)",
}
CONFIG_COLOR = {
    "A": "#7f7f7f",
    "B": "#4d4d4d",
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


# ---------------------------------------------------------------------------
# Figure 1 — UMAP of GPT-2 Large embeddings
# ---------------------------------------------------------------------------

EMBED_CACHE = Path("data/eval/runs/2026-05-22-mauve-with-pipes/embeddings_cache.npz")


def _load_config_texts(cfg: str) -> list[str]:
    out: list[str] = []
    for f in sorted((INFER_DIR / cfg).glob("*.json")):
        try:
            with open(f) as fh:
                obj = json.load(fh)
        except Exception:
            continue
        t = (obj.get("assistant_text_clean") or "").strip()
        if t and t != "<EXTRACTION_FAILED>":
            out.append(t)
    return out


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Mean-pool the last hidden state of GPT-2 Large over tokens."""
    import torch
    from transformers import GPT2Model, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2-large")
    tok.pad_token = tok.eos_token
    model = GPT2Model.from_pretrained("gpt2-large").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  device={device}, batch=8, n_texts={len(texts)}")

    outs: list[np.ndarray] = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc).last_hidden_state  # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        outs.append(pooled.cpu().numpy())
        if (i // batch_size) % 25 == 0:
            print(f"    embedded {i + len(batch)}/{len(texts)}")
    return np.concatenate(outs, axis=0)


def _build_or_load_embeddings(
    ref_sample_n: int = 2000, seed: int = 42
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if EMBED_CACHE.exists():
        print(f"LOAD cached embeddings: {EMBED_CACHE}")
        data = np.load(EMBED_CACHE, allow_pickle=True)
        ref = data["ref"]
        cfg_embs = {cfg: data[f"cfg_{cfg}"] for cfg in CONFIG_ORDER if f"cfg_{cfg}" in data.files}
        return ref, cfg_embs

    print("COMPUTE GPT-2 Large embeddings (uncached)")

    # Reference (downsampled to keep UMAP fast and the figure legible)
    ref_df = pl.read_parquet(REF_PATH)
    rng = np.random.default_rng(seed)
    if len(ref_df) > ref_sample_n:
        idx = rng.choice(len(ref_df), size=ref_sample_n, replace=False)
        ref_texts = [ref_df["text"][int(i)] for i in idx]
    else:
        ref_texts = ref_df["text"].to_list()
    print(f"  reference: {len(ref_texts)} texts (downsampled from {len(ref_df)})")
    ref_emb = _embed_texts(ref_texts)

    cfg_embs: dict[str, np.ndarray] = {}
    for cfg in CONFIG_ORDER:
        texts = _load_config_texts(cfg)
        print(f"  {cfg}: {len(texts)} texts")
        if not texts:
            continue
        cfg_embs[cfg] = _embed_texts(texts)

    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMBED_CACHE,
        ref=ref_emb,
        **{f"cfg_{cfg}": emb for cfg, emb in cfg_embs.items()},
    )
    print(f"WROTE cache: {EMBED_CACHE}")
    return ref_emb, cfg_embs


def figure_1_umap_embeddings() -> None:
    """6-panel small-multiples view: each panel shows reference as grey
    background + one config's ads in its accent colour. Side-by-side
    comparison reveals which configs' clouds sit closer to the reference
    distribution that MAUVE compares against.
    """
    ref_emb, cfg_embs = _build_or_load_embeddings()

    from umap import UMAP

    cfgs_present = [c for c in CONFIG_ORDER if c in cfg_embs]
    all_emb = np.concatenate([ref_emb] + [cfg_embs[c] for c in cfgs_present])
    print(f"UMAP fit on {all_emb.shape[0]} vectors, dim={all_emb.shape[1]}")
    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        metric="euclidean",
    )
    proj = reducer.fit_transform(all_emb)

    cuts = [len(ref_emb)]
    for cfg in cfgs_present:
        cuts.append(cuts[-1] + len(cfg_embs[cfg]))
    ref_proj = proj[: cuts[0]]
    cfg_projs: dict[str, np.ndarray] = {
        cfg: proj[cuts[j] : cuts[j + 1]] for j, cfg in enumerate(cfgs_present)
    }

    # Shared view box derived from the reference cloud (1st/99th percentile)
    # so outlier reference points don't flatten the configs.
    x_lo, x_hi = np.percentile(ref_proj[:, 0], [1, 99])
    y_lo, y_hi = np.percentile(ref_proj[:, 1], [1, 99])
    pad_x = 0.05 * (x_hi - x_lo)
    pad_y = 0.05 * (y_hi - y_lo)
    xlim = (x_lo - pad_x, x_hi + pad_x)
    ylim = (y_lo - pad_y, y_hi + pad_y)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), sharex=True, sharey=True)
    for ax, cfg in zip(axes.flat, cfgs_present):
        ax.scatter(
            ref_proj[:, 0],
            ref_proj[:, 1],
            s=3,
            c="#d6d6d6",
            alpha=0.35,
            linewidths=0,
            zorder=1,
        )
        pts = cfg_projs[cfg]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=14,
            c=CONFIG_COLOR[cfg],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.25,
            zorder=3,
        )
        ax.set_title(f"{CONFIG_LABEL[cfg]} — n={len(pts)}", fontsize=10, loc="left")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)

    # Outer labels only
    for ax in axes[-1, :]:
        ax.set_xlabel("UMAP dimension 1")
    for ax in axes[:, 0]:
        ax.set_ylabel("UMAP dimension 2")

    fig.suptitle(
        "UMAP of GPT-2 Large embeddings per configuration vs. the v3 high-tier reference cloud (grey)",
        fontsize=11,
        y=1.00,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig-4-3-1-umap-embeddings.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"WROTE {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Forest plot of per-platform CIs
# ---------------------------------------------------------------------------


def figure_2_per_platform_forest() -> None:
    df = pl.read_parquet(RUN_DIR / "mauve_scores_by_platform.parquet").filter(
        pl.col("platform") != "ALL"
    )

    n_plat = len(PLATFORM_ORDER)
    fig, axes = plt.subplots(
        n_plat, 1, figsize=(7.8, 1.8 * n_plat), sharex=True
    )

    for ax, plat in zip(axes, PLATFORM_ORDER):
        sub = df.filter(pl.col("platform") == plat)
        gold_row = sub.filter(pl.col("config") == "GOLD")
        gold_mean = (
            float(gold_row.row(0, named=True)["mauve_mean"])
            if not gold_row.is_empty()
            else None
        )
        gold_lo = (
            float(gold_row.row(0, named=True)["ci_low_mean"])
            if not gold_row.is_empty()
            else None
        )
        gold_hi = (
            float(gold_row.row(0, named=True)["ci_high_mean"])
            if not gold_row.is_empty()
            else None
        )

        # Light band marking the GOLD 95% CI — visual cue for statistical tie.
        if gold_lo is not None and gold_hi is not None:
            ax.axvspan(gold_lo, gold_hi, color="#f3e1bd", alpha=0.55, zorder=0)
        if gold_mean is not None:
            ax.axvline(gold_mean, color="#c08400", linestyle="--", linewidth=1.0, zorder=1)

        # One row per config; GOLD pinned at the bottom for visual anchor.
        cfgs = [c for c in CONFIG_ORDER]
        for i, cfg in enumerate(cfgs):
            r = sub.filter(pl.col("config") == cfg)
            if r.is_empty():
                ax.text(
                    0.02,
                    i,
                    "(no data)",
                    color="#999999",
                    fontsize=8,
                    va="center",
                )
                continue
            row = r.row(0, named=True)
            m = float(row["mauve_mean"])
            lo = float(row["ci_low_mean"])
            hi = float(row["ci_high_mean"])
            ax.errorbar(
                m,
                i,
                xerr=[[m - lo], [hi - m]],
                fmt="o",
                color=CONFIG_COLOR[cfg],
                ecolor=CONFIG_COLOR[cfg],
                capsize=3,
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.4,
                elinewidth=1.2,
                zorder=3,
            )

        ax.set_yticks(np.arange(len(cfgs)))
        ax.set_yticklabels(cfgs, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(len(cfgs) - 0.5, -0.5)
        ax.set_title(PLATFORM_LABEL[plat], loc="left", fontsize=10, fontweight="bold")
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("MAUVE score (95% bootstrap CI; gold band = GOLD CI)")
    fig.suptitle(
        "Per-platform MAUVE with 95% bootstrap CIs (GOLD CI band shaded)",
        fontsize=11,
        y=1.005,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig-4-3-2-per-platform-forest.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"WROTE {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_2_per_platform_forest()  # cheap; run first
    figure_1_umap_embeddings()       # expensive; ~3-5 min CPU first run


if __name__ == "__main__":
    main()
