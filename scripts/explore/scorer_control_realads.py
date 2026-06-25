"""Control for the robustness probe: does the scorer light up on its OWN
training distribution (real held-out ads, native split-field format)?

Two experiments:

1. NATIVE: score N real ads from the held-out random/test split using their
   pre-built ``text`` field (the exact format the model trained on), then report
   the predicted-composite distribution and Spearman vs the true v3 label. If the
   model is healthy on-distribution we expect a wide spread and Spearman ~0.7.

2. FORMAT COST: take real ads that have split headline/body/description in the
   raw corpus, score them (a) with fields split (training shape) vs (b) all text
   mashed into ``body`` (the shape Draper's synthetic outputs hit in production),
   and measure how far the score moves. This isolates the distribution-shift cost
   flagged in the Phase 2 results doc.

Run:  uv run python scripts/explore/scorer_control_realads.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

from draper.scoring_predictor.inference import load_predictor

CHECKPOINT = "data/scoring_predictor/checkpoints/random/best"
TEST_SPLIT = "data/scoring_predictor/splits/random/test.parquet"
CORPUS = "data/scored/v3/scored_ads.parquet"
N_NATIVE = 800
N_FORMAT = 300


def _calibrate(predictor, raw: np.ndarray) -> np.ndarray:
    if predictor.calibrators is not None and raw.size > 0:
        return predictor.calibrators.transform(raw)
    return np.clip(raw, 0.0, 1.0)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


def _pctiles(x: np.ndarray) -> str:
    p = np.percentile(x, [10, 50, 90])
    return f"p10={p[0]:.3f} p50={p[1]:.3f} p90={p[2]:.3f}"


def experiment_native(predictor) -> None:
    df = pl.read_parquet(TEST_SPLIT).head(N_NATIVE)
    texts = df["text"].to_list()
    targets = np.asarray(df["target_composite"].to_list(), dtype=float)

    # Score in batches via the raw forward path (text is already pre-built).
    preds = []
    for start in range(0, len(texts), 64):
        raw = predictor._forward(texts[start : start + 64])
        cal = _calibrate(predictor, raw)
        preds.extend(cal[:, 0].tolist())  # composite is head 0
    pred = np.asarray(preds, dtype=float)

    print("=" * 80)
    print(f"EXPERIMENT 1 — NATIVE real ads, training format  (n={len(pred)})")
    print("=" * 80)
    print(f"  predicted composite : mean={pred.mean():.3f} std={pred.std():.3f}  {_pctiles(pred)}")
    print(f"  true v3 composite   : mean={targets.mean():.3f} std={targets.std():.3f}  {_pctiles(targets)}")
    print(f"  Spearman(pred, true): {_spearman(pred, targets):.3f}   (Phase-1 reported ~0.72)")
    print(f"  range used          : {pred.min():.3f} .. {pred.max():.3f}")


def experiment_format_cost(predictor) -> None:
    df = pl.read_parquet(CORPUS)
    # Keep ads that actually have separate fields (a real headline AND body).
    df = df.filter(
        (pl.col("ad_copy_headline").fill_null("").str.strip_chars().str.len_chars() > 0)
        & (pl.col("ad_copy_body").fill_null("").str.strip_chars().str.len_chars() > 0)
    ).head(N_FORMAT)

    rows = df.to_dicts()
    split_items = [
        {
            "platform": r.get("platform") or "unknown",
            "vertical": r.get("vertical") or "unknown",
            "headline": r.get("ad_copy_headline"),
            "body": r.get("ad_copy_body"),
            "description": r.get("ad_copy_description"),
        }
        for r in rows
    ]
    # Monolithic: everything concatenated into body, no field structure.
    mono_items = [
        {
            "platform": r.get("platform") or "unknown",
            "vertical": r.get("vertical") or "unknown",
            "body": " ".join(
                str(r.get(c) or "").strip()
                for c in ("ad_copy_headline", "ad_copy_body", "ad_copy_description")
            ).strip(),
        }
        for r in rows
    ]

    split_pred = np.asarray([s["composite"] for s in predictor.score_many(split_items)], dtype=float)
    mono_pred = np.asarray([s["composite"] for s in predictor.score_many(mono_items)], dtype=float)
    delta = mono_pred - split_pred

    print("\n" + "=" * 80)
    print(f"EXPERIMENT 2 — FORMAT COST: split fields vs monolithic body  (n={len(rows)})")
    print("=" * 80)
    print(f"  split-fields composite : mean={split_pred.mean():.3f} std={split_pred.std():.3f}  {_pctiles(split_pred)}")
    print(f"  monolithic   composite : mean={mono_pred.mean():.3f} std={mono_pred.std():.3f}  {_pctiles(mono_pred)}")
    print(f"  mean abs shift |mono-split|: {np.abs(delta).mean():.3f}   (max {np.abs(delta).max():.3f})")
    print(f"  mean signed shift          : {delta.mean():+.3f}  (negative = monolithic scores lower)")
    print(f"  Spearman(split, mono)      : {_spearman(split_pred, mono_pred):.3f}   (rank stability across formats)")


def main() -> None:
    print(f"loading predictor from {CHECKPOINT} ...\n")
    predictor = load_predictor(CHECKPOINT, device="cpu")
    experiment_native(predictor)
    experiment_format_cost(predictor)


if __name__ == "__main__":
    main()
