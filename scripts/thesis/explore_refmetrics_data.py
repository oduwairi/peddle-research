"""Explore the reference-overlap + Upworthy data to find genuinely informative
figure designs (distributions, joint relationships, mechanism) — not summaries."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

REF = Path("data/eval/reference_scores")
VAL = Path("data/eval/validation")

print("===== reference_scores parquet schema (C) =====")
c = pl.read_parquet(REF / "C.parquet")
print(f"rows={len(c)}  cols={len(c.columns)}")
for name, dt in zip(c.columns, c.dtypes):
    print(f"  {name:28s} {dt}")

print("\n===== sample row (C[0]) =====")
row = c.head(1).to_dicts()[0]
for k, v in row.items():
    sv = str(v)
    print(f"  {k:28s} {sv[:80]}")

print("\n===== which configs / platform present =====")
for cfg in ("A", "B", "C", "GOLD"):
    df = pl.read_parquet(REF / f"{cfg}.parquet")
    plats = df["platform"].value_counts().to_dicts() if "platform" in df.columns else "NO platform col"
    print(f"  {cfg}: n={len(df)}  platform={plats}")

print("\n===== pool/multi columns? =====")
print("  cols containing 'pool' or 'multi':",
      [x for x in c.columns if "pool" in x or "multi" in x])

print("\n===== Upworthy validation JSON structure (meteor) =====")
j = json.loads((VAL / "refmetrics_meteor_upworthy.json").read_text())
print("  top-level keys:", list(j.keys()))
for k, v in j.items():
    if isinstance(v, list):
        print(f"  {k}: list len={len(v)}  sample={v[:3]}")
    else:
        print(f"  {k}: {v}")
print("\n  available upworthy json files:",
      sorted(p.name for p in VAL.glob("refmetrics_*_upworthy.json")))
