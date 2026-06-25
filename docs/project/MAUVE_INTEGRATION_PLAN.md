# MAUVE eval arm — integration plan

**Date:** 2026-05-18
**Status:** Planned, not yet implemented
**Owner:** Osayd

## Motivation

Existing eval surface (3-judge LLM panel, learned-scorer absolute arm, reference-vs-GOLD pairwise) leaves two gaps that a thesis committee can object to:

1. The LLM-judge arms are subject to documented biases (position, length, self-preference) and are wrong 16–27% of the time even at the published state of the art.
2. The learned-scorer arm uses a regressor we trained ourselves on AdFlex composite scores — vulnerable to a "circular eval" objection regardless of its ρ=0.72 / ECE=0.007 calibration.

MAUVE (Pillutla et al., NeurIPS 2021 Outstanding Paper; JMLR 2023 theory extension) closes both gaps. It measures KL divergence between two text distributions in embedding space, using an off-the-shelf pretrained encoder we did not train. It answers a different question from the existing arms — not "is ad A better than ad B?" but "does the corpus of generations come from the same distribution as the corpus of real high-performing ads?"

This is the **distribution-matching** claim that maps directly onto our Humpback / backtranslation training objective (Li et al. ICLR 2024). A fine-tuned model imitating real high-performing ads should produce outputs distributionally closer to the real-ad corpus than a generic instruction-tuned model does. MAUVE measures exactly that.

## Design decisions

1. **Reference distribution.** Top-tier (`high`) real ads from `data/scored/v3/scored_ads.parquet` (~8k rows after tier filter). Stratified by platform to match the test-brief platform mix.
2. **Granularity.** One MAUVE score per config × per platform, plus an overall score across platforms.
3. **Embedding model.** GPT-2-large (MAUVE library default). Most defensible — every MAUVE paper uses it. Compatible with the `torch` + `transformers` versions already pinned in the scoring-predictor extra.
4. **Bootstrap CIs.** Resample the generated side 100× at fixed N, report 95% CI. MAUVE is known to be sample-size sensitive; CIs are standard practice.
5. **GOLD as sanity check.** GOLD inferences are pulled from `Brief.reference_assistant` (real ads from the held-out test split). MAUVE(GOLD, reference) should be ≥ 0.9 — if not, the pipeline is broken.
6. **Configs evaluated.** A (GPT-5.5), B (base Qwen3-8B), C (fine-tuned Draper-r16), GOLD. Optionally A_pipe / B_pipe / C_pipe later for the agent-orchestration arm.

## Risks

- **Sample size.** N=215 inferences per config (n=204 for C after extraction failures) is on the low end for MAUVE. Phase 5 sanity checks at N=50/100/200 confirm ranking stability. Mitigation if unstable: regenerate test inferences at N≈1000.
- **Reference contamination.** If Draper's training set overlaps with the v3 corpus used as MAUVE reference, scores will be inflated for C. **Must verify in Phase 2** that the held-out test split is also held out from the v3 reference set.
- **Flat scores across configs.** Possible if cleaned text is too similar across models. Sanity checks at Phase 5 catch this. Mitigation: increase generation N, or use a finer-grained encoder (sentence-transformers all-mpnet-base-v2 as a secondary).

## Codebase integration points

All paths relative to `/home/user/Projects/Draper.ai/`.

### Reference corpus

- **Source:** `data/scored/v3/scored_ads.parquet` (55,160 rows, schema: `ad`, `composite_score`, `signal_scores`, `tier_probs`, `tier`, `scoring_version`).
- **Ad text fields:** `ad.ad_copy.{headline, body, description, cta}`.
- **Tier assigner:** `src/draper/scoring/tier_assigner.py:13-59` — top 20th percentile by composite_score → `tier="high"`.
- **IO helpers:** `src/draper/utils/io.py` — `read_parquet()` (Polars), `read_jsonl()`.
- **Existing loader pattern:** `src/draper/evaluation/adflex_loader.py:35-102`.

### Generated corpora

- **Cleaned text:** `data/eval/inferences_clean/{config}/{example_id}.json` — `assistant_text_clean` field. Use this as primary source.
- **Raw fallback:** `data/eval/inferences/{config}/{example_id}.json` — `assistant_text` field.
- **Loader:** `src/draper/evaluation/driver.py:130-142` — `load_inferences_for_config()`.
- **Existing text resolver:** `src/draper/evaluation/learned_scorer.py:50-76` — `_resolve_text()` already handles the clean → raw fallback. **Reuse, don't reimplement.**
- **Drop `<EXTRACTION_FAILED>` rows** (sentinel from `judge/normalize.py`).

### Eval-arm pattern to mirror

- **Closest analog:** `src/draper/evaluation/learned_scorer.py`
  - `score_configs()` (lines 133-245) — main entry point.
  - `load_scores()` (lines 248-253) — single-config Parquet reader.
  - `summarize()` (lines 276-303) — per-config + per-segment aggregates.
- **Path resolver:** `src/draper/evaluation/paths.py:93-211` — `EvalPaths` dataclass. Add `mauve_scores_root` property.
- **Config loader:** `src/draper/evaluation/config.py:53-124` — `EvalConfig` class. Add optional `mauve:` block (the `extras="ignore"` pattern at line 69 makes this backwards-compatible).

### CLI

- **Entry point:** `scripts/eval.py` (typer app). Existing pattern at lines 638-721 (`score`) and 723-793 (`score-summary`) is the template.

### Tests

- **Pattern:** `tests/evaluation/test_eval_pipeline.py`, `test_similarity.py`.
- **New file:** `tests/evaluation/test_mauve_scorer.py`.

## Phased implementation

### Phase 1 — Scaffold (½ day)

- [ ] Add `mauve-text>=0.4.0` to `pyproject.toml` as a new optional extra `[mauve]`.
- [ ] `uv lock && uv sync --extra mauve`.
- [ ] Add `mauve_scores_root` property to `src/draper/evaluation/paths.py` → `data/eval/mauve_scores`.
- [ ] Create empty `src/draper/evaluation/mauve_scorer.py` with signatures only.

### Phase 2 — Reference corpus builder (1 day)

- [ ] **Pre-check (blocking):** verify held-out test brief example_ids are not present in `data/scored/v3/scored_ads.parquet`. Abort if there's overlap and re-derive splits.
- [ ] Implement `load_reference_corpus(tier="high", platform=None) -> list[str]` in `mauve_scorer.py`:
  - Reads v3 Parquet via Polars.
  - Filters `tier == "high"`, optionally by platform.
  - Concatenates `headline + body + description` into one text blob per ad (matches the shape of cleaned generations).
- [ ] Cache results to `data/eval/mauve_ref/{platform_or_all}.parquet`.

### Phase 3 — Core scorer (1–2 days)

In `src/draper/evaluation/mauve_scorer.py`:

```python
@dataclass
class MauveResult:
    config: str
    platform: str          # "ALL" for overall
    mauve: float
    ci_low: float
    ci_high: float
    n_gen: int
    n_ref: int
    n_dropped: int
    embedding_model: str
    created_at: str

def score_config(
    config_name: str,
    reference_texts: list[str],
    platform: str = "ALL",
    n_bootstrap: int = 100,
    embedding_model: str = "gpt2-large",
    seed: int = 42,
) -> MauveResult: ...

def score_configs(
    config_names: list[str],
    reference_tier: str = "high",
    platforms: list[str] | None = None,
    out_dir: Path | None = None,
    n_bootstrap: int = 100,
) -> dict[str, Path]: ...

def load_scores(config_name: str, root: Path | None = None) -> pl.DataFrame: ...

def summarize(
    config_names: list[str],
    run_id: str,
    groupby: list[str] | None = None,
) -> dict[str, Path]: ...
```

Steps inside `score_config()`:
1. Load cleaned generations for config (reuse `learned_scorer._resolve_text()`).
2. Filter to platform if specified (drop generations whose brief platform doesn't match).
3. Drop `<EXTRACTION_FAILED>` rows; record `n_dropped`.
4. Compute point estimate: `mauve.compute_mauve(p_text=generations, q_text=reference, ...)`.
5. Bootstrap: for `i in range(n_bootstrap)`, resample generations with replacement, recompute MAUVE.
6. Return point estimate + percentile CI from bootstrap distribution.

Write `data/eval/mauve_scores/{config}.parquet` with one row per `(config, platform)` pair.

### Phase 4 — CLI wiring (½ day)

Add to `scripts/eval.py`:

```bash
# Score
uv run python scripts/eval.py mauve \
    --configs A,B,C,GOLD \
    --reference-tier high \
    --bootstrap-n 100

# Aggregate
uv run python scripts/eval.py mauve-summary \
    --configs A,B,C,GOLD \
    --run-id mauve-2026-05-18 \
    --groupby platform
```

Add to `configs/eval.yaml`:

```yaml
mauve:
  reference_tier: high
  embedding_model: gpt2-large
  bootstrap_n: 100
  platforms: [facebook, pinterest, reddit, tiktok, twitter]
  random_seed: 42
```

### Phase 5 — First run + sanity checks (½ day)

Non-negotiable checks before trusting any number:

1. **GOLD floor check:** MAUVE(GOLD, reference) ≥ 0.9. GOLD *is* real ads from a held-out split; if this isn't ~1.0, the pipeline is wrong.
2. **Random text ceiling check:** generate 200 Lorem ipsum strings → MAUVE ≤ 0.1. Confirms the metric discriminates.
3. **N sensitivity check:** run at N=50, 100, 200. Ranking across A/B/C should be stable. If it flips with sample size, abort and scale generations up.

### Phase 6 — Stretch (1 day, optional)

- [ ] Per-platform breakdown in summary tables.
- [ ] **Paired bootstrap delta tests** (Draper > GPT MAUVE significance), mirroring the paired-t setup in `SCORING_PREDICTOR_PHASE2_RESULTS_2026-05.md`.
- [ ] **Vendi Score sidekick** (TMLR 2023, Conditional-Vendi arXiv 2411.02817 Nov 2024) — measures diversity within a single config's generations. Reuses the same embedding pipeline. Adds a "diversity preserved" claim.

### Phase 7 — Thesis write-up

Out of scope for code work. Target ~1 page methods + ~2 pages results with one main table and one per-platform table. Citations:

- Pillutla et al. NeurIPS 2021 — MAUVE original
- Pillutla et al. JMLR 2023 — MAUVE theory
- Li et al. ICLR 2024 — Humpback (motivates the distribution-matching framing)
- Friedman & Dieng TMLR 2023 — Vendi Score (if Phase 6)

## Expected deliverable

After Phase 5, the headline table will be:

| Config | MAUVE (overall) | 95% CI | TikTok | Facebook | Pinterest | Twitter | Reddit |
|---|---|---|---|---|---|---|---|
| GOLD | ~0.97 | — | … | … | … | … | … |
| **C (Draper)** | **expected ~0.7–0.8** | — | … | … | … | … | … |
| B (Qwen) | expected ~0.4–0.6 | — | … | … | … | … | … |
| A (GPT-5.5) | expected ~0.4–0.5 | — | … | … | … | … | … |

If results approximate this shape (the literature strongly suggests they will for a Humpback-trained model vs. an instruction-tuned generic model), the thesis has a clean positive result:

> Draper outputs are distributionally closer to real winning ads than baseline generations, with bootstrap-CI-backed significance, using a metric with NeurIPS Outstanding Paper pedigree and no LLM judges involved.

This is the "distribution-matching" positive claim, complementary to the calibrated engagement-prediction claim from the learned-scorer arm and any controllability claim added later.

## Total time estimate

| Phase | Effort |
|---|---|
| 1. Scaffold | 0.5 day |
| 2. Reference corpus | 1.0 day |
| 3. Core scorer | 1.5 days |
| 4. CLI wiring | 0.5 day |
| 5. Sanity checks | 0.5 day |
| **Subtotal (shippable)** | **4 days** |
| 6. Stretch (paired tests, Vendi) | +1 day |

## References

- Pillutla, K. et al. (2021). MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers. NeurIPS 2021 (Outstanding Paper).
- Pillutla, K. et al. (2023). MAUVE Scores for Generative Models: Theory and Practice. JMLR.
- Li, X. et al. (2024). Self-Alignment with Instruction Backtranslation. ICLR 2024.
- Friedman, D. & Dieng, A. B. (2023). The Vendi Score: A Diversity Evaluation Metric for Machine Learning. TMLR.

## Related docs

- `docs/project/EVAL_METHODOLOGY_FIX.md` — eval methodology background
- `docs/research/SCORING_PREDICTOR_PHASE2_RESULTS_2026-05.md` — learned-scorer arm results (the paired-bootstrap pattern in Phase 6 mirrors this doc's statistics section)
- `src/draper/evaluation/README.md` — current arm taxonomy; update after Phase 5
