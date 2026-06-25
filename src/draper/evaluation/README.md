# Eval pipeline

Pairwise LLM-as-judge across the active config roster, plus reference eval
against the real winning ad and methodology validation against external
A/B-test ground truth.

## Config roster and naming convention

Configs follow a split-suffixed naming convention introduced with the v2
cutover (2026-05-24):

| Config  | Split | Description                                         |
|---------|-------|-----------------------------------------------------|
| `A_v2`  | v2    | GPT-5.5 single-shot (frontier baseline)             |
| `B_v2`  | v2    | Base Qwen3-8B via OpenRouter (no fine-tune)         |
| `C_v2`  | v2    | Draper v2 Qwen3-8B fine-tune (r=64, no tools)       |
| `GOLD_v2` | v2  | Real winning ad from v2 test split (sentinel)       |
| `A`     | v1    | GPT-5.5 single-shot (v1 re-run only)                |
| `B`     | v1    | Base Qwen3-8B (v1 re-run only)                      |
| `C`     | v1    | Draper-r16 fine-tune (v1 re-run only)               |
| `GOLD`  | v1    | Real winning ad from v1 test split (sentinel)       |
| `A_pipe` / `C_pipe` / `B_pipe` | v1 | Frontend pipeline configs (Arm 2, v1 only) |

Every v2 config name ends in `_v2`. Bare names (`A`, `B`, `C`, `GOLD`) are
v1 and preserved for re-runs. To re-run v1 evals, flip `test_split_dir` in
`configs/eval.yaml` back to `data/final/test` (215 briefs) and restore the
v1 pair list (`[A,C], [B,C], [A,B], [C,GOLD], [A,GOLD], [B,GOLD]`).

**Arm-2/pipeline configs** (`A_pipe`, `C_pipe`, `B_pipe`) are not yet ported
to v2. Firing arm-2 against the v2 test split will fail until `*_pipe_v2`
configs ship.

## Two arms, three judges, two questions

The pipeline runs two **parallel, independent** evaluation arms:

- **Tournament eval (model vs model).** Each pair in `arm1_pairs` is judged
  head-to-head. Cheap, scales, gives a clean ranking. Tells you the order of
  configs but not the absolute level.
- **Reference eval (model vs real ad).** Each config is also paired against
  `GOLD_v2` (v2) or `GOLD` (v1 re-runs) — the original winning ad from the
  held-out test split. Win-rate vs GOLD anchors absolute quality: ≥50% means
  the model is at human-pro level on at least the test distribution.

A separate **methodology validation** step runs the judges against the Upworthy
A/B test winners (`scripts/eval.py validate`) and reports per-judge accuracy
vs the real CTR winner. **Run this before trusting any judge verdict** — a
judge that can't pick A/B winners on real data is noise on Draper's outputs.

## When does GOLD validation happen?

There is **no separate gold-eval command**. `GOLD_v2` is a sentinel config in
`configs/eval.yaml`'s `arm1_pairs` (as of the 2026-05-24 v2 cutover):

```yaml
arm1_pairs:
  - [A_v2, C_v2]
  - [B_v2, C_v2]
  - [A_v2, B_v2]
  - [C_v2, GOLD_v2]
  - [A_v2, GOLD_v2]
  - [B_v2, GOLD_v2]
```

When `judge` runs against any pair containing a GOLD config, the driver
synthesizes a one-off `Inference` from `Brief.reference_assistant` (the real
ad, kept in memory but never sent to inference models). `is_gold()` in
`gold.py` matches the bare sentinel `GOLD` or any `GOLD_*` variant, so
`GOLD_v2` works automatically. Aggregation reports win-rate vs GOLD alongside
the model-vs-model rows.

GOLD is arm-1 only — URL scenarios (arm 2) have no held-out reference ad.

## Recommended workflow (v2 split, active default)

```bash
# 0. Validate the judges first. If accuracy < ~65%, fix the rubric before
#    interpreting any model verdicts.
python scripts/eval.py validate \
  --judge claude-sonnet-4-6 \
  --judge gemini-2.5-flash \
  --judge gpt-5.4-mini \
  --stream upworthy --limit 50

# 1. Inference (one-time per config; outputs cached on disk).
#    test_split_dir in eval.yaml must point at data/constructed_v2/final_v2/test (228 briefs).
python scripts/eval.py infer --configs A_v2,B_v2,C_v2 --split test

# 2. Normalize — extract clean ad copy before judging (REQUIRED).
#    Strips rationale, <think> blocks, emoji-spam, and model-collapse artifacts.
#    Caches under data/eval/inferences_clean/. ~$2 with haiku-4-5 default,
#    ~$0.50 with --extractor gemini-2.5-flash.
python scripts/eval.py normalize --configs A_v2,B_v2,C_v2,GOLD_v2

# 3a. Live judging (sync, full price).
python scripts/eval.py judge \
  --pair A_v2,C_v2 --pair B_v2,C_v2 --pair A_v2,B_v2 \
  --pair C_v2,GOLD_v2 --pair A_v2,GOLD_v2 --pair B_v2,GOLD_v2 \
  --judge claude-sonnet-4-6 --judge gemini-2.5-flash

# 3b. OR: batch judging (50% off, 24h SLA — OpenAI & Anthropic only).
python scripts/eval.py judge-batch submit \
  --pair C_v2,GOLD_v2 --judge claude-sonnet-4-6 --run-id jun-v2-smoke
python scripts/eval.py judge-batch status \
  --run-id jun-v2-smoke --pair C_v2,GOLD_v2 --judge claude-sonnet-4-6
python scripts/eval.py judge-batch collect \
  --run-id jun-v2-smoke --pair C_v2,GOLD_v2 --judge claude-sonnet-4-6

# 4. Aggregate (reads either live or batch judgments — same on-disk shape).
#    NOTE: --groupby vertical and --groupby source_tier collapse to a single
#    "unknown" bucket on the v2 split — those columns are absent from v2
#    metadata. Only --groupby platform produces meaningful segments on v2.
python scripts/eval.py aggregate --run-id jun-v2-smoke \
  --groupby platform --similarity

# 5. Pretty-print.
python scripts/eval.py report --run-id jun-v2-smoke
```

## Judge panel (May 2026)

`configs/eval.yaml` ships a 3-judge cost-aware panel:

| Judge                | Sync $/M out | Batch $/M out | Provider  | Role                  |
|----------------------|--------------|---------------|-----------|-----------------------|
| `claude-sonnet-4-6`  | $15          | $7.50         | Anthropic | Frontier anchor       |
| `gemini-2.5-flash`   | $2.50        | sync only     | Google    | Cheap GA judge        |
| `gpt-5.4-mini`       | $4.50        | $2.25         | OpenAI    | Mid-tier OAI judge    |

Cohen's kappa is computed between `primary` (Sonnet) and `secondary`
(gemini-flash); the third judge contributes to Elo and bootstrap CIs.

Cross-family coverage matters because LLM judges show measured self-preference
bias. Training data was multi-provider teacher-graded so Claude-as-primary is
fine here — no correlated-bias concern.

## On-disk layout

All eval artifacts live under `data/eval/`. Every writer routes through
`draper.evaluation.paths.EvalPaths` so the layout stays consistent.

```
data/eval/
  inferences/<config>/<example_id>.json         # latest single-shot output per config
  inferences_clean/<config>/<example_id>.json   # normalized ad copy (judge input)
  judgments/<judge>/<pair>/<example_id>.json    # pairwise verdicts
  learned_scores/<config>.parquet               # latest scoring-predictor output
  validation/<stream>_<judge>.json              # judge methodology calibration
  url_scenarios.jsonl                           # Arm 2 seed data
  runs/<run_id>/                                # frozen per-run artifacts
    manifest.json
    aggregates/{summary,ci,*}.parquet  elo.json
    diagnostics/<kind>/...
    batches/<judge>/<pair>/...
```

`run_id` follows `YYYY-MM-DD-<slug>` (e.g. `2026-05-15-hook-v2`). The
flat per-config caches (`inferences/`, `inferences_clean/`, `judgments/`,
`learned_scores/`) are keyed by config name and **overwrite on rerun** —
preserve runs across iterations by either using a new variant config
(see below) or by copying `runs/<old>/` aside.

## Iterating agent architectures

The frontend agent is reached via configs with `runner: frontend` (e.g.
`A_pipe`, `B_pipe`, `C_pipe`). To iterate the agent and compare to a
baseline run, use **variant configs**: a config name shaped
`<base>@<variant_slug>` (e.g. `A_pipe@hook-v2`, `C_pipe@no-rag`). The
variant suffix is preserved verbatim in every on-disk path, so the
baseline `A_pipe` data is never overwritten.

```yaml
# configs/eval.yaml
eval:
  configs:
    A_pipe:                         # baseline (existing)
      runner: frontend
      base_url_env: EVAL_FRONTEND_A_URL
      token_env: EVAL_SERVICE_TOKEN
      timeout_s: 300
    A_pipe@hook-v2:                 # iteration: new orchestrator prompt
      label: "GPT-4o + frontend pipeline — hook rewrite v2"
      runner: frontend
      base_url_env: EVAL_FRONTEND_A_HOOK_V2_URL
      token_env: EVAL_SERVICE_TOKEN
      timeout_s: 300
```

Point the variant's `base_url_env` at a separate frontend build (different
branch, deployed locally on another port, or a fresh Modal app). The eval
pipeline itself doesn't know anything has changed — only the config
points at a different agent.

```bash
# Run the variant's scenarios, judge it against the baseline, aggregate.
python scripts/eval.py scenarios run --configs A_pipe@hook-v2
python scripts/eval.py normalize --configs A_pipe@hook-v2
python scripts/eval.py judge \
  --pair A_pipe@hook-v2,A_pipe \
  --judge claude-sonnet-4-6 --judge gpt-5.4-mini
python scripts/eval.py aggregate --run-id 2026-05-15-hook-v2 --arm arm2
python scripts/eval.py score-summary \
  --configs A_pipe@hook-v2,A_pipe --run-id 2026-05-15-hook-v2

# Diff vs the baseline aggregate run.
python scripts/eval.py compare \
  --base 2026-05-14-clean-pipe \
  --candidate 2026-05-15-hook-v2
# → runs/2026-05-15-hook-v2/aggregates/compare_vs_2026-05-14-clean-pipe.{parquet,md}
```

The compare command joins the two runs' `summary.parquet` and
`learned_scores_summary.parquet` on the shared (config × judge) keys and
emits win-rate, learned-score, and Elo deltas.

## MAUVE arm (distribution-matching)

**Question it answers:** does Draper's output distribution look like real high-performing ads at the corpus level — not per-ad quality (pairwise) and not absolute score (learned scorer), but population-level overlap in text feature space?

**Reference corpus:** v3 high-tier ads from `data/scored/v3/scored_ads.parquet`, minus any example whose `example_id` hash appears in the held-out test split (contamination filter in `mauve_reference.py`).

**Slices:** per-platform (facebook, pinterest, reddit, tiktok, twitter) plus an `ALL` aggregate.

**Embedding model:** GPT-2-large (MAUVE library default — no API calls, runs locally).

**Uncertainty:** bootstrap CIs over `mauve.bootstrap_n` resamples (default 100) at 95% level.

**Sanity checks (Phase 5 of the integration plan):** GOLD config should score ≥0.90 (held-out real ads vs reference should nearly match themselves); Lorem Ipsum baseline should score ≤0.10. If either check fails, inspect the reference corpus loader before trusting per-config results.

**Outputs:** per-platform Parquets under `data/eval/mauve_scores/<config>/`; per-run aggregates via `mauve-summary`.

**Requires:** `[mauve]` extra (`uv pip install -e ".[mauve]"`). See `docs/project/MAUVE_INTEGRATION_PLAN.md` for the full design and phase plan.

```bash
# Run MAUVE for one or more configs (outputs to data/eval/mauve_scores/).
# Use _v2 configs against the v2 split (default test_split_dir).
uv run python scripts/eval.py mauve --configs A_v2,B_v2,C_v2,GOLD_v2

# Aggregate into a per-run summary table.
uv run python scripts/eval.py mauve-summary --run-id 2026-05-19-mauve-v2
```

## Reference-metrics arm (overlap-vs-winner)

**Question it answers:** how close is each generation to a real winning ad? For an ad the wording *is* the conversion mechanism, so similarity to a proven winner is legitimate positive evidence — per-ad (unlike MAUVE's corpus-level view), and judge-free.

**Metrics:** BLEU + chrF (`sacrebleu`), ROUGE-L (reuses `judge/similarity.rouge_l_f1`), METEOR (`nltk`), BERTScore F1 (`bert-score`, roberta-large). BLEU/chrF normalized to `[0,1]` (÷100); ROUGE-L/METEOR already `[0,1]`; BERTScore raw (no baseline rescaling — English values cluster high).

**Two references per generation:**
- `*_gold` — the single real winning ad for the brief (cleaned GOLD copy).
- `*_multi` — the `k` (default 5) nearest high-tier real ads on the same platform, ranked by MiniLM cosine to the gold. Blunts the one-to-many critique (a brief has many valid winners). Multi-ref aggregation: BLEU/chrF/METEOR consume the full ref list natively; ROUGE-L and BERTScore take the **max** over refs.

**Reference pool:** the same v3 high-tier corpus, contamination filter, and on-disk cache (`data/eval/mauve_ref/`) as the MAUVE arm.

**Memorization cross-check:** `gold_overlap_excess = rouge_l_gold − rouge_l_multi`. High on the fine-tune flags GOLD-specific echo (reproducing *this brief's* winner) vs ordinary broad-style overlap — read alongside the construction-stage n-gram leak guard. Reported column, **not a gate**.

**Grounding (`reference-validate`):** scores each Upworthy A/B variant by similarity to a held-out pool of *other* tests' winners (leave-one-pair-out) and reports per-metric Precision@1 vs the real CTR winner — the same bar the judge arm is held to. A Wilson CI excluding 0.5 means "closer to known winners" carries signal.

**Sanity anchor:** the GOLD config scores `*_gold ≈ 1.0` (self-match), analogous to MAUVE's GOLD ≥ 0.90.

**Outputs:** per-config Parquets under `data/eval/reference_scores/<config>.parquet` (one row per `(config, example_id)`); per-run aggregates via `reference-summary`; per-metric validation JSON under `data/eval/validation/refmetrics_<metric>_upworthy.json`.

**Requires:** `[refmetrics]` extra (`uv pip install -e ".[refmetrics]"`).

```bash
# Compute (fast pass without the neural metric). Uses _v2 configs on the v2 split.
uv run python scripts/eval.py reference-metrics --configs A_v2,B_v2,C_v2,GOLD_v2 --no-bertscore

# Aggregate into a per-run summary (only --groupby platform is meaningful on v2).
uv run python scripts/eval.py reference-summary --configs A_v2,B_v2,C_v2,GOLD_v2 \
  --run-id 2026-06-03-refmetrics-v2 --groupby platform

# Grounding: do the metrics predict real Upworthy A/B winners?
uv run python scripts/eval.py reference-validate --metrics bleu,chrf,rouge_l,meteor --limit 200
```

## Modules

- `briefs.py` — load test briefs and URL scenarios.
- `gold.py` — `GOLD` sentinel + brief→Inference synthesis.
- `inference/` — runners (OpenAI, vLLM, frontend pipeline).
- `judge/clients.py` — **single source of truth** for all provider clients.
  Exports lazy-init singletons (`openai_client()`, `gemini_client()`,
  `anthropic_client()`), model-prefix predicates (`is_claude`, `is_gemini`),
  `provider_for_model` (batch routing), `gemini_compat_schema` (schema
  stripper), `clip_score`, and named token-budget constants
  (`OPENAI_MAX_TOKENS=512`, `ANTHROPIC_MAX_TOKENS=1024`,
  `GEMINI_MAX_OUTPUT_TOKENS=2048`). All other judge modules import from here;
  none duplicate client or predicate logic.
- `judge/extract.py` — regex pre-cleaner (strips `<think>`, markdown
  chrome, conversational preambles). Runs before `normalize.py`.
- `judge/normalize.py` — LLM-based ad-copy extractor (`claude-haiku-4-5` by
  default). Distinguishes pedagogical rationale from actual copy; handles
  model-collapse (emoji spam, repetition loops) via `<EXTRACTION_FAILED>`
  sentinel. Caches per `(config, example_id)` with SHA256 invalidation under
  `data/eval/inferences_clean/`. Run `eval.py normalize` before judging. See
  `docs/project/EVAL_METHODOLOGY_FIX.md` for the full rationale and results.
- `judge/pairwise.py` — live pairwise judge. Dispatch by model prefix:
  `claude-*` → Anthropic tool-use forcing; `gemini-*` → Gemini
  `response_schema`; everything else → OpenAI strict `json_schema`.
- `judge/batch.py` — batch judge runs (OpenAI + Anthropic only — Gemini
  has no flat 50% batch discount). Outputs same on-disk shape as live
  judging; `aggregate` reads either path identically.
- `judge/validation.py` — methodology validation against Upworthy A/B
  winners (LLM-based, one judge at a time).
- `judge/similarity.py` — `rouge_l_f1` + cosine-to-gold diagnostics
  (emitted as columns, never as scores).
- `judge/aggregation.py` — win-rate tables, bootstrap CIs, Elo, kappa,
  per-segment groupby.
- `proxy_validation.py` — statistical primitives for stream validation
  (Spearman, Kruskal-Wallis, Precision@K, NDCG@K).
- `mauve_scorer.py` / `mauve_reference.py` — MAUVE distribution-matching arm
  + v3 high-tier reference-corpus builder with held-out contamination filter.
- `reference_metrics.py` — reference-overlap arm (BLEU/chrF/ROUGE-L/METEOR/
  BERTScore vs the GOLD ad + a nearest-neighbor multi-ref pool); reuses the
  MAUVE reference corpus and the Upworthy `ProxyValidator` for grounding.
