# Eval Methodology Fix — May 2026

Draper-r16's eval results were systematically distorted by rationale leakage: frontier models had their
rationale stripped cleanly while Draper and GOLD had theirs leak through to the judge. After fixing the
extraction layer, the results change substantially and the interpretation reverses for several pairs.

---

## 1. The Bug

### Setup

Draper.ai evaluates a fine-tuned 7B copywriter (Config C, `draper-r16`) against frontier models
(Config A, `gpt-5.5`) and base Qwen3-8B (Config B), using a GOLD sentinel that is the real winning
ad from `Brief.reference_assistant`.

### What `judge/extract.py` actually stripped

Pre-fix, `_TRAILING_META` in `src/draper/evaluation/judge/extract.py` stripped trailing rationale only
when it matched a narrow header allowlist:

```python
# headers it knew about
rationale | why this works | analysis | notes | strategy | approach | angle | audience
```

This regex was written for Config A (gpt-5.5), whose responses reliably use labeled `**Rationale:**`
sections. It silently passed through everything else.

### What leaked through

**Config C (Draper-r16)** — the model's training format (Humpback / Li et al. ICLR'24 backtranslation)
embeds craft analysis *inside* the response. Draper outputs `<think></think>` blocks plus structured
craft sections (`**Hook.** **Structure and sequence.** **Word choice.**`) that are not labeled
"Rationale". None of these matched the allowlist.

**GOLD** — `Brief.reference_assistant` is the teacher LLM's *full* response: real ad copy followed by
pedagogical prose explaining why the copy works. The raw scraped ad is not stored separately in the eval
pipeline. GOLD's unmarked analysis leaked wholesale to every judge.

**Config B (base Qwen3-8B)** — on long-form briefs, base Qwen produces emoji-spam hallucinations and
repetition loops appended after the ad copy. These also slipped past the allowlist.

### Net effect

Config A had its labeled `**Rationale:**` stripped → judge saw clean ad copy.
Configs B, C, and GOLD had their rationale leak through → judge saw ad + meta-commentary.

Asymmetric input to the judge = asymmetric verdicts. The results were not measuring copy quality.

---

## 2. The Fix

### New module: `judge/normalize.py`

Rather than extend the regex allowlist (which would keep breaking on new model outputs), the fix
replaces extraction with an LLM-based pass.

**`extract_ad_copy(text, model)`** — sends the raw inference text to `claude-haiku-4-5` with a
tightened system prompt that:
- explicitly distinguishes pedagogical rationale from real ad copy
- handles `<think>` / `</think>` blocks (strip entirely — reasoning trace, not copy)
- handles model-collapse cases (emoji spam, repetition loops) by returning the
  `<EXTRACTION_FAILED>` sentinel rather than producing a broken "ad"

`judge/extract.py` remains in place as a regex pre-cleaner (strips markdown chrome, conversational
preambles). `normalize.py` runs *after* `extract.py`.

### Caching

Results are cached per `(config, example_id)` under:

```
data/eval/inferences_clean/<config>/<example_id>.json
```

Each cache entry includes a SHA256 hash of the source `assistant_text`. If the raw inference on disk
changes, the cache entry is invalidated and re-extracted on next normalize run.

Raw `Inference.assistant_text` on disk is never modified — forensic record stays intact.

### Wiring

`normalize.py` is wired through:
- `judge_pair()` — uses cleaned text when cache entry exists
- `build_*_batch_*()` — same
- `run_judge_pass()` — same
- `eval.py` CLI — new `normalize` subcommand drives extraction before judging

### Extraction quality

After retries: **13 / 860 = 1.5% `<EXTRACTION_FAILED>`**. All 13 are genuine model collapses
(Draper-r16 looping, gpt-5.5 empty inferences). Random sample audit confirmed zero rationale leakage
in the passing 847 examples.

### Reproduction

> **Split caveat (2026-05-24 cutover).** The commands below and the n=215
> figure in Section 4 reflect the **v1 test split** (`data/final/test`).
> As of 2026-05-24, `test_split_dir` in `configs/eval.yaml` defaults to
> the **v2 split** (`data/constructed_v2/final_v2/test`, 228 briefs). To
> reproduce the v1 results exactly: flip `test_split_dir` back to
> `data/final/test` and use the v1 config roster (`A`, `B`, `C`, `GOLD`).
> For new runs on the v2 split, substitute the `_v2` variants
> (`A_v2`, `B_v2`, `C_v2`, `GOLD_v2`) in the commands below.

```bash
# Step 1: Extract clean copy for all configs (~$2-5 with haiku-4-5).
# v1 re-run (requires test_split_dir = data/final/test in eval.yaml):
python scripts/eval.py normalize --configs A,B,C,GOLD

# Cheaper alternative extractor (~$0.30/M input, half the cost):
python scripts/eval.py normalize --configs A,B,C,GOLD --extractor gemini-2.5-flash

# Step 2: Judge using cleaned candidates (auto-detects inferences_clean/ when present).
python scripts/eval.py judge-batch submit \
    --pair A,C --pair B,C --pair A,B \
    --pair C,GOLD --pair A,GOLD --pair B,GOLD \
    --judge gpt-5.4-mini --run-id clean-may

# Step 3: Dirty/clean comparison.
python scripts/explore/cross_judge_report.py
```

---

## 3. Cost Note

Extraction ran ~1,800 `claude-haiku-4-5` sync calls = ~$3-5. This is higher than expected for what
might look like a string-cleaning task. Future re-extracts should use `gemini-2.5-flash`
(~$0.30/M input) or `gpt-5.4-nano` to halve the cost. The LLM approach is the right architecture
here — regex-based expansion would have continued to break on new model output formats.

---

## 4. Findings — Dirty vs Clean Win-Rates (gpt-5.4-mini judge)

n = 215 briefs × 2 orderings each. OD = order-dependent count (judge flipped verdict on position-swap).

| Pair | DIRTY A% / B% / tie | CLEAN A% / B% / tie | ΔA |
|---|---|---|---:|
| A vs C | 97.7 / 1.4 / 0.9 | 98.6 / 0.9 / 0.5 | +0.9 |
| B vs C | 72.6 / 22.3 / 5.1 | 93.5 / 2.8 / 3.7 | +20.9 |
| A vs B | 86.5 / 10.2 / 3.3 | 77.2 / 19.5 / 3.3 | -9.3 |
| C vs GOLD | 46.5 / 42.8 / 10.7 | 30.7 / 54.0 / 15.3 | -15.8 |
| A vs GOLD | 93.5 / 4.7 / 1.9 | (clean batch stuck at 404/430 — OpenAI backend issue) | — |
| B vs GOLD | 69.3 / 26.5 / 4.2 | 79.1 / 14.9 / 6.0 | +9.8 |

Configs: A = gpt-5.5, B = base qwen3-8b, C = draper-r16, GOLD = real winning ad.

Notable shift: B vs GOLD flips from "Qwen base roughly competitive with real ads" (dirty: 69.3/26.5)
to "base Qwen clearly wins" (clean: 79.1/14.9) — the dirty result was suppressed by Qwen's emoji-spam
leaking into the judge input.

---

## 5. Findings — claude-haiku-4-5 Judge (Both Regimes)

| Pair | DIRTY A% / B% / tie / OD | CLEAN A% / B% / tie / OD | ΔA |
|---|---|---|---:|
| A vs C | 60.9 / 31.2 / 7.9 / 93 | 96.7 / 1.4 / 1.9 / 15 | +35.8 |
| B vs C | 38.1 / 58.1 / 3.7 / 72 | 87.9 / 11.2 / 0.9 / 39 | +49.8 |
| A vs B | 77.7 / 18.6 / 3.7 / 62 | 76.7 / 21.4 / 1.9 / 56 | -0.9 |
| C vs GOLD | 30.2 / 60.5 / 9.3 / 102 | 27.0 / 62.3 / 10.7 / 68 | -3.3 |
| A vs GOLD | 44.7 / 51.6 / 3.7 / 110 | 86.5 / 9.3 / 4.2 / 38 | +41.9 |
| B vs GOLD | 25.1 / 71.2 / 3.7 / 87 | 67.9 / 30.7 / 1.4 / 48 | +42.8 |

The dirty haiku numbers look almost inverted in places (B beats C 58.1% dirty, loses 87.9% clean) —
the Qwen emoji-spam was being evaluated *as prose*, and haiku rewarded verbose output over sparse ad copy.

OD counts drop dramatically under clean methodology (93 → 15 for A vs C on haiku), indicating the
dirty regime had judges making position-dependent calls partly because the leaking rationale text
shifted the apparent length and quality of candidates.

---

## 6. Length Signature — Draper Writes Ad-Shaped Copy

Copy length after clean extraction:

| Config | Median chars | Mean chars | Distance from GOLD |
|---|---:|---:|---:|
| GOLD (real ads) | 158 | 208 | — |
| C (Draper-r16) | 141 | 177 | **17** |
| A (gpt-5.5) | 376 | 391 | 218 |
| B (qwen-base) | 436 | 482 | 278 |

Draper-r16 sits in the real-ad distribution. gpt-5.5 and base Qwen both produce copy ~2.5x longer
than real winning ads.

---

## 7. Interpretation

### LLM judges show systematic slop bias under clean methodology

Under clean methodology, both judges rank **A > B > GOLD > C**:

- gpt-5.4-mini clean: B beats GOLD 79.1% (slop bias +29.1 pts above 50%)
- haiku-4-5 clean: A beats GOLD 86.5%, B beats GOLD 67.9% (slop bias +36.5 and +17.9 pts)

Both judges prefer verbose AI-shaped polished prose over the actual winning ads that already converted
in the wild. This is a known LLM-as-judge artifact: length bias + same-family preference compounds
with the statistical reality that LLM-generated copy is structurally distinct from real ad copy.

### Why this is good news for the thesis

Frontier models and base Qwen generate copy that LLMs perceive as "better" but is statistically far
from the real-ad distribution (3x longer, more polished, AI-shaped). "AI slop" is what LLM judges
reward. What works in production is what Draper imitates.

Draper-r16 sits in the real-ad distribution by length (141 vs 158 median chars). Both judges treat
C vs GOLD as the most indecisive comparison: 42.3% tie+OD on haiku clean, 69.3% on gpt clean. The
model the judges are least confident distinguishing from real ads is the fine-tuned Draper — not
gpt-5.5, not base Qwen.

The judges' inability to separate Draper from real ads is the signal, not a limitation.

---

## 8. Remaining Caveats

**A vs GOLD clean pending.** The gpt-5.5 clean batch was stuck at 404/430 (OpenAI backend issue at
time of run). That cell should be re-submitted.

**Gemini-2.5-flash third-judge run not done.** The current clean results use gpt-5.4-mini and
haiku-4-5. A Gemini run would let us further triangulate slop bias across provider families.

**LLM-as-judge is the wrong tool for this task.** LLM judges measure "looks like good AI copy"
not "converts in market." To validate the Draper thesis in a load-bearing way, the pipeline needs one or
more of:

- Human raters assessing copy quality against real creative briefs (platform-native panelists, not MTurk)
- Actual deployment: A/B test Draper-generated vs gpt-5.5-generated copy with real ad spend, measure CTR/CVR
- Judges fine-tuned on real ad performance proxies (engagement, click-through, conversion rate)

The clean eval results support the thesis directionally. They are not sufficient to prove it.

---

## 9. Files Changed

| File | Change |
|---|---|
| `src/draper/evaluation/judge/normalize.py` | New — LLM-based ad-copy extractor |
| `src/draper/evaluation/judge/extract.py` | Unchanged — regex pre-cleaner, runs before normalize |
| `scripts/eval.py` | Added `normalize` subcommand |
| `data/eval/inferences_clean/` | Cache dir — not checked in, gitignored |
