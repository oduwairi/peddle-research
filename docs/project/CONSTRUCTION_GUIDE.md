# Training Data Construction Guide

This guide explains how to generate training examples for Draper.ai's
copywriting fine-tune using the chat-first **bundle** construction
pipeline.

> **History — 2026-04 pivot:** Draper.ai previously supported five
> pattern-skill formats. The pivot narrowed the training target to
> creative copywriting only. See `CONSTRUCTION_ARCHITECTURE.md` for the
> full rationale. The prior 5-format guide is snapshotted at
> `archive/docs/CONSTRUCTION_GUIDE_5format.md`.

## What the pipeline does, in plain English

The task is backtranslation (Humpback, Li et al., ICLR'24). For each
training example:

1. `SourceSelector` picks a single high-score reference ad.
2. Python derives copywriting context from the ad itself (no RNG):
   - `source_ad_shape` — inferred from populated copy fields
     (`has_body` vs `headline_only`; carried as provenance metadata only)

   The prior `platform_framing` axis was removed after the 2026-04 audit:
   the corpus's `platform` is a scraping-source artifact (which AdFlex
   endpoint returned the ad), not a creative attribute, so conditioning
   the teacher on it trained the student on label noise. Platform-
   specific formatting is handled at inference via prompt.
3. That context + the real ad + the BACKTRANSLATION style rules are
   packed into a self-contained **bundle** string.
4. You paste the bundle into the chosen chat agent (Claude / GPT /
   Gemini). The agent returns `<user_prompt>` + `<assistant_response>`
   tags — a purely factual brief and an assistant reply that delivers
   the real ad verbatim with a short rationale.
5. `ingest` parses the tags, verifies backtranslation fidelity (word
   coverage + verbatim signature), and saves a `TrainingExample`.

The student model learns: factual brief → real copy produced as part of
a normal LLM response.

## Quick Start

```bash
# Always activate venv first
source .venv/bin/activate

# 0. Capacity study (read-only) — validate the threshold before spending credits
python scripts/construct.py cluster-report

# 1. Pre-compute cluster manifests
python scripts/construct.py cluster

# 2. Check current progress
python scripts/construct.py status

# 3. Prepare 10 bundles; auto-selects underrepresented provider
python scripts/construct.py prepare 10 --provider auto

# 4. After pasting a bundle and getting a tagged response, ingest it
python scripts/construct.py ingest --prompt-index 0 --file /tmp/response.txt

# 5. Validate saved examples
python scripts/construct.py validate

# 6. Filter + build when complete
python scripts/construct.py filter
python scripts/construct.py build
```

## The copywriting format

| What it teaches | Source ads per bundle | Style |
|---|---|---|
| Deliver a real high-performing ad as an assistant response, driven by a purely factual user brief | 1 ad (composite_score ≥ 0.70, total copy ≥ 60 chars) | backtranslation only |

The current target in `configs/construction.yaml` is 3000 post-filter
— refine it with `cluster-report` based on fingerprint-unique capacity.

**Why backtranslation only:** the task is *producing* copy, not
*reasoning about* it. The real ad is the fixed point; the teacher
reverse-engineers a plausible brief. Data-grounded / context-distilled /
natural styles do not apply and `CopywritingConstructor.build_prompt`
raises if one is passed.

## CLI Commands

### `cluster-report` — Capacity study (read-only pre-flight)

```bash
python scripts/construct.py cluster-report
```

Simulates strict-pass bundle emission for copywriting using the current
`configs/construction.yaml` thresholds. Reports unique-ad pool,
fingerprint-unique bundle capacity, `% of raw target` the strict pool can
satisfy, and platform distribution.

Run this **before** `cluster`. If copywriting reports < 100 % of its raw
target, widen `score_min` or lower `copywriting_min_copy_chars`.

### `cluster` — Pre-compute manifests

```bash
python scripts/construct.py cluster
```

Writes three manifests to `data/constructed/_clusters/`:

- `advertiser_clusters.jsonl` — grouping dimension (min_size = 3)
- `vertical_clusters.jsonl` — grouping dimension (min_size = 30)
- `copywriting_ads.jsonl` — copywriting source (ad IDs + scores)

### `status` — Check progress

```bash
python scripts/construct.py status
```

Shows generated / target / remaining, cluster artifact counts, and
provider mix with drift from target.

### `prepare` — Get bundles for a batch

```bash
python scripts/construct.py prepare [batch_size] [options]
```

- `batch_size` — number of bundles to prepare (default 10)
- `--provider` — `claude` / `gpt` / `gemini` / `auto` (default auto)
- `--personas-path` — override persona library path

Each bundle is emitted as the self-contained prompt string: the
BACKTRANSLATION style rules, the ad-derived context directive, the real
ad marked as the gold target, and the required output tags. A
`_last_prepared.json` sidecar captures per-prompt-index metadata for
`ingest`.

### `ingest` — Save a chat-agent response

```bash
# From file
python scripts/construct.py ingest --prompt-index N --file /path/to/response.txt

# From stdin (paste mode)
python scripts/construct.py ingest --prompt-index N
# Paste response then Ctrl-D
```

Loads the sidecar, parses the tagged output, runs the backtranslation
fidelity check (word coverage + verbatim signature), and writes a
`TrainingExample`.

### `validate` — Check examples

```bash
python scripts/construct.py validate
```

### `batch-submit` / `batch-list` / `batch-collect` / `batch-cancel` — API batch mode

Uses each provider's native Batch API for ~50 % cheaper generation with a
24 h SLA.

```bash
python scripts/construct.py batch-submit 50 --model claude-haiku-4-5-20251001
python scripts/construct.py batch-list
python scripts/construct.py batch-collect
python scripts/construct.py batch-cancel <batch_id>
```

### `filter` — Quality filtering

```bash
python scripts/construct.py filter
```

Runs the sequential filter stages against `data/constructed/copywriting/
examples.jsonl`.

### `build` — Assemble final dataset

```bash
python scripts/construct.py build
```

Packs copywriting examples into a HuggingFace `DatasetDict` with
stratified 85 / 7.5 / 7.5 train / val / test splits. Saves to
`data/final/`.

## Quality Control

The `filter` command runs sequential stages. Research basis: AlpaGasus,
Superfiltering, GRAPE — full LLM-as-judge has weak ROI when teacher
output is already above the quality floor.

1. **Structural validation** — correct message format.
2. **Minimum length** — assistant response ≥ 80 chars (copywriting's
   floor; the global default of 200 chars is too tight for
   backtranslation responses).
3. **Language detection** — English only.
4. **Rubric** — copywriting's rubric is empty by design; structural
   fidelity lives in the ingestion word-coverage / verbatim-signature
   check.
5. **Format-specific** — schema-leak + ad-centrality guards on
   backtranslation responses (the teacher must deliver the ad as the
   assistant reply, not describe it).
6. **Response-text dedup** — TF-IDF cosine > 0.80.
7. **Prompt-text dedup** — TF-IDF cosine > 0.85.
8. **Source-ad-set dedup** — reject examples sharing source-ad set with
   an earlier-accepted one.

## Generating Responses (For Chat Agents)

When a chat agent receives a bundle, it must:

1. **Write the user prompt as purely factual product information** —
   zero creative knowledge, no tone guidance, no audience framing, no
   phrasing copied from the ad. A product owner describing their own
   product.
2. **Deliver the real ad verbatim in the assistant response** — wrap it
   naturally: a short lead-in, the ad exactly as-is, and a 2–4
   paragraph rationale grounded in visible details of the execution. No
   alternatives.
3. **Preserve source copy exactly** — emojis, punctuation, and line
   breaks are load-bearing.
4. **Return the required tags only** — no text before, between, or
   after `<user_prompt>` / `<assistant_response>`.

## File Locations

| Path | Contents |
|------|----------|
| `data/scored/v3/scored_ads.jsonl` | ~40K scored ads (v3 hybrid KM-survival scorer) |
| `data/constructed/_clusters/` | Pre-computed cluster manifests (3 JSONL files) |
| `data/constructed/copywriting/examples.jsonl` | Generated training examples with provenance |
| `data/constructed/copywriting/_last_prepared.json` | Sidecar with per-prompt-index metadata |
| `data/constructed/copywriting/filtered.jsonl` | Post-quality-filter examples |
| `data/final/` | Final HuggingFace Dataset (after `build`) |
| `configs/construction.yaml` | Target, score band, clustering params, provider rotation, filter thresholds |
| `configs/personas.yaml` | Marketing personas (kept for metadata parity) |

## Configuration

Key knobs in `configs/construction.yaml`:

```yaml
construction:
  scored_ads_path: data/scored/v3/scored_ads.jsonl
  overgeneration_buffer: 1.25

  formats:
    copywriting:
      target: 3000
      score_min: 0.70
      valid_styles: [backtranslation]
      style_ratios:
        backtranslation: 1.00

  clustering:
    min_vertical_cluster: 30
    max_per_vertical: 200     # caps head verticals to prevent training-mix skew
    max_per_advertiser: 50
    min_advertiser_cluster: 3
    format:
      copywriting_min_copy_chars: 60

  provider_rotation:
    claude_ratio: 0.40
    gpt_ratio:    0.35
    gemini_ratio: 0.25

  quality_filter:
    min_response_length: 200
    dedup_threshold: 0.80
    prompt_dedup_threshold: 0.85
    cross_format_source_dedup: true
```

Config loading validates:
- `style_ratios` sums to 1.0.
- Every `style_ratios` key is in `valid_styles` (and vice versa — zero-ratio entries can't appear in `valid_styles`).
- Provider ratios sum to 1.0.

Operators use `ConstructionConfig.raw_target_for(TaskFormat.COPYWRITING)`
to know how many bundles to prepare — this returns `target *
overgeneration_buffer`.

## Resume / Picking Up Where You Left Off

Everything is checkpoint-based. Run `status` to see where things stand,
then `prepare` the remaining work. The system automatically:

- Knows how many examples exist (via `.checkpoint.json` sidecars).
- Tracks which ad IDs have been used (via `source_ad_ids` in saved
  examples).
- Tracks bundle fingerprints already emitted to prevent re-emitting
  identical bundles.
- Advances the RNG seed with `generated_count` so dice rolls stay fresh
  across resumes.
- Appends to existing files (never overwrites).

## Why this architecture

Narrow-domain SFT favours quality over breadth (**LIMA** NeurIPS'23,
**AlpaGasus** ICLR'24, **Cherry-LLM** NAACL'24, **Superfiltering**
ACL'24). Backtranslation (**Humpback**, Li et al. ICLR'24) pins the
real ad as the fixed point so teacher invention is bounded — the
teacher cannot fabricate copy; it can only write a plausible brief that
would have produced the ad. Ad-derived context (platform + populated
copy fields) replaces RNG persona / seed / evol-operator rolls that the
student never sees at inference anyway.
