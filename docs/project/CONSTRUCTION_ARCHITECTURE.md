# Construction Pipeline Architecture

Design document for Draper.ai's copywriting training-data construction pipeline.
Complements `CONSTRUCTION_GUIDE.md`: the guide tells you *how to use* the
pipeline; this doc explains *how it's built* and *why*.

> **History — 2026-04 pivot:** Draper.ai previously shipped five pattern-skill
> formats (positioning, copywriting, diagnostic, optimization,
> channel-format-fit). The pivot narrowed the training target to creative
> copywriting only. The retired format packages, their docs, and related
> dead machinery are preserved under `archive/`; this document covers the
> active pipeline only. The 5-format architecture document is snapshotted
> at `archive/docs/CONSTRUCTION_ARCHITECTURE_5format.md`.

## Goal

Turn ~40K scored ads into high-quality instruction-tuning examples for
creative copywriting. The resulting fine-tune is:

- Grounded in real ad-performance data (not generic marketing advice
  distilled from the teacher's pretraining).
- Structurally faithful — the real ad is the fixed point; the teacher
  reverse-engineers a plausible brief (Humpback / Li et al., ICLR'24).
- Robust to product/brief variety via ad-derived conditioning rather than
  RNG-rolled personas and seeds.

## Design principles

1. **Chat-first, API-optional.** Primary workflow uses a chat subscription
   (Claude / GPT / Gemini). API-batch mode is a secondary speedup path
   sharing the same bundle format.
2. **Everything in one chat-agent pass.** The teacher receives the full
   context (ad-derived directive, the real ad, style rules, output format)
   and produces the complete training example in a single turn.
3. **Deterministic Python for dice; LLM for semantics.** All local state
   advances under seeded RNG. The teacher only writes the brief and the
   response.
4. **Continuous score bands, not discrete tiers.** Copywriting's filter
   uses `composite_score ≥ threshold`, not the 3-tier label.
5. **Backtranslation is structural, not stylistic.** The real ad is the
   assistant response; the teacher reverse-engineers a purely factual
   brief. Other styles (data-grounded, context-distilled, natural) do not
   apply.
6. **Ad-derived context, not RNG axes.** Voice, shape, and depth come from
   the source ad itself (platform + populated copy fields), not persona /
   seed / evol-operator rolls.
7. **Research-backed.** Key architectural choices anchored to 2023–2026
   papers (cited below).

## End-to-end data flow

```
data/scored/v3/scored_ads.jsonl   (~40K ads; continuous composite_score + tier label)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  cluster-report  (read-only capacity study — optional pre-flight)    │
│      Simulates strict-pass bundle capacity at the proposed            │
│      copywriting threshold. Writes nothing; informs threshold tuning  │
│      before you spend teacher credits.                                │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  cluster  (AdClusterer.compute_and_save)                             │
│      Emits:                                                          │
│        advertiser_clusters.jsonl    (grouping dim, kept for reuse)   │
│        vertical_clusters.jsonl      (grouping dim, min_size=30)      │
│        copywriting_ads.jsonl        ← Copywriting source manifest    │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  SourceSelector.select_batches(COPYWRITING, consumed_ids, count)     │
│      Structural-cleanliness gate → consumed-id / fingerprint dedup.  │
│      Returns list[list[ScoredAd]] — one ad per bundle.               │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  prepare command (chat mode)   OR    batch-submit (API mode)         │
│      For each single-ad bundle:                                      │
│        - Derive copywriting context (source_ad_shape) from the ad — │
│          no RNG, platform-agnostic (see 2026-04 audit).              │
│        - Build the teacher bundle (style rules + the ad).            │
│        - Chat: emit to console + _last_prepared.json sidecar         │
│        - Batch: submit to provider's Batch API + pending registry    │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
teacher response (tagged: <user_prompt>, <assistant_response>)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ingest / batch-collect                                              │
│      parse_bundle_output → pipeline.ingestion_check                  │
│      (word-coverage + verbatim-signature fidelity) →                 │
│      TrainingExample →                                               │
│      data/constructed/copywriting/examples.jsonl                     │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  filter command — sequential stages                                  │
│      Structural / min-length / language / rubric / format-specific   │
│      (schema-leak + ad-centrality for backtranslation) / response    │
│      and prompt dedup / source-ad-set dedup.                         │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  build command                                                       │
│      Stratified 85/7.5/7.5 split → HuggingFace DatasetDict →         │
│      data/final/                                                     │
└──────────────────────────────────────────────────────────────────────┘
```

## The copywriting format

| Skill | Input | Output | Source manifest |
|---|---|---|---|
| Tactical copy production (backtranslation) | One high-score reference ad | A factual brief + the ad delivered as the assistant response, wrapped in a lead-in and a short rationale | `copywriting_ads.jsonl` |

### Filter threshold

| Score band | Extra filter |
|---|---|
| `composite_score ≥ 0.70` | total copy (headline + body + description + cta) ≥ 60 chars |

The tier label is retained in the data as a readable summary for teacher
prompts but is never a filter input.

### Backtranslation, not styles

Copywriting runs backtranslation-only (Humpback, Li et al., ICLR'24). The
task is *producing* copy, not *reasoning about* it: the real ad is the
fixed point and the teacher reverse-engineers a plausible brief. Earlier
iterations that layered data-grounded / context-distilled / natural
styles over copywriting were retired — the ad itself already carries the
structural signal that makes a copywriting example teach. The code path
is locked down: `CopywritingConstructor.build_prompt` raises if a
non-`BACKTRANSLATION` style is passed.

## Ad-derived context

Copywriting skips persona / seed / evol-operator / difficulty RNG. Voice,
shape, and rationale depth come from the source ad itself. The single
deterministic axis is:

| Axis | Source | Purpose |
|---|---|---|
| **source_ad_shape** | Populated copy fields (`headline` / `body` / `description` / `cta`) | Provenance metadata — `has_body` vs `headline_only`. Not rendered into the teacher prompt (the teacher reads the ad directly); kept on `ExampleMetadata` for downstream filtering / analytics. |

Lives in `formats/copywriting/dice.py` (`derive_copywriting_context`).
`CopywritingPipeline.roll_bundle_axes` returns a `BundleAxes` whose
`copywriting_context` field carries the derived axis; `render_axes_block`
currently emits nothing because there is no teacher-visible directive to
render. It is kept as a seam so a future conditioning axis can be added
without touching `bundle.py`.

The prior `platform_framing` axis was removed after the 2026-04 audit:
the corpus's `platform` is a scraping-source artifact (which AdFlex
endpoint returned the ad — `adflex.py:_parse_ad` takes `platform` as a
caller argument, not a field on the ad), not an intrinsic creative
attribute. Same advertisers appear under multiple platform tags. Teaching
the student to produce "platform-native" copy off a label that encodes
endpoint routing trained on label noise. Platform-specific formatting is
handled at inference via prompt.

Scenario / rationale-depth / brief-register / brief-ask / brief-tension
axes — all previously rolled here — were removed because the student
never sees those rolls at inference and each axis was a fresh surface
for teacher invention.

## Bundle structure

`bundle.build_bundle(BundleContext)` assembles a self-contained prompt
block:

```
# Training-example generation — task: copywriting

<preamble explaining the task>

## Style rules                ← backtranslation rules
                              ← pipeline.render_axes_block(ctx):
                                 currently empty; seam for a future
                                 conditioning axis
## Source ad                  ← the gold target (real ad, preserved verbatim)
## Output format              ← <user_prompt>, <assistant_response>
```

Multi-turn follow-ups are structurally incompatible with backtranslation
(a turn-2 assistant response can't preserve the real ad verbatim) and were
removed from the bundle builder.

## TrainingExample provenance

Every saved example carries provenance in `ExampleMetadata`:

```python
class ExampleMetadata(BaseModel):
    source_ad_ids: list[str]
    source_tiers: list[str]
    source_scores: list[float]
    platform: str
    vertical: str
    construction_model: str             # Declared provider
    prompt_style: PromptStyle           # Always BACKTRANSLATION for active data
    persona_id: str                     # Sampled but not rendered — kept for schema parity
    seed_idx: int                       # Always -1
    evol_op: str                        # Always ""
    difficulty: str                     # Always "standard"
    turn_structure: str                 # Always "single"
    followup_type: str                  # Always ""
    source_ad_shape: str                # Derived from ad
    construction_timestamp: str
```

Fields that never vary for copywriting are retained so old JSONL rows
keep parsing and a future format could reuse them without a migration.

## Module map

| Module | Role |
|---|---|
| `construction/schemas.py` | Pydantic models — `TrainingExample`, `ExampleMetadata`, `PromptStyle`, `TaskFormat` (single value), `FormatConfig`, `ClusteringConfig`, `ConstructionConfig` |
| `construction/personas.py` | `Persona`, `PersonaLibrary` — loads `configs/personas.yaml`. Sampled for metadata parity, not rendered. |
| `construction/difficulty.py` | Difficulty dice + directives. Copywriting is sparse-disallowed so sampling always lands on `standard`. |
| `construction/provider_rotation.py` | Provider classifier + deficit-based suggestion |
| `construction/bundle.py` | `BundleContext` + `build_bundle` + `parse_bundle_output` (single-turn output tags only) |
| `construction/base_constructor.py` | Abstract constructor interface; `assign_styles` (collapses to single valid style for copywriting) |
| `construction/formats/base.py` | `FormatPipeline` ABC — selector / persona / rubric / ingestion / quality / render hooks every format overrides |
| `construction/formats/registry.py` | `get_pipeline(task_format)` dispatch |
| `construction/formats/copywriting/` | Backtranslation mode — owns `constructor`, `selector` (structural-cleanliness filter), `dice` (derived `source_ad_shape`, no RNG, platform-agnostic), `ingestion` (word-coverage + verbatim signature), `quality_filter` (schema-leak + ad-centrality), `rubric` (empty by design), `pipeline` (glue; overrides `roll_bundle_axes` + `render_axes_block`) |
| `construction/source_selector.py` | Shared ad-lookup + dedup primitives. `SourceSelector.select_batches(task_format, ...)` dispatches through `get_pipeline(task_format).select_batches(...)` |
| `construction/clusterer.py` | Score-threshold clustering; emits advertiser / vertical / copywriting manifests |
| `construction/cluster_report.py` | Read-only capacity simulator |
| `construction/quality_filter.py` | Sequential filter stages |
| `construction/dataset_builder.py` | Stratified split + HuggingFace assembly |
| `scripts/construct.py` | CLI: `status`, `cluster`, `cluster-report`, `prepare`, `ingest`, `validate`, `batch-*`, `filter`, `build` |

## Quality filter

Deliberately cheap. No LLM-as-judge calls at filter time.

1. **Structural** — correct message roles + order.
2. **Min length** — floor from `FormatPipeline.min_length_floor` (copywriting drops the default 200-char floor to 80 for backtranslation responses, which are naturally tight).
3. **Language** — English only (`langdetect`).
4. **Rubric** — `FormatPipeline.rubric_check`. Copywriting's rubric is empty by design; structural fidelity lives in the ingestion check.
5. **Format-specific** — `FormatPipeline.extra_quality_filters`. Copywriting adds schema-leak + ad-centrality guards for backtranslation responses.
6. **Response-text dedup** — TF-IDF cosine > 0.80.
7. **Prompt-text dedup** — TF-IDF cosine > 0.85 on user prompts.
8. **Source-ad-set dedup** — reject any example sharing source-ad set with an earlier-accepted one.

## Adding a format

The registry dispatch pattern is still in place, so a future format can
drop in under `src/draper/construction/formats/<name>/` and register a
subclass of `FormatPipeline` from the package `__init__`. Copywriting is
the canonical template. Shared modules pick up the override automatically
— no changes to `source_selector.py`, `ingestion.py`, `dice.py`, or
`quality_filter.py` are needed.

The archived packages under `archive/construction/formats/` are the prior
implementations of positioning / diagnostic / optimization /
channel-format-fit. They depend on older versions of the shared
orchestrators and will not run against the current single-format code
without porting.

## Capacity study (`cluster-report`)

Read-only pre-flight check. Given the current copywriting threshold and
raw-generation target (`target × overgeneration_buffer`), the report
shows:

- Unique-ad pool under the filter
- Fingerprint-unique bundle capacity under the strict pass
- `% of raw target` the strict pool can satisfy
- Platform distribution of eligible ads

Run it before paying teacher credits. If copywriting's capacity is < 100 %
of its raw target, widen the threshold or drop the total-copy floor in
`configs/construction.yaml` before running `cluster`.

## Resumability

File-backed and checkpoint-driven:

- `data/constructed/copywriting/examples.jsonl.checkpoint.json` —
  `{"generated_count": N}` advanced on every save.
- `data/constructed/copywriting/_last_prepared.json` — per-prompt-index
  sidecar with derived-context fields for ingest reconciliation.
- `consumed_ad_ids()` / `consumed_bundle_fingerprints()` on the
  copywriting constructor — read existing examples to skip reuse.
- RNG seed advances with `generated_count` (used only where RNG still
  participates — currently persona sampling for metadata parity).

## Research foundations

| Decision | Source |
|---|---|
| Continuous score over discrete tiers | **URIAL** (Lin et al., ICLR'24) — SFT surfaces capabilities conditioned on signals the base model can't see; discretising a continuous engagement score throws away resolution the teacher already uses |
| Narrow-domain SFT with one format | **LIMA** (Zhou et al., NeurIPS'23), **AlpaGasus**, **Cherry-LLM** (NAACL'24), **Superfiltering** (ACL'24) — narrow-domain SFT favours quality over format count |
| Cut landscape-Q&A style | **Gekhman et al.** (EMNLP'24, arXiv:2405.05904) — SFT on unseen content slowly learns it AND raises hallucination on unrelated prompts; content-currency is RAG's job at inference, not SFT's |
| Copywriting backtranslation (ad-derived context, no RNG persona/scenario axes) | **Humpback** (Li et al., ICLR'24) — reverse-engineer a brief from a real ad so the real ad is the fixed point; **LIMA** (NeurIPS'23) — curated examples beat scraped orthogonal grids |
| Avoid template overfit | **Sclar et al.** (arXiv:2310.11324) — up to 76pp performance swing from prompt format alone |
| Skip full LLM-as-judge | **AlpaGasus** (arXiv:2307.08701); **GRAPE** (arXiv:2502.04194, 2026) — naive judge filtering backfires on smaller students |
| Multi-provider rotation | **"Synthetic Eggs in Many Baskets"** (arXiv:2511.01490, 2025) |

## API batch mode

`batch-submit` / `batch-list` / `batch-collect` / `batch-cancel` share
dice + ingestion code with the chat path:

- `construction/dice.py::prepare_bundles` — rolls dice + builds `BundleContext`
- `construction/ingestion.py::ingest_response` — parses tagged output and saves a `TrainingExample`
- `construction/batch/` — provider-agnostic Protocol + OpenAI / Anthropic clients + registry

Adding a new provider is a drop-in: implement `BatchClient`, register in
`factory.py`.

## Known limitations and future work

- **Raw target is a placeholder.** Lock it by running `cluster-report` —
  adjust threshold + target in `configs/construction.yaml` before shipping
  teacher credits.
- **TF-IDF dedup is a proxy for semantic dedup.** Sentence-embedding dedup
  (e.g., `bge-small`) would catch more semantic near-duplicates but adds a
  torch/transformers dependency.
- **Rubric is empty by design.** Fidelity lives in the ingestion
  word-coverage / verbatim-signature check rather than required-section
  keyword matching.
- **DPO stage parked for v1.5.** Use the engagement-derived score as a
  preference signal after SFT trains and evaluates.

## Testing

Construction-pipeline coverage:

| Suite | Covers |
|---|---|
| `test_bundle.py` | Bundle builder output, tag parser round-trip |
| `test_difficulty.py` | Distribution ratios, directives per tier, copywriting sparse-disallow behaviour |
| `test_rubrics.py` | Copywriting rubric (empty by design) + pipeline registration |
| `test_provider_rotation.py` | Provider classification, deficit-based suggestion, ratio validation |
| `test_quality_filter.py` | Structural / min-length / dedup / extra-filter stages |
| `test_construction_schemas.py` | Pydantic round-trip, config loading, single-format enforcement |
| `test_dataset_builder.py` | Stratified 85/7.5/7.5 split assembly |
| `test_clusterer.py` | Score-threshold clustering, min_size floors, copywriting copy-length floor |
| `test_source_selector.py` | Copywriting strict dedup, consumed-ID guarantees, single-ad bundles |
| `test_dice_and_ingestion.py` | Shared prepare/ingest helpers; backtranslation fidelity on a matching response |
| `test_copywriting_dice.py` | Ad-derived context derivation |
