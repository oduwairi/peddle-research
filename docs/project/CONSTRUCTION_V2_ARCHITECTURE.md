# Construction Pipeline v2 — Architecture & Implementation Plan

> Status: design approved 2026-05-18; two-slot assistant turn locked 2026-05-19; **unified single-pass teacher deployed 2026-05-21** • Owner: o.duwairi • Companion to `CONSTRUCTION_ARCHITECTURE.md` (v1, kept for reference)

## 0. Guiding principle: Draper is a standalone copywriter, not a tool fragment

The v2 fine-tune is designed as a **complete, deployable artifact** that can stand alone behind any caller — the current freeform agent, a future agent, a notebook, a direct API curl. The model's output must read as a competent copywriter's response on its own, not as a fragment that only makes sense after an orchestrator wraps it.

Practical consequence: the assistant turn has one structural anchor — `<think>...</think>` — followed by a freeform deliverable region (see §3.2). The deliverable region carries the ad verbatim and may wrap it with whatever short framing prose fits the brief's `task` and the ad's natural register (a lead-in line, a closing line, or neither). It does *not* emit raw copy plus the assumption that an orchestrator will dress it up. If today's agent strips the framing prose, that's a rendering decision the agent owns — the model still produces the full response.

This decoupling buys: (a) portability across deployment surfaces, (b) honest evaluation (judges read one response, not "what the agent shows after stripping"), and (c) robustness against future agent rewrites — the durable artifact is the checkpoint, not the integration glue.

## 1. Motivation

The v1 backtranslation pipeline produced a model that underperforms in production:

| Symptom | Source |
|---|---|
| ~5% inference returns `<EXTRACTION_FAILED>` (model collapse / non-ad output) | `docs/research/SCORING_PREDICTOR_PHASE2_RESULTS_2026-05.md` |
| Agent loop *degrades* Draper's score (-0.040 vs direct +0.036) | `docs/research/RQ2_OFFLINE_2x2_RESULTS_2026-05.md` |
| Flat `engagement_velocity` head — model learns durable copy, not trending phrasing | `SCORING_PREDICTOR_PHASE2_RESULTS_2026-05.md` |
| Teacher paraphrases / writes rationale-only / ignores brief during construction | `scripts/explore/diagnose_claude_rejects.py` |
| Base Qwen3-8B-Instruct lacks native thinking; ChatML template strips `<think>` blocks | `configs/training.yaml`, `src/draper/construction/quality_filter.py` |

The **root cause** (user diagnosis): in v1 training data, the brief→ad jump is too sudden. The brief contains generic product info; the ad lands as a non-sequitur — there is no strategic signal in the input that mathematically constrains the response space toward the source ad. The model learns *"ads come out of the blue"* rather than *"this brief naturally produces this ad."*

Secondary problems that v2 also fixes:

1. v1's brief is free-text prose at inference, but the frontend constructs it from a structured object first — a serialization step that adds nothing and creates skew if `renderProductFacts` drifts from training-time formatting.
2. v1's variety knobs were either content-dependent (persona, scenario — pushed off-distribution when forced) or vestigial (`source_ad_shape`, `conversation_register` collapse to ad metadata, not real diversity). **v2 abandons synthetic variety knobs entirely** — variety comes from the natural span of the source corpus.
3. v1 has no thinking channel — quality filter explicitly strips `<thinking>` blocks. The model never learns to verbalize why a particular ad answers a particular brief.

v2 is a clean-sheet rebuild. v1 code is preserved under `archive/construction-v1/` for reference and rollback; v1 model + endpoint stay live for 24h after v2 cutover.

## 2. v1 → v2 diff at a glance

| Dimension | v1 | v2 |
|---|---|---|
| User-turn shape | Natural-language prose flattened from a structured object | Canonical JSON, `{product, bridge, platform}` |
| Brief content | Product facts only (cardinal rule: "Draper sees product-only briefs") | Product facts **+ strategic bridge fields** (positioning, audience, angle, buyer pain) — derived from the ad, never quoting it |
| Assistant turn | Source ad, verbatim | `<think>` (first-person internal reasoning) + freeform deliverable region carrying the ad verbatim, optionally wrapped with short framing prose |
| Variety mechanism | Two deterministic-from-ad axes (`source_ad_shape`, `conversation_register`) — no real RNG | None. Variety is inherited from the natural diversity of the source corpus (different ads → different natural voices, lengths, depths). |
| System prompt | Dynamic, role-dependent | Static — same string at training and inference |
| Fidelity contract | Word coverage ≥60% + 6-word verbatim signature over the entire assistant response | Same check, but scoped to the **ad portion only** (think trace excluded) |
| Teacher calls per ad | 1 (fused brief + rationale + ad emission) | 1 (single-pass: `<brief>` + `<think>` + deliverable in one call) |
| Examples per ad | 1 | 1 (no rolls; teacher reads the ad and shapes the rationale to match) |
| Base model | `unsloth/Qwen3-8B-unsloth-bnb-4bit` (Instruct, no native thinking) | TBD — Phase 0 bake-off across thinking-native candidates |
| Frontend prompt | `renderProductFacts(product) -> prose` | `serializeBriefForDraper(brief, platform) -> canonical JSON` |
| Cardinal rule | "Draper sees product-only briefs" | "Draper sees product facts + positioning/audience/angle/pain bridge fields as structured JSON; never the ad's copy itself" |

## 3. Architecture

### 3.1 Brief schema (the user turn)

```python
class BriefProduct(BaseModel):
    name: str                        # required — product/service name
    description: str                 # required — one-paragraph what it is and does
    category: str | None             # broad market category
    key_features: list[str]          # concrete features from the product page
    unique_selling_points: list[str] # differentiators / positioning claims
    price_info: str | None           # pricing summary if available
    tone_signals: list[str]          # voice cues from the OWNER's framing, not from the ad
    # New in v2:
    category_context: str | None     # one-line landscape framing
    proof_points: list[str]          # testimonials / numbers / named entities visible on the page
    offer: str | None                # promo / free trial / discount
    platform_hint: str | None        # which surface the ad will run on (carried for routing, not coaching)

class BriefBridge(BaseModel):
    """Strategic facts derived from the source ad. Never quote the ad's copy."""
    positioning: str        # one-line strategic frame ("premium alternative to spreadsheet workflows")
    target_audience: str    # specific buyer persona ("Series-A marketing leads three months in")
    angle: str              # creative direction label ("problem-aware skeptic", "aspirational founder identity")
    buyer_pain: str         # concrete friction being addressed ("Sundays lost to payroll spreadsheets")

class Brief(BaseModel):
    task: str               # free-form natural-language request — e.g. "Write a Reddit
                            #   post warning trailer haulers about tongue-weight failures."
                            #   No enum, no slug; phrased the way a real caller would type
                            #   it in chat. The static system prompt is skill-agnostic, so
                            #   the student learns task-string → output-shape from data.
    product: BriefProduct
    bridge: BriefBridge
    platform: str           # "meta" | "tiktok" | "x" | "google" | "pinterest" | "reddit"
```

**Why the bridge fields are the load-bearing change.** Without them, `product.description + product.key_features` alone leave the response space enormous — any ad for the product is plausible. With them, the brief specifies *the strategic frame* of the answer (problem-led vs benefit-led, who it's for, what pain it speaks to) — which mathematically corresponds to a much narrower band of plausible ad copy. The source ad falls inside that band by construction, so brief → ad becomes a learnable mapping instead of a sample from a near-uniform prior.

**Bridge-field discipline (enforced at construction time):**
- Bridge fields are facts ABOUT marketing intent, not the ad's copy.
- The brief-extraction teacher prompt forbids 5-grams (or longer) from the ad to appear in any bridge field. Ingestion rejects briefs that leak.
- Bridge labels are short — `angle` is a *label* like "aspirational founder identity," not a hook line; `buyer_pain` is a *description* like "Sundays lost to payroll spreadsheets," not a tagline.

### 3.2 Response schema (the assistant turn)

```
<think>
{first-person, present-tense internal reasoning. The way a copywriter
 actually thinks at the desk — "I want to lead with consequence, not
 features… no, that hook's too sales-y, let me try…". Names product
 facts and bridge fields, weighs tradeoffs, sometimes discards options.
 Hidden by UI convention.}
</think>

{freeform deliverable region. For the ad-copy skill, this carries the
 source ad character-for-character verbatim, optionally framed with a
 short lead-in line and/or closing line if the brief's task and the
 ad's natural register call for one. Peer-to-peer voice; no greetings,
 no apologies, no offers to revise.}
```

**Why two slots and not three.** `<think>` is the only universal structural anchor we lock across all future Draper skills. The deliverable region is freeform on purpose — it lets the model wrap the artifact the way a competent practitioner would given the task. Forcing a third tag (we tried `<note>` in early smokes) produced two failure modes: notes collapsed into chatbot filler ("Hope this helps!", "Let me know if you want revisions!") that contaminated the artifact, and the rigid three-slot shape pushed the model to always emit a note even when the register didn't warrant one.

- **`<think>` is internal.** Written for the model itself, hidden from the user by UI convention. First-person, decisional voice. If this block were written as third-person analytical prose ("The ad leverages fear-based education…"), it would be a portfolio case study masquerading as reasoning — the category confusion that made the early smoke unreadable.
- **The deliverable carries the artifact.** For the ad-copy skill, that's the source ad reproduced verbatim. Any framing prose around the ad is the model's call based on the task and register — short and natural, or absent entirely.

**Why thinking is required.** The disconnect in v1 — *"answers come out of the blue"* — is fundamentally a missing-explanation problem. Even with rich bridge fields in the brief, the student needs to learn *the mapping function* from (product, bridge) to ad. The `<think>` trace is that mapping made explicit: it walks through which product fact justifies the hook, which bridge field justifies the structure, which audience cue justifies the lexical register. With assistant-only loss, both `<think>` and the deliverable are part of the unmasked target — the model is trained to verbalize its decisions internally AND to produce the artifact, which generalizes better than learning the surface mapping alone.

**Fidelity contract.** The deliverable region must contain the source ad verbatim — ≥60% word coverage AND at least one 6-word contiguous phrase matching the source. The check runs only on the deliverable region (never on the think trace, which would trivially satisfy coverage by naming features).

**Platform-label contract.** For mapped platforms (Meta, TikTok, X, Pinterest, Reddit), the deliverable must carry every bold Title-Case label that corresponds to a populated source-ad field (e.g., `**Primary text:**` for Meta's `headline` slot). `check_platform_labels_present` enforces this at ingest; field labels must not appear inside `<think>`.

### 3.3 Ad-driven response shape (replacing dice)

There is no dice grid, no recipe sampler, no per-example RNG knob. Instead, the **teacher infers the response shape directly from the source ad's natural voice and complexity**. A cupcake DTC ad → playful tone, short rationale, light reasoning depth. A B2B compliance ad → neutral tone, longer rationale, deeper reasoning. A meme-driven TikTok hook → terse rationale matching the brevity; an explainer Reddit post → walked-through reasoning matching the length.

The rationale-generation teacher prompt instructs:

> Read the ad. Notice its natural voice (formal / neutral / playful), length, and the depth of marketing thinking behind it. Write the `<think>` section in first-person, present-tense decisional voice — the way a copywriter actually thinks while drafting — and MATCH that natural shape (same voice, proportional length, reasoning depth that fits the ad's strategic complexity). Then produce the deliverable: the source ad reproduced character-for-character verbatim, with optional short framing prose around it if the task and register call for one.

**Why this is better than dice.**

- **No artificial pairings.** Dice could roll "formal" on a goofy cupcake ad — an unnatural pairing the model would have to memorize as legitimate. With ad-driven shape, every (rationale, ad) pairing is internally consistent.
- **Variety still exists.** It comes from the ~25k natural source-ad voices in the corpus, not from a synthetic 81-cell grid.
- **One example per ad.** No N rolls. 1 ad = 1 brief = 1 example. Simpler, cheaper, lower train/inference skew.
- **Static system prompt.** No renderer, no contract test for byte-equal Python/TS prompt strings.
- **At inference, the model infers shape from the brief.** The brief carries `tone_signals`, `category`, `platform_hint`, and `angle` — together these triangulate the right voice/length/depth the same way the teacher did from the ad during construction. This is why stage 1's `tone_signals` field is now non-optional: it's the inference-time style anchor.

### 3.4 Unified single-pass teacher

The production pipeline uses **one batch call per source ad**. The single-pass teacher emits three regions in a single response:

1. `<brief>...</brief>` — a JSON object with `task`, `product` (including `tone_signals`), `bridge`, and `platform`. Parsed and cached to `single_pass.briefs_cache_path` (default `data/constructed_v2/copywriting/briefs.jsonl`).
2. `<think>...</think>` — first-person internal reasoning grounded in the brief the teacher just emitted (not in the source ad).
3. Freeform deliverable — the source ad reproduced character-for-character verbatim, optionally wrapped with short framing prose.

The **user message** presented to the teacher embeds the source ad via `render_labeled_ad(ad)` — the ad's copy laid out with platform-native field labels in bold Title Case (e.g., `**Primary text:**`, `**Headline:**` for Meta; `**Tweet:**`, `**CTA:**` for X). The teacher reproduces these labels verbatim in its deliverable, training the student to emit the same structure that the frontend's `emit_campaign` parser expects. For OTHER-group platforms and unmapped `(source, group)` pairs, `render_labeled_ad` falls back to the unlabeled `ad_copy_text` blob.

`collect_batch` splits this into two files so the existing `ingest_responses` loop runs unchanged: `briefs.jsonl` carries `{ad_id, brief}` rows; `responses_raw.jsonl` carries the reassembled `<think>` + deliverable string.

**Provider model bindings** are configured under `providers` in `configs/construction_v2.yaml`. `submit` resolves the model from `config.providers[provider]`; `--model` overrides it. `validate_batch_model` (`src/draper/construction/batch/factory.py`) enforces a provider denylist before any network round-trip.

**Slice partitioning.** `submit --slice i/N` takes the `i`-th of `N` disjoint contiguous chunks of the shared `selection.parquet`. The range form `--slice i-j/N` takes the union of chunks `i` through `j` inclusive (useful when a provider can absorb a larger share in a single batch run). This lets multiple providers submit in parallel without overlapping ad_ids. The `selection_lineage_hash` column on `selection.parquet` is verified at submit and collect time; `SelectionLineageMismatch` is raised if the selection was re-run between the two (pass `--allow-lineage-drift` to override).

**Legacy two-stage path.** `BriefExtractionConfig` and `RationaleConfig` survive as optional config fields so two-stage YAMLs still load. Phase 4 will delete this path once the single-pass corpus is confirmed healthy.

## 4. Construction pipeline

Two skills share this pipeline. The pipeline shape is identical; only the
`config.skill` field and the `SkillGateBundle` it resolves differ:

| Skill | Config file | Deliverable | Captioning step | Platform-label gate |
|---|---|---|---|---|
| `copywriting` | `configs/construction_v2.yaml` | Freeform ad copy (verbatim) | No | Yes |
| `image_brief` | `configs/construction_v2_image.yaml` | `<image_brief>` prose ART-DIRECTION BRIEF | Yes — `caption-submit` / `caption-collect` before `submit` | No (skipped — `labels=None`) |

`submit_single_pass`, `collect_batch`, and `ingest_responses` in `pipeline.py` contain zero per-skill branches. All dispatch goes through the `SkillGateBundle` returned by `ingest.skills.get_bundle(config.skill)`.

### 4.0 End-to-end flow (dataset → merged checkpoint)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0 — SOURCE CORPUS                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
            data/scored/v3/scored_ads.parquet           (~55k v3-scored AdFlex ads)
                          │
                          ▼
                ┌──────────────────────────┐
                │ SourceSelector           │            stratified by platform;
                │ construct_v2 select      │            min v3 composite ≥ threshold
                └──────────────────────────┘            (configs/construction_v2[_image].yaml)
                          │
                          ▼
            data/constructed_v2/_audit/selection.parquet     (~25k ad_ids)

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0.5 — VLM CAPTIONING  (image_brief skill only)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                          │   (copywriting skill skips this stage)
                          ▼
                ┌──────────────────────────┐
                │ captions/builder.py      │            filters to image-capable URLs
                │ construct_v2             │            (IMAGE_URL_SUFFIXES); skips
                │  caption-submit          │            already-captioned ads by default
                │  caption-collect         │            (--recaption to overwrite)
                └──────────────────────────┘            uses same BatchRegistry lifecycle
                          │
                          ▼
            data/captions/v1/captions.parquet           (ad_id → caption text)

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — SINGLE-PASS TEACHER  (1 batch call per ad)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────────────┐
                │ SkillGateBundle          │            resolved from config.skill
                │  .prepare_source_ads()   │            image_brief: joins captions
                │  .build_request()        │            onto each SourceAd; drops
                └──────────────────────────┘            ads with no caption
                          │
                          ▼
                ┌──────────────────────────┐
                │ submit_single_pass       │            provider model resolved from
                │  → Batch API (provider)  │            config.providers[provider]
                └──────────────────────────┘            copywriting emits:
                          │                               <brief> JSON + <think> + verbatim ad
                          │                             image_brief emits:
                          │                               <brief> JSON + <think> + <image_brief> PROSE
                          ▼
                ┌──────────────────────────┐
                │ collect_batch            │            splits into two files:
                │  bundle.parse_response() │            briefs.jsonl {ad_id, brief}
                └──────────────────────────┘            responses_raw.jsonl {ad_id, content}
                          │
                          ▼
            data/constructed_v2/<skill>/briefs.jsonl        (parsed <brief> JSON)
            data/constructed_v2/<skill>/responses_raw.jsonl (~25k reassembled responses)

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3 — PARSE + VALIDATE                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────────────┐
                │ bundle.parse_response()  │   reject ─► reasons:
                │ → ParsedResponse(        │              missing_think
                │     think,               │              missing_deliverable
                │     deliverable)         │              teacher_failed
                │                          │              think_too_short
                │  copywriting: expects    │              pre_think_noise
                │    freeform ad copy      │
                │  image_brief: expects    │
                │    <image_brief> prose   │
                │    ART-DIRECTION BRIEF   │
                └──────────────────────────┘
                          │
                          ▼
                ┌──────────────────────────┐
                │ bundle.leak()            │   reject ─► leak:positioning
                │ 5-gram(ad, bridge)       │              leak:angle  …
                │ None → stage skipped     │   image_brief: skipped (leak=None —
                │ (image_brief: skipped)   │   copy is a legitimate verbatim input)
                └──────────────────────────┘
                          │
                          ▼
                ┌──────────────────────────┐
                │ bundle.fidelity()        │   copywriting: word_coverage<60%
                │  skill-specific contract │               no_6gram_signature
                │                          │               (think trace excluded)
                │                          │   image_brief: prose carries ≥30% of
                │                          │               the literal caption content
                │                          │               words (caption-overlap); no JSON
                └──────────────────────────┘
                          │
                          ▼
                ┌──────────────────────────┐
                │ bundle.grounding()       │   copywriting: no_bridge_field_ref
                │  copywriting: ≥1 bridge  │   (product-fact requirement dropped
                │   field ref in <think>   │    2026-05-22 — bridge anchor only)
                │  image_brief: ≥1         │   image_brief: no_brand_guidelines_ref
                │   creative.brand_guide-  │   (a brand_guidelines content word
                │   lines word in <think>  │    must surface in the <think> trace)
                └──────────────────────────┘
                          │
                          ▼
                ┌──────────────────────────┐
                │ bundle.content_bridge()  │   copywriting: None → stage skipped
                │  None → stage skipped    │   image_brief: content_bridge_ungrounded /
                │  verifies the factual    │     _text_missing_from_deliverable /
                │  content bridge          │     _text_under_reported
                └──────────────────────────┘
                          │
                          ▼
                ┌──────────────────────────┐
                │ bundle.labels()          │   copywriting: missing_labels:<slot,...>
                │  None → stage skipped    │   (OTHER-group ads: skip)
                │  (image_brief: skipped)  │   image_brief: always skipped (labels=None)
                └──────────────────────────┘
                          │ pass (target ≥80% yield)
                          ▼
            data/constructed_v2/<skill>/examples.jsonl    (~20k examples)

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4 — QUALITY FILTER                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────────────┐
                │ quality_filter           │            TF-IDF dedup signature on
                │  dedup signature =       │             (canonical_brief_json, parsed.ad)
                │   tfidf(brief_json, ad)  │            drop if similarity >0.95
                │  + length sanity         │            length cap = 8192 tokens
                │  + content-safety        │            content classifier (profanity,
                └──────────────────────────┘                       hate, sexual, scam)
                          │
                          ▼
            data/constructed_v2/copywriting/filtered.jsonl    (~18k examples)

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5 — DATASET ASSEMBLY                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────────────┐
                │ dataset_builder          │            row shape (3 messages):
                │  stratify by platform    │             system  = STATIC_SYSTEM_PROMPT
                │  split 90/5/5            │             user    = canonical_json(brief)
                │  emit HF DatasetDict     │             assistant = <think>R</think>
                └──────────────────────────┘                         {deliverable}
                          │
                          ▼
            data/final_v2/   ┌── train/  (~16.2k rows)
                             ├── val/    (~0.9k rows)
                             └── test/   (~0.9k rows, held-out for eval)
              + audit: data/constructed_v2/_audit/stratification.md

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6 — TRAINING  (configs/training_v2.yaml)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────────────┐
                │ scripts/train.py         │            base = Phase-0 winner
                │  Unsloth + TRL QLoRA     │            r=64, α=128, lr=1.5e-4
                │  assistant_only_loss     │            max_length=8192, 3 epochs
                │  <think> inside          │            rented H100, ~1 day
                │   assistant turn         │
                └──────────────────────────┘
                          │
                          ▼
            outputs/draper-v2/   adapter checkpoints
                          │
                          ▼
                ┌──────────────────────────┐
                │ scripts/train.py merge   │            merged FP16 weights;
                │  --push                  │            pushed to HF for vLLM
                └──────────────────────────┘
                          │
                          ▼
            HF hub: <org>/draper-v2-merged

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7 — EVAL + DEPLOY                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────────────┐            held-out 500-ex test split
                │ scripts/eval.py          │            learned scorer + 3-judge panel
                │  + agent_smoke.py        │            target: composite ≥ v1 + 5%,
                └──────────────────────────┘                    C_pipe ≥ C (vs v1 -0.040)
                          │  ── fails → ROLLBACK (revert env flip, keep v1)
                          ▼  passes
                ┌──────────────────────────┐
                │ deploy/modal_vllm.py     │            app: draper-vllm-v2
                │  download_weights →      │            L4 GPU, OpenAI-compat
                │  modal deploy            │
                └──────────────────────────┘
                          │
                          ▼
            frontend: OPENAI_BASE_URL flips to v2 endpoint
                      + brief-rendering.ts ships canonical JSON serializer
                      + static system prompt pinned to STATIC_SYSTEM_PROMPT
                      (v1 endpoint kept live 24h as fallback)
```

### 4.1 Compact view (construction-only zoom)

```
 scored ads ──► [caption-submit/collect] ──► captions.parquet  (image_brief only)
                      │
                      ▼
 scored ads ──► SkillGateBundle.prepare_source_ads()   (enrich or identity)
                      │
                      ▼
               single_pass teacher ──► briefs.jsonl (cached <brief> JSON)
               (one batch call/ad)            │
                      │                       ▼
                      └──── responses_raw ──► bundle.parse_response()
                                                    │
                                                    ▼
                                          bundle.build_brief()  (validate + inject ad_copy)
                                          + bundle.leak()       [skipped if None]
                                          + bundle.fidelity() + bundle.grounding()
                                          + bundle.content_bridge() [skipped if None]
                                          + bundle.labels()     [skipped if None]
                                                    │
                                                    ▼
                                            quality_filter
                                                    │
                                                    ▼
                                            dataset_builder
                                                    │
                                                    ▼
                                        data/final_v2[_image]/ (HF DatasetDict)
```

### 4.2 Module map (`src/draper/construction_v2/`)

Code is organized into five responsibility subpackages plus the top-level `config.py`, `pipeline.py`, and `platform_labels.py`. Subpackage `__init__` files re-export the public surface; the CLI in `scripts/construct_v2.py` is a thin dispatch layer over `pipeline.py`.

| Module | Responsibility |
|---|---|
| `config.py` | `ConstructionV2Config` + subconfigs (`SelectionConfig`, `ProviderConfig`, `SinglePassConfig`, `BatchConfig`, `FilterConfig`, `DatasetConfig`). `skill: str = "copywriting"` selects the `SkillGateBundle` used throughout the pipeline — changing this is the only config change needed to switch skills. Legacy `BriefExtractionConfig` / `RationaleConfig` retained as optional fields until Phase 4. `from_yaml()` is the single config loader. |
| `pipeline.py` | CLI-facing orchestration: path helpers, `submit_single_pass` / `collect_batch` (async), `ingest_responses` (parse → leak → fidelity → grounding → content_bridge → labels loop; `IngestStats.labels_failed` / `content_bridge_failed` count gate rejections; the content_bridge and labels stages are skipped when their bundle callable is `None`), `verify_selection_lineage`, `parse_slice_spec` / `apply_slice` (`i/N` single-chunk or `i-j/N` range partitioning), `registry_for`, `SelectionLineageMismatch` / `PartialFailureThreshold` exceptions. |
| `platform_labels.py` | Platform-native field-label projection. `PlatformLabelGroup` enum (META / TIKTOK / X / PINTEREST / REDDIT / GOOGLE / OTHER). `PLATFORM_LABEL_MAP` maps `(AdSource, PlatformLabelGroup)` → `tuple[LabelSlot, ...]`. `render_labeled_ad(ad)` renders the source ad with bold Title-Case labels (used in the teacher user message). `check_platform_labels_present(deliverable, ad)` → `LabelResult` — verifies that every populated-slot label appears in the deliverable (ingest gate). Falls back to the unlabeled `ad_copy_text` blob for OTHER-group ads and unmapped pairs. |
| `schemas/brief.py` | `BriefProduct`, `BriefBridge`, `Brief` pydantic models. Single source of truth; TS schema mirrors this. Also exports `STATIC_SYSTEM_PROMPT`, `canonical_json`, `SUPPORTED_PLATFORMS`. `BriefProduct` coerces null lists and joins list-form `category_context`; `Brief` hoists top-level `tone_signals` into `product`. |
| `schemas/image_brief.py` | `ImageBriefInput` — the brief the image-brief writer conditions on: `task` + `objective` + `product` (full `BriefProduct`, unchanged) + `ad_copy` (finished ad copy, verbatim platform-labeled, injected by `build_brief` from the source ad — named `ad_copy` not `copy` to avoid shadowing `BaseModel.copy`) + `platform` + `creative` (`CreativeDirection` — the one nested visual-direction block). `CreativeDirection` carries `orientation` (the canvas; injected from platform, never authored), `brand_guidelines` (the STYLE bridge — reusable visual identity/feel, teacher-authored), and the factual CONTENT bridge: `on_creative_text` (verbatim non-copy on-image text strings) + `key_facts` (load-bearing content facts the copy+product don't supply, each stated as a founder would brief a designer — a fact, not a picture). Both content lists default `[]`; the bridge carries WHAT the creative must convey, never HOW it is composed nor how a fact is rendered (layout/shape/color/framing/placement stay the deliverable's job). `canonical_image_brief_input_json` serializes it (delegates to the shared `canonical_dict_json` in `schemas/brief.py`). Also houses `ImageBrief` — an optional parsed/cache view of the prose deliverable (two fields: `brief` prose + `negative` exclusion list); `canonical_image_brief_json` for that shape. |
| `schemas/records.py` | `ExampleRecord`, `RejectionRecord` — post-ingest row shapes. `ExampleRecord.brief` is a canonical dict (`dict[str, Any]`) — not a typed `Brief` — so both skills' briefs can be stored without importing their specific model; builder and quality filter render via `canonical_dict_json`. `RejectionRecord.stage` literal: `parse` / `fidelity` / `grounding` / `leak` / `labels` / `content_bridge` / `filter`. |
| `teacher/single_pass.py` | **Production teacher (copywriting).** `SINGLE_PASS_TEACHER_SYSTEM` (includes platform-label vocabulary and the rule that labels must not appear inside `<think>`), `build_single_pass_request(ad, model, …)` (user message embeds `render_labeled_ad(ad)` — the source ad laid out with platform-native labels), `parse_single_pass_response(content)` → `SinglePassParseResult`. |
| `teacher/image_brief_single_pass.py` | **Production teacher (image_brief).** `build_image_brief_request(ad, model, …)` — embeds the ad metadata, the VERBATIM platform-labeled copy (`render_labeled_ad(ad)`), and the LITERAL VLM caption from `raw[CAPTION_RAW_KEY]` (the sole visual ground truth). The teacher emits a `<brief>` with `task`, `objective`, `product`, and the `creative` block's authored keys (`brand_guidelines` + `on_creative_text` + `key_facts`); `ad_copy`, `creative.orientation`, and `platform` are NOT emitted by the teacher — they are injected at ingest via `build_brief`. `<think>` may reason from the copy. `parse_image_brief_response(content)` → `ImageBriefParseResult` — extracts `<brief>` JSON + `<think>` + the `<image_brief>` deliverable, parses the deliverable as prose (the literal caption re-registered observational → directive) and extracts the trailing `Avoid:` exclusion line into `exclusions`. Satisfies `TeacherParseResult`. |
| `teacher/brief_extractor.py` | **LEGACY (two-stage).** `BRIEF_EXTRACTION_SYSTEM_PROMPT`, chat-mode `extract_brief`. Retained until Phase 4. |
| `teacher/rationale_prompt.py` | **LEGACY (two-stage).** `RATIONALE_TEACHER_SYSTEM`, `build_rationale_request`. Retained until Phase 4. |
| `ingest/skills.py` | **Skill registry.** `SkillGateBundle` dataclass carrying 9 callables: `prepare_source_ads`, `build_request`, `parse_response`, `build_brief` (ingest-time: validates the cached brief dict into the skill's brief model and injects any field the teacher does not author — image_brief injects the verbatim platform-labeled `ad_copy` + the platform-derived `creative.orientation`, and hoists any stray top-level `brand_guidelines`/`on_creative_text`/`key_facts` into the `creative` block), `fidelity`, `grounding`, `leak` (optional — `image_brief` sets `None`; the 5-gram leak guard is skipped because copy is a legitimate verbatim brief input; copywriting uses `check_bridge_leak`), `labels` (optional — `image_brief` sets `None`; skips the platform-label stage), `content_bridge` (optional — set only by `image_brief`; verifies the factual content bridge). `register(bundle)` / `get_bundle(skill)` / `registered_skills()`. Module-level `_register_copywriting_bundle()` and `_register_image_brief_bundle()` populate the registry on import. `TeacherParseResult` protocol: `brief / think / deliverable / errors`. |
| `ingest/response_parser.py` | `parse_response(text) -> ParsedResponse(think, deliverable) \| ParseRejection`. Rejects: `missing_think`, `missing_deliverable`, `teacher_failed`, `think_too_short`, `pre_think_noise`. (copywriting skill) |
| `ingest/fidelity.py` | `check_deliverable_fidelity` — word-coverage ≥60% + 6-word verbatim signature on the deliverable only. `check_think_grounding(think, brief)` — passes iff at least one bridge field surfaces in the think trace (product-fact requirement dropped 2026-05-22). Both used by the copywriting bundle. |
| `ingest/image_brief_fidelity.py` | `check_image_brief_fidelity` — two stages: the `<image_brief>` prose region must extract non-empty, and the prose must carry ≥30% (`MIN_CAPTION_OVERLAP`) of the literal caption's content words (caption-overlap); no JSON schema. `check_image_brief_brief_alignment` — grounding gate: `<think>` must reference a content word from `creative.brand_guidelines`; returns `GroundingResult` with `bridge_match` kept for cross-skill symmetry. `check_image_brief_content_bridge(brief, deliverable, source_ad)` → `ContentBridgeResult` — verifies the factual content bridge: (1) **factuality** — each `on_creative_text`/`key_facts` atom's content words overlap the caption; (2) **over-report** — each `on_creative_text` string appears in the deliverable; (3) **under-report** — every double-quoted on-image string in the deliverable not in `ad_copy` is covered by an `on_creative_text` item (`key_facts` is deliberately not credited — verbatim text belongs in `on_creative_text`; the Avoid line is excluded). The no-composition rule (`key_facts` states facts, never the visual realization) is enforced structurally + by prompt, not machine-gated. Used by the image_brief bundle. |
| `ingest/leak_guard.py` | 5-gram overlap check between bridge-field copy and the source ad. Runs first in the ingest loop so leaky briefs short-circuit before the more expensive fidelity check. |
| `ingest/__init__.py` | Re-exports the ingest public surface including `LabelResult` and `check_platform_labels_present` from `platform_labels.py`. |
| `captions/builder.py` | VLM captioning for image-brief source ads. `IMAGE_URL_SUFFIXES` — frozenset of image-capable URL extensions (`.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`). `iter_captionable_ads(ads)` — filters source ads to those with image URLs. `build_caption_request(ad, model, prompt)` — constructs the vision-enabled batch request embedding the creative URL. `parse_caption_response(content)` → caption text. `enrich_source_ads_with_captions(ads, captions_parquet, require_caption)` — joins `data/captions/v1/captions.parquet` onto source ads by `ad_id`; drops ads with no caption when `require_caption=True`. `write_caption_rows(rows)` — appends to `captions.parquet`. Also exports `estimate_caption_cost_usd`, `load_captions_lookup`, `captions_path`, `CAPTIONS_OUTPUT_PATH`, `CAPTION_TASK_FORMAT` (`vlm_caption_v1` — keeps caption batches distinct in the shared registry), `CAPTION_PROMPT_VARIANTS`, `LITERAL_CAPTION_PROMPT`, `STRATEGIC_CAPTION_PROMPT`. |
| `captions/__init__.py` | Re-exports the full public surface of `captions/builder.py`. |
| `dataset/source_selector.py` | `SourceAd`, `select_source_ads(config)` (stratified by platform, writes `selection_lineage_hash` column; drops ads whose labeled render is empty for mapped platforms; `_has_structural_artifact` filter dropped `url_in_headline` and `hashtag_dump` — both ~95% false-positive on platform-native patterns), `load_source_ads_by_id`, `PlatformConcentration` exception. |
| `dataset/quality_filter.py` | TF-IDF dedup on `(canonical_brief_json, deliverable)`, length cap, content-safety classifier. |
| `dataset/builder.py` | Stratification key = `platform`. Emits 3-message `[system, user, assistant]` rows for the HF DatasetDict. |

**Removed vs the dice-era design:** `dice.py` and `sampler.py` modules don't exist.

### 4.3 Reused from v1 unchanged

- `src/draper/construction/source_selector.py` — drop the `_pairs_to_batches` paths only.
- `src/draper/construction/batch/` — provider-agnostic batch infrastructure. **Extended for vision:** `content_blocks.py` translates `[image_url, text]` content lists to each provider's native wire format; `BatchRequest.messages` is now `list[dict[str, Any]]` (was `list[dict[str, str]]`). The anthropic, gemini, and openai batch clients updated accordingly.
- `src/draper/scoring/schemas.py` — `ScoredAd` is unchanged.
- `src/draper/training/{config,data_loader,trainer,merge,hub}.py` — only the config file differs.

### 4.4 Archived from v1

Moved to `archive/construction-v1/`:
- `src/draper/construction/{personas,clusterer,cluster_report,dice,difficulty,religious_scripture,bundle}.py`
- `src/draper/construction/formats/copywriting/{dice,constructor}.py`
- `configs/personas.yaml`
- `scripts/construct.py` → `archive/scripts/construct_v1.py`
- The single-pass brief+rationale teacher prompt.

### 4.5 Entry point

`scripts/construct_v2.py` — Typer CLI. The full pipeline for either skill:

```
# --- Selection (both skills) ---
construct_v2 select        # pick source-ad batch; writes selection.parquet + lineage hash
                           # --target N  --allow-unbalanced  --force-unbalanced
                           # --exclude-from-run <run_id>  (repeatable; excludes collected
                           #   AND in-flight ad_ids from each named run)

# --- Captioning (image_brief skill only; run between select and submit) ---
construct_v2 caption-submit   # submit a VLM captioning batch over selection.parquet
                              # --provider anthropic|openai|gemini  (required)
                              # --slice i/N or i-j/N
                              # --prompt literal|strategic  (caption prompt variant; default: literal)
                              # --model <override>
                              # --skip-existing / --recaption  (default: skip already-captioned)
                              # --bypass-selection  (caption full image corpus, not just selection)

construct_v2 caption-collect <batch_id>   # poll + write captions.parquet; remove on success

# --- Teacher submission (both skills) ---
construct_v2 submit        # submit a single-pass batch for one provider's ad slice
                           # --provider anthropic|openai|gemini  (required)
                           # --slice i/N or i-j/N  (default 0/1; disjoint chunk(s)
                           #                        of selection.parquet)
                           # --run-id <id>          (scope to data/constructed_v2/runs/<id>/)
                           # --model <override>     (override providers[<provider>].model)
                           # --allow-lineage-drift

construct_v2 collect <batch_id>   # poll + persist single-pass results; remove on success
                                  # --run-id <id>

construct_v2 list          # show batches (--run-id; --registry-path for legacy);
                           # polls providers for live status by default
                           # --no-refresh  skip live poll, read cached status only
construct_v2 cancel <batch_id>    # cancel a tracked batch

# --- Ingest + quality filter + build (both skills) ---
construct_v2 ingest        # parse responses + build_brief + leak[skip if None] + bundle.fidelity
                           # + bundle.grounding + content_bridge[skip if None]
                           # + bundle.labels[skip if None]
                           # reports labels_failed (0 for image_brief) + content_bridge_failed (0 for copywriting)
                           # --run-id  --input (override responses_raw.jsonl path)

construct_v2 filter        # quality filter pass (dedup + length + content-safety)
                           # --run-id

construct_v2 build         # assemble final HF DatasetDict (data/final_v2 or data/final_v2_image)
                           # --run-id  --output  --input

construct_v2 render        # render collected single-pass briefs+responses as reviewable Markdown
                           # --run-id  --output

# Hidden legacy aliases (kept one migration phase):
construct_v2 batch-list    → list
construct_v2 batch-cancel  → cancel
```

Dropped from the pre-single-pass design: `extract`, `rationale`, `batch-submit` with `--stage` flag — replaced by the unified `submit` command.

### 4.6 Configs

Two config files drive two skills. Both are loaded via `ConstructionV2Config.from_yaml()`.

**`configs/construction_v2.yaml`** — copywriting skill:

```yaml
construction_v2:
  skill: copywriting          # selects SkillGateBundle; default omits this line

  output_dir: data/constructed_v2
  final_dir: data/final_v2
  audit_dir: data/constructed_v2/_audit

  selection:
    target_count: 3000          # override with `select --target N` for smoke runs
    min_composite: 0.7          # v3 score cutoff
    stratify: platform
    allow_unbalanced: false
    max_platform_share: 0.50    # raises PlatformConcentration if exceeded
    # ... v1-parity quality gates (english_only, drop_unsafe, etc.)

  providers:
    anthropic:
      model: claude-sonnet-4-6
      max_tokens: 4000
      temperature: 0.4
    openai:
      model: gpt-5.4
      max_tokens: 4000
      temperature: 0.4
    gemini:
      model: gemini-3.1-pro-preview
      max_tokens: 4000
      temperature: 0.4

  single_pass:
    forbid_ngram_overlap: 5
    briefs_cache_path: data/constructed_v2/copywriting/briefs.jsonl

  batch:
    stuck_timeout_minutes: 360
    auto_force_cancel: true
    max_partial_error_rate: 0.05

  filter:
    dedup_similarity_threshold: 0.95
    max_tokens: 8192
    min_deliverable_chars: 40

  dataset:
    train_ratio: 0.90
    val_ratio: 0.05
    test_ratio: 0.05
    seed: 42
```

**`configs/construction_v2_image.yaml`** — image_brief skill (key diffs from above):

```yaml
construction_v2:
  skill: image_brief

  output_dir: data/constructed_v2_image
  final_dir: data/final_v2_image
  audit_dir: data/constructed_v2_image/_audit

  selection:
    min_composite: 0.6        # relaxed: 0.70 leaves only 2,822 image-capable ads,
                              # below the 3,000 target; 0.60 gives 5,184 (headroom)

  single_pass:
    briefs_cache_path: data/constructed_v2_image/image_brief/briefs.jsonl

  filter:
    min_deliverable_chars: 20  # image_brief prose is shorter than freeform ad copy
```

The legacy `brief_extraction` and `rationale` sections are intentionally absent from both files. Two-stage YAMLs still load (those fields are optional) but no new configs should include them.

## 5. Training pipeline (`configs/training_v2.yaml`)

Diffs from `configs/training.yaml`:

| Field | v1 | v2 | Reason |
|---|---|---|---|
| `base_model` | `unsloth/Qwen3-8B-unsloth-bnb-4bit` | TBD by Phase 0 bake-off | Native thinking required |
| `max_length` | 4096 | 8192 | `<think>` traces inflate sequences |
| `lora_r` | 32 | 64 | Richer joint target (rationale + ad) |
| `lora_alpha` | 64 | 128 | rsLoRA convention `2 * r` |
| `learning_rate` | 2.0e-4 | 1.5e-4 | Compensate for longer effective sequences |
| `output_dir` | `outputs/qwen3-8b-copywriting` | `outputs/draper-v2` | |
| `dataset_dir` | `data/final` | `data/final_v2` | |
| `assistant_only_loss` | true | true | Kept — verify per base model that `<think>` is inside the assistant turn |

Final HF row shape:

```python
{
  "messages": [
    {"role": "system",    "content": STATIC_SYSTEM_PROMPT},
    {"role": "user",      "content": canonical_json(brief)},
    {"role": "assistant", "content": (
        f"<think>\n{rationale}\n</think>\n\n"
        f"{deliverable}"
    )},
  ],
  "metadata": {"example_id": ..., "ad_id": ..., "platform": ...},
}
```

`STATIC_SYSTEM_PROMPT` describes the two-slot contract (`<think>` + freeform deliverable). Byte-identical between training and inference (see `src/draper/construction_v2/schemas/brief.py::STATIC_SYSTEM_PROMPT` — the canonical string), and mirrored in the frontend's `system-prompt.ts`. Any drift and the writer goes off distribution.

Chat-template contract per candidate (resolved in Phase 0):

- **Qwen3-8B-Thinking** — native `<|im_start|>assistant\n<think>...</think>...<|im_end|>`. Drop-in.
- **DeepSeek-R1-Distill-Qwen-7B** — native `<｜Assistant｜><think>...</think>...<｜end▁of▁sentence｜>`. Drop-in.
- **Llama-3.1-8B-Instruct** — no native think channel. Requires a custom tokenizer template patch wrapping `<think>...</think>` inside the assistant turn. Lean against unless bake-off shows dominance.

## 6. Frontend cutover

Goal: zero train/inference skew. The bytes the model sees at inference must be character-identical to the bytes it saw during training.

### 6.1 Changes

- `frontend/lib/agent/brief-rendering.ts`:
  - Remove `renderProductFacts` (lines 54–78).
  - Extend `BriefProductSchema` (line 81) with `category_context`, `proof_points`, `offer`, `platform_hint`.
  - Add `BriefBridgeSchema` mirroring `BriefBridge`.
  - Add `serializeBriefForDraper(brief, platform) -> string` returning canonical JSON (sorted keys, no `null` elision — byte-identical to pydantic `model_dump(mode="json", exclude_none=False)` with `sort_keys=True`).
- `frontend/lib/agent/tools/draft-campaign.ts`:
  - Lines 180–200 (`renderBrief`): return `serializeBriefForDraper(brief, platform)`.
  - Drop the platform-specific creative-coaching prose (lines 163–178). Platform shape is driven by the existing structured-extraction pass (lines 213+).
  - `generateText` call (line 558): pass `system: STATIC_SYSTEM_PROMPT` (exported from a shared module that is byte-equal to the Python constant).
- `frontend/lib/agent/tools/ask-draper.ts`:
  - Same swap at lines 106 / 312–317.
  - Fold the orchestrator's `request` field into `bridge.angle` rather than carrying a free-form coaching field.
- `frontend/lib/agent/system-prompt.ts`:
  - v1's `TRAINING_SYSTEM_PROMPT` constant is retired in favor of `STATIC_SYSTEM_PROMPT` mirrored from Python.
- `CLAUDE.md` — update the cardinal rule section to v2 wording.

### 6.2 Cutover sequence (single PR, single deploy)

1. v2 model trains and merges to `outputs/draper-v2/merged`.
2. v2 vLLM endpoint deployed at a separate URL (e.g. `draper-vllm-v2`) — not yet routed.
3. Frontend env flips `OPENAI_BASE_URL` to v2 endpoint.
4. Frontend ships JSON serializer change + `STATIC_SYSTEM_PROMPT` constant in the same deploy.
5. v1 endpoint left running 24h as fallback.

**The frontend serializer change and the v2 model are incompatible with their counterparts.** They must ship together.

### 6.3 Contract test

`tests/contract/brief_serialization.json` — fixture with N representative briefs. Both sides load it and assert that `serializeBriefForDraper(brief)` (TS) equals `canonical_json(brief)` (Python) byte-for-byte. Plus a simple equality test that `STATIC_SYSTEM_PROMPT` matches byte-for-byte across Python and TS.

## 7. Implementation phases

### Phase 0 — Base-model bake-off

**Goal:** pick the base model on a 4k-example mini-dataset before committing to full construction.

**Duration:** 3–4 days wall-clock • **Cost:** ~$30 in teacher calls + ~6–10 GPU-hours

**Tasks:**

1. Build out `src/draper/construction_v2/` foundations (schemas, response parser, prompts).
2. Implement `brief_extractor.py` + `rationale_prompt.py` with chat-mode (non-batch) execution.
3. Run `scripts/bake_off_v2.sh`:
   - Select ~5,000 ads from `data/scored/v3/scored_ads.parquet`.
   - Brief extraction batch → ~5,000 briefs cached.
   - Rationale batch (1:1) → ~5,000 raw responses.
   - Filter → ~4,000 examples (assume 80% yield).
   - Build → `data/final_bakeoff/`.
4. For each candidate in `{Qwen3-8B-Thinking, DeepSeek-R1-Distill-Qwen-7B}` (+optionally `Llama-3.1-8B-Instruct`):
   - 1-epoch QLoRA SFT via `scripts/train.py --config configs/training_v2.yaml --base-model <candidate> --epochs 1 --output-dir outputs/bakeoff_<candidate>`.
5. `scripts/bake_off_eval.py`:
   - Generate completions on the 500-example held-out test split at T=0.4, top_p=0.9.
   - Score parsed ad portion via the learned scorer (`scripts/serve_scoring_predictor.py`).
   - Run a 3-criteria LLM-as-judge (Claude Opus 4.7 or GPT-5.5): (a) ad-style genre fidelity, (b) `<think>` coherence and shape-fit to the ad, (c) overall helpfulness.
   - Emit `data/bakeoff/leaderboard.md`.

**Exit criteria:**
- Winner with ≥5% scorer composite lift over Qwen3-8B-Instruct base on the 500-example test set.
- Winner with `<think>`-coherence judge score ≥3.5/5.
- Bridge-field leakage rate <2% (5-gram overlap with source ad).

**Deliverables:** `data/bakeoff/leaderboard.md`, locked `base_model` in `configs/training_v2.yaml`.

### Phase 1 — Build v2 construction pipeline at production quality

**Goal:** harden Phase 0 prototypes into production code with full test coverage and CLI parity with v1.

**Duration:** ~1 week

**Tasks:**

1. Complete all `src/draper/construction_v2/` modules per §4.2.
2. Implement `scripts/construct_v2.py` CLI with all subcommands.
3. Write `configs/construction_v2.yaml`.
4. Write the contract-test fixture `tests/contract/brief_serialization.json` and Python+TS sides of the test.
5. Unit tests: brief schema round-trip, response parser edge cases (missing think, sentinel failure), fidelity check on synthetic ads, grounding check (positive + negative).
6. Integration test: end-to-end on 10 source ads using chat-mode (non-batch) teachers — assert 9+/10 yield.

**Exit criteria:**
- `make lint && make typecheck && make test` passes.
- Smoke: `scripts/construct_v2.py submit --provider anthropic --limit 1 --run-id smoke` submits one single-pass batch; `collect` + `ingest` produces ≥1 fidelity-passing example.
- Frontend contract test green: TS and Python emit byte-identical canonical JSON for all fixture briefs and the same `STATIC_SYSTEM_PROMPT`.

### Phase 2 — Run full construction

**Goal:** produce the v2 training corpus.

**Duration:** 2–3 days wall-clock • **Cost:** ~$200 in teacher calls (Anthropic Batch, 50% discount)

**Targets:**
- ~25,000 unique source ads from `data/scored/v3/scored_ads.parquet`.
- 1:1 brief → rationale → ~25,000 raw rationale responses.
- After filter (assume ~75% yield, slightly lower than dice-era because grounding is stricter on a single shot) → ~18,000 training examples.

**Tasks:**

1. `scripts/construct_v2.py select --target 25000` → `data/constructed_v2/_audit/selection.parquet` (with lineage hash).
2. `scripts/construct_v2.py submit --provider anthropic --slice 0/3` (+ openai `1/3`, gemini `2/3`) → submits disjoint ad chunks as single-pass batches.
3. `scripts/construct_v2.py collect <batch_id>` for each in-flight batch → writes `briefs.jsonl` + `responses_raw.jsonl`.
4. `scripts/construct_v2.py ingest` → parse + leak guard + fidelity + grounding.
5. `scripts/construct_v2.py filter` → quality filter pass.
6. `scripts/construct_v2.py build --output data/final_v2/` → HF DatasetDict with train/val/test splits stratified by `platform`.

**Exit criteria:**
- Construction yield ≥75% after fidelity + grounding rejections.
- Stratification report at `data/constructed_v2/_audit/stratification.md` shows balance across train/val/test (no platform below 8% of its split).
- Bridge-field leak rate <2% on a 200-example manual audit.

**Deliverables:** `data/final_v2/` HF DatasetDict, audit reports under `data/constructed_v2/_audit/`.

### Phase 3 — Train at scale

**Goal:** produce a merged Draper-v2 model that beats v1 on held-out eval.

**Duration:** 1 day wall-clock on a rented H100 • **Cost:** ~$30 GPU rental

**Tasks:**

1. `scripts/train.py --config configs/training_v2.yaml` (3 epochs, base model from Phase 0).
2. `scripts/train.py merge --push` → uploads merged weights to HF for vLLM.
3. `modal run deploy/modal_vllm.py::download_weights` then `modal deploy deploy/modal_vllm.py` to a new app name (`draper-vllm-v2`).
4. Run the eval pipeline on the 500-example held-out test split + the agent-smoke harness.

**Exit criteria:**
- Held-out learned-scorer composite ≥ v1 composite + 5%.
- `engagement_velocity` head ≥ v1 + 3% (v1's weak point).
- Extraction-failure rate <2% (v1: ~5%).
- Agent-loop wrapping no longer degrades the score: target `C_pipe ≥ C` (v1: `C_pipe - C = -0.040`).

**Deliverables:** merged checkpoint on HF, deployed Modal endpoint URL, `docs/research/DRAPER_V2_TRAINING_RESULTS.md`.

### Phase 4 — Frontend cutover

**Goal:** route production traffic to v2 with zero distribution skew.

**Duration:** 1–2 days (mostly QA)

**Tasks:**

1. Implement frontend changes per §6.1 in a single PR.
2. Pass the contract test (TS ↔ Python byte-equality).
3. Stage on a preview deploy; smoke 20 representative briefs end-to-end.
4. Promote: flip `OPENAI_BASE_URL` env to v2 Modal URL.
5. Update CLAUDE.md cardinal rule and the `frontend/lib/agent/README.md`.

**Exit criteria:**
- Preview-deploy smoke: 20/20 briefs produce a campaign card (no `<EXTRACTION_FAILED>`).
- Agent-smoke harness on the new endpoint matches Phase 3 results within ±2%.
- v1 endpoint stays live 24h post-cutover for fallback.

**Rollback gate:** if v2 LLM-as-judge or scorer numbers don't clear v1 by ≥5% in production traffic during the first 24h, revert `OPENAI_BASE_URL` + the frontend serializer change in a single 10-line diff.

## 8. Critical files

### To create

- `src/draper/construction_v2/__init__.py`
- `src/draper/construction_v2/config.py`
- `src/draper/construction_v2/pipeline.py`
- `src/draper/construction_v2/schemas/{__init__,brief,records}.py`
- `src/draper/construction_v2/teacher/{__init__,brief_extractor,rationale_prompt,single_pass}.py`
- `src/draper/construction_v2/ingest/{__init__,response_parser,fidelity,leak_guard}.py`
- `src/draper/construction_v2/dataset/{__init__,source_selector,quality_filter,builder}.py`
- `scripts/construct_v2.py`
- `scripts/bake_off_v2.sh`
- `scripts/bake_off_eval.py`
- `configs/construction_v2.yaml`
- `configs/training_v2.yaml`
- `tests/contract/brief_serialization.json`
- `tests/construction_v2/test_*.py` (suite)
- `docs/research/DRAPER_V2_TRAINING_RESULTS.md` (Phase 3 deliverable)

### To modify

- `frontend/lib/agent/brief-rendering.ts`
- `frontend/lib/agent/tools/draft-campaign.ts`
- `frontend/lib/agent/tools/ask-draper.ts`
- `frontend/lib/agent/system-prompt.ts`
- `frontend/lib/agent/schemas.ts`
- `frontend/lib/agent/README.md`
- `CLAUDE.md`
- `deploy/modal_vllm.py` (point at v2 merged weights; deploy as `draper-vllm-v2`)

### To archive (`archive/construction-v1/`)

- `src/draper/construction/{personas,clusterer,cluster_report,dice,difficulty,religious_scripture,bundle}.py`
- `src/draper/construction/formats/copywriting/{dice,constructor}.py`
- `configs/personas.yaml`
- `scripts/construct.py` (→ `archive/scripts/construct_v1.py`)

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Bridge fields leak ad copy** — teacher paraphrases the ad headline into `angle` or `buyer_pain` | Brief-extraction prompt explicitly forbids quoting; ingestion-time check rejects briefs whose bridge fields share a 5-gram with the source ad. Reject rate budgeted at <10%; if higher, tighten prompt or reduce to 4-gram. |
| **Teacher fails to vary response shape** — every rationale ends up the same medium-length neutral voice regardless of source ad | Bake-off judge scores `<think>` coherence and shape-fit. If shape collapses, escalate by adding 2–3 in-prompt examples that explicitly demonstrate playful-short vs formal-long contrast. |
| **At inference, the brief is too thin to drive response shape** — model defaults to training-average voice | Brief-extraction prompt requires non-empty `tone_signals` and rejects empty results. `tone_signals` are the inference-time style anchor. Frontend brief builder must populate them (or the orchestrator must infer them from retrieved context). |
| **JSON serialization drift between Python and TS** | Codegen one from the other (datamodel-code-generator) + a contract-test fixture loaded by both sides. CI gate on byte-equality across the fixture set. |
| **Thinking trace bloats sequence length** | `max_length=8192` is a guess; Phase 0 reports p95 sequence length. If p95 >7k tokens, bump to 12k and lower per-device batch to 1. |
| **Teacher refuses to emit verbatim ads** (Anthropic content policy on commercial copy) | Reuse v1's exact "MOST IMPORTANT RULES" framing inside `RATIONALE_TEACHER_SYSTEM`. Track rejection batches via `scripts/explore/diagnose_claude_rejects.py`. |
| **v2 underperforms v1 on the held-out scorer** | Rollback gate before traffic flip. v1 endpoint stays live 24h. 10-line revert PR (env flip + `renderBrief` revert). |
| **Bridge fields don't improve correlation enough** — model still treats the brief as decoration | The think trace is the second line of defense — it forces the model to verbalize the bridge-fact-to-hook mapping. If both fail in Phase 0, reconsider whether bridge fields need to be expanded (e.g. add `competitive_context`, `objection_handled`). |
| **`<think>` collapses into third-person analytical commentary** — early smoke output read like portfolio case studies ("The ad leverages X… The tone signals align with Y…"), not like internal monologue | The rationale-teacher prompt now explicitly forbids the third-person register and requires first-person, decisional voice with phrases like "I want…", "let me try…", "no, that's too sales-y". The smoke renderer surfaces the think block inline (not in a `<details>` collapse) so register problems are visible on first read. |
| **Framing prose contaminates the ad** — model lapses into "Here's a draft: …" or starts the ad with a meta-preamble that breaks the verbatim contract | Verbatim-ad anchoring (teacher reproduces the source ad character-for-character) plus the deliverable-only fidelity check (≥60% word coverage + 6-word signature) holds the line in training. Any framing prose lives outside the verbatim span; fidelity rejects rows where the ad text was paraphrased instead of reproduced. |

## 10. Future work

- **Per-format extension.** v2 is single-format (copywriting). If a second format is added later, the format-registry pattern from v1 (`src/draper/construction/formats/registry.py`) still applies — bridge fields become format-specific (e.g. landing-page copy uses different bridge axes than ad copy).
- **Online RLHF / DPO** after Phase 4 is stable, using production scorer feedback as the preference signal.
- **Optional response-shape steering at inference.** If the orchestrator wants explicit control (e.g. "make this one punchier"), it can append a single sentence to the user message ("Voice: punchy, short.") rather than re-introducing a parallel dice channel. Defer until production data shows a need.
- **Bake-off rerun cadence.** As new strong open-weight bases ship (Qwen4, DeepSeek-V4, Mistral-Reasoning), rerun Phase 0 quarterly.

---

See also: [`HUMAN_EVAL_STUDY_PLAN.md`](HUMAN_EVAL_STUDY_PLAN.md) — RQ1 pairwise preference study plan (Draper v2 vs frontier baseline vs GOLD), including Prolific design, analysis plan, and OSF pre-registration checklist.
