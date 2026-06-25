# Draper.ai — Phased Implementation Plan

## Context

Draper.ai is a master's thesis project (26 weeks) building a domain-specialized 7B **agentic** marketing reasoning model combining QLoRA fine-tuning with tool-use capabilities and live web search. The fine-tuned model acts as a **function-calling agent** — it receives a marketing task, formulates a plan, decides which tools to call (web search, URL scraping), processes results, and iterates until it produces a complete structured output. This is not a linear pipeline; the model drives the execution.

Currently zero code exists — only thesis proposal, data sources doc, project vision doc, and a literature review. The goal is to go from documentation to a fully evaluated prototype answering three research questions: (1) can a fine-tuned 7B agent match frontier LLMs for marketing tasks, (2) does fine-tuning with tool-use outperform either alone, (3) can proxy signals approximate ad performance.

**Status as of 2026-03-25:**
- AdFlex API: **purchased** ($300 Pro plan, 500K credits/month)
- Phase 1 scraping pipeline: **largely complete** — AdFlex client, Meta Library, Google Transparency, TikTok Library all built with tests passing. BigSpy client also built (deprioritized).
- Exploration notebooks: Meta Ad Library (01), AdFlex API (02), IRA dataset (03), Upworthy (04) created
- GPU for fine-tuning: **cloud rental** (Vast.ai or Lambda Labs, provision in Phase 3)
- Progress tracking: informal (no Jira)

> **2026-04 pivot — copywriting-only scope:** Construction has narrowed to the **copywriting format only** (backtranslation / Humpback). The multi-format architecture (positioning, diagnostic, optimization, channel_format_fit) described in this plan is preserved in `archive/construction/formats/` and deferred to post-thesis work. The project structure and phase descriptions below reflect the original multi-format design.

---

## Agent Architecture

The fine-tuned 7B model operates as a **function-calling agent** with access to two tools:

| Tool | Description | When Model Should Call It |
|------|-------------|--------------------------|
| `web_search(query)` | Search the web via Tavily/Serper, returns top results with snippets | Competitive intel, market trends, industry data |
| `scrape_url(url)` | Extract structured content from a webpage via trafilatura | Analyze product pages, competitor sites, landing pages |

**Agent loop at inference:**
1. Model receives task (e.g., "Generate a campaign for this URL: ...")
2. Model reasons about what information it needs → generates a tool call (or multiple)
3. Tool is executed, result injected back into context
4. Model continues reasoning — may call more tools or produce final output
5. Loop ends when model emits a structured final response (no more tool calls)

**Training approach:** Fine-tuning focuses on **domain knowledge only** (5 pattern-skill task formats: positioning, copywriting, diagnostic, optimization, channel_format_fit). Tool-use ability comes from the base model's pre-trained function-calling capability — no trajectory fine-tuning needed. This keeps training data focused, the ablation clean, and leaves trajectory fine-tuning as a post-thesis extension if tool-use quality proves insufficient.

---

## Project Structure

```
Draper.ai/
├── pyproject.toml
├── .env.example                      # API keys template
├── .gitignore
├── Makefile                          # lint, test, typecheck
├── configs/
│   ├── scraping.yaml                 # Scraping params, rate limits, verticals
│   ├── scoring.yaml                  # Proxy score weights, tier thresholds
│   ├── training.yaml                 # QLoRA hyperparams, data mix ratios
│   ├── eval.yaml                     # Judge model, dimensions, test sets
│   └── agent.yaml                    # Tool definitions, loop config, max steps
├── src/draper/
│   ├── scraping/                     # AdFlex, ad library clients (data collection)
│   ├── scoring/                      # Composite scorer, tier assigner
│   ├── construction/                 # Training data builder (copywriting format only, post-2026-04)
│   │   ├── formats/                  # Per-format pipelines (one package per format)
│   │   │   ├── copywriting/          # backtranslation-mode — active
│   │   │   ├── positioning/          # archived → archive/construction/formats/
│   │   │   ├── diagnostic/           # archived → archive/construction/formats/
│   │   │   ├── optimization/         # archived → archive/construction/formats/
│   │   │   └── channel_format_fit/   # archived → archive/construction/formats/
│   ├── training/                     # QLoRA train, sweep, merge
│   ├── tools/                        # Tool implementations available to the agent at inference
│   │   ├── web_search.py             # Tavily/Serper wrapper
│   │   ├── url_scraper.py            # trafilatura-based page extraction
│   │   ├── registry.py               # Tool registry + JSON schema definitions
│   │   └── executor.py               # Execute tool calls, format results
│   ├── agent/                        # Agent loop, orchestration, output schemas
│   │   ├── loop.py                   # Core agent loop (reason → call tool → observe → repeat)
│   │   ├── output_schema.py          # Pydantic models for structured outputs
│   │   └── prompts.py                # System prompts with tool definitions
│   ├── evaluation/                   # LLM judge, proxy validation, metrics
│   └── utils/                        # LLM client, I/O, logging
├── data/
│   ├── raw/                          # Per-source scraped data (JSONL)
│   ├── scored/                       # After composite scoring (Parquet)
│   ├── constructed/                  # Per-format training examples
│   ├── final/                        # HF Dataset ready for training
│   └── validation/                   # IRA + Upworthy datasets
├── notebooks/
├── scripts/
├── tests/
└── writing/                          # Already exists
```

## Key Technical Decisions (validated via deep research — March 2026)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base model | Bake-off: **Qwen3-8B** (primary) vs LLaMA 3.1 8B (fallback) | Qwen3-8B has first-class function-calling + thinking mode toggle; Apache 2.0. Qwen 2.5 is superseded, Mistral 7B v0.3 is abandoned |
| Training stack | **Unsloth + TRL SFTTrainer + HuggingFace PEFT** | Unsloth: 2x speed, 70% less VRAM (official HF integration Feb 2026). TRL has native tool-calling dataset support |
| QLoRA config | r=32, `use_rslora=True`, `packing=True`, `use_liger_kernel=True`, `assistant_only_loss=True` | r=32 preferred over 64 (less overfitting with 10K examples); new TRL/PEFT features are free improvements |
| GPU | **RTX 4090** ($0.40-0.80/hr) or A100 40GB on Vast.ai/RunPod | A100 80GB is overkill. Total training cost ~$2-10 |
| Inference serving | **vLLM** with guided JSON generation | Best for tool-use with small models; constrained generation prevents malformed tool calls |
| Data sources | **AdFlex** (primary — sole ad intelligence source with engagement metrics across 5 platforms) + Meta Ad Library (free structural supplement) | AdFlex provides engagement data for Facebook, TikTok, X, Pinterest, Reddit at 100 credits/call. Meta Ad Library lacks engagement data but is free. Official ad libraries (Google, TikTok) provide supplementary structural data |
| Search API | **Tavily** (primary) + **Exa.ai** (competitive intel via semantic `findSimilar`) | Exa's semantic search is superior for discovering competitor content |
| Content extraction | **trafilatura** (static pages) + **Jina Reader API** (JS-heavy pages, free) | Many ad landing pages are JS-heavy SPAs that trafilatura can't handle |
| Agent loop | **Custom implementation** (not LangChain) | Validated: no framework is clearly superior for a simple tool-calling loop (~50-80 lines of Python) |
| Construction LLM | **Claude** (via subscription, batch prompting) | Better structured extraction; batches of 50-100 ads per prompt |
| Judge LLM | **GPT-4o** (primary) + **Gemini** (secondary cross-validation) | GPT-4o avoids circularity with Claude construction. Multi-judge catches systematic biases |
| Eval harness | **PromptFoo or DeepEval** | Build marketing-specific metrics as custom scorers rather than building eval framework from scratch |
| Data format | JSONL → Parquet → HF Dataset | Unchanged — still the standard |
| Tool-use training | **None** — rely on Qwen3-8B's native function-calling | Keeps ablation clean. Mitigation: mix ~20% tool-use examples into training data; use `init_lora_weights="pissa"` |
| Config | YAML + Pydantic Settings | Reproducible experiments |

---

## Phase 1: Scraping Pipeline Setup (Weeks 1–4)

### Week 1 — Project bootstrap + API access
- Init git repo, directory structure, `pyproject.toml`, `.env.example`
- Sign up: AdFlex (€99/mo Pro plan), Apify (free tier), Tavily, Exa.ai (free tier)
- Download IRA dataset (3,517 ads) + Upworthy archive → `data/validation/`
- Build: `utils/llm_client.py` (unified Anthropic/OpenAI async client), `utils/io.py`, `utils/logging.py`
- Set up linting (ruff), type checking (mypy), Makefile

### Week 2 — AdFlex client + exploratory scraping
- Build `scraping/adflex.py`: search, detail, paginated search with cursor (`last_hit`), checkpoint/resume
- Build `scraping/rate_limiter.py`: token bucket, configurable per-API
- Define canonical `RawAd` Pydantic schema (all fields: copy, format, platform, country, active_days, engagement, advertiser info)
- Exploratory scrape via AdFlex: test all 5 platforms, analyze field coverage and engagement quality in notebook

### Week 3 — Supplementary scrapers + knowledge corpus
- Build scrapers: `meta_library.py` (Apify), `google_transparency.py` (SerpApi), `tiktok_library.py` (Apify)
- Build `knowledge_corpus.py`: URL list → trafilatura extraction → Claude structured extraction → JSONL
- Test each with 50–100 items. Evaluate data quality

### Week 4 — Pipeline orchestration + dataset exploration
- Build `scripts/scrape.py` CLI (typer): per-source commands, resume, progress bars
- Create `configs/scraping.yaml`: target verticals, rate limits, counts per vertical, AdFlex platform/filter config
- IRA exploration notebook: distributions of spend/impressions/clicks, map to RawAd schema, prototype composite scoring on ground truth
- Upworthy exploration notebook: understand structure, plan eval usage

**Phase 1 gate:** AdFlex client works E2E with resume across all 5 platforms. Supplementary scrapers functional. IRA/Upworthy loaded and explored. `make lint && make typecheck` passes.

---

## Phase 2: Dataset Construction at Scale (Weeks 5–9)

### Week 5 — Full-scale scraping + composite scorer
- Launch AdFlex scraping: 20K–30K ads across 11 verticals using continuous collection with cursor checkpointing (run → pause → assess → resume)
- Build `scoring/composite_scorer.py`: longevity (log-scaled), engagement volume (log-scaled), engagement velocity, re-delivery bonus, advertiser persistence, early death penalty. Weighted combination from `configs/scoring.yaml`
- Build `scoring/tier_assigner.py`: score distribution → top 20% high, middle 50% medium, bottom 30% low
- Score all collected data. Analyze distributions in notebook

### Week 6 — IRA proxy validation + score calibration
- Build `evaluation/proxy_validation.py`: compute composite scores on IRA data, correlate with real spend/impressions/clicks (Spearman, Pearson, precision@K)
- Calibrate scoring weights based on IRA correlations
- `scoring/benchmark_calibrator.py`: sanity-check tiers against WordStream/LocaliQ industry benchmarks
- **This is a thesis contribution for RQ3** — document thoroughly

### Week 7 — Training data construction (Formats 1–3)
> **Post-2026-04:** Only Format 2 (copywriting) was implemented; Formats 1 and 3 are deferred and their code is archived.

- Build `construction/clusterer.py`: group ads by (advertiser, vertical, platform); pre-compute per-format manifests
- Format 1 — **Positioning** (`formats/positioning.py`): 3–5 high-score ads grouped by vertical → angle / messaging selection. Target: 2K *(archived)*
- Format 2 — **Copywriting** (`formats/copywriting/`): single high-score ad as reference → reverse-engineer the brief that produced it (Humpback backtranslation). Target: 3K *(active — target updated to 3K)*
- Format 3 — **Diagnostic** (`formats/diagnostic.py`): single ad (any score) → performance critique / root-cause analysis. Target: 1.5K *(archived)*
- Use Claude Haiku for bulk, Sonnet for quality-critical. Manual review 30 per format

### Week 8 — Training data construction (Formats 4–5)
> **Post-2026-04:** These formats are deferred; code is archived under `archive/construction/formats/`.

- Format 4 — **Optimization** (`formats/optimization.py`): high/low ad pair (same advertiser or vertical) → conservative + aggressive rewrites of the weak ad. Target: 2K *(archived)*
- Format 5 — **Channel/format fit** (`formats/channel_format_fit.py`): cross-platform advertiser cluster → primary/secondary platform + creative-format recommendation. Target: 1K *(archived)*
- Build shared construction infrastructure: `bundle.py` (teacher-bundle assembly), `dice.py` (style/persona/seed/evol/difficulty/multi-turn rolls), `ingestion.py` (response parsing), `batch/` (async batch path for cost efficiency)

### Week 9 — Quality filtering + dataset assembly
- Build `construction/quality_filter.py`: min length, no empty fields, language detection, deduplication (cosine similarity), LLM quality rating on 10% sample
- Build `construction/dataset_builder.py`: combine formats → HF Dataset with metadata columns (source, format_type, vertical, tier, construction_model). 85/7.5/7.5 train/val/test split, stratified
- Statistics notebook: per-format, per-vertical, per-tier counts and distributions
- If < 8K examples after filtering, generate targeted synthetic fill

**Phase 2 gate:** 8K–15K training examples. No format < 1K. No vertical < 500. Quality filter pass rate > 80%. Dataset saved as HF Dataset in `data/final/`.

---

## Phase 3: Fine-Tuning + RAG Integration (Weeks 10–13)

### Week 10 — GPU setup + base model bake-off
- Provision **RTX 4090** ($0.40-0.80/hr) or A100 40GB on Vast.ai/RunPod
- Install: CUDA, PyTorch, **Unsloth**, transformers, peft, bitsandbytes, trl
- Build `training/data_loader.py`: load HF Dataset, format to chat template via `apply_chat_template()`, multi-task mixing, tokenization
- **2-day bake-off:** Fine-tune **Qwen3-8B** vs **LLaMA 3.1 8B** on 500 examples (1 epoch each). Evaluate with quick LLM judge on 50 held-out. Select winner. Also verify function-calling still works post-fine-tune
- Create `configs/training.yaml`: base_model, lora_r=32, lora_alpha=16, lr=2e-4, epochs=3, batch_size=4, grad_accum=4, max_seq=4096, bf16=true, use_rslora=true, packing=true, use_liger_kernel=true, assistant_only_loss=true

### Week 11 — Full QLoRA fine-tuning (Run 1)
- Build `training/train.py`: **Unsloth + SFTTrainer** + QLoRA + 4-bit NF4 + wandb/Trackio logging + checkpointing
- Include ~20% tool-use examples in training mix to preserve function-calling ability
- Use `init_lora_weights="pissa"` for better initialization
- Full training: all examples, 3 epochs (~1–3h on RTX 4090)
- Build `training/merge_adapter.py`: merge LoRA into base model for inference
- **Set up vLLM** for serving with guided JSON generation (critical for reliable tool calls)
- Manual eval: 20 diverse test inputs across all 5 pattern-skill tasks + 10 tool-calling tests. Note failure modes

### Week 12 — Hyperparameter sweep + data ablations
- Build `training/sweep.py`: orchestrate multiple runs
- Sweep: lr [1e-4, 2e-4, 5e-4], LoRA rank [32, 64, 128], epochs [2, 3, 5], data mix ratios
- **Data composition ablation** (thesis-critical): train without Source 2 (knowledge corpus), train without synthetic data. Evaluate each
- Select best config, train final model

### Week 13 — Agent tools + integration test
- Build `tools/web_search.py`: Tavily + Serper implementations, returns structured `SearchResult` list
- Build `tools/url_scraper.py`: httpx + trafilatura → structured page content extraction
- Build `tools/registry.py`: tool definitions as JSON Schema (matches model's native function-calling format)
- Build `tools/executor.py`: takes a tool call from model output → dispatches to correct tool → returns formatted result
- **Integration test:** Give fine-tuned model a system prompt with tool definitions, a marketing task, and verify it (a) decides to call tools using its native function-calling, (b) produces better output with tool results than without

**Phase 3 gate:** Fine-tuned model produces quality outputs across all 6 tasks. Best hyperparams documented. Tools are callable. Model's native function-calling works with the defined tools. All 4 ablation configs ready.

---

## Phase 4: Agent Loop + Pipeline Integration (Weeks 14–16)

### Week 14 — Agent loop + output schemas
- Build `agent/loop.py` — **core agent loop**:
  ```
  while not done and steps < max_steps:
      response = model.generate(messages, tools=tool_definitions)
      if response.has_tool_calls:
          results = executor.run(response.tool_calls)
          messages.append(assistant=response, tool_results=results)
      else:
          done = True  # model produced final answer
  ```
  - Configurable `max_steps` (default 5) to prevent infinite loops
  - Handles: model calling multiple tools in one turn, malformed tool calls (retry), tool execution errors
- Build `agent/output_schema.py`: `CampaignOutput` Pydantic model (executive summary, audience profile, channel strategy, messaging framework, ad copy variants, competitive context)
- Build `agent/prompts.py`: system prompts that (a) define tools, (b) instruct model to use them for gathering competitive intel and analyzing product pages, (c) specify structured output format
- Test with 10 diverse URLs: verify the model autonomously scrapes the page, searches for competitors, and synthesizes a campaign

### Week 15 — Multi-task agent support + prompt engineering
- Extend agent loop to all task types: campaign gen, critique, comparison, strategic Q&A, copy gen, channel reasoning
- Each task type gets a tailored system prompt guiding tool-use strategy (e.g., critique tasks may not need web search; campaign gen should always search for competitors)
- Iterate prompts against 5 inputs per task type — tune for tool-use quality and output quality
- Build `scripts/demo.py`: interactive CLI with rich output showing agent reasoning + tool calls

### Week 16 — Edge cases, configs, E2E testing
- Handle: blocked pages, empty search results, model refusing to use tools, model calling tools excessively, invalid structured output (retry with correction prompt)
- Build `tests/test_pipeline.py`: unit tests (mocked tools) + integration (10 URLs through full agent loop)
- Prepare all 4 ablation configs:
  - **Config A:** GPT-4o as agent with same tools + prompts (frontier baseline)
  - **Config B:** Base 7B (not fine-tuned) as agent with tools (tool-use only baseline)
  - **Config C:** Fine-tuned 7B with NO tools (domain knowledge only, single-shot generation)
  - **Config D:** Fine-tuned 7B as agent WITH tools (**Draper.ai full system**)
- Profile E2E latency per config (target < 90s including tool calls)

**Phase 4 gate:** All 4 configs produce valid structured outputs for all 6 tasks. Agent loop handles edge cases. Demo CLI shows reasoning + tool calls. Tests pass.

---

## Phase 5: Evaluation (Weeks 17–21)

### Week 17 — LLM-as-judge framework
- Build `evaluation/test_scenarios.py`: 50–100 scenarios (20 campaign gen, 15 critique, 15 comparative, 15 strategic Q&A, 10 copy gen, 10 channel reasoning)
- Set up **PromptFoo or DeepEval** as eval harness (handles running evals, storing results, comparing runs)
- Build `evaluation/judge.py`: **GPT-4o** (primary) + **Gemini** (secondary) pairwise evaluation, 5 dimensions (strategic relevance, creativity, actionability, channel appropriateness, predicted performance), position swap for each pair, chain-of-thought before scoring
- Run all pairwise evaluations: 6 pairs × scenarios × 2 orderings

### Week 18 — Analysis + multi-task + agent behavior evaluation
- Build `evaluation/metrics.py`: win rates, Elo ratings, bootstrap CIs, per-task breakdowns
- Test hypotheses: D > A (beats frontier), D > C (tools add value), D > B (fine-tuning adds value), C > B (relative contribution)
- Multi-task analysis: per-task win rates, task interference detection
- **Agent behavior analysis** (new): across all configs that use tools (A, B, D):
  - Avg tool calls per task, tool call relevance (did the search query make sense?), information utilization (did the model actually use what it retrieved?)
  - Does fine-tuning (D vs B) change tool-use *patterns* even without trajectory training? (interesting finding either way)

### Week 19 — Proxy label validation (RQ3)
- Complete proxy validation analysis (correlations, calibrated vs uncalibrated)
- Proxy label ablation: full model vs. model trained without proxy-labeled data → judge evaluation
- Document RQ3 findings

### Week 20 — Real-world campaign deployment
- Select 2–3 scenarios, deploy from all 4 configs ($25–50 each, 7-day min)
- Total: 8–12 campaigns, $200–600 budget
- Build `evaluation/headline_eval.py`: Upworthy benchmark evaluation
- Monitor campaigns daily

### Week 21 — Results collection + synthesis
- Collect campaign metrics: impressions, CTR, CPC
- Does config ranking match judge ranking?
- Create `notebooks/05_eval_results.ipynb`: all thesis figures/tables (win rate heatmaps, Elo rankings, campaign performance charts)

**Phase 5 gate:** All 3 RQs answered with evidence. Figures and tables ready for thesis.

---

## Phase 6: Thesis Writing (Weeks 22–26)

- Weeks 22–23: Draft Ch 4 (Methodology) + Ch 5 (Results)
- Weeks 24–25: Draft Ch 1 (Intro), Ch 3 (Architecture), Ch 6 (Discussion), Ch 7 (Conclusion). Full revision
- Week 26: Final proofread + submit

---

## Key Dependencies & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| AdFlex credit budget overrun | Insufficient data if credits/call higher than estimated | Monitor credit balance after each collection loop; adjust platform weights if needed |
| LLM construction cost overrun | Budget blow-out | Haiku for bulk, Sonnet for critical; batch API |
| GPU availability | Phase 3 delay | RTX 4090 on Vast.ai/RunPod (~$0.40-0.80/hr); total cost ~$2-10 |
| Overfitting on small dataset | Poor model quality | Early stopping, dropout, aggressive augmentation |
| Campaign policy rejections | Phase 5 incomplete | Manual review before submission; backup scenarios |
| Proxy scores don't correlate | Weak RQ3 answer | Valid negative finding; document honestly |
| Fine-tuning degrades tool-use | Model stops calling tools or calls them poorly after QLoRA | Mix ~20% tool-use examples in training data; use `init_lora_weights="pissa"`; serve via vLLM with guided JSON generation; if still degraded, reduce LoRA rank or try CorDA KPM mode |
| AdFlex platform engagement gaps | Some platforms (Reddit) have weak engagement data | Focus budget on high-engagement platforms (Facebook, TikTok); use platform-specific filters and orderings to maximize data quality |

## Verification Strategy

Each phase gate requires:
1. **Automated checks**: linting, type checking, tests pass
2. **Data quality**: manual review of samples, distribution analysis in notebooks
3. **Functional demos**: E2E pipeline runs, outputs are coherent
4. **Documentation**: configs committed, decisions recorded, notebook analysis saved
