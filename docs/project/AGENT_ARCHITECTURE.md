# Draper Agent Architecture

## 1. The problem this solves

The V1 agent was a single chat call with up to 7 tool steps. Draper drove everything: it scraped URLs, searched the web, and wrote the final ad in the same model pass. This produced generic output — Draper is trained for creative writing, not tool-chain orchestration, and asking it to navigate a multi-step research loop pulled it out of distribution.

The current architecture separates the two concerns that were conflated: **orchestration** (deciding what to research, authoring the brief, calling tools) and **creative writing** (producing the actual ad copy). Each is handled by the model best suited to it.

## 2. The two-role split

```
┌─────────────────────────────────────────────────────────┐
│  USER INPUT                                             │
│  url · uploaded files · product description             │
│  + platform choice (Meta / TikTok / X / Google /        │
│    Pinterest / Reddit)                                  │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (gpt-5.4-mini / general LLM)              │
│                                                         │
│  · Drives the freeform loop                             │
│  · Calls scrape_url, attach_file, web_search,           │
│    exa_similar, generate_image                          │
│  · Authors product facts in founder voice (informed     │
│    by research — not passed as separate fields)         │
│  · Picks angle + audience as craft metadata only        │
│    (never sent to Draper)                               │
│  · Routes creative asks to the right Draper endpoint    │
│  · Inspects diagnostics from draft_campaign / ask_draper │
│  · Calls emit_campaign (full ad), emit_image            │
│    (image-only), or emit_copy (loose copy) to ship      │
│                                                         │
│  For quick replies: streams plain text, no tools.       │
└──────────────┬────────────────────────┬─────────────────┘
               │ draft_campaign         │ ask_draper
               ▼                        ▼
┌──────────────────────────────────────────────────────────┐
│  DRAPER (fine-tuned writer, Modal vLLM)                  │
│                                                          │
│  · Receives a first-person natural-language brief:       │
│    product facts only — no angle, no audience framing,   │
│    no research appended (see "The cardinal rule" below)  │
│  · Writes free-form prose ad copy (+ visual_brief for    │
│    campaigns in v1; v2 draft_campaign: visual_brief      │
│    always null — image brief is a separate               │
│    ask_draper(mode:"visual_brief") call)                 │
│  · Same training-time system prompt for both endpoints   │
│  · Every writer output is uncommitted/internal —         │
│    held in CreativeProvenance, returned as diagnostics   │
│  · emit_* terminators read content verbatim from slots   │
└──────────────┬───────────────────────────────────────────┘
               │ assembled CampaignOutput (draft_campaign path)
               ▼
┌─────────────────────────────────────────────────────────┐
│  CAMPAIGN CARD                                          │
│  (rendered via emit_campaign)                           │
└─────────────────────────────────────────────────────────┘
```

**Why this split?** Draper's creative taste lives in its weights — it was trained on brief → real-world ad copy pairs and is specialized for that task. The orchestrator system prompt only drives operational steps (which tools to call, when to stop researching, how to author the product brief). Putting "what good looks like" coaching in the orchestrator prompt would be redundant and would conflict with what Draper already knows.

### Writer variant dispatch

The active writer implementation is selected per-request by `frontend/lib/agent/writer/index.ts`. `getWriterVariant()` reads the `WRITER_VARIANT` env var; anything other than `"v2"` resolves to `"v1"`. Both `makeDraftCampaignTool` and `makeAskDraperTool` are exported from this module as variant-dispatching factories — callers never import from the variant-specific files directly.

| `WRITER_VARIANT` | Writer model | Brief format | Assistant turn shape |
|---|---|---|---|
| `v1` (default) | v1 fine-tune | Product-only natural language | Free-form prose |
| `v2` | Qwen3-8B (`oduwairi/draper-v2-qwen/merged-bf16`) | Canonical JSON (`product` + `bridge` fields) | `<think>…</think>` + labeled ad copy (`Headline:`, `CTA:`, …) |

The v2 writer's `<think>` block is stripped by `writer/v2/post-process.ts` before field extraction and rendered in the UI via `components/chat/think-block.tsx` + `lib/chat/parse-think-segments.ts`.

### The cardinal rule: Draper sees product-only briefs (v1) or canonical-JSON briefs (v2)

**v1:** Draper was trained on briefs containing **only product facts a founder would write** (`BACKTRANSLATION_STYLE_RULES` in `src/draper/construction/bundle.py:32-93`). The training spec explicitly forbade tone guidance, audience framing, suggested angles, and phrasing copied from the ad.

**v2:** Draper v2 (Qwen3-8B) was trained on **canonical JSON briefs** that include `product` facts plus `bridge` fields (positioning, audience, angle, buyer pain) derived from the source ad. Bridge fields frame strategic intent — they are never quoted copy from the ad.

**The invariant in both variants:** retrieval results (competitor copy, buyer voice, market trends) do NOT become fields passed to Draper. They inform **how the orchestrator authors** the brief — the way a well-prepared founder would describe their product knowing the market.

- Founder voice ("It's an AI marketing agent that replaces a paid-media team for early-stage startups.") → product facts ✓
- Strategist voice ("Lean into the angle that customers are tired of agency burnout.") → creative brief, NOT product facts ✗

### `craft` is metadata, not brief input

`angle` and `audience` live in a `craft: { angle, audience }` object on `draft_campaign` and `emit_campaign`. They are attached to the campaign card for display but bypass the brief renderer entirely — Draper never sees them.

### `research` is a forcing function

Both `draft_campaign` and `ask_draper` require at least one `ResearchItem` (`.min(1)` on the Zod array). The items are NOT rendered into Draper's prompt. They exist to compel the orchestrator to actually call web_search / scrape_url / exa_similar before drafting. `renderResearchAside` (which previously wove research into Draper's prompt as casual asides) was deleted.

### Draper's two endpoints

The orchestrator routes every creative ask through one of two Draper tools — never authors copy itself.

| Tool | Use when | What the orchestrator receives |
|---|---|---|
| `draft_campaign` | Full platform-shaped campaign → always followed by `emit_campaign` | Diagnostics (`DraftDiagnostics`: field lengths, cap violations, collapse, score, preview); campaign body held in `lastCampaignDraft`. v1: `visual_brief` also set in `lastVisualBrief`; v2: `visual_brief` always null — image brief from separate `ask_draper(mode:"visual_brief")` call |
| `ask_draper` | Variants, rewrites, taglines, hooks, slogans, brainstorms | Diagnostics summary (`{ kind:"copy"; snippetCount; score; collapse; preview }`) — copy held in `lastCopyDraft`; rendered only via `emit_copy`. v2 visual-brief mode stores prose in `lastVisualBrief` |

The system prompt (`system-prompt.ts`) contains an explicit routing table mapping ask shapes to the right tool.

## 3. Turn flow

A turn takes one of three shapes. The orchestrator decides first.

### Quick reply
Smalltalk, clarifying questions, no product context. Plain text, no tool calls.

### Sub-campaign creative (`ask_draper` path)

1. **Read the product.** `scrape_url` (`mode: "product"`) or `attach_file`. Once per source.
2. **Gather grounding research.** At least one targeted `web_search` for buyer voice or competitor copy. Use findings to inform the product description — not as a separate brief field.
3. **Ask Draper.** `ask_draper` with a concrete `request`, a well-authored founder-voice `product`, and at least one `research` item (forcing function — not rendered for Draper). Optional `notes` for genuine user constraints (must-not-says, prior copy for rewrites, user-requested tone). The result is a diagnostics summary (score, collapse flag, preview) — the copy is held internally.
4. **Emit the copy.** Call `emit_copy` (zero inputs — reads the copy verbatim from `lastCopyDraft`). The scored "creative proof" sheet renders from `tool-emit_copy` in `message-bubble.tsx`. The orchestrator never relays, re-types, or re-streams the copy — doing so double-prints it and drops per-line scores.

### Full campaign (`draft_campaign` → `emit_campaign` path)

1. **Read the product.** `scrape_url` with `mode: "product"` on the user's URL, or `attach_file` on each attachment ID.
2. **Find buyer voice.** `web_search` for reviews, Reddit threads, G2, forums. Queries target category frustration, not brand positioning.
3. **Read competitors.** `scrape_url` (`mode: "snippet"`) on 2–3 competitor pages for verbatim headlines. Batched in one step where possible.
4. **Expand competitor set (optional).** `exa_similar` on a known URL discovers semantically similar pages (requires `EXA_API_KEY`).
5. **Synthesize.** Pick an angle and audience silently — these become `craft.angle` and `craft.audience` labels for the campaign card, NOT inputs to Draper. Bake the same insight into the `product` description / USPs / category in founder voice.
6. **Draft campaign.** `draft_campaign` with `product` (founder-voice, informed by research), `craft` (angle + audience labels), and optional `notes` (genuine user constraints only). v1: Draper generates prose + `visual_brief`. v2: Draper generates labeled copy only (`visual_brief: null`); call `ask_draper(mode: "visual_brief")` to obtain the image brief.
7. **Generate image.** `generate_image` (gpt-image-2) for visual platforms (Meta, TikTok, X, Pinterest, Reddit). v1: pass Draper's `visual_brief` from draft_campaign verbatim. v2: pass the prose from the `ask_draper(mode: "visual_brief")` call. Skipped for Google.
8. **Self-check.** Orchestrator enumerates every concrete user ask in `requirements_check.user_asks[]` and marks each `satisfied`. If any are false, loops back to step 6 with a sharpened product description.
9. **Emit.** Call `emit_campaign` with only `{ meta, requirements_check }` — no copy fields. The campaign body is read verbatim from `lastCampaignDraft`; over-cap fields are fit deterministically before schema validation (`fitCampaignBody` in `writing/fit-caps.ts`). Turn ends.

Step budget: ~14. Typical full-campaign turns are 6–9 steps. The loop stops on a successful `emit_campaign` or when the ceiling is reached.

### Image-only (`emit_image` path)

When the user asks for an image by itself ("just an image", "a different / more engaging image" — no copy ask), the orchestrator ships the image alone instead of wrapping it in a campaign:

1. **Get a visual brief.** v2: `ask_draper(mode: "visual_brief")` (fold in the brand Visual identity when present). v1: `draft_campaign` — its `visual_brief` is the only brief source; `emit_campaign` is NOT called.
2. **Generate image.** `generate_image` (size by platform). Logo composited automatically; v2 palette/type already in the brief.
3. **Emit.** `emit_image` with a one-line caption. The tool reads the image URL + size off the most-recent `generate_image` call via `CreativeProvenance.lastImage` — the orchestrator passes only the caption, never the URL — and renders a standalone image card (`components/chat/image-card.tsx`). Turn ends. No scorer runs (the predictor is text-only; a bare image has no copy to score).

This path exists because forcing an image-only ask through `draft_campaign`/`emit_campaign` either upsells an unwanted campaign or thrashes when a copy field blocks the emit while the user just wanted a new picture (observed in production). The loop stops on a successful `emit_campaign`, `emit_image` (`successfulEmitImage`), or `emit_copy` (`successfulEmitCopy`) in `freeform.ts`.

## 4. The Draper tools in detail

Both tools call Draper with the same training-time system prompt verbatim:

> "You are an ad copywriter. When a user describes a product or campaign, you write ad copy and a short rationale explaining why the execution works."

No JSON schema mode for either — that pulls Draper out of distribution. Shared brief-rendering primitives live in `brief-rendering.ts` (`renderProductFacts`, `BriefProductSchema`, `ResearchItemSchema`).

### `draft_campaign` (`draft-campaign.ts`)

**v1 path — two model calls internally:**

**Call 1 — Draper writes.** The brief fields are rendered into a first-person natural-language brief: product facts (as a well-informed founder would phrase them) + optional user constraints + platform-specific ask. `craft` and `research` are not rendered. Draper's response is free-form prose including copy fields AND a one-line `visual_brief`.

**Call 2 — Orchestrator extracts.** A `generateObject` call on the orchestrator slots Draper's prose into per-platform Zod schemas. The extraction prompt instructs verbatim slotting — no rewriting. The assembled output is validated against `CampaignBodySchema` before returning.

**v2 path — best-of-N with deterministic parse:**

**Draws (N=4).** Fires 4 parallel Draper draws at temperatures [0.5, 0.7, 0.85, 1.0]. Each draw is scored by the learned predictor; index 1 (second-best — top-quartile pessimism margin) is selected. The image brief is NOT authored here — `visual_brief` is always null; a separate `ask_draper(mode: "visual_brief")` call produces it.

**Parse.** The winner's labeled ad copy (e.g. `Headline: ...`, `CTA: ...`) is parsed deterministically via `parseLabeledCopy` in `writing/v2-label-parser.ts`. Google RSA has no label vocabulary and falls back to an LLM extraction pass.

On success: stores `CampaignDraft` (body + `DraftDiagnostics`) in `creativeProvenance.lastCampaignDraft` and returns the diagnostics to the orchestrator — cap violations, collapse status, score, and a short preview. Over-cap fields are **not** a rejection; `emit_campaign` fits them deterministically via `fitCampaignBody`. On structural failure (not a length issue): returns `{ error: string; violations?: CapViolation[] }`. Built per-request via `makeDraftCampaignTool`.

### `ask_draper` (`ask-draper.ts`)

**Best-of-N sampling.** Fires N=6 parallel Draper draws at temperatures [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]. Each successful draw is scored by the learned predictor in a single batched call. The pick is index 2 of the sorted candidates (third-best — top-tertile boundary). This is a deliberate pessimism margin: at ρ≈0.72 the absolute top-scoring candidate is disproportionately likely to exploit predictor blind spots rather than being genuinely best (Khalaf et al., NeurIPS 2025). Fallback when scoring fails: middle-temperature draw (T=0.7). When all N draws fail, returns `{ error: string }`.

The brief rendered for Draper contains `request` + `product` + optional `notes` — same product-only logic as `draft_campaign`. No extraction pass. On success, the winning prose is stored in `creativeProvenance.lastCopyDraft` (with per-line `snippets[]` and aggregate `score`) and a diagnostics summary is returned to the orchestrator: `{ kind: "copy"; snippetCount; score; collapse: { isCollapsed; reason }; preview }`. The orchestrator inspects the collapse flag (if `isCollapsed`, re-ask for one deliverable at a time), then calls `emit_copy` to ship. **Nothing renders from the `ask_draper` result directly.** `emit_copy` reads `lastCopyDraft` verbatim and renders the scored "creative proof" sheet. The v2 visual-brief mode stores prose in `lastVisualBrief` (single unscored draw, fed to `generate_image`, not shown). Built per-request via `makeAskDraperTool`.

## 5. Emit-rejection recovery

`freeform.ts` classifies `emit_campaign` failures by gate type (`classifyEmitRejection`) and injects one-shot tactical repair advice on the very next step. This is faster than the generic 2-consecutive-failure threshold, which fires too late and causes the model to end the turn instead of retrying.

| Gate | Recovery advice injected |
|---|---|
| `provenance` | Must call `draft_campaign` first |
| `schema` | Fix exactly the listed field errors; re-call `emit_campaign` (don't restructure the rest) |
| `requirements` | Re-call `draft_campaign` with sharpened brief addressing each unmet ask; re-emit. Never end the turn with a plain-text apology |
| `unknown` | Read the error; fix what it names; call `emit_campaign` again |

`emit_campaign`, `emit_image`, and `emit_copy` are excluded from the generic consecutive-failure counter via `GATED_FAILURE_TOOLS` in `trajectory.ts` — their `{ error }` responses are expected retry channels, not tool malfunctions.

**Step-cap fallback with context.** When the loop exhausts the step budget after a `requirements`-class rejection, `buildStepCapFallback` surfaces the specific unmet asks (up to 3) in user-friendly language so the user knows what to address on their next message, rather than receiving a generic "please try again."

## 6. Model roles

| Role | Env vars | Recommended default | Drives |
|---|---|---|---|
| **Writer (Draper)** | `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `MODEL_ID` | Fine-tuned Draper model (Modal vLLM) | `draft_campaign` and `ask_draper` — creative writing only |
| **Orchestrator** | `ORCHESTRATOR_BASE_URL` / `ORCHESTRATOR_API_KEY` / `ORCHESTRATOR_MODEL_ID` | `gpt-5.4-mini` (OpenAI direct) | Freeform loop, all tools, extraction pass inside `draft_campaign` |

Orchestrator falls back to the writer trio if `ORCHESTRATOR_*` is unset.

**Orchestrator model selection.** Default is `gpt-5.4-mini` (OpenAI's purpose-built subagent model). Gemini 2.5 Flash is a viable backup. Avoid Gemini 3 Flash / 3.1 Flash-Lite — an open `thought_signature` bug breaks parallel tool calls, which the research step relies on.

Env getters live in `frontend/lib/env.ts`.

## 7. Tools

All tools live in `frontend/lib/agent/tools/`. The orchestrator has access to:

| Tool | Purpose |
|---|---|
| `scrape_url` | Jina Reader fetch. `mode: "product"` → structured `ProductInfo`; `mode: "snippet"` → raw text excerpt. |
| `attach_file` | Extracts text from uploaded files (PDF, DOCX, TXT, MD). |
| `web_search` | Tavily-backed web search. Requires `TAVILY_API_KEY`. |
| `exa_similar` | Exa.ai `/findSimilar` — finds semantically similar pages from a known URL. Requires `EXA_API_KEY`. Only registered when the key is present. |
| `generate_image` | gpt-image-2 via OpenAI direct. Visual_brief source: v1 = from `draft_campaign`; v2 = from a preceding `ask_draper(mode:"visual_brief")` call. Brand: v1 appends brand palette/fonts style suffix to prompt; v2 (`brandInBrief`) composites brand logo bottom-right via `sharp` (soft-fails). Size: `square` (Meta, Reddit), `landscape` (X), `portrait` (TikTok, Pinterest). Skipped for Google. |
| `ask_draper` | Universal Draper endpoint for sub-campaign creative work. Best-of-N=6, predictor-scored, pessimism-adjusted. Copy stored in `lastCopyDraft`; returns diagnostics summary `{ kind:"copy"; snippetCount; score; collapse; preview }` or `{ error }`. v2 visual-brief mode stores prose in `lastVisualBrief`. Nothing renders from this result directly — follow with `emit_copy`. Built per-request via `makeAskDraperTool`. |
| `draft_campaign` | Full campaign Draper bridge. Stores campaign body in `lastCampaignDraft`; returns `DraftDiagnostics` (cap violations, collapse, score, preview) or `{ error; violations? }` on structural failure. Built per-request via `makeDraftCampaignTool`. |
| `emit_campaign` | Ships the campaign card to the UI. Input: `{ meta, requirements_check }` only — NO copy fields. Reads campaign verbatim from `lastCampaignDraft`; calls `fitCampaignBody` to trim over-cap fields deterministically; then enforces requirements gate; then auto-scores. Returns structured `{ error }` on gate failures (never throws). Built per-request via `makeEmitCampaignTool`. |
| `emit_image` | Image-only terminator. Ships a standalone image (no campaign) for bare image asks. Reads URL + size off `creativeProvenance.lastImage`; orchestrator passes only a one-line `caption`. Gate: rejects if no image was generated this turn. Returns `{ kind: "image"; image_url; caption; size }` or `{ error }` (never throws). No scorer. Built per-request via `makeEmitImageTool`. |
| `emit_copy` | Loose-copy terminator. Ships scored copy lines (variants / hooks / rewrites) as the turn's deliverable (the "creative proof" sheet). Zero orchestrator input — reads prose verbatim from `lastCopyDraft`. Collapse backstop: refuses to render a degenerate draft. Returns `{ kind: "copy"; prose; score; snippets? }` or `{ error }` (never throws). Built per-request via `makeEmitCopyTool`. |
| `score_copy` | Scores copy the orchestrator is weighing itself (e.g. user-pasted lines). 1–32 snippets per call. Do NOT use on `ask_draper` output (held in provenance, shipped via `emit_copy`) or after `draft_campaign` (auto-scored at `emit_campaign`). Built per-request via `makeScoreCopyTool`. |

## 8. `emit_campaign` gates

`emit_campaign` enforces two runtime gates (plus a deterministic fit pass) before schema validation. All return structured `{ error: string }` so the orchestrator can read the failure and retry — none throw.

**Pre-gate — Cap fit.** `fitCampaignBody` runs unconditionally. Optional fields over cap (Meta `description`, X `card_description`, Google `path1`/`path2`) are dropped to `""`; required fields are boundary-truncated. Cap violations can never block emission.

**Gate 1 — Provenance slot.** Rejects if `creativeProvenance.lastCampaignDraft` is null. Prevents the orchestrator from fabricating copy. Only a successful `draft_campaign` call populates the slot.

**Gate 2 — Requirements check.** Requires `requirements_check: { user_asks: string[]; satisfied: boolean[]; notes: string | null }`. If any `satisfied[i]` is false, the gate rejects. Forces the orchestrator to enumerate requirements before emitting and catches "I know it's missing X but I'll emit anyway" patterns. This is the one remaining **soft, bounded** gate — caps and collapse became pure no-gate signals.

## 9. Schema split

Two distinct campaign schemas serve different stages of the pipeline:

| Schema | Used by | Contains |
|---|---|---|
| `CampaignBodySchema` | `draft_campaign` output | Platform fields only — no `requirements_check` |
| `CampaignOutputSchema` | `emit_campaign` input/output | Platform fields + `requirements_check` |

Per-platform body schemas: `MetaCampaignBodySchema`, `TikTokCampaignBodySchema`, `XCampaignBodySchema`, `GoogleCampaignBodySchema`, `PinterestCampaignBodySchema`, `RedditCampaignBodySchema`. All live in `schemas.ts`.

## 10. Platform support

Six platforms: Meta, TikTok, X, Google (Responsive Search Ads), Pinterest (Standard Pin), and Reddit (Promoted Post). `PLATFORM_SPEC` in `platforms.ts` carries per-platform char caps (`hard` = enforced by schema, `soft` = in-feed display targets) and `arrayLimits` for Google RSA.

Per-platform notes:

- **Google RSA:** no visual asset (image generation skipped); requires `sourceUrl` in the brief for `final_url`.
- **Pinterest:** 2:3 portrait pin (`portrait` size). Title is shown in-feed (only first 40 chars visible); description is not shown — Pinterest uses it as an algorithmic relevance signal. Requires `sourceUrl` for `final_url`.
- **Reddit:** 1:1 image (`square` size); single headline. CTA is optional — `"None"` is a sentinel that suppresses the button. Requires `sourceUrl` for `final_url`.

## 11. Warmup architecture

Modal vLLM scales down after 120s idle (`deploy/modal_vllm.py`). The warmup heartbeat keeps it alive.

**Module:** `frontend/lib/agent/writer-warmup.ts`, exported function `maybeWarmWriter(source)`. Synchronous — kicks off a background ping if outside the 60s dedupe window and not already in flight. Returns `WarmupStatus: "fired" | "in-flight" | "recent" | "disabled"`.

**Triggers:**
- NextAuth session callback (`frontend/auth.ts`) — fires on every `/api/auth/session` hit. `SessionProvider` refetches every 90s.
- `app/api/warmup/route.ts` — thin wrapper; can be hit explicitly as a liveness probe.

**Dedupe:** 60s window. The 90s SessionProvider refetch interval plus 60s dedupe means at most one real ping per minute even under heavy UI polling.

## 12. Tool failure UX

`frontend/lib/agent/tool-display.ts` is the single source of truth for user-facing fallback copy.

Tool errors arrive as `{ error: string }` from tool results. Those strings are written for the orchestrator LLM (retry instructions, provider exception details) — not safe to show users directly. Cards call `toolDisplay.<tool>.failure(reason?)` instead:

| Tool | Copy |
|---|---|
| `scrape_url` | "Couldn't read that page." |
| `attach_file` | "Couldn't parse that file." |
| `web_search` | "Search came back empty." / "Search didn't return results." |
| `exa_similar` | "No similar pages found." / "Cross-reference didn't return results." |
| `generate_image` | "visual generation failed — copy still ready" |
| `emit_campaign` (iterating) | "Iterating on the draft…" |
| `emit_image` (iterating) | "Composing the image…" |

## 13. Billing / quota metering

Each chat turn accumulates a **Cost-Equivalent Token (CET)** total written to `usage.tokensUsed` at end-of-turn. 1 CET = $1e-6.

`freeform.ts` creates a `CostMeter` per request inside `withCostMeter(costMeter, ...)` (AsyncLocalStorage). Every async tool dispatch shares the same meter without threading.

**What records into the meter:**

| Source | Rate key |
|---|---|
| `generate_image` | `cetForImage(size)` |
| `web_search` | `cetForTavily(depth)` |
| `exa_similar` | `cetForExa()` |
| `scrape_url` / `attach_file` (Jina fetch) | `cetForJina()` |
| `draft_campaign` (Draper GPU — v1: 1 draw, v2: N=4 parallel draws) | `cetForDraperDraft()` |
| `ask_draper` (per successful draw, N=6 × `cetForDraperDraft()`) | `cetForDraperDraft()` |
| `ask_draper` predictor call (best-of-N scoring) | `cetForScoringPredictorCall()` (~100 CET) |
| `ask_draper` snippet scoring (per call when ≥2 labeled variants detected) | `cetForScoringPredictorCall()` (~100 CET) |
| `score_copy` (scoring predictor) | `cetForScoringPredictorCall()` (~100 CET) |
| `emit_campaign` auto-score (scoring predictor) | `cetForScoringPredictorCall()` (~100 CET) |
| Orchestrator loop tokens | `cetForOrchestratorTokens(in, out)` |

All rates live in `frontend/lib/billing/cost-rates.ts`. Single-file edit to retune from real invoices.

Plan quotas (`frontend/lib/billing/plans.ts`, `ASSUMED_CET_PER_CAMPAIGN = 40_000`):

| Plan | Monthly CET | Approx. drafts | Price |
|---|---|---|---|
| Free | 120,000 | ~3 | $0 |
| Pro | 6,000,000 | ~150 | $20/mo |
| Pro+ | 20,000,000 | ~500 | $49.99/mo |
| Enterprise | Unlimited | Custom | Custom |

## 14. Trajectory logging

Every step is written to `agent_trace` (`stepIndex`, `stage`, `text`, `toolCalls`, `toolResults`, `inputTokens`, `outputTokens`, `modelId`, `latencyMs`). Stage keys: `freeform`, `draft`, `emit`; errors prefixed `error:`. Best-effort — DB failure logs and never blocks streaming.

Inspect with `scripts/diagnostics/inspect_traces.py` (TUI trace inspector, Postgres-backed).

## 15. Robustness

- **Transient HTTP retries** — `fetchWithRetry` in `http.ts` retries once on 408/425/429/5xx and network errors (not timeouts).
- **Repair on consecutive tool failures** — After `toolFailureRepairThreshold` (default 2) consecutive failures on the same tool, `prepareStep` injects a corrective instruction for the next step. Fires once per tool per conversation. `emit_campaign`, `emit_image`, and `emit_copy` are exempt (`GATED_FAILURE_TOOLS`).
- **Per-rejection emit repair** — `classifyEmitRejection` + `buildEmitRepairAdvice` in `freeform.ts` inject gate-specific advice on the step immediately following an `emit_campaign` rejection.
- **Observability log** — `freeform.ts` `onFinish` warns when the orchestrator produced substantial assistant text (>200 chars, `looksLikeCreativeOutput` heuristic) without any Draper call this turn. Not a hard gate — surfaces routing violations for offline review.

## 16. What this is NOT

- Not LangGraph, not Mastra, not a new framework. ~300 lines of TypeScript on top of the AI SDK v6 already in use.
- Not a self-critique loop. There is no critic model in the current architecture; gate logic provides the quality bar.
- Not training-time work. The freeform loop runs on the deployed Draper model; research and brief authoring happen at inference time.

## 17. Entry points and file map

```
frontend/
  app/api/chat/route.ts         — Route Handler → runFreeformAgent()
  app/api/warmup/route.ts       — Thin wrapper around maybeWarmWriter()
  auth.ts                       — NextAuth config; session callback triggers maybeWarmWriter()
  lib/agent/
    freeform.ts                 — runFreeformAgent(): the loop, emit-rejection classifier,
                                  step-cap fallback, observability log
    system-prompt.ts            — buildAgentSystemPrompt() / buildAgentSystemPromptV1() /
                                  buildAgentSystemPromptV2(): routing table, cardinal rule,
                                  brand-context section (projectContext + brandVisualIdentity
                                  for v2), session-context section, tool guidance;
                                  variant-aware (v1/v2)
    suppress-intermediate-text.ts — suppresses partial orchestrator text during tool-call steps
    schemas.ts                  — CampaignBodySchema, CampaignOutputSchema,
                                  per-platform body/emit schemas, RequirementsCheckSchema
    platforms.ts                — PlatformId, PLATFORM_SPEC, constraint helpers
    brief-rendering.ts          — renderProductFacts, BriefProductSchema, ResearchItemSchema
                                  (v1 path only; renderResearchAside deleted)
    tool-display.ts             — user-facing fallback copy for tool failure states
    writer-warmup.ts            — maybeWarmWriter(): shared Modal vLLM warmup module
    provenance.ts               — CreativeProvenance (lastCampaignDraft / lastCopyDraft /
                                  lastVisualBrief / lastImage), CampaignDraft, CopyDraft,
                                  DraftDiagnostics, newCreativeProvenance()
    config.ts                   — tunables (timeouts, step caps, retry thresholds)
    trace-stages.ts             — TRACE_STAGE constants
    http.ts                     — fetchWithRetry, withTimeout
    trajectory.ts               — trace persistence, failure tracking,
                                  GATED_FAILURE_TOOLS (emit_campaign + emit_image + emit_copy)
    writer/
      index.ts                  — getWriterVariant(); makeDraftCampaignTool /
                                  makeAskDraperTool variant-dispatching factories
      v2/
        draft-campaign.ts       — makeDraftCampaignToolV2: best-of-N=4 (temps [0.5,0.7,
                                  0.85,1.0], pick index 1), deterministic label parse
                                  (parseLabeledCopy) + LLM fallback; visual_brief always null
        ask-draper.ts           — makeAskDraperToolV2: mode (copy default | visual_brief);
                                  copy = best-of-N=6, predictor-scored; visual_brief = single
                                  draw T=0.7, unscored
        system-prompt.ts        — training-time system prompt for the v2 writer
        post-process.ts         — stripThink(): removes <think>…</think> before extraction
        render-brief.ts         — serializes brief object to canonical JSON for v2 writer
        brief-schema.ts         — Zod schema for v2 brief wire format (incl. mode field)
    writing/
      ask-draper-result.ts      — AskDraperResult discriminated type (copy | visual_brief | error)
      best-of-n.ts              — runBestOfN(): shared best-of-N engine; scoring optional
                                  (pass null to take first valid draw, e.g. visual_brief mode)
      detect-collapse.ts        — detectCollapse(): CollapseReport (duplicateRatio, reason);
                                  flags degenerate/duplicate draws from over-broad asks
      draft-diagnostics.ts      — buildDraftDiagnostics(): assembles DraftDiagnostics from
                                  body + deliverable + score for return to orchestrator
      errors.ts                 — CapViolation, classifyDraftError, extractCapViolations,
                                  formatCapViolationError (dead — caps no longer gate emission)
      extraction.ts             — assembleCampaign, EXTRACTION_SCHEMA, extractVisualBrief
                                  (v1 LLM extraction path)
      fit-caps.ts               — fitCampaignBody(): deterministic over-cap field truncation /
                                  drop called by emit_campaign; campaignFieldLengths() read-only
      score-snippets.ts         — scoreDeliverableSnippets(): per-snippet predictor scoring
      v2-label-parser.ts        — parseLabeledCopy, assembleV2Campaign (v2 deterministic parse)
    tools/
      scrape-url.ts             — scrape_url
      attach-file.ts            — attach_file
      web-search.ts             — web_search
      exa-similar.ts            — exa_similar
      generate-image.ts         — generate_image
      ask-draper.ts             — v1 makeAskDraperTool (dispatched via writer/index.ts)
      draft-campaign.ts         — v1 makeDraftCampaignTool (dispatched via writer/index.ts)
      emit-campaign.ts          — emit_campaign (makeEmitCampaignTool factory); reads
                                  lastCampaignDraft; fitCampaignBody pre-gate; requirements
                                  gate; auto-scores post-gates
      emit-copy.ts              — emit_copy (makeEmitCopyTool factory); reads lastCopyDraft;
                                  collapse backstop; loose-copy terminator
      emit-image.ts             — emit_image (makeEmitImageTool factory); reads lastImage;
                                  image-only terminator
      score-copy.ts             — score_copy (makeScoreCopyTool factory)
      index.ts                  — agentTools registry, type exports
    scoring/
      predictor-client.ts       — HTTP client for the scoring predictor service
      extract-text.ts           — snippetToPredictorItem(): maps copy + platform/kind
                                  to the ScoreItem wire format
  lib/env.ts                    — typed env getters (two role groups + writerVariant())
  lib/chat/
    parse-think-segments.ts     — parses v2 writer output into { type, content }[] segments
    segment-copy.ts             — segmentCopy(), ScoredSnippet, AD_COPY_LABELS:
                                  dependency-free copy segmenter shared by server
                                  (writing/score-snippets.ts) and client (message-bubble.tsx);
                                  AD_COPY_LABELS must stay byte-aligned with
                                  src/draper/construction_v2/platform_labels.py
  lib/billing/
    cost-rates.ts               — CET rate table (one file to retune from invoices)
    cost-meter.ts               — CostMeter + AsyncLocalStorage context helpers
    plans.ts                    — plan definitions, CET quotas, ASSUMED_CET_PER_CAMPAIGN
    usage.ts                    — getUsage, canSendMessage, recordTokenUsage (DB)
  components/chat/
    think-block.tsx             — renders the <think> reasoning segment from v2 writer output
```
