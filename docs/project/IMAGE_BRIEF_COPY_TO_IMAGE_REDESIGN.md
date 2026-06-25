# Image-Brief Skill — Copy→Image Brief Redesign

**Status:** Training side IMPLEMENTED 2026-06-01 (smoke pending) — frontend wiring deferred
**Date:** 2026-06-01
**Owner:** @oduwairi
**Scope decision:** Full reframe (dedicated image-brief brief; drop the research→copy bridge)

**Locked decisions (§10):** (1) no explicit `copy_on_creative` field — implicit;
(2) `ad_copy` is the copywriting skill's exact platform-labeled output (one string);
(3) `bridge` deleted entirely; (4) `product` stays FULL (essential per-skill context).

---

## 1. TL;DR

The `image_brief` skill currently conditions the writer on a **research→copy
brief** (`task + product + bridge + platform`) reverse-engineered from the ad,
and it **never shows the writer the finished ad copy**. But the image skill is
always invoked *after* copywriting — the copy already exists. We are training
(and running) the visual step on the wrong inputs.

This doc proposes a dedicated image-brief brief whose conditioning is the
**finished copy + a slim product anchor + a copy→image creative direction**.
The `bridge` (angle / buyer_pain / positioning / target_audience) — a device for
narrowing *copy* space — is dropped and replaced by a `creative` block (a
`CreativeDirection`) that narrows *visual* space. The deliverable (`<think>` +
`<image_brief>` prose re-registered from the literal VLM caption) is unchanged.

---

## 2. Current contract (what's wrong)

### 2.1 Training

The image-brief teacher reuses the copywriting `Brief` verbatim
(`src/draper/construction_v2/teacher/image_brief_single_pass.py:11-13`;
`schemas/brief.py`). Its grounding contract explicitly:

- **Excludes the copy from the student.** The `<brief>` fields are derived from
  the copy + metadata, but the copy itself is not part of the brief the student
  conditions on (`image_brief_single_pass.py:64-72`). At inference the student
  sees only `product + bridge + platform`.
- **Forbids the bridge from referencing the copy.** `angle / buyer_pain /
  positioning / target_audience` "must NEVER quote or paraphrase the ad's copy"
  (`:186-188`), enforced at ingest by the 5-gram leak guard
  (`single_pass.forbid_ngram_overlap: 5`).

So the student learns `brief(product + bridge) → image_brief`, with the copy
discarded by construction.

### 2.2 Inference

The frontend already enforces "image after copy": `draft_campaign` (v2) returns
copy only and defers the visual
(`frontend/lib/agent/writer/v2/draft-campaign.ts:54-57`); the visual brief is a
separate `ask_draper(mode: "visual_brief")` call
(`writer/v2/ask-draper.ts:73-102`). At that point the finished copy sits in
`ctx.creativeProvenance.lastDraperProse` and the structured copy is available
from the assembled `CampaignBody` — **but neither is passed to Draper.** The
visual-brief call rebuilds the same `product + bridge` brief
(`brief-schema.ts:190-216`). Training and inference are consistent with each
other, and consistently wrong.

### 2.3 Why the bridge is the wrong device here

`CONSTRUCTION_V2_ARCHITECTURE.md:88` states the bridge's job: without it,
`product → ad` samples from a near-uniform prior; the bridge narrows the *copy*
space enough to make `brief → copy` learnable. That is a research→copy device.

Once the copy exists, that narrowing is **already realized in the copy's
words**. Re-deriving angle/buyer_pain to condition the *image* is redundant and
lossy. The image skill needs a different narrowing device — copy→visual
treatment. The architecture doc even anticipates this (`:832`): "bridge fields
become format-specific... different bridge axes than ad copy." This redesign is
the documented extension path.

---

## 3. Evidence: copy and visual are coupled

In the real winning creatives, the on-image text is a **transformed compression
of the campaign message** — related to the copy but not identical (from
`data/constructed_v2_image/smoke/_audit_shared/single_pass_comparison_literal.md`):

| ad | copy (headline) | on-image text in real creative |
|----|-----------------|-------------------------------|
| `147739` | "The best app to improve your language skills after Duolingo" | **GET FLUENT FASTER / PLAY CLOZEMASTER!** |
| `29284` | "Make it at home. Save 5 bucks. Everytime." | **BUBBLE MILK TEA / FOR LESS THAN A BUCK** |
| `1747094` (Target console) | "Woven Drawer Console Table" + product description | *no body copy on creative* — only a creator handle + logo |

Two lessons:

1. **The copy is the strongest predictor of the visual** (especially the
   on-image text), and the model can only learn the copy→on-image-text mapping
   if it sees the copy alongside the caption-derived deliverable. The bridge
   forces it to hallucinate which words land on the creative.
2. **The mapping is creative, not literal** (the on-image text is a *new* hook,
   not a verbatim headline) — and **not every winner bakes copy in** (the Target
   table has none). So the lesson is *learn the copy→on-creative-text
   relationship*, never "dump the headline onto the image." Both require the
   copy present as conditioning.

---

## 4. Proposed redesign

### 4.1 New image-brief brief (`<brief>` JSON)

Deliverable unchanged. Only the conditioning changes:

```jsonc
{
  "task": "Give me the creative/image brief for the reddit ad below.",
  "platform": "reddit",        // INJECTED at ingest from the source ad, never authored

  // INJECTED at ingest — what the ad is trying to DO. Shapes the creative
  // archetype even when the copy is subtle. Teacher-authored (reverse-engineered).
  "objective": "promo_offer",  // awareness | promo_offer | launch | social_proof | conversion

  // NEW — the finished ad copy, VERBATIM, in the SAME platform-labeled form the
  // copywriting skill EMITS (render_labeled_ad). A single STRING, not an object:
  // image-skill input == copy-skill output, byte for byte. INJECTED at ingest.
  "ad_copy": "Headline: Get fluent faster — play Clozemaster!\nCTA: Install",

  // FULL product object — UNCHANGED from the copywriting brief. Essential
  // per-skill background context for visual subject choice. tone_signals REQUIRED.
  "product": {
    "name": "Clozemaster",
    "description": "...",
    "category": "language-learning game",
    "key_features": ["..."],
    "unique_selling_points": ["..."],
    "price_info": null,
    "tone_signals": ["bold", "playful", "retro-gaming"],   // REQUIRED non-empty
    "category_context": "...",
    "proof_points": [],
    "offer": null
  },

  // NEW — replaces `bridge`. The one nested block grouping the canvas + the two
  // bridges (a `CreativeDirection`). Carries WHAT the creative must look like and
  // contain, never HOW it is composed (layout/framing/lighting stay the deliverable).
  "creative": {
    "orientation": "portrait",                 // canvas — INJECTED at ingest from platform
    "brand_guidelines": "...",                 // REQUIRED — STYLE bridge (recurring visual identity)
    "on_creative_text": ["GET FLUENT FASTER"], // CONTENT bridge (text), default []
    "key_facts": ["a built-in vocabulary game"] // CONTENT bridge (facts), default []
  }
}
```

**Field rationale**

- `ad_copy` — the whole point, and the coherence win: it is the **copywriting
  skill's exact deliverable** (the platform-labeled copy string from
  `render_labeled_ad` / `platform_labels.py`), so copy-skill output feeds
  image-skill input byte for byte. A single string, not a structured object.
  Sourced at training from the source ad's labeled render; at inference from the
  prior `draft_campaign`'s labeled copy. (Named `ad_copy`, not `copy`, to avoid
  shadowing pydantic's `BaseModel.copy`.)
- `objective` — REQUIRED. What the ad is trying to DO (awareness / promo_offer /
  launch / social_proof / conversion). The purpose shapes the creative archetype
  even when it isn't obvious from the finished copy. Teacher-authored.
- `product` — FULL, unchanged from the copywriting brief. The copy doesn't
  isolate what the thing physically *is* ("Get fluent faster" never says "app");
  `category` / `category_context` / `name` / `key_features` ground visual
  subject choice, and the structured facts are essential per-skill context.
  `tone_signals` stays REQUIRED — the visual register anchor.
- `creative` — the `CreativeDirection` block that replaces `bridge`. It groups
  the canvas (`orientation`) with the two narrowing bridges below.
- `creative.orientation` — the canvas the creative must fill (`square` /
  `landscape` / `portrait`). Derived from platform; INJECTED at ingest via
  `aspect_ratio_for_platform`, never authored by the teacher.
- `creative.brand_guidelines` — REQUIRED. The STYLE bridge: the brand's recurring
  visual identity / feel (aesthetic register + art-style/medium + type feel).
  Keeps the caption reconstructable from the brief at the reusable-brand level
  (the role the bridge played for copy). Teacher-authored; never this ad's
  composition, never world knowledge.
- `creative.on_creative_text` — CONTENT bridge (text). Verbatim strings burned
  into the creative that are NOT part of the ad copy (overlay headlines, button
  labels). Default `[]` when the only text in the creative is the copy.
  Teacher-authored from the caption.
- `creative.key_facts` — CONTENT bridge (facts). Load-bearing content facts the
  creative must convey that the copy + product don't supply (named entities, real
  counts, concrete claims), each stated the way a founder briefing a designer
  would — a fact, never the visual realization ("a 28-day curriculum covering ~28
  named AI tools", never "a central green square with the OpenAI logo"; layout/
  shape/color/framing/placement stay the deliverable's job). Default `[]` when
  copy + product already supply the content. Teacher-authored from the caption.
- **No `copy_on_creative` field (Q1 locked).** Which copy becomes on-image text
  is left to the `<image_brief>` deliverable, learned implicitly from (copy in,
  caption-derived deliverable out). Add the field later only if the implicit
  signal proves weak.

### 4.2 What's dropped / kept

| element | before | after |
|---------|--------|-------|
| `task` | "image brief for our X ad" | references the campaign ("…for the ad below") |
| `objective` | — | **added** — awareness / promo_offer / launch / social_proof / conversion (teacher-authored) |
| `product` | full (10 fields) | **full (unchanged)** — essential per-skill context |
| `bridge` | angle / buyer_pain / positioning / audience | **dropped** |
| `ad_copy` | — | **added** — verbatim, platform-labeled string (= copy-skill output), injected at ingest |
| `creative` (`CreativeDirection`) | — | **added** — `orientation` (injected) + `brand_guidelines` (required STYLE bridge) + `on_creative_text` (default `[]`) + `key_facts` (default `[]`) |
| `<think>` | reasons from brief only | reasons from brief **incl. copy** |
| `<image_brief>` deliverable | re-register caption observational→directive | **unchanged** |
| VLM captioning pipeline | literal caption is supervision target | **unchanged** |

---

## 5. Teacher contract changes

`IMAGE_BRIEF_TEACHER_SYSTEM` (`image_brief_single_pass.py:43-200`) is rewritten:

1. **Inputs shown to the teacher:** ad metadata + **verbatim copy** + literal
   VLM caption. (Copy is already in `SourceAd`; the user message builder
   `build_image_brief_user_message` adds a "## Ad copy" block.)
2. **Grounding contract flips for `ad_copy`:** the copy is now a first-class brief
   field carried verbatim — *not* something to keep out. `product` /
   `creative` stay grounded (the CONTENT-bridge lists are `[]` when unsupported);
   world knowledge still forbidden.
3. **`<think>` rule relaxes:** "reason from the brief — which now includes the
   copy." The think may say "the headline promises X, so the frame shows X."
   This is more faithful to how an art director works and removes the awkward
   "never look at the copy" constraint.
4. **Leak guard dropped for this skill.** It guarded copy→copy leakage
   (irrelevant when input *is* the copy and output is a visual). `creative`
   should still carry strategic visual labels, not a copy echo — enforced by
   prompt instruction, not an ngram gate. (`single_pass.forbid_ngram_overlap`
   becomes a no-op / unused for image_brief.)
5. **Worked example updated** to show copy in, and an `<image_brief>` whose
   on-image text is the transformed hook (e.g. Clozemaster), demonstrating the
   non-literal copy→on-creative-text mapping.

---

## 6. Ingest-gate changes

| gate | file | change |
|------|------|--------|
| fidelity (caption overlap ≥30%) | `ingest/image_brief_fidelity.py` (`check_image_brief_fidelity`) | **unchanged** |
| grounding (`<think>` references the brief) | `image_brief_fidelity.py` (`check_image_brief_brief_alignment`) | **updated** — a dedicated check (does NOT delegate to `check_think_grounding`). Keys on `creative.brand_guidelines`: passes iff a content word from `brand_guidelines` surfaces in `<think>`. Failure reason `no_brand_guidelines_ref`. |
| content bridge (NEW) | `image_brief_fidelity.py` (`check_image_brief_content_bridge`) | **added** — verifies the factual content bridge (`on_creative_text` + `key_facts`) is grounded in the caption and consistent with the deliverable. Rejection reasons: `content_bridge_ungrounded` (atom not in the caption), `content_bridge_text_missing_from_deliverable` (an `on_creative_text` string the prose never renders), `content_bridge_text_under_reported` (a quoted on-image string in the prose, not the copy, that no bridge item covers). Counter `IngestStats.content_bridge_failed`; `RejectionRecord` stage `"content_bridge"`. Skipped for copywriting (`content_bridge=None`). |
| platform labels | — | still skipped (`labels=None`) |
| leak guard (5-gram) | `single_pass` config | **dropped** for image_brief (`leak=None`) |

The `SkillGateBundle` shape (`ingest/skills.py`) **grew** to carry the new
routing: it gained `build_brief` (validate the cached brief dict into the skill
model + inject the fields the teacher doesn't author), optional `leak` (`None`
for image_brief; copywriting uses `check_bridge_leak`), and optional
`content_bridge` (set only by image_brief). It now carries nine callables
(`prepare_source_ads`, `build_request`, `parse_response`, `build_brief`,
`fidelity`, `grounding`, `leak`, `labels`, `content_bridge`) plus `name` +
`system_prompt`. For image_brief, the gate chain that actually runs is
fidelity → grounding → content_bridge (leak + labels are `None`, so skipped).

---

## 7. Schema + serializer + contract-test changes

- **Python:** add `ImageBriefInput` (+ the nested `CreativeDirection`) and
  `canonical_image_brief_input_json` to `schemas/image_brief.py` (which already
  houses the deliverable view). `product` reuses the copywriting `BriefProduct`
  verbatim (full, unchanged); `ad_copy` is a plain `str` (the platform-labeled
  copy block, validated non-empty); `bridge` is absent. The copywriting `Brief` +
  `canonical_json` and their contract tests are **untouched**.
- **TypeScript twin:** new `frontend/lib/agent/writer/v2/image-brief-schema.ts`
  + serializer (mirrors `render-brief.ts` canonical-JSON rules). Byte-identical
  to the Python output.
- **Contract tests:** add `tests/contract/` fixtures for the image-brief brief
  canonical JSON (Python ↔ TS).
- **Parser:** `parse_image_brief_response` (`:330-404`) keeps splitting
  `<brief>` / `<think>` / `<image_brief>`; the `<brief>` JSON is now validated
  against `ImageBriefInput` instead of the copywriting `Brief`.

**Training-mix note (verify):** the single fine-tune trains on *both* skills
under the same `STATIC_SYSTEM_PROMPT`, discriminated by `task` + brief shape.
The new shape (presence of `ad_copy` + `creative`, absence of `bridge`)
should *help* the model distinguish image-tasks from copy-tasks. Confirm no
confusion / collision in a small mixed eval after regeneration.

---

## 8. Frontend inference contract (follow-on)

Replace the `ask_draper(mode: "visual_brief")` path with a **copy-aware visual
brief**:

- **Provenance gate (preferred):** promote the visual brief to a dedicated tool
  (e.g. `draft_image_brief`) that **requires `creativeProvenance.draftSucceeded`**
  and pulls the structured copy from the prior `draft_campaign` — mirroring the
  emit-time provenance gate. The orchestrator supplies `objective` + the
  `creative` block's authored fields (+ full product, already built for
  `draft_campaign`); `platform` and `creative.orientation` are derived, not
  authored. It does **not** re-author the copy — the tool reuses the prior
  draft's exact platform-labeled copy as the `ad_copy` string.
- The tool builds `ImageBriefInput` via the new TS serializer and calls the
  writer (single draw, unscored — predictor is a copy model, as today).
- `generate_image` continues to consume the returned prose `<image_brief>`.

(Implementation order: training first, then this wiring. Captured here so the
two stay coherent.)

---

## 9. Regeneration & rollout

1. ✅ **DONE (2026-06-01)** — Rewrite teacher prompt + schema + grounding gate
   + pipeline wiring (training side). Changes:
   - `schemas/image_brief.py` — `ImageBriefInput` + the nested `CreativeDirection`
     + `canonical_image_brief_input_json`; `schemas/brief.py` — generic
     `canonical_dict_json`.
   - `teacher/image_brief_single_pass.py` — new system prompt (copy in, bridge
     out, `<think>` may use copy, `ad_copy` injected not emitted), user message
     shows labeled copy, parser flags a missing `creative` block / strips
     stray copy.
   - `ingest/image_brief_fidelity.py` — grounding (`check_image_brief_brief_alignment`)
     keys on `creative.brand_guidelines`; new `check_image_brief_content_bridge`
     content-bridge gate.
   - `ingest/skills.py` — `SkillGateBundle` gains `build_brief` (validate +
     inject verbatim `ad_copy` + `creative.orientation`), optional `leak`, and
     optional `content_bridge`; image_brief sets `leak=None` + `labels=None` and
     wires `content_bridge=check_image_brief_content_bridge`.
   - `pipeline.py` — `load_briefs` returns raw dicts; ingest builds the brief via
     `bundle.build_brief`, routes leak via `bundle.leak`.
   - `schemas/records.py` — `ExampleRecord.brief` is a canonical dict; builder +
     quality filter render via `canonical_dict_json`.
   - Tests green (181 construction_v2), mypy + ruff clean.
2. ⏳ **NEXT (operator)** — Re-run the **10-ad smoke**
   (`scripts/explore/single_pass_smoke.py` or
   `construct_v2 submit --run-id smoke-copytoimg …` over the image config),
   render with `render_single_pass_md.py`, eyeball the copy→visual coupling.
   *The existing `data/constructed_v2_image/smoke/` artifacts are stale under
   the old contract.* Requires captions parquet + provider API keys (paid).
3. Full image_brief run (`configs/construction_v2_image.yaml`), ingest, filter,
   build.
4. Frontend `draft_image_brief` wiring + TS contract test (deferred).
5. Retrain the v2 fine-tune on the combined (copywriting + new image_brief)
   dataset; redeploy Modal vLLM.

**Config touch points:** `configs/construction_v2_image.yaml` —
`single_pass.forbid_ngram_overlap` becomes unused for this skill (leave or
remove). No other YAML changes anticipated.

---

## 10. Decisions (locked 2026-06-01)

1. **`copy_on_creative` — implicit.** No explicit field. On-image text is
   carried by the `<image_brief>` deliverable and learned from (copy in,
   caption-derived deliverable out). Revisit only if the implicit signal proves
   weak.
2. **`ad_copy` — platform-labeled string.** Exactly the copywriting skill's
   deliverable form (`render_labeled_ad`), a single string. Image-skill input ==
   copy-skill output, byte for byte. (Structured-object option rejected. Named
   `ad_copy`, not `copy`, to avoid shadowing pydantic's `BaseModel.copy`.)
3. **`bridge` — full removal.** No vestigial fields.
4. **`product` — FULL, unchanged** from the copywriting brief. Per-skill
   structured grounding is essential context; `tone_signals` required, all other
   fields null-allowed per the existing grounding contract.

---

## 11. Out of scope

- Video / non-still creative (still deferred; image-capable filter unchanged,
  `source_selector.py:395-407`).
- The captioning pipeline (`captions/`) — literal caption stays the supervision
  target.
- The copywriting skill and its `Brief` / contract tests — untouched.
