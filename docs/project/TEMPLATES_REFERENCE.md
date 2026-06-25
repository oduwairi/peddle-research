# Training Data Construction — Templates Reference

Flat reference for every prompt template, directive, and rule the
copywriting construction pipeline uses. Covers what teachers actually see
when generating training examples.

> **History — 2026-04 pivot:** Draper.ai previously supported five
> pattern-skill formats. The pivot narrowed the training target to
> creative copywriting only; this reference covers the active pipeline
> only. The 5-format reference is snapshotted at
> `archive/docs/TEMPLATES_REFERENCE_5format.md`.

**Source of truth:** the Python constants in
`src/draper/construction/`. If this doc and the code disagree, the code
wins — regenerate this doc.

## Table of Contents

1. [Bundle structure](#1-bundle-structure)
2. [Backtranslation style rules](#2-backtranslation-style-rules)
3. [Ad-derived context directive](#3-ad-derived-context-directive)
4. [Copywriting template + system prompt](#4-copywriting-template--system-prompt)
5. [Ingestion fidelity checks](#5-ingestion-fidelity-checks)
6. [Quality-filter guards](#6-quality-filter-guards)
7. [Required output tag format](#7-required-output-tag-format)

---

## 1. Bundle structure

Every teacher bundle is assembled in this order by
`bundle.py::build_bundle(BundleContext)`:

```
# Training-example generation — task: copywriting

<preamble: "You are helping generate one high-quality training example...">

## Style rules
<BACKTRANSLATION_STYLE_RULES — see §2>

## Platform framing & source-ad shape
<single ad-derived directive — see §3>

## Source ad (the gold target — preserve its copy in the response)
<the real high-performing ad, rendered verbatim>

## Output format
<required tag structure — see §7>
```

---

## 2. Backtranslation style rules

Copywriting runs backtranslation-only (Humpback, Li et al., ICLR'24).
The rules are defined as `bundle.BACKTRANSLATION_STYLE_RULES`:

```
# Training example — copywriting

You are generating one question/answer training pair for fine-tuning
a copywriter model. The model is being trained to write successful
ads (like the one below) naturally, as part of a normal LLM response.

The real, high-performing ad is pasted below. Write a user/assistant
pair such that, if the student model reproduced this pair at
inference, it would deliver this ad to the user.

## The pair

- **User prompt** — purely factual about the product the user is
  trying to sell. Zero creative knowledge. The user describes what
  the product is, what it does, what it contains, what it comes in —
  the things a product owner knows about their own product. No tone
  guidance, no audience framing, no suggested angles, no phrasing
  copied from the ad.

- **Assistant response** — the assistant answers appropriately with
  the ad copy presented as part of the response, the way any helpful
  LLM would. Present the copy the way a reader would see it. Wrap it
  naturally: a lead-in, the ad, and a substantive rationale (2-4
  paragraphs) explaining why this execution works on this platform
  — grounded in specific, visible details of the ad. Commit to the
  ad; no alternatives.
```

---

## 3. Ad-derived context directive

Copywriting skips persona / seed / evol-operator / difficulty RNG.
`formats/copywriting/dice.py` derives a single provenance axis from the
ad itself:

- **`source_ad_shape`** is inferred from populated copy fields
  (`headline` / `body` / `description` / `cta`) — `has_body` or
  `headline_only`. Carried on `ExampleMetadata` for downstream
  filtering / analytics; not rendered into the teacher prompt (the
  teacher can read the ad's shape off the ad).

The prior `platform_framing` axis was removed after the 2026-04 audit:
the corpus's `platform` is a scraping-source artifact (which AdFlex
endpoint returned the ad), not a creative attribute. Conditioning the
teacher on it trained the student on label noise. Platform-specific
formatting is handled at inference via prompt.

`CopywritingPipeline.render_axes_block` currently returns no directive
text — it is kept as a seam so a future conditioning axis can be
rendered without touching `bundle.py`.

---

## 4. Copywriting template + system prompt

Defined in `formats/copywriting/constructor.py::CopywritingConstructor`.

**System prompt** (the one saved in every `TrainingExample`'s system
message): a concise copywriter-persona prompt. See the constant
`SYSTEM_PROMPT` in the module for the canonical text.

**Example template** (what the teacher sees when asked to produce a
training pair):

```
## Real ad (ground truth — this exact copy is the answer)
{ad_facts}

Write the user prompt as purely factual product information (zero
creative content — no tone, no angle, no phrasing copied from the
ad). Then write the assistant response that delivers this ad
naturally as part of a normal LLM reply: a short lead-in, the ad
exactly as-is, then one to three short paragraphs pointing at
specific moves in this execution. Commit to the ad; no
alternatives.
```

`{ad_facts}` is rendered by `_ad_facts(ad)` — headline / body /
description / CTA as plain prose, without field labels.

---

## 5. Ingestion fidelity checks

`CopywritingPipeline.ingestion_check` enforces backtranslation
fidelity. Defined in `formats/copywriting/ingestion.py`:

1. **Word coverage** — at least `BACKTRANS_MIN_WORD_COVERAGE` of the
   ad's content words must appear in the assistant response. Teachers
   that fabricate copy rather than reproduce the real ad fail this
   check.
2. **Verbatim signature** — a canonical fingerprint of the ad (content
   words in order) must appear in the assistant response. Catches
   paraphrased-but-not-verbatim responses that would pass word coverage
   alone.

Failures are logged with the coverage ratio and returned as
`IngestResult(verbatim_failed=True, error=…)`. The example is not
saved.

---

## 6. Quality-filter guards

`CopywritingPipeline.extra_quality_filters` runs after the shared
filters. Defined in `formats/copywriting/quality_filter.py`:

- **Schema-leak guard** — rejects responses that mention "headline",
  "body copy", "CTA", or similar field labels in prose. The assistant
  should deliver the ad, not describe its anatomy.
- **Ad-centrality guard** — rejects responses that hedge with "this
  ad might be", "here's one possible direction" or similar
  alternative-offering language. Backtranslation commits to the ad.

`min_length_floor` returns 80 chars for backtranslation — down from
the 200-char global default, because backtranslation responses are
naturally tight (a lead-in + the ad + a short rationale).

---

## 7. Required output tag format

The teacher must return these tags in order, with no text before,
between, or after:

```
<user_prompt>[the brief]</user_prompt>

<assistant_response>[the response]</assistant_response>
```

No self-rating, no model-name, no multi-turn follow-up tags. `ingest`
/ `batch-collect` parse these via `bundle.parse_bundle_output`.
