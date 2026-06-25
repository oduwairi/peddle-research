# Human Evaluation Study Plan — RQ1 (Pairwise Preference)

**Status:** Locked, awaiting OSF pre-registration
**Owner:** oduwairi@gmail.com
**Created:** 2026-05-19

---

## 1. Research Question

**RQ1 (primary):** Does fine-tuning a domain-specific model on real high-performing ad copy produce output that human readers prefer over a frontier general-purpose model on the same brief?

**RQ1b (secondary, absolute anchor):** How close is the fine-tuned model's output to copy that already won in production (the original real ad the brief was reverse-engineered from)?

**Bonus (LLM-judge calibration):** Do automated LLM judges agree with human preference on the same item-pairs? This validates the LLM-judge results used throughout the rest of the thesis.

---

## 2. Design Summary

| Item | Value |
|---|---|
| Arms | 3 — `C`, `A`, `GOLD` |
| Comparisons (pairs) | 2 — `C vs A`, `C vs GOLD` |
| Briefs sampled | 50, stratified by platform |
| Item-pairs total | 100 (50 briefs × 2 pairs) |
| Raters per item-pair | 5 |
| Total judgments | 500 |
| Respondents needed | ~34 (15 pairs each) |
| Budget | ~$165 (Prolific incl. fees) |
| Build effort | ~600 LOC, ~2 weeks calendar |

---

## 3. Arms

| Arm | Source | Role |
|---|---|---|
| **C** | Fine-tuned Draper (`draper-r16`), Modal vLLM, single-shot, no tools | Our system. Cached at `data/eval/inferences_clean/C/`. |
| **A** | GPT-5.5 single-shot via OpenAI runner | Frontier baseline. Cached at `data/eval/inferences_clean/A/`. |
| **GOLD** | Real high-performing ad copy stored in `Brief.reference_assistant` | Absolute-quality anchor. Synthesized at item-build time via `gold_inference_from_brief()`. |

Excluded from this study: `B` (base Qwen3-8B), `A_pipe` / `B_pipe` / `C_pipe` (agent pipeline). The agent question is a separate RQ, evaluated with LLM judges only — calibrated by the human↔LLM agreement statistics this study produces.

---

## 4. Pairs

| Pair | Primary/Secondary | Question it answers |
|---|---|---|
| **C vs A** | Primary | Does fine-tuning beat the frontier? *(RQ1 headline)* |
| **C vs GOLD** | Secondary | Does Draper match a real winning ad? *(absolute anchor)* |

Dropped from the original plan: **A vs GOLD**. Saves ~$80 with limited marginal value — we can infer roughly where A sits relative to GOLD from existing LLM-judge data.

---

## 5. Sample Size & Power

- **50 briefs** stratified-sampled from the 215-brief held-out test set (`data/final/test/`). Stratify by `platform` (Meta / TikTok / X / Google / Pinterest / Reddit). Fix the sample with a deterministic seed at item-generation time.
- **2 pairs × 50 briefs = 100 item-pairs.**
- **5 raters per item-pair = 500 judgments.**
- **Per-pair power:** 250 judgments → detectable win-rate gap vs 50% is ~6pp at α=0.05, power=0.8. Sufficient to distinguish "wins clearly" from "ties."

Optional bump if budget allows: 80 briefs (~$250) — strengthens per-platform subgroup analysis.

---

## 6. Participants

- **Platform:** Prolific.
- **Screeners:** active social-media user, 18–55, US/UK/CA, has purchased an online-ad-promoted product in the last 6 months.
- **Dose per respondent:** 15 item-pairs (~5–6 min), $1.50 payout.
- **Total respondents:** ~34 (500 / 15, with some over-recruitment for attrition).

Audience is not matched per-brief (e.g., we don't try to recruit skincare buyers for skincare briefs). Disclosed in limitations.

---

## 7. Protocol per Item-Pair

Respondent sees:

1. **The brief** — product description, audience, platform — rendered in the same way the wizard renders briefs.
2. **Ad A and Ad B** — side-by-side cards, no model labels, no source attribution. Cards render the same way `frontend/components/chat/` renders generated campaigns, minus model metadata.
3. **Forced choice:** *"If you saw both of these in your feed, which are you more likely to click?"* — A | B. No tie option (ties recoverable post-hoc via Likert spread).
4. **Likert 1–5 on each ad:** persuasiveness, clarity, on-brand fit.

**Randomization:**
- Position (A=left vs A=right) randomized per item-pair, balanced per respondent.
- Pair-type order randomized within respondent's 15-item batch.

**Quality control:**
- Two attention checks per respondent: one obviously-broken ad ("`asdf asdf asdf`") vs a normal ad — failing either → respondent's data excluded.
- Time-floor: median item-pair time <2s → excluded.
- Time-ceiling: respondent took >30 min total → excluded (likely walked away).

**Blinding:** no model names anywhere in the UI, the URL, or the source HTML.

---

## 8. Implementation

Build the survey as a new route in the existing Next.js app. Reuses Postgres, Drizzle, and the campaign-card components.

### 8.1 Reused infrastructure

- `data/eval/inferences_clean/{A,C}/<example_id>.json` — pre-extracted ad copy, ready to display.
- `Brief.reference_assistant` — GOLD copy.
- `src/draper/evaluation/judge/aggregation.py` — `pair_results_to_dataframe`, `win_rates_table`, `bootstrap_win_rate_ci`, `elo_ratings`. **Model-agnostic** — feeds human verdicts unchanged.
- `src/draper/evaluation/gold.py::gold_inference_from_brief()` — synthesizes a GOLD `Inference` from a Brief.
- `frontend/lib/db/schema.ts` (Drizzle) + Postgres — extend with study tables.
- NextAuth + Postgres scaffolding — bypassed with a token-from-URL flow for Prolific.

### 8.2 New build

**Backend / data:**
1. **Drizzle migration:** add tables
   - `survey_items(id, pair_type, brief_id, brief_json, ad_a_text, ad_a_arm, ad_b_text, ad_b_arm, created_at)`
   - `survey_sessions(id, prolific_pid, started_at, completed_at, excluded_reason)`
   - `survey_responses(id, session_id, item_id, chosen, position_swap, likert_a, likert_b, time_ms, created_at)`
2. **`scripts/eval.py survey-items`** — new subcommand. Stratified-samples 50 briefs from the held-out test set with a fixed seed, builds 100 `(brief, ad_a, ad_b, pair_type)` rows from `inferences_clean/`, writes to `survey_items`. Idempotent.

**Frontend routes:**
3. `/study/[token]?PROLIFIC_PID=…` — entry. Creates session, allocates 15 items with counterbalancing, redirects to `/study/[token]/q/1`.
4. `/study/[token]/q/[n]` — single-pair page. Renders brief + two ad cards + forced choice + 3 Likerts. POSTs to `/api/study/submit`, advances.
5. `/study/[token]/done` — shows Prolific completion code.
6. `/api/study/submit` — writes one `survey_responses` row, returns next item or completion.

**Analysis:**
7. **`scripts/eval.py survey-analyze`** — pulls responses from Postgres, applies exclusion rules, feeds verdicts through existing `aggregation.py` for win-rates and Bradley-Terry/Elo. Adds:
   - Krippendorff's α (new helper, ~30 LOC using `simpledorff` or a manual implementation).
   - Cohen's κ between human majority and each of the 3 LLM judges on the same 100 item-pairs.
   - Per-platform subgroup win rates (collapse to "visual" vs "text-only" if N is thin).
   - Likert means per dimension with paired t-tests, Bonferroni-corrected.
8. **Plots:** win-rate forest plot with CIs; Likert means bar chart; human↔LLM agreement matrix.

---

## 9. Analysis Plan (what goes in the thesis)

**Primary outcome:**
- Per-pair win rate with 95% bootstrap CI (1000 resamples, seed 42).
- Bradley-Terry latent skill score → rank ordering of {A, C, GOLD}.

**Secondary outcomes:**
- Likert means per dimension (persuasiveness, clarity, on-brand fit), paired t-test with Bonferroni correction.
- Krippendorff's α on forced-choice across raters. Interpretation thresholds noted in limitations (α<0.4 = poor; 0.4–0.6 = moderate; >0.6 = substantial).
- Per-platform subgroup win rates (descriptive, no inferential test — underpowered).

**Bonus — LLM-judge calibration:**
- Cohen's κ for (human majority vote) vs (each LLM judge's vote) on the same 100 item-pairs.
- Per-pair agreement rate. Used to justify trusting LLM-judge results elsewhere in the thesis.

---

## 10. Pre-Registration Checklist (OSF)

To submit before any data collection:

- [ ] Arms locked: `{A, C, GOLD}`; pairs: `{C-A, C-GOLD}`
- [ ] N = 50 briefs × 2 pairs × 5 raters = 500 judgments
- [ ] Sampling frame: 215-brief held-out test set, stratified by platform, deterministic seed
- [ ] Primary outcome: per-pair win rate, 95% bootstrap CI
- [ ] Secondary: Likert dims (persuasiveness, clarity, on-brand), Krippendorff α, LLM-judge κ
- [ ] Exclusion: failed attention check OR median item time <2s OR total time >30 min
- [ ] Counterbalancing: position randomized per item-pair, balanced per respondent
- [ ] Analysis script + survey UI committed to a tagged release before launch
- [ ] No interim peeking at results; analysis run once on full dataset

---

## 11. Timeline & Budget

| Day | Deliverable |
|---|---|
| 1–2 | Drizzle migration + `survey-items` script + survey route shell |
| 3 | UI polish, counterbalancing, attention checks |
| 4 | Pilot N=5 with friends; check timing/clarity; fix UX |
| 5 | OSF pre-registration submitted; tag release |
| 6 | Prolific study launched |
| 7–9 | Data collection (Prolific typically fills in 24–48h) |
| 10–11 | Analysis script, plots, thesis write-up section |

**Budget breakdown:**
- 500 judgments × $0.25 ≈ $125
- Prolific platform fee (~33%) ≈ $40
- **Total: ~$165**

---

## 12. Limitations (to disclose in the thesis)

1. **Self-reported click intent ≠ actual CTR.** Marketing research has known this gap for decades. Framing: this is a *relative* comparison under matched conditions, not an absolute CTR prediction.
2. **Audience not matched per brief.** A general Prolific panel judges all platforms/verticals. Brand-fit Likert is the only audience-sensitive measure and may be noisy.
3. **Ad copy shown without visual.** All platforms in the study include images in production; this study evaluates copy only. Cross-modal preference is out of scope.
4. **GOLD survivor bias.** GOLD ads are A/B-test winners — they already beat alternatives in production. A model that ties with GOLD has matched the *winning* sample of human copy, not the average.
5. **Population: Prolific US/UK/CA workers.** Generalizes to a Western online-consumer population, not global.

---

## 13. Related work (for the thesis methodology section)

- Clark et al., *"All That's Human Is Not Gold"* (ACL 2021) — protocol for human eval of NLG.
- van der Lee et al., *"Best practices for the human evaluation of automatically generated text"* (INLG 2019) — methodology checklist.
- Zheng et al., *"Judging LLM-as-a-Judge"* (NeurIPS 2023) — the paper we calibrate our LLM judges against.
- Chatbot Arena / LMSYS (Chiang et al., 2024) — pairwise + Bradley-Terry at scale.
- Louviere & Flynn — Best-Worst Scaling (alternative we considered but rejected for simplicity).
- WMT Direct Assessment (Bojar et al.) — continuous-rating alternative we considered.

---

## 14. Open questions (not blocking)

- Should we collect optional free-text *"why?"* on a subset of judgments for qualitative analysis? Adds ~30s per item but yields quotes for the discussion section. **Default: no, defer to a possible follow-up.**
- IRB / ethics review at the university — confirm whether human-subjects review is required for a paid online preference study with no PII and no sensitive content. **Action: check with advisor.**
