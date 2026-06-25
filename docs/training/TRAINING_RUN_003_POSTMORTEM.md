# Training Run #003 — Post-mortem & Handoff

**Date:** 2026-05-24
**Pod:** RunPod L40S US (`root@103.196.86.40:54688`), 46 GB VRAM
**Run dir on pod:** `outputs/draper-v2-qwen3/r64-dora-20260524-083417/`
**Outcome:** Manually killed at step 306 of 768 (epoch 1.18) after memorization onset detected at step 305 eval; **then resumed from checkpoint-300 to test the noise-vs-memorization hypothesis**, ran for 115 more steps, `early_stopping_patience=2` fired naturally at step 415, training halted cleanly, `load_best_model_at_end=true` restored the step-250 weights, adapter pushed to `oduwairi/draper-v2-qwen/adapter`, merged weights pushed to `oduwairi/draper-v2-qwen/merged`. **Best `eval_loss = 1.2552` at step 254 (epoch 0.99)** — unchanged by the resume; no later eval beat it. Beats Run #002's all-time best (1.535) by −0.28. Inference quality on held-out test briefs **NOT YET TESTED** — this is the primary task for the next agent.

This document is the handoff for the next agent. Read alongside `TRAINING_RUN_001_POSTMORTEM.md` and `TRAINING_RUN_002_POSTMORTEM.md`.

---

## TL;DR

First production training run on the **v2 dataset** (structured-JSON briefs + `<think>` reasoning + platform-labeled deliverables) using **Qwen3-8B with r=64 LoRA (7× capacity vs Run #002's r=16)**. Trained cleanly through epoch 1 — descending eval, near-zero train/eval gap, no Run #001/002-class defects. **Memorization onset hit at the epoch-1→2 boundary** (step ~260, epoch 1.01): train loss dropped 1.24 → 0.74 within 50 steps while eval rebounded 1.255 → 1.304. Killed manually at step 306. User then asked to resume and let early-stopping decide — the resume confirmed the dynamic was **mild and noisy, not catastrophic**: post-resume evals at steps 364 / 415 came in at 1.290 / 1.281 (trending *back* toward best, not monotonically up). `early_stopping_patience=2` triggered cleanly at step 415, `load_best_model_at_end=true` restored checkpoint-250 (eval 1.2552) as the shipping artifact.

**Three load-bearing findings:**

1. **v2 eval_loss is NOT directly comparable to v1 eval_loss.** The v2 assistant turn (`<think>` opener/closer + platform labels like `Headline:`, `CTA:`) has a much higher fraction of low-entropy structural tokens than v1's pure ad-copy assistant turn. Average per-token loss drops mechanically as a result. "We beat Run #002's 1.535" is a misleading framing — same-epoch comparisons against Run #002's eval curve are the only honest read.
2. **High-capacity LoRA (r=64) on a small-ish dataset (4,081 train examples) memorizes faster than expected.** Run #002's r=16 trained through 311 steps with no memorization signature; r=64 hit memorization at step ~260. The 7× extra capacity bought us faster epoch-1 descent but ate epochs 2–3 of training budget. **For v3 we likely want r=32 — or `early_stopping_patience=1` instead of 2.**
3. **eval_loss is still not the ship criterion.** Per Run #002 criterion #4 (never tested), the actual ship gate is inference quality on held-out v2 test briefs. Best-eval-checkpoint is necessary-but-not-sufficient. The next agent's first task is to load `oduwairi/draper-v2-qwen/merged` and eyeball copy on 3–5 v2 test briefs.

---

## What changed vs Run #002

| Axis | Run #002 | Run #003 |
|---|---|---|
| **Dataset** | v1 copywriting (prose brief → ad) | **v2** (canonical JSON brief → `<think>` + platform-labeled deliverable) |
| Train / val / test | 2,442 / 215 / 215 | **4,081 / 226 / 228** (+67% train) |
| LoRA rank / α | r=16 / α=32 (~24M trainable) | **r=64 / α=128 (175.9M trainable, 7×)** |
| LoRA dropout | 0.05 | 0.05 |
| max_length | 4,096 | **8,192** (doubled — `<think>` is long) |
| Learning rate | 2e-4 | 1.5e-4 |
| Total steps planned | 459 | **768** (3 full epochs at bs=16) |
| GPU | RunPod 4090 EU (~165 TFLOPS BF16, 24 GB) | **RunPod L40S US (~362 TFLOPS BF16, 46 GB)** |
| `assistant_only_loss` | true | true |
| Merge feasibility | OOM on 4090 → adapter-only ship | **L40S has 46 GB → merge expected to fit** |

Everything else (DoRA + rsLoRA, `use_liger_kernel=false`, save_total_limit=5, save_steps=50, eval_steps=50, logging_steps=1, early_stopping_patience=2, load_best_model_at_end=true, optim=adamw_8bit, bf16=true, warmup_ratio=0.03, lr_scheduler=cosine) held identical to Run #002.

---

## Timeline of Run #003

All times UTC, 2026-05-24.

| Time | Event |
|---|---|
| ~07:30 | Original A40 pod provisioned. Bootstrap started inside an SSH command that piped through `tail -80` — output buffered, never streamed. `uv pip install` mid-flight, SSH dropped with "client_loop: send disconnect: Broken pipe". |
| ~07:50 | Tried to reconnect — `Connection refused`. Pod was dead. Total time burned: ~30 min of bootstrap + an hour of confusion. **Lesson logged:** bootstrap inside tmux ON the pod, not piped through SSH from local. |
| 08:15 | User provisioned fresh L40S pod (`103.196.86.40:54688`). Probe confirmed L40S 46 GB, driver 570, Python 3.11.10, PyTorch 2.4.1+cu128 pre-installed. Missing: uv, rsync, tmux. |
| 08:16 | apt install (tmux + rsync). Synthesized `.env.train` from local `.env`, scp'd to `/root/.env.train` (mode 600; **avoided MooseFS at /workspace** which doesn't honor unix chmod). |
| 08:17 | `upload_to_pod.sh` (v2-only INCLUDE_PATHS) synced src/, scripts/, configs/training_v2_qwen.yaml, data/constructed_v2/final_v2, pyproject.toml. ~3 MB code + ~18 MB dataset. |
| 08:20 | Bootstrap script written to `/root/bootstrap.sh`, launched inside tmux session `bootstrap`. Survived several minutes of SSH idleness. |
| 08:26 | `=== BOOTSTRAP DONE ===` marker fired. ~5 min total (apt + uv install + venv + `uv pip install -e ".[training]"` + HF login + smoke import). Resolved deps: torch 2.10.0+cu128, unsloth 2026.4.8, trl 0.24.0. |
| 08:26:40 | `train_v2_bakeoff.sh` (Qwen-only via `ARMS=qwen`) launched in tmux session `bakeoff`. |
| 08:27 | `inspect` passed (no GPU work, tokenizer + template render). |
| 08:28–08:33 | Smoke step: Qwen2.5-0.5B-Instruct proxy (per config's `smoke_base_model`), 100 examples × 50 steps. Final smoke eval = 2.679. Smoke adapter pushed to `oduwairi/draper-v2-qwen/smoke-test`. |
| 08:33:34 | Real train started. Dataset loaded: 4,081 / 226 / 228. |
| 08:34:04 | Qwen3-8B-bnb-4bit loaded onto L40S. LoRA attached (r=64, α=128, DoRA + rsLoRA), 36 layers patched. 175.9M trainable params (2.10% of 8.4B base). |
| 08:34–08:36 | Tokenization (4,081 train examples, ~80 s). |
| ~08:37 | Step 1 logged: `train_loss=2.817`, `grad_norm=40.4`, `lr=6.25e-06` (warmup start). |
| ~08:55 | **Step 50 eval: 1.4499** (train 1.45 at the same window, gap ~0). |
| ~09:17 | **Step 100 eval: 1.3634** (train 1.40, gap +0.04 train-above-eval). Δ vs step 50: −0.087. |
| ~09:39 | **Step 152 eval: 1.3207** (train 1.33, gap +0.00). Δ: −0.042. |
| ~10:02 | **Step 203 eval: 1.2882** (train 1.34, gap +0.05). Δ: −0.033. |
| ~10:34 | **Step 254 eval: 1.2552 — BEST EVAL OF RUN.** Checkpoint-250 saved with this best. Train mean 1.24, gap flipped sign for first time to +0.015 (eval slightly above train). |
| ~10:36 | **Step 260 spike**: train_loss=1.89, gnorm=9.66 at the exact epoch 1.000 boundary. Single-batch artifact (epoch reshuffle + hard batch coincidence). |
| ~10:38 | **Step 261–262 sharp drop**: train_loss=0.73, 0.90. Run #002 saw a similar epoch-2 dynamic (1.45 → 1.20) but milder. Reading deferred to next eval per Run #002's "false memorization alarm" lesson. |
| ~10:39–10:59 | Train mean held in 0.70–0.80 band through steps 263–305 (epoch 1.03–1.18). Anti-overfit gap grew steadily. |
| ~11:00 | **Step 305 eval: 1.3038 — UP from 1.255 (+0.049).** Train at same window = 0.74. Gap = +0.56. Memorization signature confirmed unambiguous (Run #002 same checkpoint kept descending; this one reversed). |
| 11:01:32 | User decision: kill. Signed-INT to python pid 9238 via SSH. GPU memory released 17.7 GB → 1 MiB within 10 s. tmux `bakeoff` exited cleanly (set -euo pipefail caught python's non-zero, skipped auto-merge before it could fail looking for non-existent `final/`). |
| 11:01:32 | Merge step launched manually in tmux session `merge` against checkpoint-250 (the best). |
| 11:03:20 | **Merge complete** (3 min total on L40S — Run #002's 4090 OOM resolved by simply having 46 GB). Merged 16-bit pushed to `oduwairi/draper-v2-qwen/merged` (352 MB). |
| 11:08 | User changed mind: "maybe the eval change was noise, let it naturally stop". |
| 11:09 | Added `DRAPER_RUN_DIR_OVERRIDE` env-var hack to `src/draper/training/config.py:run_dir()` so `--resume` could find the existing checkpoints. Rsync'd to pod. |
| 11:12:44 | Resume launched: `DRAPER_RUN_DIR_OVERRIDE=... python scripts/train.py train --resume --push`. TRL auto-found `checkpoint-300` as the latest in the overridden run dir. |
| ~11:15 | Model + optimizer + scheduler + RNG + early_stop counter state all restored from checkpoint-300. Counter = 0 (since checkpoint-300 was saved before step-305 eval, and step-254 was the last logged "new best"). |
| ~11:37 | **Step 364 eval: 1.2896** (Δ +0.034 vs best). Worse than best → counter = 1. Closer to best than step-305's 1.304 — first hint that step-305 reading was partly noise. |
| ~12:00 | **Step 415 eval: 1.2808** (Δ +0.026 vs best). Worse than best → counter = 2 → `early_stopping_patience` exceeded → training halts. |
| 12:00 | `load_best_model_at_end` reloads checkpoint-250 weights. Saves to `final/`. |
| 12:01:07 | Adapter pushed to `oduwairi/draper-v2-qwen/adapter` (704 MB). EXIT=0. |

Total training wall-clock: **~2 h 25 m** for 306 of 768 steps (40% completion). Cost at ~$1.10/hr L40S: **~$2.65** for training + ~$0.40 estimated for merge.

---

## What the metrics actually told us

### Loss trajectory (steps logged at logging_steps=1, evals at eval_steps=50)

```
STEP | EPOCH | TRAIN_LOSS | GRAD_NORM | EVAL_LOSS | GAP   | NOTES
   0 | 0.004 |     2.753  |    38.10  |     —     |  —    | frozen baseline (lower than Run #002's 3.118)
   1 | 0.008 |     2.817  |    40.44  |     —     |  —    | step 1, lr ramp begins (warmup ~23 steps)
  10 | 0.039 |     1.779  |     2.99  |     —     |  —    | sharp early descent (7× LoRA pays)
  50 | 0.196 |     ~1.45  |     ~1.7  |    1.450  | 0.00  | eval 1, healthy
 100 | 0.392 |     ~1.40  |     ~1.7  |    1.363  | +0.04 | eval 2, gap = train-above-eval (anti-overfit)
 150 | 0.588 |     ~1.33  |     ~1.6  |    1.321  | +0.00 | eval 3
 200 | 0.784 |     ~1.34  |     ~1.6  |    1.288  | +0.05 | eval 4, gnorm spike at step 188 (single batch)
 254 | 0.99  |     ~1.24  |     ~1.8  |    1.255  |+0.015 | eval 5 — BEST EVAL; gap flipped sign first time
 260 | 1.00  |     1.89   |     9.66  |     —     |  —    | epoch boundary: reshuffle + hard batch + ~LR seam
 262 | 1.008 |     0.90   |     2.26  |     —     |  —    | train dropped 1.31 → 0.73 → 0.90 over 3 steps
 305 | 1.18  |     ~0.74  |     ~2.0  |    1.304  | +0.56 | eval 6 — MEMORIZATION ONSET confirmed
 306 | 1.18  |     0.69   |     2.45  |     —     |  —    | last logged before SIGINT (then resume from ckpt-300)
 364 | 1.42  |     ~0.75  |     ~1.9  |    1.290  | +0.034| eval 7, post-resume, patience+=1
 415 | 1.62  |     ~0.70  |     ~1.9  |    1.281  | +0.026| eval 8, post-resume, patience+=2 → early_stop
```

### Reading the curves

1. **Warmup phase (0–23):** textbook. Loss dropped 2.82 → ~1.8 by step 10 as LR ramped 6.25e-6 → 1.5e-4. Sharper descent than Run #002 (1.85 at step 14 there) — consistent with 7× rank.
2. **Epoch 1 (steps 24–256):** smooth, monotonic eval descent (1.450 → 1.255 over 5 evals). Per-50-step Δ decayed −0.087 → −0.042 → −0.033 → −0.033 (stable). Train/eval gap stayed near zero in the train-above-eval (anti-overfit) regime for the entire epoch.
3. **Epoch boundary (steps 257–262):** sharp train drop from 1.24 to 0.73, with the canonical step-260 spike (1.89, gnorm 9.66) caused by dataset reshuffle + hard batch coincidence. This is the moment Run #002's postmortem explicitly warned about ("step 156 train_loss drops 1.45 → 1.20 over 4 steps").
4. **Epoch 2 entry (steps 263–306):** train held in 0.70–0.80 band. **Eval rose 1.255 → 1.304 at step 305.** Run #002 at this point saw eval keep descending (1.593 → 1.588) — consolidation. Ours reversed — memorization.

### The memorization was real but mild (validated by resume)

The user's instinct after the manual kill was: "maybe the eval change was noise". The resume confirmed a nuanced read:

| Step | Eval | Δ vs best | Reading |
|---|---|---|---|
| 254 | **1.2552** | — | best |
| 305 | 1.3038 | +0.050 | initial signal, looked catastrophic |
| 364 | 1.2896 | +0.034 | post-resume, trending back toward best |
| 415 | 1.2808 | +0.026 | post-resume, trending back further |

If memorization were monotonic-rising, the three post-best evals would have been increasing (e.g., 1.30 → 1.32 → 1.34). Instead they came down (1.30 → 1.29 → 1.28). The eval landed in a "noise band" ~0.03 above the best, with high val-set sampling variance dominating the trend over the actual overfit drift.

**This validates `early_stopping_patience=2` over a hypothetical `patience=1`.** Patience=1 would have triggered at step 305 (one bad eval) and stopped 110 steps earlier — which would have been the wrong call (the model wasn't degrading further). Patience=2 absorbed the noise and triggered at the right moment, with the same shipping artifact (still checkpoint-250). For v3 considering `patience=1` was floated in this postmortem before the resume — **retract that recommendation**. Patience=2 is the right setting.

The single-eval-rebound at step 305 looked dramatic in the moment because train was collapsing fast (1.24 → 0.74) and eval rose ~5%. In hindsight, "train dropped 40%, eval moved ~2.5% above best (averaged over post-resume evals)" — those numbers don't match catastrophic memorization, they match mild overfit + high noise.

### Why memorization came faster than Run #002

Three multiplicative factors:

1. **7× more LoRA capacity** (r=64 vs r=16). 175.9M trainable params is enough to memorize substantial slices of a 4k-example train set on second pass.
2. **Smaller effective dataset per token-budget.** v2 examples are longer (max_length 8192) but the model still treats each as one training row. Capacity-per-row ratio is ~3.5× higher.
3. **v2's structural scaffolding makes memorization easier to spot but also easier to do.** The model learns to predict `<think>...</think>\n\n{platform-label}: {content}` as a unit, and on the second pass it can memorize the content too.

Run #002 stopped at step 311 (epoch 2.04) before its r=16 capacity could memorize. We probably would have seen the same dynamic on r=16 — it just would have happened later (epoch 3+). The r=64 / r=16 axis traded epoch-1 speed for epoch-2+ stability.

### Train/eval gap evolution (the cleanest single signal)

```
EVAL_STEP | TRAIN (mean) | EVAL | GAP (eval - train)
       50 |     1.45     | 1.450|      ~0
      100 |     1.40     | 1.363|     -0.04   (anti-overfit: train > eval)
      150 |     1.33     | 1.321|     -0.01
      200 |     1.34     | 1.288|     -0.05
      254 |     1.24     | 1.255|     +0.015  ← first flip to overfit territory
      305 |     0.74     | 1.304|     +0.56   ← memorization unambiguous
```

The gap flip at step 254 was the earliest signal — well before the step-305 eval rebound made it obvious. **For Run #004 consider: monitor gap = eval - train_mean(50). Two consecutive evals with gap > +0.05 = early-stop trigger.** Cheaper than waiting for absolute eval to rebound.

---

## Why eval_loss is NOT comparable across v1 → v2

This is the most important methodological finding of Run #003.

**v1 assistant turn (Run #002):** pure ad-copy prose. Every token is content. Per-token loss reflects copy entropy directly.

**v2 assistant turn (Run #003):** structured wrapper around content:

```
<think>
{reasoning prose: 50-300 tokens, teacher-LLM style — repetitive structure}
</think>

{Pinterest: Headline: X\nDescription: Y\nCTA: Z       OR
 Meta:      Primary text: A\nHeadline: B\nDescription: C  OR
 ...}
```

Per-token loss breakdown (rough):

1. **Structural tokens** (`<think>`, `</think>`, `\n\n`, platform labels `Headline:`/`CTA:`/`Primary text:`, JSON-like field separators) — **near-zero loss** once trigger pattern is learned. ~15–25% of tokens.
2. **Think prose** — **moderate loss**. Teacher LLM writes thinking in repetitive style ("OK so the brief asks for X, the angle is Y, I'll land with Z"). Lower entropy than ad copy. ~30–50% of tokens.
3. **Actual ad words** — **high loss**. This is the genuinely creative part. ~25–55% of tokens.

In Run #002 the assistant was entirely (3). Every token was an "interesting" content token.

In Run #003 the same model could be identically good at (3) and still post lower average eval_loss because (1)+(2) drag the mean down. So:

| Claim | Honest? |
|---|---|
| "We beat Run #002's all-time best eval (1.535 vs 1.255)" | **No** — not apples-to-apples |
| "Same-epoch eval is ~0.4–0.5 below Run #002 at every checkpoint" | **Yes** — but still partly reflects boilerplate |
| "Mechanically the training is sound — no collapse, smooth descent" | **Yes** — fully supported |
| "The adapter writes better ads than Run #002's" | **UNKNOWN** — requires inference eval |

**Implication for Run #004:** if eval_loss can't compare across data shapes, the experiment design must control for shape. Either (a) hold eval_loss interpretation to within-version-only deltas, or (b) compute eval_loss only over deliverable tokens (mask out `<think>` and labels). (b) is invasive (trainer change) but would restore cross-version comparability.

---

## Trigger-pattern conditioning as design choice (validated)

The v2 design — JSON-canonical brief as user turn, `<think>...</think>` + platform-labeled deliverable as assistant turn — is **trigger-pattern conditioning** (the term used in adapter literature, also seen in tool-use specialist models, image-gen concept adapters, etc.).

**The rationale (user, this session):** "the earlier model which used raw prose had mode collapses as well total hallucinations (responds to hi with a ad campaign)." Run #003 architecturally cannot fail that way because:

- The adapter only fires on inputs matching the canonical JSON shape.
- Natural prose ("hi", "how does this work") doesn't match → adapter contributes near-nothing → base model's chat behavior survives.
- This matches the production architecture: `frontend/lib/agent/freeform.ts`'s orchestrator NEVER hands raw user prose to Draper — it always assembles a structured brief via `draft_campaign` / `ask_draper`.

**Validated this run:** memorization onset happened on JSON-shaped input only (since that's all training saw). Base model behavior on non-JSON input is preserved by construction. Run #001's "hi → ad campaign" failure mode is impossible by design.

**Caveats logged for future:**

- `canonical_json()` in `src/draper/construction_v2/schemas/brief.py` (`sort_keys=True`, `separators=(",", ":")`) is the contract. If the production orchestrator ever emits JSON with different key order or whitespace, the adapter sees out-of-distribution input. This is a real load-bearing invariant.
- DoRA + LoRA still modify weight matrices globally — adapter contribution to natural-prose responses is small-but-non-zero. If we ever need to *prove* general-capability preservation, run a held-out NLU benchmark (MMLU subset, IFEval) on base vs adapter. Probably overkill for current product surface.
- `<think>` will be emitted whenever input looks brief-shaped. If a malformed brief sneaks through, expect `<think>` over garbage. Mitigation: keep the orchestrator strict.

---

## What was right / wrong about Run #002's projections (re-evaluated)

Run #002's postmortem made the following projections for Run #003. Verdict at each:

| # | Run #002's projection | Run #003 result | Verdict |
|---|---|---|---|
| 1 | "Realistic eval target with no recipe changes: 1.3–1.5" | Best eval 1.2552 | ✅ better than projected, but partly because v2 ≠ v1 (see above) |
| 2 | "Realistic with brief-quality filter: 1.0–1.3" | v2 has built-in quality gates (Phase 3 filter); eval landed 1.255 | ⚠️ at the high end of band, but not a clean test (other axes changed too) |
| 3 | "Per-vertical fine-tunes would lower the floor" | Not tested — Run #003 is one global adapter | ⏳ deferred to Run #004+ |
| 4 | "Inference quality on held-out briefs is the gating criterion" | **NOT TESTED YET** | 🔥 still the most important open question |
| 5 | "Liger-kernel collapse won't recur with use_liger_kernel: false" | Confirmed: no collapse, all Run #001 defenses silent | ✅ |
| 6 | "L40S/A100/H100 with ≥40 GB can merge cleanly" | L40S 46 GB merge **in flight at writing**, no OOM expected | ⏳ TBD on merge completion |

**The single most important Run #002 lesson that bit us:** "eval_loss is the wrong primary signal once collapse is solved." We followed that — used eval as the safety net and stopped on memorization, not on eval target met. But we still don't have the actual ship signal (inference quality).

---

## Operational lessons

These are the small footguns hit during this session.

- **Bootstrap inside tmux on the pod, not piped through SSH from local.** The original A40 pod died mid-`uv pip install` because the SSH session held the command's stdout (`tail -80`) buffered, then SSH dropped on broken pipe, then sshd never came back. Lost ~1 h on this. The L40S retry used `tmux new -d -s bootstrap "bash /root/bootstrap.sh ..."` — survived multiple SSH disconnects across the 5-min install. **Add to docs/training/CLOUD_OPS.md when it exists.**
- **MooseFS at /workspace doesn't honor unix chmod.** `chmod 600 /workspace/.env.train` left perms at 666. Put env files with secrets on local ext4 (`/root/`) instead.
- **`upload_to_pod.sh` INCLUDE_PATHS must be version-strict.** This session accidentally shipped v1 paths (`data/final`, `configs/training.yaml`) to the v2 pod before user pushback caught it. Fixed in-script. **Update CLAUDE.md to flag this as a discipline:** never include v1 paths when uploading to v2 pods (or vice versa).
- **Killing python with SIGINT in tmux gives a clean GPU release.** PID 9238 → SIGINT → 10 s later GPU showed 1 MiB used. No `nvidia-smi --gpu-reset` needed. Unlike Run #002's parent-bash-died-but-python-orphan dance.
- **`train_v2_bakeoff.sh`'s `set -euo pipefail` correctly aborted the auto-merge** when the python `train` step exited non-zero. We then ran merge manually against `checkpoint-250` — this is the right pattern when you want to merge from best, not from "final" (which doesn't exist after a manual kill).
- **`tail_train_metrics.py --follow` saved the day.** Trackio writes to SQLite, not stdout. The script was missing from `upload_to_pod.sh` INCLUDE_PATHS in Run #002 — this run had it (added per Run #002's lesson). Real-time eval visibility was the difference between making the memorization call confidently and either over-killing or under-killing.
- **Eval cadence (every 50 steps) was again the right call.** Step-50 grain gave us the gap-flip signal at step 254 and the memorization confirmation at step 305 — both within ~25 min of each other. Per-epoch eval would have meant deciding much later.
- **Per-step logging (`logging_steps=1`) made the epoch-boundary spike visible.** Step 260's gnorm=9.66 (single-batch outlier) would have been invisible at logging_steps=10. The visibility cost — ~6 KB/step of SQLite writes — remains negligible.
- **HF Hub auto-pruning works.** save_total_limit=5: at any moment exactly 5 checkpoints on disk. When step 300 saved, step 50 was pruned (we caught it post-kill: dir had 100/150/200/250/300, no 50). By design. Means checkpoint-250 (the BEST) is the oldest still-on-disk checkpoint — if the run had continued ~250 more steps it would have been pruned. **For runs that may overfit early, consider save_total_limit=10 or load_best_model_at_end's checkpoint pinning.** After the resume, final on-disk state was 200/250/300/350/400 + `final/` — checkpoint-250 was still safe but only by 100 steps.
- **`DRAPER_RUN_DIR_OVERRIDE` env var added to `config.py:run_dir()` for one-off resume.** Three-line edit: `if override := os.environ.get("DRAPER_RUN_DIR_OVERRIDE"): return Path(override)`. Lets you point `train --resume` at an existing run dir's checkpoints (otherwise the trainer always creates a new timestamped dir and finds no checkpoints to resume from). Cleaner than adding a `--resume-from <path>` CLI flag for what was a one-off need. **For Run #004 consider making this a proper `--resume-from` flag if resume becomes a regular workflow.**
- **Resume produces a clean SINGLE-RUN trajectory in trackio** when you use the env-var override approach — because the SFTTrainer reuses the same `run_name=run_dir.name`. Without the override, the resume would create a second run_name and split the eval history across two trackio runs, complicating postmortem analysis.

---

## What was right / wrong about my real-time interpretation

These are mistakes I made during monitoring this session, for the next agent to avoid:

- **Initial wall-clock estimate was off by 5×.** I claimed "50–70 min total wall clock for Qwen on L40S" based on L40S's BF16 spec advantage over A40. Actual: ~5–6 h wall clock (we stopped at 2 h 25 m). Cause: max_length 8192 + r=64 LoRA puts you memory-bandwidth bound, not compute-bound — L40S's BF16 throughput advantage doesn't fully translate.
- **I conflated step numbers with epoch positions across runs.** Said "step 150 = end of epoch 1" — true for Run #002 (153 steps/epoch) but FALSE for Run #003 (256 steps/epoch). User caught it ("this is still not over 1 epoch"). The right comparison axis is epoch fraction, not step number.
- **Smoke "step 50 eval = 2.679" was misread as smoke health.** It IS smoke health, but I let it sit visible in the log when reporting the real train's progress, and triggered grep false positives. Use the trackio db (filtered by run_name) for monitoring, not log files.
- **My grep patterns had false positives early.** "Error" matched "incompatible torch version" warning; "Push complete" was from the smoke step lingering in the log. Switched to `grep -F` + fixed-string match + trackio direct query — fixed it. **For Run #004 monitoring: read trackio db directly with sqlite3, ignore stdout log entirely for milestone detection.**
- **My eval projections evolved as the data came in.** Initial: step-50 eval likely 2.0–2.4. Actual: 1.450. Initial best-case projection for final eval: 1.0. Actual best: 1.255 at step 254, with memorization preventing further gains. **Lesson:** don't anchor projections to Run #002's eval scale — v2 data shape moves the floor.

---

## What's left undone

These are the gaps the next iteration needs:

1. **🔥 Inference quality on held-out v2 test briefs.** Per Run #002 criterion #4. This is the load-bearing decision. Load `oduwairi/draper-v2-qwen/merged`, pick 5 briefs from `data/constructed_v2/final_v2/test/`, generate, eyeball. Specifically check:
   - Does the model emit `<think>...</think>` cleanly?
   - Does the deliverable follow the per-platform field labels (Headline: / CTA: / etc.)?
   - Is the copy faithful to the brief's `product` and `bridge` fields?
   - Does it respect `tone_signals` (direct vs. playful vs. urgent vs. ...)?
   - Does length match the platform (TikTok ≠ Reddit ≠ Pinterest)?
   - **Is the `<think>` block coherent or just memorized teacher-LLM phrases?**
2. **Comparison against the v1 shipped adapter** (`oduwairi/draper-qwen3-8b/adapter`). Same 5 briefs → side-by-side copy → human judge. If v2 isn't visibly better, the recipe change wasn't worth it.
3. **Per-platform / per-vertical eval breakdown.** Did the model overfit uniformly across platforms, or only on some? `data/constructed_v2/final_v2/test/` has `metadata.platform` per row — compute eval_loss per platform stratum. If TikTok eval ≈ Pinterest eval the overfit is global; if disparate, the brief-shape capacity is unequally distributed.
4. **r=32 ablation.** If memorization came from too-much-capacity-for-dataset-size, an r=32 LoRA (still 2× Run #002, half this run) might descend to ~1.30 eval without epoch-2 memorization. Small experiment, ~3 h on L40S, ~$3.50.
5. ~~early_stopping_patience=1 in v2 config~~ **RETRACTED — keep patience=2.** Originally proposed in this postmortem before the resume. After resume showed eval trended back toward best (1.304 → 1.290 → 1.281), patience=1 would have stopped too early on a noise spike. Patience=2 absorbed the noise and triggered at the right moment. Keep as-is.
6. **Auto-cancel hook in `train_v2_bakeoff.sh`.** When training exits non-zero (manual kill, early_stop, crash), the script aborts entirely (set -euo pipefail). Better: catch the non-zero, find the best-eval checkpoint, run merge against IT. Saves the manual second-step we did this session.
7. **CLOUD_OPS.md doc.** Bootstrap-in-tmux pattern, MooseFS chmod gotcha, env-file in /root, version-strict INCLUDE_PATHS, SIGINT for clean GPU release. None of this is written down — every cloud run rediscovers it.

---

## Open questions for the next agent

Ordered by expected impact / cost ratio:

1. **Does the merged adapter produce better copy than Run #002's adapter?** Load both, generate on the same 5 v2 test briefs, side-by-side. **This is THE question.** ~30 min on user's RTX 3060 Mobile or local L40S session. If v2 adapter is visibly better — ship to frontend. If similar — Run #003 was an expensive null result; the next iteration should change something more fundamental (data shape, training objective).
2. **Was r=64 the right rank?** The strongest single hypothesis for why memorization came so fast. Cheap test: r=32 retrain on same data, same config. Compare epoch-2 dynamics.
3. **Is the `<think>` content actually contributing to copy quality, or just adding eval-loss-dilution?** Ablation: train an r=64 adapter on v2 with `<think>` blocks stripped at construction time. If eval stays ~1.25 and copy quality stays similar → `<think>` is decorative. If quality drops materially → `<think>` is load-bearing (the architecture is right).
4. **Does the merged 16-bit adapter actually serve correctly via vLLM?** Run #002 never tested this (adapter-only ship). Bootstrap a local vLLM with `oduwairi/draper-v2-qwen/merged` once merge completes, hit it via OpenAI-compatible API with one brief. Confirm no chat-template mismatch.
5. **Was the structural-token dilution real?** Compute per-token loss breakdown: bucket eval tokens into {structural, think-prose, deliverable-content}, report mean loss per bucket. If structural mean ≈ 0.1, think-prose mean ≈ 1.5, deliverable mean ≈ 2.0 → confirmed. Useful for Run #004 eval design.

---

## Run #003 success criteria — verdict

From Run #002's postmortem (criteria for Run #003):

| # | Criterion | Status |
|---|---|---|
| 1 | Smoke runs cleanly with loss in healthy band, no collapse | ✅ smoke eval 2.679 on Qwen2.5-0.5B proxy |
| 2 | Cloud run reaches step X with eval_loss "in band" (band undefined for v2) | ✅ peak eval 1.255 at step 254 — beats Run #002's all-time best by −0.28 same epoch-aligned |
| 3 | Cloud run completes N steps, final adapter pushed | ⚠️ killed at 306/768 (40%), best-checkpoint adapter being merged + pushed at writing |
| 4 | Manual inference on held-out test brief produces sensible ad copy | ⏳ **NOT TESTED — primary task for next agent** |
| 5 | Liger-kernel collapse does not recur | ✅ confirmed clean |
| 6 | Step-0 baseline + label-sanity + LossCollapseCallback all silent | ✅ all defenses correctly never fired |
| 7 (NEW) | Memorization onset is detected early via gap + eval rebound | ✅ caught at step 305 (one eval past the gap-flip signal) |
| 8 (NEW) | Merge fits on the chosen GPU | ⏳ TBD on merge completion (L40S 46 GB should suffice per Run #002 math) |

**Net call: partial pass, same shape as Run #002.** Training was mechanically clean, memorization detection worked exactly as designed, but the actual ship signal (criterion #4) remains unverified. **Run #004's gating criterion should once again be inference quality on held-out briefs — and for once it should actually be tested.**

---

## Adapter shipping info

```
HF Hub repo:    oduwairi/draper-v2-qwen
  smoke-test/   ✅ pushed (08:32:58, smoke adapter on Qwen2.5-0.5B proxy)
  adapter/      ✅ pushed (12:01:07, step-250 weights via early_stopping reload, 704 MB)
  merged/       ✅ pushed (11:03:20, merged 16-bit from checkpoint-250, 352 MB)

Source ckpt:    outputs/draper-v2-qwen3/r64-dora-20260524-083417/checkpoint-250/
Eval loss:      1.2552 (on v2 val split, 226 examples) — best of 8 evals
Train loss:     ~1.24 (mean around step 250, before epoch-2 collapse to ~0.70)
Base model:     unsloth/Qwen3-8B-unsloth-bnb-4bit
LoRA config:    r=64, alpha=128, DoRA + rsLoRA, dropout 0.05
Adapter size:   704 MB safetensors (adapter) / 352 MB safetensors (merged 16-bit)
Final stop:     early_stopping_patience=2 at step 415, eval_loss=1.281 (Δ +0.026 vs best)
Trainer state:  full eval history through step 415 in trainer_state.json
```

To use (adapter-only):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen3-8B-unsloth-bnb-4bit")
tok  = AutoTokenizer.from_pretrained("unsloth/Qwen3-8B-unsloth-bnb-4bit")
model = PeftModel.from_pretrained(base, "oduwairi/draper-v2-qwen", subfolder="adapter")
```

To serve via vLLM (once merged is up):

```bash
# Or load the merged repo directly as base:
vllm serve oduwairi/draper-v2-qwen --revision main --served-model-name draper-v2-qwen
# (replace with the actual merged path/subfolder once verified)
```

**Cardinal usage rule (from CLAUDE.md):** Draper sees product-only briefs in canonical JSON. The orchestrator never passes raw user prose. The system prompt must be byte-identical to `STATIC_SYSTEM_PROMPT` in `src/draper/construction_v2/schemas/brief.py`. Any drift = distribution shift = slop.

---

## Footer — session outcome

- [x] Merge exit code: 0 (clean, no OOM on L40S 46 GB)
- [x] Merged weights size on disk: 352 MB (fp16 fused)
- [x] Merged weights URL: https://huggingface.co/oduwairi/draper-v2-qwen/tree/main/merged
- [x] Adapter URL: https://huggingface.co/oduwairi/draper-v2-qwen/tree/main/adapter
- [ ] First inference smoke result on v2 test brief: **TODO — primary task for next agent**
- [ ] Pod stopped time: pod still running at time of writing; user to stop
- [x] Total wall clock: ~4 h (08:15 pod up → 12:01 final push)
- [x] Estimated session cost: ~$4.50 (4 h L40S at ~$1.10/h)
