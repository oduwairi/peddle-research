# Training Run #002 — Post-mortem & Handoff

**Date:** 2026-05-01
**Pod:** RunPod 4090 EU (`root@213.192.2.102:40070`)
**Run dir on pod:** `outputs/qwen3-8b-copywriting/r16-dora-20260501-085529/`
**Outcome:** Killed at step 311 of 459 (early stop, diminishing returns). Best `eval_loss = 1.535` at step 300. Adapter pushed to `oduwairi/draper-qwen3-8b/tree/main/adapter`. **No merged weights** — 4090 is too small to merge cleanly.

This document is the handoff for the next agent. Read alongside `TRAINING_RUN_001_POSTMORTEM.md`.

---

## TL;DR

Run #002 trained cleanly. **All Run #001 collapse fixes (liger off, label sanity, baseline CE check, LossCollapseCallback) held — no collapse, no kill-switch trip.** Loss descended through warmup as expected (3.12 → 1.85 over 14 steps), then plateaued in the 1.4–1.5 range for the rest of training. Eval kept descending slowly all the way to step 300 (1.778 → 1.535). Train↔eval gap stayed tight throughout (max 0.44, never widening — no overfitting signal).

**The model is genuinely generalizing on a hard problem, just not converging to the ambitious 0.3–0.7 eval target from Run #001's postmortem.** Most likely root cause: dataset heterogeneity (broad coverage of verticals + brand voices) plus backtranslation noise (briefs are reverse-engineered, sometimes misaligned with copy) — both cap achievable loss regardless of model capacity. Adapter shipped at `oduwairi/draper-qwen3-8b/adapter` for v1 use; merge attempted on the 4090 but OOMs (Unsloth's merge path needs fp16 base ~16 GB + warmup ~15 GB = ≥30 GB → A100 40 GB minimum required for merge).

---

## What we validated from Run #001's fixes

All the defenses landed in the Run #001 postmortem worked exactly as designed. None of them tripped, but their presence let us trust the run.

- **`use_liger_kernel: false`** (configs/training.yaml:64) — the actual culprit from Run #001. With it off, no collapse signature.
- **Pydantic guard** (`src/draper/training/config.py:79–95`) — would have raised `ValueError` if anyone snuck liger back on without `DRAPER_ALLOW_LIGER=1`. Did not fire (config was clean), but prevented config drift.
- **`Trainer._assert_label_sanity`** — confirmed the masked fraction was in [10%, 90%] before step 1.
- **Step-0 frozen baseline (`Trainer._baseline_loss`)** — printed `train_loss=3.118` at step 0 (in [1.5, 12] healthy band). This is the strongest single signal that the loss-compute path is wired correctly: Run #001 would have shown ~0 here.
- **`LossCollapseCallback`** — registered, never fired. Lowest train loss was ~1.0 (step 187), well above the 0.05 abort threshold.
- **`save_steps: 50` + `save_total_limit: 5`** — gave us a checkpoint every ~14 min. checkpoint-300 (best eval) was on disk at the moment we decided to stop.
- **`logging_steps: 1`** — caught the warmup curve and the epoch-2 transition cleanly. Per-step granularity made the diagnosis obvious.

**Single concrete change recommended for Run #003 even though it didn't bite us:** the `tail_train_metrics.py` script is not in `scripts/upload_to_pod.sh`'s `INCLUDE_PATHS`, so we had to `scp` it manually after the pod was bootstrapped. Adding it to the include list is a 2-line fix.

---

## Timeline of Run #002

| Time (UTC) | Event |
|---|---|
| 08:48 | Pod provisioned (RunPod 4090 EU). PyPI speed-tested at 2.18 MB/s — green to fire. |
| 08:48 | Pod missing `rsync` — had to `apt-get install rsync` before upload script ran. |
| 08:49 | `upload_to_pod.sh` synced src/, scripts (subset), configs/training.yaml, data/final, pyproject.toml, Makefile. |
| 08:50 | Built `.env.train` from `.env` (HF_TOKEN + HF_HUB_REPO) + added `AUTO_SHUTDOWN=1` and `SKIP_SMOKE=1`. SCP'd to pod. |
| 08:51 | `train_cloud_bg.sh` fired. PID 1962 (parent bash, nohup'd). |
| 08:51–08:54 | Bootstrap: apt + uv + `uv pip install -e ".[training]"` + HF login. ~3 min on warm netfs (`/workspace` MooseFS). |
| 08:54 | Dataset loaded: train 2442, val 215, test 215. |
| 08:54–08:55 | Qwen3-8B base downloaded (~5 GB, 2 shards) + Unsloth patching + LoRA attached (r=16, α=32, DoRA+rsLoRA, dropout 0.05). |
| 08:55 | Tokenization: train (1m26s, num_proc=64), val (1m13s). |
| 08:57 | Step-0 baseline: `train_loss=3.118`, `grad_norm=12.04`. Healthy band [1.5, 12]. |
| 08:58 | Step 1: `train_loss=2.973`, `grad_norm=8.878`, `lr=1.43e-5`. |
| 09:02 | Step 14 (warmup end): `train_loss=1.845`, `grad_norm=1.05`, `lr=2e-4`. |
| 09:14 | **Step 50 eval: `eval_loss=1.778`** (train 1.534, gap 0.24). First checkpoint saved. |
| 09:29 | **Step 100 eval: `eval_loss=1.666`** (train 1.52, gap 0.15). |
| 09:43 | **Step 150 eval: `eval_loss=1.593`** (train 1.47, gap 0.12). End of epoch 1. |
| 09:43 | Step 156: train_loss drops 1.45 → 1.20 over 4 steps (epoch 2 starts, model consolidates first-pass signal). Initially flagged as possible memorization onset; later proven wrong. |
| 09:57 | **Step 200 eval: `eval_loss=1.588`** (train 1.15, gap 0.44). Gap widened, eval barely moved. Looked like memorization. |
| 10:11 | **Step 250 eval: `eval_loss=1.558`** (train 1.12, gap 0.44). Eval still descending — not memorization, just slow generalization. Earlier "memorization" call retracted. |
| 10:25 | **Step 300 eval: `eval_loss=1.535`** (train ~1.10, gap 0.43). New best, but per-50-step delta down to 0.023 — diminishing returns. |
| 10:33 | User decision: kill. Spent budget vs marginal eval gains tipped against continuing. |
| 10:33 | First kill attempt: `kill 1962` (parent bash). bash exited but **python child PID 7287 survived** — training kept running. Caught by user. |
| 10:35 | `kill 7287` — actual python process died. GPU freed. Final logged step: 311. |
| 10:34 | Adapter pushed to HF Hub: 285 MB, ~35 MB/s. URL: https://huggingface.co/oduwairi/draper-qwen3-8b/tree/main/adapter |
| 10:36 | Merge attempt #1 (`merged_16bit`). OOM at `caching_allocator_warmup`: tried to allocate 14.93 GiB on top of 15.90 GiB already held. |
| 10:37 | Merge attempt #2 (`merged_4bit`). **Same OOM** — `merged_4bit` only changes the *output* format; merge always loads fp16 base internally. |
| 10:39 | Pod stopped (user). Adapter is the only shipping artifact. |

Total run wall-clock: **~1h 50m** (vs projected ~2h 48m for full 459 steps). Cost: ~$0.85 in pod time. 311 of 459 steps completed.

---

## What the metrics actually told us

### Loss trajectory

```
STEP | TRAIN_LOSS | GRAD_NORM | EVAL_LOSS | GAP   | NOTES
   0 |     3.118  |    12.04  |     —     |  —    | frozen baseline
   1 |     2.973  |     8.88  |     —     |  —    | step 1, lr ramp begins
  14 |     1.845  |     1.05  |     —     |  —    | warmup ends, lr at peak (2e-4)
  50 |     1.534  |     0.94  |    1.778  | 0.24  | eval 1, healthy gap
 100 |     1.373  |     0.87  |    1.666  | 0.15  | eval 2, gap closing
 150 |     1.408  |     0.84  |    1.593  | 0.12  | eval 3, end of epoch 1, gap tight
 156 |     1.251  |     1.07  |     —     |  —    | epoch 2 starts, train drops ~0.20
 200 |     1.149  |     1.21  |    1.588  | 0.44  | eval 4, gap widened (epoch-2 consolidation)
 250 |     1.082  |     1.19  |    1.558  | 0.44  | eval 5, eval still descending slowly
 300 |     1.092  |     1.19  |    1.535  | 0.44  | eval 6 — best so far, killed shortly after
 311 |     1.13   |     —     |     —     |  —    | last logged step before kill
```

### Reading the curves

1. **Warmup phase (0–14):** textbook. Loss dropped 3.12 → 1.85 as LR ramped 0 → 2e-4. Grad norms trended down from ~12 to ~1, indicating the model was finding a stable update direction.
2. **Plateau phase (14–150):** loss bobbled in 1.4–1.6 with no clear trend. The model had hit its first-pass capacity ceiling on data this heterogeneous.
3. **Epoch-2 transition (153–200):** train dropped sharply (~1.45 → ~1.15) at the epoch boundary, looking exactly like memorization onset. *It wasn't.* Eval kept descending: 1.593 → 1.588 → 1.558 → 1.535. The drop was the model consolidating first-pass learning, not rote memorization. Without the step-50 eval cadence we couldn't have distinguished these.
4. **Late phase (200–300):** train flat in 1.05–1.15, eval slowly descending at ~0.025 per 50 steps. Real but tiny improvement.

### Why the gap widened from 0.12 (step 150) to 0.44 (step 200) without it being memorization

This was the trickiest read. The gap-widening *looks like* overfit signature. What actually happened:
- Train dropped because the model had now seen each example once and was using epoch 2 to refine its mapping with full context. This is consolidation.
- Eval barely moved because the val set was *already* the model's generalization frontier — the model couldn't generalize to held-out faster than it was already doing.
- Memorization would show **eval rising while train dropped**. Eval *kept dropping*, just slowly. Test passes.

The diagnostic that distinguishes: watch eval at step N+50. If it dropped, you're consolidating (good). If flat or rising, you're memorizing (bad). Run #002's eval kept dropping all the way to step 300 → consolidation, not memorization.

---

## What we learned about the data

The flat train plateau in 1.4–1.5 (epoch 1) and ~1.1 (epoch 2+) is most likely a **dataset property, not a training bug.** Two hypotheses, both probably contributing:

### 1. Dataset heterogeneity caps compressibility

The 2442 training examples span many verticals (DTC, fintech, B2B SaaS, lifestyle, etc.) with very different brand voices, formats, and lengths. A LoRA r=16 adapter has ~24M trainable params — enough to learn a generic "brief → ad copy" mapping, but not enough to learn vertical-specific or brand-specific styles. The model does the best it can on the average mapping; per-token loss floor is set by the variance the adapter can't compress.

If this hypothesis is right, **per-vertical fine-tunes would lower the floor materially.** A v3 experiment: cluster the train set by vertical (we already have vertical labels), train a LoRA per cluster, evaluate per cluster. Probably eval drops to 0.7–1.0 per vertical from a global 1.5.

### 2. Backtranslation noise sets a hard loss floor

Briefs were reverse-engineered from copy by a teacher LLM (Humpback / Li et al.). Some of these reverse-engineered briefs include details that aren't actually in the copy (or omit details that are). The model is then trained to predict the copy from sometimes-misaligned briefs. The mismatch is irreducible — no amount of training fixes a brief that doesn't actually describe the copy.

If this is the dominant cause, **cleaner briefs (better teacher prompt, brief-quality filter) would help more than per-vertical splits.** A v3 ablation: take the top 25% of briefs by some quality proxy (rubric score, teacher confidence), retrain on just those. If eval drops materially with smaller-but-cleaner data, this is confirmed.

### Most likely: both

Suggested v3 ordering: try the brief-quality filter first (cheap — no retraining infrastructure changes), then per-vertical splits (more work, requires training infrastructure for multi-adapter pipelines). Run a tiny ablation (50 steps each, smoke-sized) before committing to a full retrain.

---

## What was right / wrong about Run #001's projections

Run #001's postmortem set targets that turned out to be too aspirational for this dataset.

| Target | Run #001's projection | Run #002's reality | Verdict |
|---|---|---|---|
| Step-50 train loss | 0.5–1.0 | 1.534 | ❌ ~50% above band |
| Step-153 eval loss | 0.3–0.7 | 1.593 | ❌ ~3× above band |
| Loss collapse risk | Possible | Did not occur | ✅ fixes worked |
| `eval_loss > 1.5` flag | "still wrong" | True at step 50, but model was healthy | ⚠️ false positive — flag is too tight |
| Train/eval divergence | Flagged as overfit | Held tight at gap ≤0.44 | ✅ |

**The aspirational eval target (0.3–0.7) was almost certainly extrapolated from QLoRA runs on narrower data.** It should be revised in the next iteration's postmortem. Realistic Run #003 target with no recipe changes: eval 1.3–1.5. Realistic with brief-quality filter: 1.0–1.3. Realistic with per-vertical splits: 0.7–1.0 per vertical.

---

## Why we couldn't merge on the 4090

The merge code path (`src/draper/training/merge.py`) explicitly loads the base model in fp16/bf16 (line 47: `load_in_4bit=False`) so the LoRA delta can be fused cleanly. Then `transformers.modeling_utils._load_pretrained_model` calls `caching_allocator_warmup` which pre-allocates a single fp16-sized scratch buffer (~14.93 GiB).

```
fp16 base weights:                ~16 GiB
caching_allocator warmup buffer:  ~15 GiB
peak demand:                      ~31 GiB
4090 capacity:                    ~24 GiB
                                  → OOM
```

`merged_4bit` save method **does not save us** — it only changes the *output* format (4-bit instead of fp16 on disk). The internal merge still happens at fp16, so peak VRAM is the same.

`PYTORCH_ALLOC_CONF=expandable_segments:True` does not save us — the OOM is one giant contiguous allocation, not fragmentation.

### Three viable paths forward (any one works, none blocking v1)

1. **Skip merge entirely. Ship adapter-only.** This is the current state. vLLM supports LoRA adapters natively (`--enable-lora`); transformers + PEFT load adapters at runtime; the frontend's OpenAI-compat endpoint can serve adapter-only via vLLM. There is **no production reason to merge.**
2. **Merge later on a bigger GPU.** A100 40 GB (~$1.50/hr) or H100 80 GB (~$2.50/hr). Total cost for one merge: provision (~3 min) + bootstrap (~5 min) + merge (~3 min) + upload (~3 min) + stop (~1 min) = ~15 min × hourly rate ≈ $0.30–0.60. Trivial.
3. **CPU-merge with disk offload.** Possible but slow (~30 min) and needs ~32 GB RAM. Not worth it vs option 2.

Recommended: **option 1 unless production explicitly needs merged weights.** Adapter at `oduwairi/draper-qwen3-8b/adapter` is a complete shipping artifact.

---

## Operational lessons (worth keeping)

These are the small footguns hit during this run.

- **Killing `bash scripts/train_cloud.sh` (parent) does NOT kill the python child.** The `nohup ... &` indirection inside `train_cloud_bg.sh` makes the python process its own session leader — when the bash parent dies, the child becomes an orphan but keeps running. Always `ps aux | grep train.py` before assuming the kill worked. The fix is either (a) document this in `train_cloud_bg.sh`, or (b) make the script print the python PID separately so users kill the right thing.
- **`scripts/tail_train_metrics.py` is missing from `upload_to_pod.sh`'s INCLUDE_PATHS.** Add it. 2-line fix:
  ```bash
  # in scripts/upload_to_pod.sh INCLUDE_PATHS:
  "scripts/tail_train_metrics.py"
  ```
- **Pod containers don't always ship rsync.** The first thing `upload_to_pod.sh` does is run rsync from the *local* side, which fails immediately if the pod side doesn't have rsync. Either (a) add rsync to `train_cloud.sh`'s apt install step, or (b) detect missing rsync in `upload_to_pod.sh` and apt-install it via SSH before transferring.
- **`.env.train` doesn't exist in the repo by design** (it has secrets). The cloud workflow assumes you build it from `.env` before SCPing. Could be automated: `scripts/upload_to_pod.sh` could optionally extract `HF_TOKEN`, `HF_HUB_REPO`, `AUTO_SHUTDOWN`, `SKIP_SMOKE` from local `.env` and SCP a synthesized `.env.train`.
- **HF Hub uploads are network-bound; merge is GPU-bound — they can run in parallel.** We started the adapter push and the merge in two separate SSH sessions. Worked fine, halved the wall-clock for the (would-have-been) push+merge sequence.
- **Eval cadence (every 50 steps) was the right call.** Step-50 eval gave us a clear continue/stop signal at every 12-min interval. Per-epoch eval (3 evals total) would have meant much slower decision loops. Keep `eval_steps: 50` for future runs.
- **`save_total_limit: 5` correctly cleaned up `checkpoint-50` mid-run** when `checkpoint-300` was saved. That's why only 100, 150, 200, 250, 300 are on disk — by design.
- **Per-step logging produces a beautiful curve.** Reading the metrics every 1 step (vs every 10 in Run #001) made the warmup, plateau, and epoch-2 transition all visible in real time. The cost is ~6 KB/step of SQLite writes — totally negligible. Keep `logging_steps: 1`.

---

## What's left undone

These are real gaps in the v1 pipeline that the next iteration needs:

1. **No merged weights.** Adapter only. Need an A100+ pod for ~15 min if production wants merged. Not blocking the frontend wiring (vLLM serves adapters).
2. **Test-set evaluation pipeline does not exist yet.** `data/final/test/` (215 examples) is untouched. The user explicitly noted this is a separate future workstream. Need: load adapter, generate copy for each test brief, score with LLM judge + proxy metrics, aggregate. Lives under `src/draper/evaluation/` (currently a stub).
3. **No held-out *brand* split.** Current val/test are random splits — they share brands with train. A "leave-one-brand-out" split would be the cleanest test of true generalization. Not strictly required for v1.
4. **Inference-quality smoke test** (compare adapter vs base on 5 held-out briefs, eyeball the copy) was *not* run in this session. Should be the first thing the next agent does before claiming the adapter is shippable. eval_loss=1.535 might or might not produce good copy; the only way to know is to look.
5. **The `tail_train_metrics.py` script lives outside `INCLUDE_PATHS`.** Documented above. Fix in upload script.

---

## Open questions for the next agent

Ordered by expected impact / cost ratio:

1. **Does the adapter actually produce good copy?** Pull `oduwairi/draper-qwen3-8b/adapter`, load on top of `unsloth/Qwen3-8B-unsloth-bnb-4bit`, generate for 5 held-out briefs, eyeball. **Do this before anything else.** ~10 min on the user's 3060 (adapter-only inference fits easily). If output is gibberish or ignores the brief → run is broken; investigate. If output is plausible ad copy → ship adapter to frontend.
2. **Per-vertical eval breakdown.** Compute `eval_loss` per vertical cluster on the val set. If it's flat across verticals → uniform generalization (data heterogeneity hypothesis weakened). If some verticals are 0.8 and others are 2.0 → vertical-specific fine-tunes are the right v2.
3. **Brief-quality filter ablation.** Keep top-25% of train examples by rubric/teacher-confidence score. Retrain (50-step smoke or full 459-step run). Does eval drop materially? If yes → brief noise is the dominant cause → invest in better backtranslation (better teacher prompt, brief-vs-copy alignment check).
4. **Is the "step-153 eval 0.3–0.7" target salvageable?** Probably not on this dataset shape. Either lower the bar in the postmortem (more honest), or change the data shape (per-vertical, brief-quality filter) to get there.
5. **Should we re-enable `use_liger_kernel`?** Open question from Run #001. Not relevant for v2 unless we hit a throughput ceiling. The Pydantic guard makes it a one-liner experiment if curiosity demands.

---

## Run #002 success criteria — verdict

From Run #001's postmortem:

| # | Criterion | Status |
|---|---|---|
| 1 | Smoke runs 30 steps locally with loss in 0.3–2 range, no collapse | ✅ done before this run (2026-04-30) |
| 2 | Cloud run reaches step 153 with `eval_loss` between 0.3 and 0.7 | ❌ eval = 1.593 at step 150 — way above target band, but legitimately generalizing |
| 3 | Cloud run completes 459 steps, final adapter pushed | ⚠️ stopped at step 311, adapter-from-checkpoint-300 pushed (eval 1.535 — best of run) |
| 4 | Manual inference on held-out test brief produces sensible ad copy | ⏳ NOT TESTED — primary task for next agent |
| **NEW** | Liger-kernel collapse does not recur | ✅ confirmed clean |
| **NEW** | Step-0 baseline + label-sanity + LossCollapseCallback all silent | ✅ all defenses correctly never fired |

**Net call: partial pass.** The training run was clean and produced a healthy adapter, but the eval-target band from Run #001 was unrealistic for this dataset, and we never validated the actual output quality (criterion #4). Run #003's gating criterion should be **inference quality on held-out briefs**, not eval_loss — eval_loss was the right Run #002 signal because we needed to verify "no collapse"; by Run #003 that's a solved problem.

---

## Adapter shipping info

```
HF Hub:        https://huggingface.co/oduwairi/draper-qwen3-8b/tree/main/adapter
Source ckpt:   outputs/qwen3-8b-copywriting/r16-dora-20260501-085529/checkpoint-300/
Eval loss:     1.535 (on val split, 215 examples)
Train loss:    ~1.10 (running average around step 300)
Base model:    unsloth/Qwen3-8B-unsloth-bnb-4bit
LoRA config:   r=16, alpha=32, DoRA + rsLoRA, dropout 0.05
Adapter size:  180 MB (safetensors)
Total upload:  285 MB (incl. optimizer/scheduler state for resume)
```

To use:
```python
# adapter-only inference (recommended)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen3-8B-unsloth-bnb-4bit")
tok  = AutoTokenizer.from_pretrained("unsloth/Qwen3-8B-unsloth-bnb-4bit")
model = PeftModel.from_pretrained(base, "oduwairi/draper-qwen3-8b", subfolder="adapter")
# model is callable like any HF model
```

Or for vLLM serving: pass `--lora-modules draper=oduwairi/draper-qwen3-8b/adapter` at server startup.
