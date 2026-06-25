# Training Run #001 — Post-mortem & Handoff

**Date:** 2026-04-30
**Run dir on lost pod:** `outputs/qwen3-8b-copywriting/r16-dora-20260430-122659/`
**Outcome:** Killed at step ~155 of 459. Eval at end of epoch 1 = 4.05 (worse than base). Model not shipped.

This document is the handoff for the next agent. Read it before starting Run #002.

---

## TL;DR

We fired the first cloud QLoRA fine-tune of Qwen3-8B on 2,442 copywriting examples. **The training metric went to ~0 within 20 steps; eval_loss at end of epoch 1 was 4.05 — the model was actively getting worse, not better.**

Root cause is most likely a **3-way interaction between `train_on_responses_only` (Unsloth's masker), `padding_free=True`, and `use_liger_kernel=True`** in TRL 0.24 + Unsloth 2026.4.8. The masker's label layout desyncs from the loss-compute layout under that combination, producing a near-zero metric on training while gradients still update LoRA weights — but in the wrong direction.

The fixes (now committed) turn off `padding_free` and `use_liger_kernel` and instrument the run for finer-grained observability. Run #002 is the validation pass.

---

## What we shipped before this run

- `src/draper/training/trainer.py` — Unsloth + TRL `SFTTrainer` wrapper, calls `train_on_responses_only` post-construction for assistant-only loss.
- `src/draper/training/hub.py` — pushes adapter folder to HF Hub.
- `src/draper/training/config.py` — Pydantic mirror of `configs/training.yaml`.
- `scripts/train.py` — Typer CLI: `smoke` / `train` / `merge` subcommands, all support `--push`.
- `scripts/upload_to_pod.sh` — rsync from laptop to RunPod (no GitHub).
- `scripts/train_cloud.sh` — pod-side bootstrap (apt + uv + pip install training extras + hf login + smoke + train + push + auto-shutdown).
- `scripts/train_cloud_bg.sh` — `nohup` wrapper so SSH disconnect doesn't kill training.

Smoke (Qwen2.5-0.5B, 2 steps) passed locally and on the cloud pod, with `train_loss=0.91`.

---

## Timeline of Run #001

| Time (UTC) | Event |
|---|---|
| 11:40 | First pod (RunPod 4090, EU region) — bootstrap stalled at PyPI ~0.3 MB/s for 25+ min. Killed. |
| 12:11 | Second pod (different region) — PyPI ~2.5 MB/s, HF ~12 MB/s. Bootstrap succeeded in ~7 min. |
| 12:21 | `python scripts/train.py train --push` started (PID 6447). |
| 12:22 | Dataset loaded (train 2442, val 215, test 215). |
| 12:26 | Qwen3-8B base model downloaded (~5 GB, both safetensors shards). |
| 12:28 | Trainer initialized. Tokenization + masking complete. |
| 12:29 | First training step. ~24s for step 1, settled to ~16s/step by step 5. |
| 12:31 | Step 10: `train_loss=1.008`, `grad_norm=1.93`, `lr=1.29e-4`. **Looked normal.** |
| 12:34 | Step 20: `train_loss=0.0058`, `grad_norm=0.026`. **First red flag — 100× drop in 10 steps.** |
| 12:37–12:50 | Steps 30–110: loss bounces in 0.0001–0.012 range. Grad_norm mostly tiny (0.0004–0.05) with occasional small spikes. |
| 12:51 | Step 140: `grad_norm=2.148` (1700× higher than nearby steps) — confirmed real gradient signal exists on some batches but train_loss still ~0.0001. |
| 13:11 | **Eval at step 153: `eval_loss=4.053`.** Definitive evidence of bad training. |
| 13:13 | Killed PID 879. Pod stopped. |
| 13:13 | Pod was reclaimed by RunPod pool during the brief Stop window — no checkpoint recovered. |

---

## Diagnostic findings

### What we ruled out

1. **"All labels are masked → zero-token loss."**
   We dumped a real example through the masker and confirmed:
   - Total tokens: 464
   - Masked (system + user): 125 tokens (27%)
   - **Unmasked (loss target): 339 tokens (73%)**
   The masker is finding `<|im_start|>assistant\n` correctly. Plenty of tokens are unmasked.

2. **"Markers don't match Qwen3 chat template."**
   Verified: `<|im_start|>user\n` and `<|im_start|>assistant\n` both appear verbatim in the rendered template (Qwen3 also injects `<think>\n\n</think>\n\n` after the assistant marker, but everything from there to `<|im_end|>` is correctly unmasked).

3. **"Off-by-one label shift in `train_on_responses_only`."**
   Read the source at `unsloth_zoo/dataset_utils.py:336`:
   ```python
   labels[assistant_k : user_j] = input_ids[assistant_k : user_j]
   ```
   `labels = input_ids` for the unmasked region is **correct** because HF's `Trainer.compute_loss` does the standard shift (`logits[:, :-1]` predicts `labels[:, 1:]`). This is the canonical autoregressive setup.

4. **"Model genuinely learning fast on teacher-distilled data."**
   Eval_loss = 4.05 disproves this. If train were real, eval would also show low loss (a model that perfectly fits 320 train examples on a narrow domain would also do okay on held-out from the same domain).

### What we suspect

The root cause is most likely the interaction between three subsystems:

1. **`train_on_responses_only`** (Unsloth) — sets `labels` array per-example with the assumption of a standard `[batch, seq]` 2D layout.
2. **`padding_free=True`** (TRL SFTConfig) — concatenates examples into a single 1D flat sequence, with `position_ids` reset at example boundaries. Labels arrive in the same flat layout.
3. **`use_liger_kernel=True`** (Liger) — replaces the standard `nn.CrossEntropyLoss` with a fused kernel that fuses the LM-head linear + softmax + CE. The shift happens *inside* the fused kernel.

When all three are on:
- Unsloth's masker writes `labels` after dataset prep, expecting standard 2D batch layout.
- TRL's `padding_free` collator may then re-pack labels into 1D form. If the masker's `-100` positions don't get re-mapped correctly, the loss-compute may end up with labels mostly equal to inputs at the same position (no shift relative to inputs), making the "predict next token" task trivially solvable — model copies inputs to outputs, loss → 0.
- Meanwhile gradients **do** flow (we observed grad_norm spikes up to 2.148), but they're optimizing a degenerate objective. After 153 steps of this, the LoRA has learned a wrong-direction transformation, raising eval loss above the base model's natural perplexity on chat data (~2–3 typically). Hence `eval_loss=4.05`.

This is a **plausible** cause; we have not fully proven it. The fix below disables the two riskiest layers (`padding_free` and `liger`) so we get a clean baseline first. If Run #002 trains correctly with both off, we can re-enable them one at a time to isolate.

### Trackio metrics from Run #001 (for reference)

```
STEP | TRAIN_LOSS | GRAD_NORM  | EPOCH    | EVAL_LOSS
  10 | 1.007976   | 1.930251   | 0.066    | -
  20 | 0.005777   | 0.026298   | 0.131    | -
  30 | 0.000947   | 0.004680   | 0.197    | -
  40 | 0.000171   | 0.013026   | 0.262    | -
  50 | 0.000056   | 0.000408   | 0.328    | -
  60 | 0.012328   | 0.004715   | 0.393    | -
  70 | 0.000927   | 0.069523   | 0.459    | -
  80 | 0.001713   | 0.001380   | 0.524    | -
  90 | 0.000947   | 0.097553   | 0.590    | -
 100 | 0.000758   | 0.044856   | 0.655    | -
 110 | 0.000465   | 0.011841   | 0.721    | -
 120 | 0.000802   | 0.005609   | 0.786    | -
 130 | 0.000262   | 0.001236   | 0.852    | -
 140 | 0.000190   | 2.148278   | 0.917    | -
 150 | 0.000058   | 0.000268   | 0.983    | -
 153 | -          | -          | 1.000    | 4.053
```

`eval_loss=4.05` is **worse than zero-shot base Qwen3-8B-Instruct** on similar held-out chat data (typical zero-shot CE on assistant-only tokens is ~2–3). This is the smoking gun.

---

## Fixes committed for Run #002

All changes are in this branch (uncommitted at time of writing). Lint + mypy pass.

### `configs/training.yaml`

```yaml
# SFT-specific
packing: false
padding_free: false           # was implicitly true via SFTConfig default
use_liger_kernel: false       # was true
assistant_only_loss: true     # unchanged

# Eval / checkpointing
eval_strategy: epoch          # unchanged — eval costs ~90s, per-epoch is right
save_strategy: steps          # was: epoch
save_steps: 50                # new — checkpoint every ~14 min
save_total_limit: 5           # was: 3
logging_steps: 1              # was: 10 — catch loss collapse at step 2 not step 10

# Smoke overrides
smoke:
  n_examples: 50              # was: 10
  max_steps: 30               # was: 2 — 2 steps couldn't catch the collapse
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
```

### `src/draper/training/config.py`

Added `padding_free: bool = False` and `save_steps: int = 50` to the Pydantic schema. Updated defaults for `use_liger_kernel`, `save_strategy`, `save_total_limit`, `logging_steps` to match new YAML.

### `src/draper/training/trainer.py`

Threaded `cfg.padding_free` and `cfg.save_steps` into the `SFTConfig` constructor. No other logic change.

### Throughput cost of these changes

- `padding_free: false` → ~20% slower (more padding waste in batches with mixed lengths)
- `use_liger_kernel: false` → ~15% slower (no fused CE)
- `logging_steps: 1` → trivial, just more SQLite writes
- `save_strategy: steps`, `save_steps: 50` → ~9 saves over 459 steps × ~20s each = ~3 min total

Net: step time goes from ~16s → ~22s. Total run estimate **~2.8 hr** (was ~2 hr). Cost ~$2 instead of ~$1.40. Worth it for getting a real model.

---

## What Run #002 should do

### Pre-flight (laptop, ~5 min)

1. **Validate the smoke locally first.** With `max_steps: 30` and `logging_steps: 1`, smoke should now show step-by-step loss. If the loss collapses to ~0 by step 5 even on the 0.5B model, we have repro on cheap hardware — debug there.
   ```bash
   uv run python scripts/train.py smoke
   ```
2. **Confirm trackio metrics are reasonable** at step 30 of smoke (loss should be in 0.3–2 range, decreasing). If yes → cloud is safe to fire.

### On the cloud pod

1. **Provision a fresh RunPod 4090 pod.** Use the same template (PyTorch 2.4.0 / CUDA 12.4 / py3.11). Don't Stop early — Stopping a pod risks losing the GPU back to the pool (this is what cost us Run #001's checkpoint).
2. **Speed-test PyPI before committing.** First pod we used had 0.3 MB/s PyPI throughput, second had 2.5 MB/s. Always run:
   ```bash
   ssh <pod> 'curl -o /tmp/t -w "%{speed_download}\n" -sL https://pypi.org/simple/torch/'
   ```
   If the number is < 1 MB/s, kill the pod and try a different region — bootstrap will take 30+ min otherwise.
3. **Upload + fire detached:**
   ```bash
   POD_HOST=root@<ip> POD_PORT=<port> SSH_KEY=~/.ssh/id_ed25519 bash scripts/upload_to_pod.sh
   scp -P <port> -i ~/.ssh/id_ed25519 /tmp/draper.env.train root@<ip>:/workspace/Draper.ai/.env.train
   ssh -p <port> -i ~/.ssh/id_ed25519 root@<ip> 'cd /workspace/Draper.ai && set -a && . ./.env.train && set +a && bash scripts/train_cloud_bg.sh'
   ```

### Monitoring during the run

- **Step 1–10:** loss should start ~1.5–2.5 and trend down monotonically (with noise). If it drops below 0.1 by step 5, abort — the bug isn't fixed.
- **Step 50:** first checkpoint saves. `train_loss` should be ~0.5–1.0. If it's <0.05, abort.
- **Step 100:** loss should be flattening. Range 0.4–0.8 is healthy.
- **Step 153 (end of epoch 1):** **eval_loss is the critical signal.** Should be in 0.3–0.7 range. If `eval_loss > 1.5`, something is still wrong; if `eval_loss < 0.05`, also wrong (deflated metric). The 0.3–0.7 band is where you want to be.
- **Steps 153–306:** epoch 2 train_loss should be marginally lower than epoch 1's average.
- **Step 306 (eval):** if `eval_loss` *rose* from epoch 1, overfitting started. `load_best_model_at_end=True` will revert to epoch 1 at the end.

Read trackio metrics live:

```bash
ssh <pod> 'cd /workspace/Draper.ai && .venv/bin/python -c "
import sqlite3, json
con = sqlite3.connect(\"/root/.cache/huggingface/trackio/huggingface.db\")
for step, blob in con.execute(\"SELECT step, metrics FROM metrics ORDER BY step\"):
    m = json.loads(blob)
    if \"eval/loss\" in m: print(f\"EVAL step={step} loss={m[\\\"eval/loss\\\"]:.4f}\")
    elif \"train/loss\" in m: print(f\"  step={m.get(\\\"train/global_step\\\",0)} loss={m[\\\"train/loss\\\"]:.4f} grad={m.get(\\\"train/grad_norm\\\",0):.4f}\")
"'
```

(The `train.log` file does NOT contain loss values when `report_to=trackio` — TRL replaces stdout logging with the trackio callback. SQLite is the only source for live metrics.)

---

## Other lessons from Run #001 (worth keeping)

- **PyPI speed varies wildly between pods.** We had a 10× difference between two pods in different regions. Always speed-test before committing.
- **`/workspace` is a network filesystem on RunPod (MooseFS).** First `uv pip install -e ".[training]"` is slow on netfs because torch ships ~30K small header files. Once installed and the pod is Stopped (carefully), the venv persists across resumes — second run is ~30 sec.
- **RunPod Community Cloud Stop = real risk.** Stopping releases the GPU back to the pool. We Stopped Run #001's pod for ~1 second to make local edits and **lost the GPU** to another customer. Don't Stop unless you're done. If you must pause for >30 min, accept that you may lose the slot and have to re-provision (with full bootstrap cost on a fresh pod).
- **Smoke must run more than 2 steps to be meaningful.** The bug in Run #001 first appeared between step 10 and step 20. A 2-step smoke had no chance of catching it. New smoke runs 30 steps with per-step logging.
- **TRL with `report_to=trackio` doesn't log metrics to stdout.** All loss values must be read from the SQLite db at `/root/.cache/huggingface/trackio/huggingface.db`. The progress bar in `train.log` only shows step counts and `s/it`, not loss/grad. Plan monitoring around SQLite, not log tailing.
- **Volume disk choice matters:** 20 GB volume is enough for code + dataset (~50 MB) + venv (~5 GB) + base model (~5 GB) + ~5 step-50 checkpoints (~1 GB) with headroom. 50 GB container disk is needed for transient `uv` cache during install.
- **`hub_strategy: every_save` would push every checkpoint to HF Hub — survives pod loss.** We didn't add this for Run #002 (current step-50 saves on local volume are good enough for our cadence) but it's a good cheap-insurance toggle for runs >3 hr or on flaky pods. Adds ~30s × N saves to total runtime.

---

## Open questions for the next agent

1. **Does the bug repro on smoke now?** Run smoke locally (`uv run python scripts/train.py smoke`) and watch the per-step loss. If the 0.5B model's loss collapses to <0.05 by step 5, we have a free repro and can debug without burning cloud GPU.
2. **Which of `padding_free` or `use_liger_kernel` is the actual culprit?** Once Run #002 succeeds with both off, do an ablation: turn one back on at a time on smoke (30 steps each, ~3 min per ablation). Whichever re-introduces the collapse is the broken one. Then file an upstream bug.
3. **Is there a faster fine-tune target?** Once we have a working pipeline, the venv + base-model cache reuses across runs — second cloud run on the same pod is ~30 sec to fire. Iteration cycle on hyperparams should be ~3 hr per attempt.

---

## Run #002 success criteria

- Smoke runs 30 steps locally with loss trending downward in 0.3–2 range. No collapse to ~0.
- Cloud run reaches step 153 with `eval_loss` between 0.3 and 0.7.
- Cloud run completes 459 steps, final adapter pushed to `oduwairi/draper-qwen3-8b/tree/main/adapter`.
- Manually loading the final adapter and running inference on a held-out test brief produces sensible ad copy (not gibberish, not memorized training example).

If all four pass, ship the adapter and call Run #002 done.

---

## Addendum (2026-04-30, ~16:30–17:15 UTC) — local repro & corrected root cause

After the postmortem, before firing Run #002, we ran a local ablation on RTX 3060 (6GB) + Qwen2.5-0.5B + xformers + torch 2.10. **Same TRL 0.24 + Unsloth 2026.4.8 as cloud Run #001.** Smoke was bumped to `bs=2, n_examples=100, max_steps=50` so `padding_free` would actually have something to do (it's a no-op at bs=1) and so the post-step-10 collapse window was reachable.

### Ablation results (`scripts/ablate_smoke.sh`)

| Run | `padding_free` | `use_liger_kernel` | step10 | step30 | step50 | outcome |
|---|---|---|---|---|---|---|
| FF | F | F | 2.617 | 2.723 | 2.807 | ✅ healthy |
| TF | T | F | 2.616 | 3.034 | 3.075 | ✅ healthy |
| FT | F | T | **0.015** | **0.0004** | **0.0006** | 🔴 **COLLAPSED** |
| TT | T | T | **0.018** | **0.0006** | **0.0005** | 🔴 **COLLAPSED** |

**`use_liger_kernel=True` alone reproduces the Run #001 collapse pattern.** `padding_free` is innocent: FF and TF train identically, FT and TT collapse identically. The original three-way-interaction theory was wrong. The actual culprit is Liger's fused LM-head CE interacting with Unsloth's `train_on_responses_only` masker — likely the `assistant_masks` column drop documented in [TRL #3781](https://github.com/huggingface/trl/issues/3781), but reproduced here on a version that should have the fix. Worth filing upstream with this repro recipe.

Compare cloud Run #001 step 20 (loss=0.006, grad_norm=0.026) vs local FT step 17 (loss=0.002, grad_norm=0.17) — same fingerprint, smaller bs collapses faster.

### Code defenses landed for Run #002

- **Pydantic guard** (`src/draper/training/config.py`) — `TrainingConfig.use_liger_kernel=True` raises `ValueError` unless `DRAPER_ALLOW_LIGER=1` is set. Hard-blocks the bad flag at config-load time, before any model loads.
- **Loss-collapse kill-switch** (`src/draper/training/trainer.py:LossCollapseCallback`) — registered unconditionally; raises `RuntimeError("LOSS_COLLAPSE: ...")` if 3 consecutive logged losses fall below 0.05 within the first 30 steps. Would have fired ~step 20 on Run #001, capping wasted GPU-time at <60s. Cloud script's auto-shutdown trap then stops the pod.
- **Label-sanity check** (`Trainer._assert_label_sanity`) — runs after `train_on_responses_only` wrap; asserts masked fraction is in [10%, 90%] and prints batch shape. Catches masker breakage before step 1.
- **Step-0 frozen baseline** (`Trainer._baseline_loss`) — runs `model(**batch)` in eval mode before the first optimizer step; asserts CE loss in [1.5, 12]. Catches degenerate loss compute (Run #001 symptom) at step 0.
- **Explicit `assistant_only_loss=False` in SFTConfig** — guard-rail comment so a future edit can't silently turn on TRL's native masker on top of Unsloth's.
- **Smoke instrumentation** — `bs=2`, `max_steps=50`, `logging_steps=1`, `eval_steps=50` (was per-epoch; aligned with `save_steps` so `load_best_model_at_end` validates).
- **Eval-strategy fix** — `eval_strategy: steps` (was `epoch`), required by `load_best_model_at_end + save_strategy=steps`. The earlier postmortem-fix commit silently introduced this incompatibility; smoke didn't catch it because the prior smoke was 2 steps and pre-fix.
- **Tighter version pins** (`pyproject.toml`) — `trl>=0.24,<0.25`, `unsloth>=2026.4,<2026.5`, `liger-kernel>=0.5,<0.6`. Locks the ablated environment.
- **`scripts/tail_train_metrics.py`** — typer CLI replacing the postmortem's hand-escaped SSH SQL one-liner. Defaults to most recent run, supports `--follow`.
- **`scripts/ablate_smoke.sh`** — re-runnable harness if we want to retest after a TRL/Unsloth/Liger update. ~12 min on RTX 3060.

### Updated success criteria for Run #002

Same as the original four below, *plus*:
- Pre-flight: `Label sanity OK` and `Step-0 baseline CE loss: ~2–4` print before training starts.
- No `LOSS_COLLAPSE` raised in the first 30 steps (the new safety floor).
- `eval_loss` at step 50 is finite and roughly comparable to step-50 train_loss; if it's near zero or above 5, abort.

### Open questions (not blocking Run #002)

1. **Why does `use_liger_kernel` collapse with assistant-only-loss masking?** TRL #3781 was supposedly fixed; either we're hitting a related path or the fix doesn't cover the Unsloth-patched trainer. Worth a minimal-repro upstream issue.
2. **Does the same bug occur on FA2 (cloud)?** We can't test locally without a working FA2 install. The Pydantic guard + kill-switch protect us either way, but the answer matters for whether Liger is *ever* re-enabled on cloud.
3. **Pod-loss recovery still untested.** `save_steps=50` + `load_best_model_at_end` works in smoke but `--resume` against a real partial-run dir was never exercised.

