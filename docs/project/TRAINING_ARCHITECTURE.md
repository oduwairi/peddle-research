# Training Pipeline Architecture

Plain-English design document for Draper.ai's QLoRA fine-tuning pipeline.
Companion to `CONSTRUCTION_ARCHITECTURE.md`: that doc explains how training
*data* is built; this one explains how the *model* is trained from it.

> All non-default choices in this doc were validated against current
> documentation and open issues in **April 2026**. Where defaults shifted
> against the original `IMPLEMENTATION_PLAN.md`, the rationale is inline.

---

## What we're doing in one sentence

Take 2,442 chat-format ad-copy examples → use a tiny adapter (~16M trainable
weights) to nudge a 4-bit-quantized 8-billion-parameter base model toward
producing on-pattern ad copy → merge the adapter back into the base → serve
the merged model from vLLM behind the existing Next.js frontend.

## End-to-end pipeline

```
                ┌─────────────────────────────────────────┐
                │  data/final/  (HF DatasetDict, Arrow)   │
                │  ─────────────────────────────────────  │
                │  train       2,442 rows                 │
                │  validation    215 rows                 │
                │  test          215 rows                 │
                │  columns: messages, platform, vertical, │
                │           source_tiers, …               │
                └────────────────────┬────────────────────┘
                                     │
              draper.training.data_loader.load_dataset_dict()
                                     │
                                     ▼
       ┌──────────────────────────────────────────────────────┐
       │         Qwen3-8B base model (~8.2 B params)          │
       │   weights stored in 4-bit NF4 (~5 GB on disk/VRAM)   │
       │   ─────────────────────────────────────────────────  │
       │   loaded via Unsloth's FastLanguageModel:            │
       │   unsloth/Qwen3-8B-unsloth-bnb-4bit                  │
       └────────────────────┬─────────────────────────────────┘
                            │
            FastLanguageModel.get_peft_model(...)
            attaches a LoRA adapter to every linear layer
                            │
                            ▼
          ┌────────────────────────────────────────────┐
          │   LoRA adapter (DoRA + rsLoRA, r=16)       │
          │   ────────────────────────────────────     │
          │   trainable parameters:    ~16 M           │
          │   = 0.19 % of base weights                 │
          │   the only thing gradient descent updates  │
          └────────────────────┬───────────────────────┘
                               │
                       TRL SFTTrainer
                  (assistant_only_loss=True,
                   packing=False, liger_kernel=True)
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
            forward pass   backward pass   AdamW-8bit step
            (4-bit base   (gradients only   (updates the
             stays frozen)  flow into LoRA   16 M LoRA
                            adapter)         weights)
                               │
                  3 epochs × 2,442 examples
                  effective batch 16 (per-device 2 × grad-accum 8)
                  early-stopping on eval_loss (safety net)
                               │
                               ▼
       ┌────────────────────────────────────────────────┐
       │   outputs/qwen3-8b-copywriting/                │
       │     r16-dora-{timestamp}/                      │
       │       checkpoint-N/   (epoch checkpoints)      │
       │       final/          (best adapter)           │
       │       config.snapshot.yaml                     │
       │       trackio/        (loss curves)            │
       └────────────────────┬───────────────────────────┘
                            │
              draper.training.merge.merge_adapter()
                            │
                            ▼
       ┌────────────────────────────────────────────────┐
       │   outputs/qwen3-8b-copywriting/merged/         │
       │     model.safetensors  (~16 GB bf16)           │
       │     tokenizer/                                 │
       │     merged_meta.json   (provenance)            │
       └────────────────────┬───────────────────────────┘
                            │
                     vLLM serve
                            │
                            ▼
              frontend/lib/agent  (Next.js)
                ↓
              user
```

---

## The stack, explained

| Component | What it is | Why we use it |
|---|---|---|
| **Qwen3-8B** | 8-billion-parameter open-weight LLM from Alibaba (Apache 2.0). Native function-calling. | Strongest 8B for fine-tuning + tool-use as of April 2026. Function-calling is essential — frontend tools (`scrape_url`, `web_search`, `exa_similar`, `emit_campaign`) depend on it. |
| **QLoRA** (Quantized LoRA) | Train an adapter on top of a 4-bit-quantized base model. The base stays frozen and quantized; only the adapter is updated and stored in higher precision. | Cuts VRAM from ~32 GB (full bf16 fine-tune) to ~16 GB. Lets us train on a single 24-32 GB consumer GPU instead of an 80 GB A100. |
| **LoRA** (Low-Rank Adaptation) | A small "delta" matrix (`ΔW = B·A`) that gets added to each linear layer. Instead of training all 8B weights, we train ~16M low-rank parameters. | 100× fewer trainable parameters → far less data needed, far less overfitting risk, far less storage per checkpoint (~50 MB vs ~16 GB). |
| **DoRA** (Weight-Decomposed LoRA) | Splits each weight update into magnitude + direction components. Lets the adapter change *how strongly* a feature fires without changing *which* features fire. | 2025–2026 ablations show DoRA preserves base capabilities better than vanilla LoRA — load-bearing for keeping function-calling intact. |
| **rsLoRA** (Rank-Stabilized LoRA) | Replaces LoRA's `α/r` scaling with `α/√r`. | Lets us pick rank without re-tuning learning rate. Pairs with `α = 2r` as the modern default. |
| **PiSSA** (initialization scheme) | Originally planned. Drops out of the stack — broken on bnb-4bit base models per [PEFT #1999](https://github.com/huggingface/peft/issues/1999). DoRA fills the same role (capability preservation). | — |
| **Unsloth** | A drop-in optimization layer over transformers + PEFT. ~2× faster training, ~70% less VRAM via custom CUDA kernels and gradient checkpointing. | Free speed/memory wins. Ships pre-quantized models (`unsloth/Qwen3-8B-unsloth-bnb-4bit`) with Dynamic-2.0 calibration. |
| **TRL `SFTTrainer`** | HuggingFace's supervised fine-tuning trainer. Wraps the standard HF `Trainer` with chat-template handling and assistant-only loss masking. | Native support for the `messages`-column format we already produce in `data/final/`. No re-tokenization needed. |
| **`assistant_only_loss=True`** | Tells the trainer to only compute loss on assistant turns (not on user/system turns). | Critical for chat-format SFT — we don't want to waste gradient on memorizing user prompts. |
| **Liger Kernel** | Replaces transformers' attention/MLP/loss kernels with Triton-fused versions. | Free 10–20% speedup with no accuracy impact. |
| **AdamW-8bit** | bitsandbytes's 8-bit-quantized AdamW optimizer. | Cuts optimizer state memory ~4× vs fp32 AdamW. The optimizer states are often the second-biggest VRAM consumer after the model itself. |
| **bf16** (bfloat16) | 16-bit float with 8-bit exponent (same range as fp32, less precision). | Standard training precision on Ampere+ GPUs. Stable, no scaling tricks needed. |
| **Trackio** | HuggingFace's lightweight experiment tracker (Sept 2025). Local-first, Gradio UI. | Free, no account, runs alongside the trainer. Native TRL integration via `report_to="trackio"`. |
| **vLLM** | High-throughput inference server. OpenAI-compatible `/v1/chat/completions` endpoint. | The frontend already speaks OpenAI-compatible. Pointing `OPENAI_BASE_URL` at vLLM is a one-env-var swap. |

---

## The numbers, explained

| Number | Value | What it means | Why this number |
|---|---|---|---|
| **Base parameters** | ~8.2 B | Total weights in Qwen3-8B. | Sweet spot: large enough to be capable, small enough for single-GPU QLoRA. |
| **Quantization** | 4-bit NF4 | Each base weight stored in 4 bits using NormalFloat-4 (information-optimal for normal-distributed weights). | Cuts base model from ~16 GB (bf16) to ~5 GB. Frees VRAM for activations and the LoRA adapter. |
| **LoRA rank `r`** | 16 | The "width" of the low-rank delta. Lower = fewer trainable params + less overfit risk. | Unsloth's 2026 guide explicitly recommends r=16 for datasets <5K examples. We have 2,442. |
| **LoRA alpha** | 32 | Scaling factor for the LoRA delta. Modern convention: `α = 2r`. | Daniel Han (Unsloth) confirms `α = 2r` with rsLoRA is current best practice. |
| **Trainable parameters** | ~16 M | LoRA adapter weights. | 0.19% of the base model. Storage per checkpoint: ~50 MB. |
| **Train rows** | 2,442 | Copywriting examples after construction + quality filter. | Small. Drives every "go conservative" choice (r=16, ≤3 epochs, DoRA). |
| **Validation rows** | 215 | Held-out for eval_loss and early-stopping. | ~7.5% of data, stratified by `task_format + platform`. |
| **Test rows** | 215 | Held-out for final manual rubric judgment. | Same split logic. **Never seen during training.** |
| **Sequence length** | 4,096 tokens | Max tokens per example. Longer is truncated. | Comfortably fits the longest brief+response in our data. |
| **Per-device batch** | 2 | Examples processed in one forward+backward pass on the GPU. | What fits in 24 GB at seq=4096 with packing off. |
| **Grad accumulation** | 8 | Number of forward+backward passes before one optimizer step. | Effective batch = 2 × 8 = **16** examples per gradient update. Standard for 8B QLoRA. |
| **Steps per epoch** | ~152 | `ceil(2,442 / 16)`. | One step = one optimizer update. |
| **Total training steps** | ~456 | 152 × 3 epochs. | Tiny by foundation-model standards; that's the whole point of LoRA. |
| **Learning rate** | 2e-4 | How aggressively the LoRA weights move per step. | Unsloth + Thinking Machines Lab + Sebastian Raschka all converge on 2e-4 as the QLoRA standard. We'd drop to 1e-4 only if loss is unstable. |
| **Schedule** | cosine, 3% warmup | LR ramps up over the first 3% of steps, then decays cosine-style to 0. | Standard. Warmup prevents early instability; cosine decay finds a flat minimum. |
| **Epochs** | 3 (with patience-2 early stopping) | Number of full passes through the training set. | Unsloth: "1–3 epochs, more shows diminishing returns and overfitting risk for <5K examples." |
| **Optimizer** | AdamW-8bit | Standard adaptive optimizer with 8-bit-quantized state. | ~4× memory savings vs fp32 AdamW. |
| **Precision** | bf16 | 16-bit brain-float for activations and gradients. | Standard on Ampere+. Stable. |
| **Estimated wall-clock** | ~1.5–3 hours | On RTX 5090 / 4090. | 7,326 example-passes (2,442 × 3) at ~1–2 examples/sec. |
| **Estimated cost** | ~$1–1.50 | RunPod RTX 5090 Community at ~$0.49/hr. | Tiny dataset = tiny bill. The construction phase cost orders of magnitude more. |

---

## Why each non-obvious decision was made

### Why we dropped `packing=True`
TRL issue [#3728](https://github.com/huggingface/trl/issues/3728): packing
and `assistant_only_loss=True` are mutually exclusive in current TRL. For
chat-format SFT, masking out user/system turns matters more than the ~1.5×
throughput gain from packing. Cost: longer wall-clock (~3h instead of ~1.5h).
Benefit: gradient is only spent on the assistant copy we actually want to
teach, not on memorizing prompts.

### Why we dropped PiSSA initialization
Original plan: use PiSSA to keep Qwen3-8B's native function-calling intact
without mixing tool-call examples into training. PEFT issue
[#1999](https://github.com/huggingface/peft/issues/1999) confirms PiSSA
is broken on bnb-4bit-loaded models — the SVD runs on the quantized tensor
and produces NaN base weights. **DoRA does the same job differently**:
decomposing weight updates into magnitude + direction limits how far the
adapter can drift from the pretrained representation.

### Why we're not mixing in tool-call examples
The hedge would be a 5–10% slice of generic instruction or function-call
data (e.g. xLAM, Glaive). We're skipping it because:
1. Our data is small enough that 5% of it wouldn't be many tool-call
   examples either way.
2. DoRA + r=16 + ≤3 epochs is the strongest no-mix defense in 2026.
3. Verification step 9 (function-calling smoke test post-merge) catches
   regression cheaply. If it fails, the follow-up plan adds the mix.

### Why DoRA over vanilla LoRA
NVIDIA's DoRA writeup + the 2025–2026 PEFT-for-RLVR survey (arXiv
2512.23165) show DoRA beating LoRA by +4.4 on Llama 3 8B common-sense
reasoning, with materially less catastrophic forgetting. For our use case
(narrow domain shift on top of a general-purpose model), the forgetting
delta is what matters.

### Why eval_loss isn't the real stop signal
Community reports across QLoRA practitioners (Sebastian Raschka, Thinking
Machines Lab): eval_loss often rises after epoch 2 *while downstream
quality keeps improving*. We track eval_loss for diagnostics and use it
for early-stopping as a runaway-cost safety net, but the actual
go/no-go decision is the manual rubric pass on `data/final/test`
(verification step 8 in the plan).

---

## Module map

| Path | Role |
|---|---|
| `configs/training.yaml` | Single source of truth for hyperparameters and paths. |
| `src/draper/training/config.py` | Pydantic `TrainingConfig` mirror of the YAML. |
| `src/draper/training/data_loader.py` | Loads `data/final/`, validates shape, smoke-subsets, renders templated examples. |
| `src/draper/training/trainer.py` | `Trainer` class wrapping Unsloth + TRL `SFTTrainer`. Heavy ML imports deferred to method bodies so the module is importable on a CPU-only laptop. |
| `src/draper/training/merge.py` | `merge_adapter()` — collapses adapter into base for vLLM. Writes provenance JSON. |
| `scripts/train.py` | Typer CLI: `inspect`, `smoke`, `train`, `merge`. |

## CLI surface

```bash
# 1. Verify the chat template renders correctly (locally, fast)
uv run python scripts/train.py inspect

# 2. Validate config + dataset shape on a CPU-only laptop (no model load)
uv run python scripts/train.py smoke --dry-run

# 3. End-to-end smoke on any GPU (~1 min, tiny model)
uv run python scripts/train.py smoke

# 4. The real run (cloud GPU, ~$1, ~1.5–3 h)
uv run python scripts/train.py train

# 5. Resume if interrupted
uv run python scripts/train.py train --resume

# 6. Merge adapter into base for vLLM
uv run python scripts/train.py merge --adapter outputs/.../final
```

## Verification flow (what "done" means)

```
   inspect ──── chat template wraps assistant turns visibly ✓
       │
       ▼
   smoke --dry-run ──── config + dataset valid (no model load) ✓
       │
       ▼
   smoke ──── 2 steps complete on tiny model, no crashes ✓
       │
       ▼
   train (cloud) ──── eval_loss decreases, Trackio loss curve sane ✓
       │
       ▼
   merge ──── merged_meta.json written, weights saved ✓
       │
       ▼
   manual rubric ──── 5 hand-picked test briefs: tighter, more
                      on-pattern copy than Qwen3-8B base ✓
       │
       ▼
   function-calling smoke ──── merged model still emits well-formed
                                tool calls for `scrape_url` ✓
       │
       ▼
   ready for vLLM serving
```

## What's deliberately out of scope

- **Hyperparameter sweep** — Phase 3 Week 12 in `IMPLEMENTATION_PLAN.md`.
- **Data-composition ablations** (without knowledge corpus, without synthetic).
- **Bake-off vs Llama 3.1 8B** — committed to Qwen3-8B based on its native
  function-calling story.
- **vLLM serving config / production deployment** — separate plan.
- **Tool-use trajectory fine-tuning** — deferred; revisit only if the
  function-calling smoke test fails.

## References

- Hu et al., 2021. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- Dettmers et al., 2023. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
- Liu et al., 2024. [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353).
- Kalajdzievski, 2024. [Rank-Stabilized LoRA](https://huggingface.co/blog/damjan-k/rslora).
- Unsloth. [Qwen3 fine-tuning guide](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune).
- Unsloth. [LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).
- TRL docs. [SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer), [Trackio integration](https://huggingface.co/docs/trl/en/trackio_integration).
- PEFT issue [#1999](https://github.com/huggingface/peft/issues/1999) — PiSSA broken on bnb-4bit.
- TRL issue [#3728](https://github.com/huggingface/trl/issues/3728) — packing × assistant_only_loss conflict.
