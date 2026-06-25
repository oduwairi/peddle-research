# `scoring_predictor` — Text-only ad-performance regressor

## What it does and why it exists

The v3 `HybridScorer` labels ads using post-publication signals: Kaplan-Meier survival from real `last_seen − first_seen` timestamps, plus real likes/comments/shares/reactions/views. These signals are structurally undefined for Draper's synthetic outputs — there is no publication, no audience, no elapsed time.

This module trains a text-only DeBERTa-v3-base regressor on the 55k v3-scored AdFlex corpus so that v3-equivalent scores can be produced from ad copy alone. The classifier asks: "does this text resemble real high-performing ads?" — which is the training-quality question for Draper.

The regressor is not a redundant signal. It is complementary to LLM-as-judge: both proxies have different blind spots, and divergence between them is informative.

**Phase 1 (this module):** train + offline validate. **Done** — random split: composite ρ=0.7223 on test, ECE=0.0074.
**Phase 2 (May 2026):** wired as an absolute-scorer arm in `src/draper/evaluation/learned_scorer.py` AND as a live frontend feature (auto-score on `emit_campaign`, `score_copy` orchestrator tool, score badge UI). CLI: `eval.py score` (writes per-config Parquets) + `eval.py score-summary` (per-config / per-segment aggregation). First-run results in `docs/research/SCORING_PREDICTOR_PHASE2_RESULTS_2026-05.md`.

## Output heads

| Head | Target | Notes |
|------|--------|-------|
| `composite` | v3 composite score [0, 1] | Primary head; drives early stopping (Pearson on val) |
| `survivability` | KM-derived percentile-ranked [0, 1] | Already normalized; no extra transform needed |
| `engagement_volume` | v3 engagement_volume signal [0, 1] | Masked for Reddit + "other" (see below) |
| `engagement_velocity` | v3 engagement_velocity signal [0, 1] | Masked for Reddit + "other" (see below) |

The 4-head design preserves explainability: when a score shifts, the per-head breakdown shows whether it is driven by survivability or engagement.

## Platform masking

Reddit and "other" platforms are excluded from the v3 engagement signals (the `HybridScorer` redistributes their weights to survivability for these platforms). Training mirrors this: the `engagement_volume` and `engagement_velocity` loss terms are masked to zero for Reddit and "other" rows. The `composite` and `survivability` heads are always trained.

A hard guard in `inference.py` emits a warning when scoring Reddit ads and notes the reduced signal set.

## Text input format

Each ad is represented as a single string:

```
[platform] [vertical] | headline \n body \n description
```

Empty fields are omitted. Platform and vertical are prepended as context tokens so the model learns platform-conditional behavior without per-platform model splits (Reddit has 3,390 rows — data-starved for its own model).

## Split strategies

Three splits are materialized to `data/scoring_predictor/splits/`:

| Split name | Description | What it tests |
|------------|-------------|---------------|
| `random` | 80/10/10 stratified random | Overall regressor quality; primary validation gate |
| `heldout-platform` | Reddit held out as test set | Generalization to an engagement-signal-masked platform |
| `heldout-vertical` | Facebook (largest vertical) held out as test set | Generalization across ad verticals |

## How to run

Install the extra dependency group:

```bash
uv pip install -e ".[scoring-predictor]"
```

Full workflow:

```bash
# 1. Materialize splits (fast, CPU-only)
uv run python scripts/predict.py splits

# 2. Train on a split (~30–45 min on RTX 3060 Mobile, $0 cost)
uv run python scripts/predict.py train --split random

# 3. Fit calibrators and evaluate offline
uv run python scripts/predict.py eval-offline --split random

# 4. Repeat for held-out splits
uv run python scripts/predict.py train --split heldout-platform
uv run python scripts/predict.py eval-offline --split heldout-platform

uv run python scripts/predict.py train --split heldout-vertical
uv run python scripts/predict.py eval-offline --split heldout-vertical

# 5. Run inference on a JSONL of ad texts
uv run python scripts/predict.py predict --input path/to/ads.jsonl
```

Config: `configs/scoring_predictor.yaml` (model name, lr, batch size, epochs, head weights, sample-weight column).

## Data on disk

```
data/scoring_predictor/
  splits/          # Parquet files: {random,heldout-platform,heldout-vertical}/{train,val,test}.parquet
  checkpoints/     # HF Trainer checkpoints + isotonic calibrators per split
  logs/            # Training logs (loss curves, validation Pearson per epoch)
```

Source corpus: `data/scored/v3/scored_ads.parquet` (~55k rows).
Pretrained weights: `microsoft/deberta-v3-base`, downloaded to `~/.cache/huggingface/` (~600 MB, public, no HF account required).
Checkpoints + splits: ~5–10 GB total.

## Validation gates (Phase 2 entry criteria)

`eval-offline` reports the following. All gates must pass before wiring into `evaluation/`.

| Metric | Threshold | If failed |
|--------|-----------|-----------|
| `random` split Spearman on `composite` | ≥ 0.55 | < 0.45: abandon — text alone doesn't carry enough signal |
| `heldout-platform` (Reddit) Spearman | ≥ 0.30 | Set hard Reddit-unsupported guard in `inference.py`; do not open Phase 2 |
| Calibration ECE on `composite` after isotonic | < 0.05 | Required before any score is shown in a report or product surface |
| Predicted-score variance on ~500 Draper outputs | > 0.01, not bimodal at extremes | Model is broken on synthetic inputs; do not ship to Phase 2 |

## `eval-offline` output format

A JSON report written to `data/scoring_predictor/checkpoints/{split}/eval_report.json`:

```json
{
  "split": "random",
  "heads": {
    "composite":          {"spearman": 0.0, "pearson": 0.0, "mae": 0.0},
    "survivability":      {"spearman": 0.0, "pearson": 0.0, "mae": 0.0},
    "engagement_volume":  {"spearman": 0.0, "pearson": 0.0, "mae": 0.0},
    "engagement_velocity":{"spearman": 0.0, "pearson": 0.0, "mae": 0.0}
  },
  "per_platform": {
    "facebook": {"composite_spearman": 0.0, "n": 0},
    "reddit":   {"composite_spearman": 0.0, "n": 0}
  },
  "calibration": {"ece_composite": 0.0},
  "tier_auc": {"top_0.75": 0.0, "bottom_0.25": 0.0}
}
```

## Serving

The predictor is exposed as an HTTP service so the Next.js frontend can call it at request time without importing Python. Both paths use the same `build_app` factory in `server.py`.

### Local dev (Service #6 "Scorer" in the dev dashboard)

The dev dashboard (`infra/start-dev.sh`) auto-starts the local server on port 8001 and writes `SCORING_PREDICTOR_URL=http://localhost:8001` and `SCORING_PREDICTOR_API_KEY=dev-local-not-secret` to `frontend/.env.local` when those vars are blank.

To start it manually:

```bash
uv run python scripts/serve_scoring_predictor.py \
    --checkpoint data/scoring_predictor/checkpoints/random/best \
    --port 8001 \
    --api-key dev-local-not-secret
```

The server exits non-zero with a clear message if the checkpoint directory is missing (train one first with `scripts/predict.py train --split random`).

### Production (Modal)

```bash
# One-time: create the API key secret
modal secret create scoring-predictor-api-key SCORING_PREDICTOR_API_KEY=<random>

# After each retrain: push weights into the Modal Volume
modal run deploy/modal_scoring_predictor.py::upload_checkpoint

# Deploy / redeploy the service
modal deploy deploy/modal_scoring_predictor.py
```

Modal prints a stable HTTPS URL (`*--draper-scoring-predictor-serve.modal.run`). Set `SCORING_PREDICTOR_URL` to that URL and `SCORING_PREDICTOR_API_KEY` to the secret value in the production frontend env.

### Wire format

```
POST /score
X-API-Key: <key>
{"items": [{"platform": "meta", "vertical": "ecommerce", "headline": "...", "body": "..."}]}

→ {"scores": [{"composite": 0.72, "survivability": 0.68, "engagement_volume": 0.74, "engagement_velocity": 0.71}], "latency_ms": 31.4}
```

`GET /healthz` returns `{"status": "ok", "checkpoint": "<label>"}` once the model is loaded.

Max 64 items per `/score` call. The frontend `score_copy` tool caps at 32 snippets.

## Public API

```python
from draper.scoring_predictor import load_predictor, score_text

predictor = load_predictor("data/scoring_predictor/checkpoints/random/")
scores = score_text(
    headline="Your headline",
    body="Body copy here",
    description="Description",
    platform="facebook",
    vertical="ecommerce",
    predictor=predictor,
)
# scores: {"composite": 0.72, "survivability": 0.68, "engagement_volume": 0.74, "engagement_velocity": 0.71}
```

Inference is CPU-friendly (~30ms/ad); a full eval pass (~1,300 inferences) completes in under a minute on laptop CPU.
