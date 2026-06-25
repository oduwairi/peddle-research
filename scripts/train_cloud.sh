#!/usr/bin/env bash
# Cloud-pod bootstrap for Draper.ai QLoRA training.
#
# Designed to be run inside a freshly-rented RunPod / Vast.ai / Lambda Labs
# GPU pod (Ubuntu 22.04 + CUDA 12.x, e.g. runpod/pytorch:2.4-py3.11-cuda12.4)
# AFTER you've rsynced the project files to the pod from your laptop with
# scripts/upload_to_pod.sh.
#
# Required env vars:
#   HF_TOKEN          HuggingFace token (write scope) — needed to push results.
#   HF_HUB_REPO       Target repo for upload, e.g. oduwairi/draper-qwen3-8b.
#
# Optional env vars:
#   REPO_DIR          Where the project lives on the pod
#                     (default: /workspace/Draper.ai — the persistent volume,
#                     so outputs survive a pod stop. Falls back to $HOME if
#                     /workspace is not a directory.)
#   SKIP_SMOKE=1      Skip the on-pod smoke test (don't, unless you've run it once).
#   SKIP_PUSH=1       Train but don't push to HF Hub.
#   SKIP_TRAIN=1      Run setup + smoke and stop — useful for assessing the pod
#                     before committing to a long full-training run.
#   MERGE=1           After train+push, merge adapter into 16bit weights and push merged/.
#   AUTO_SHUTDOWN=1   After successful training+push, halt the pod (best-effort:
#                     prefers `runpodctl stop pod $RUNPOD_POD_ID`, falls back to
#                     `shutdown -h now`). Pair with NTFY_TOPIC for a "done" ping.
#   NTFY_TOPIC=...    ntfy.sh topic name to ping at smoke pass / train complete /
#                     errors / shutdown. Subscribe at https://ntfy.sh/<topic> on
#                     phone/web — no signup. Pick something unguessable.
#
# Usage on the pod (after rsync), foreground:
#   export HF_TOKEN=hf_xxx HF_HUB_REPO=oduwairi/draper-qwen3-8b
#   cd /workspace/Draper.ai && bash scripts/train_cloud.sh
#
# Detached (close laptop and walk away):
#   bash scripts/train_cloud_bg.sh

set -euo pipefail

# Prefer /workspace (persistent volume) so outputs survive a pod stop.
DEFAULT_REPO_DIR="/workspace/Draper.ai"
if [[ ! -d /workspace ]]; then
    DEFAULT_REPO_DIR="$HOME/Draper.ai"
fi
REPO_DIR="${REPO_DIR:-$DEFAULT_REPO_DIR}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------- ntfy notifications ----------------------------------------
# Cheap pings to a phone/browser via ntfy.sh — opt-in via NTFY_TOPIC.
# Always exits 0 so a failed ping never aborts training.
notify() {
    local msg="$1"
    if [[ -n "${NTFY_TOPIC:-}" ]]; then
        curl -fsS -m 10 \
            -H "Title: Draper.ai training" \
            -d "$msg" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
    fi
    log "$msg"
}

on_error() {
    local exit_code=$?
    local lineno=$1
    if [[ -n "${NTFY_TOPIC:-}" ]]; then
        curl -fsS -m 10 \
            -H "Title: Draper.ai training FAILED" \
            -H "Priority: high" \
            -H "Tags: warning,skull" \
            -d "Error at line $lineno (exit $exit_code). Check /workspace/train.log." \
            "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
    fi
}
trap 'on_error $LINENO' ERR

auto_shutdown() {
    if [[ -z "${AUTO_SHUTDOWN:-}" ]]; then
        return
    fi
    log "AUTO_SHUTDOWN set — halting pod in 30s (Ctrl-C to abort)"
    notify "Auto-stopping pod in 30s. Adapter is on HF Hub: ${HF_HUB_REPO:-?}"
    sleep 30
    if command -v runpodctl >/dev/null 2>&1 && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
        log "Stopping via runpodctl"
        runpodctl stop pod "$RUNPOD_POD_ID" || shutdown -h now
    else
        log "runpodctl unavailable — falling back to shutdown -h now"
        shutdown -h now
    fi
}

# ---------- 0. sanity --------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it first." >&2
    exit 1
fi
if [[ -z "${HF_HUB_REPO:-}" && -z "${SKIP_PUSH:-}" ]]; then
    echo "ERROR: HF_HUB_REPO is not set (or set SKIP_PUSH=1)." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found — are you sure this pod has a GPU?" >&2
fi
if [[ ! -f "$REPO_DIR/pyproject.toml" || ! -d "$REPO_DIR/src/draper" ]]; then
    echo "ERROR: $REPO_DIR doesn't look like the Draper.ai project." >&2
    echo "       Run scripts/upload_to_pod.sh from your laptop first." >&2
    exit 1
fi
if [[ ! -d "$REPO_DIR/data/final" ]]; then
    echo "ERROR: $REPO_DIR/data/final missing — the training dataset wasn't uploaded." >&2
    echo "       Re-run scripts/upload_to_pod.sh; it should include data/final/." >&2
    exit 1
fi

# ---------- 1. system deps --------------------------------------------
log "Installing system deps"
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends curl ca-certificates build-essential
fi

# ---------- 2. uv -----------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

cd "$REPO_DIR"

# ---------- 3. python venv + training extras --------------------------
log "Creating venv and installing training extras (this is the slow step)"
if [[ ! -d .venv ]]; then
    uv venv --python 3.11
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[training]"

# ---------- 4. HF login (writes ~/.cache/huggingface/token) -----------
# `huggingface-cli` was retired; the replacement CLI is `hf`. Fall back to
# the Python API if neither binary is available.
log "Logging in to HuggingFace Hub"
if command -v hf >/dev/null 2>&1; then
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=True)"
fi

# ---------- 5. on-pod smoke test --------------------------------------
# `--push` exercises the HF Hub upload code path with a tiny adapter so we
# catch token/repo/perm bugs before paying for a 2-hour run. The smoke
# artifact lands under HF_HUB_REPO/smoke-test/ so it doesn't pollute the
# real adapter path.
if [[ -z "${SKIP_SMOKE:-}" ]]; then
    log "Smoke test (tiny model, 2 steps, push to smoke-test/ to validate HF)"
    PUSH_FLAG=""
    if [[ -z "${SKIP_PUSH:-}" ]]; then
        PUSH_FLAG="--push"
    fi
    python scripts/train.py smoke ${PUSH_FLAG}
    notify "Smoke pass — full training starting on $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo GPU)"
fi

# ---------- 6. full training run --------------------------------------
if [[ -n "${SKIP_TRAIN:-}" ]]; then
    log "SKIP_TRAIN set — stopping after smoke. Re-run without SKIP_TRAIN to start the full QLoRA run."
    exit 0
fi
log "Full QLoRA training run on $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
PUSH_FLAG=""
if [[ -z "${SKIP_PUSH:-}" ]]; then
    PUSH_FLAG="--push"
fi
python scripts/train.py train ${PUSH_FLAG}
notify "Training complete — adapter pushed to https://huggingface.co/${HF_HUB_REPO:-?}/tree/main/adapter"

# ---------- 7. optional: merge + push merged weights ------------------
if [[ -n "${MERGE:-}" ]]; then
    log "Locating final adapter to merge"
    ADAPTER_DIR="$(ls -1dt outputs/qwen3-8b-copywriting/r*-dora-* | grep -v -- '-smoke' | head -1)/final"
    if [[ ! -d "$ADAPTER_DIR" ]]; then
        echo "ERROR: could not find a non-smoke adapter dir to merge" >&2
        exit 1
    fi
    log "Merging adapter at $ADAPTER_DIR"
    python scripts/train.py merge --adapter "$ADAPTER_DIR" ${PUSH_FLAG}
    notify "Merged 16-bit weights pushed"
fi

log "Done."
auto_shutdown
log "Pod still running (no AUTO_SHUTDOWN). Stop it from the RunPod UI to halt billing."
