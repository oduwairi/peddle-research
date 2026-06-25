#!/usr/bin/env bash
# Sequential v2 bake-off driver: trains three QLoRA arms on the v2 dataset.
#
#   Arm A — Qwen3-8B           (configs/training_v2_qwen.yaml)
#   Arm B — Gemma 4 bare-tag   (configs/training_v2.yaml)
#   Arm C — Gemma 4 native ch. (configs/training_v2_gemma_native.yaml)
#
# Each arm is independent: inspect → smoke --push → train --push → merge --push.
# Per-arm logs land under outputs/draper-v2-${arm}/bakeoff.log. HF Hub pushes
# go to ${HF_HUB_REPO_OWNER}/draper-v2-${arm} (one repo per arm).
#
# Assumes the environment is already prepared (uv venv active, dependencies
# installed, HF token logged in). Use scripts/train_cloud.sh first if you're
# bootstrapping a fresh cloud pod.
#
# Required env vars:
#   HF_HUB_REPO_OWNER   HuggingFace user/org for the per-arm repos
#                       (e.g. "oduwairi"). Three repos will be created on
#                       first push: draper-v2-{qwen,gemma_bare,gemma_native}.
#
# Optional env vars:
#   ARMS=qwen,gemma_bare,gemma_native    Comma-separated subset to run.
#                                         Default: all three.
#   SKIP_SMOKE=1                          Skip the per-arm smoke (don't unless
#                                         you've already validated the env).
#   SKIP_PUSH=1                           Train + merge but don't push.
#   SKIP_MERGE=1                          Skip the final merge step per arm.
#   SKIP_NATIVE_RENDER=1                  Skip re-rendering the Gemma-native
#                                         dataset (assumes it's already on disk).
#   NTFY_TOPIC=...                        ntfy.sh topic for progress pings.
#
# Usage:
#   export HF_TOKEN=hf_xxx HF_HUB_REPO_OWNER=oduwairi
#   bash scripts/train_v2_bakeoff.sh
#
# Run a single arm:
#   ARMS=qwen bash scripts/train_v2_bakeoff.sh

set -euo pipefail

# ---------- 0. sanity -------------------------------------------------
if [[ -z "${HF_HUB_REPO_OWNER:-}" && -z "${SKIP_PUSH:-}" ]]; then
    echo "ERROR: HF_HUB_REPO_OWNER is not set (or set SKIP_PUSH=1)." >&2
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" && -z "${SKIP_PUSH:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set (or set SKIP_PUSH=1)." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found — are you sure this pod has a GPU?" >&2
fi

# ---------- helpers ---------------------------------------------------
log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

notify() {
    local msg="$1"
    if [[ -n "${NTFY_TOPIC:-}" ]]; then
        curl -fsS -m 10 \
            -H "Title: Draper.ai v2 bake-off" \
            -d "$msg" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
    fi
    log "$msg"
}

on_error() {
    local exit_code=$?
    local lineno=$1
    if [[ -n "${NTFY_TOPIC:-}" ]]; then
        curl -fsS -m 10 \
            -H "Title: Draper.ai v2 bake-off FAILED" \
            -H "Priority: high" \
            -H "Tags: warning,skull" \
            -d "Error at line $lineno (exit $exit_code). See outputs/draper-v2-*/bakeoff.log." \
            "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
    fi
}
trap 'on_error $LINENO' ERR

PUSH_FLAG=""
if [[ -z "${SKIP_PUSH:-}" ]]; then
    PUSH_FLAG="--push"
fi

# ---------- 1. arm definitions ----------------------------------------
# arm_id -> "config_path|output_dir"
declare -A ARM_CONFIG=(
    [qwen]="configs/training_v2_qwen.yaml|outputs/draper-v2-qwen3"
    [gemma_bare]="configs/training_v2.yaml|outputs/draper-v2-gemma4-bare"
    [gemma_native]="configs/training_v2_gemma_native.yaml|outputs/draper-v2-gemma4-native"
)

IFS=',' read -r -a ARMS_LIST <<< "${ARMS:-qwen,gemma_bare,gemma_native}"

# ---------- 2. pre-flight: render Gemma-native dataset ----------------
# Only if Arm C is in the run list and the dataset isn't already on disk
# (or SKIP_NATIVE_RENDER is set).
NATIVE_DATASET=data/constructed_v2/final_v2_gemma_native
NATIVE_REQUESTED=0
for a in "${ARMS_LIST[@]}"; do
    if [[ "$a" == "gemma_native" ]]; then
        NATIVE_REQUESTED=1
    fi
done

if [[ "$NATIVE_REQUESTED" == "1" && -z "${SKIP_NATIVE_RENDER:-}" ]]; then
    if [[ -d "$NATIVE_DATASET" ]]; then
        log "Gemma-native dataset already exists at $NATIVE_DATASET (use SKIP_NATIVE_RENDER=1 to keep)"
    else
        log "Pre-flight: rendering Gemma-native dataset (Arm C input)"
        if ! python scripts/construct_v2/render_for_gemma.py; then
            log "Gemma-native render FAILED — dropping Arm C from the run"
            notify "Gemma-native render failed; running Qwen + Gemma-bare only"
            NEW_ARMS=()
            for a in "${ARMS_LIST[@]}"; do
                if [[ "$a" != "gemma_native" ]]; then
                    NEW_ARMS+=("$a")
                fi
            done
            ARMS_LIST=("${NEW_ARMS[@]}")
        fi
    fi
fi

# ---------- 3. per-arm loop -------------------------------------------
run_arm() {
    local arm="$1"
    local config="${ARM_CONFIG[$arm]%%|*}"
    local outdir="${ARM_CONFIG[$arm]##*|}"
    local repo="${HF_HUB_REPO_OWNER:-}/draper-v2-${arm}"
    local logfile="${outdir}/bakeoff.log"

    mkdir -p "$outdir"

    log "=========================================================="
    log "Arm: ${arm}    config: ${config}    repo: ${repo}"
    log "Tail logs:   tail -f ${logfile}"
    log "=========================================================="

    # Scope HF_HUB_REPO to this arm only.
    local -a env_kv=(HF_HUB_REPO="$repo")

    # Inspect (always — quick, surfaces template-skew before any GPU work).
    log "[${arm}] inspect"
    env "${env_kv[@]}" python scripts/train.py inspect --config "$config" 2>&1 | tee -a "$logfile"

    # Smoke (push to ${repo}/smoke-test/ to also validate HF auth).
    if [[ -z "${SKIP_SMOKE:-}" ]]; then
        log "[${arm}] smoke ${PUSH_FLAG}"
        env "${env_kv[@]}" python scripts/train.py smoke --config "$config" ${PUSH_FLAG} \
            2>&1 | tee -a "$logfile"
        notify "[${arm}] smoke passed"
    fi

    # Full training run.
    log "[${arm}] train ${PUSH_FLAG}"
    env "${env_kv[@]}" python scripts/train.py train --config "$config" ${PUSH_FLAG} \
        2>&1 | tee -a "$logfile"
    notify "[${arm}] training complete → https://huggingface.co/${repo}/tree/main/adapter"

    # Merge: locate the most recent non-smoke run dir + its final/ adapter.
    if [[ -z "${SKIP_MERGE:-}" ]]; then
        local adapter_dir
        adapter_dir="$(ls -1dt "${outdir}"/r*-dora-* 2>/dev/null | grep -v -- '-smoke' | head -1)/final"
        if [[ ! -d "$adapter_dir" ]]; then
            echo "[${arm}] ERROR: could not locate adapter dir under ${outdir}" | tee -a "$logfile" >&2
            return 1
        fi
        log "[${arm}] merge ${adapter_dir} ${PUSH_FLAG}"
        env "${env_kv[@]}" python scripts/train.py merge \
            --adapter "$adapter_dir" \
            --config "$config" \
            --save-method merged_16bit \
            ${PUSH_FLAG} 2>&1 | tee -a "$logfile"
        notify "[${arm}] merged weights pushed → https://huggingface.co/${repo}/tree/main/merged"
    fi
}

for arm in "${ARMS_LIST[@]}"; do
    if [[ -z "${ARM_CONFIG[$arm]:-}" ]]; then
        echo "ERROR: unknown arm '${arm}' (valid: qwen, gemma_bare, gemma_native)" >&2
        exit 1
    fi
    run_arm "$arm"
done

notify "Bake-off complete. Arms: ${ARMS_LIST[*]}"
log "All arms done. Per-arm logs:"
for arm in "${ARMS_LIST[@]}"; do
    outdir="${ARM_CONFIG[$arm]##*|}"
    log "  ${arm} → ${outdir}/bakeoff.log"
done
