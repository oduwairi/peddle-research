#!/usr/bin/env bash
# Rsync the project files from this laptop to a rented GPU pod.
#
# This is the "send the code + dataset" step. Run it from the laptop, not the
# pod. Sends only what training needs: src/, scripts/, configs/, pyproject.toml,
# README.md, and data/final/ (the HF DatasetDict). Skips frontend, raw scrapes,
# old training outputs, virtualenvs, etc.
#
# Required:
#   POD_HOST         SSH target, e.g. root@123.45.67.89  (or the host alias from
#                    your ~/.ssh/config — RunPod gives you the full ssh command).
#
# Optional:
#   POD_PORT         SSH port if non-default (RunPod gives you a custom port).
#   POD_DIR          Where on the pod to write to (default: ~/Draper.ai).
#   SSH_KEY          Path to the private SSH key (default: ~/.ssh/id_ed25519
#                    or whatever ssh picks up automatically).
#   DRY_RUN=1        Print what rsync would do, don't actually transfer.
#
# Usage:
#   POD_HOST=root@1.2.3.4 POD_PORT=12345 bash scripts/upload_to_pod.sh

set -euo pipefail

if [[ -z "${POD_HOST:-}" ]]; then
    echo "ERROR: POD_HOST is not set. Example: POD_HOST=root@1.2.3.4 bash $0" >&2
    exit 1
fi

# Default to the persistent volume (/workspace) so anything we upload + the
# outputs it produces survive a pod stop.
POD_DIR="${POD_DIR:-/workspace/Draper.ai}"
RSYNC_OPTS=(-rlptDvz --partial --human-readable --no-owner --no-group)
if [[ -n "${DRY_RUN:-}" ]]; then
    RSYNC_OPTS+=(--dry-run)
fi

# Build the SSH command rsync uses internally.
SSH_CMD=(ssh -o StrictHostKeyChecking=accept-new)
if [[ -n "${POD_PORT:-}" ]]; then
    SSH_CMD+=(-p "$POD_PORT")
fi
if [[ -n "${SSH_KEY:-}" ]]; then
    SSH_CMD+=(-i "$SSH_KEY")
fi
RSYNC_OPTS+=(-e "${SSH_CMD[*]}")

# Source root (this script lives in scripts/, so go one up).
SRC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Whitelist of files/dirs we want on the pod. Anything not listed here is not
# transferred — keeps the upload tiny and avoids leaking unrelated stuff.
INCLUDE_PATHS=(
    "src/draper"
    "scripts/train.py"
    "scripts/train_cloud.sh"
    "scripts/train_cloud_bg.sh"
    "scripts/tail_train_metrics.py"
    "scripts/eval.py"
    "configs/eval.yaml"
    "pyproject.toml"
    "Makefile"
    "README.md"
)

# Some pod images don't ship rsync. Detect and apt-install before transferring,
# otherwise the rsync invocation below fails on the pod side without a clear
# error. Skip if DRY_RUN — no need to mutate the pod for a dry run.
if [[ -z "${DRY_RUN:-}" ]]; then
    if ! "${SSH_CMD[@]}" "$POD_HOST" "command -v rsync >/dev/null 2>&1"; then
        echo "==> Pod is missing rsync; installing via apt-get"
        "${SSH_CMD[@]}" "$POD_HOST" "apt-get update -qq && apt-get install -y -qq rsync"
    fi
fi

echo "==> Uploading to ${POD_HOST}:${POD_DIR}"
for p in "${INCLUDE_PATHS[@]}"; do
    if [[ ! -e "$SRC_ROOT/$p" ]]; then
        echo "WARNING: $p does not exist locally; skipping" >&2
        continue
    fi
    # Make sure the parent dir on the pod exists.
    parent_dir="$(dirname "$p")"
    if [[ "$parent_dir" != "." ]]; then
        "${SSH_CMD[@]}" "$POD_HOST" "mkdir -p '$POD_DIR/$parent_dir'"
    fi
    echo "  - $p"
    if [[ -d "$SRC_ROOT/$p" ]]; then
        rsync "${RSYNC_OPTS[@]}" "$SRC_ROOT/$p/" "$POD_HOST:$POD_DIR/$p/"
    else
        rsync "${RSYNC_OPTS[@]}" "$SRC_ROOT/$p" "$POD_HOST:$POD_DIR/$p"
    fi
done

echo "==> Upload complete. Now SSH into the pod and run scripts/train_cloud.sh."
