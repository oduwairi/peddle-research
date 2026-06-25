#!/usr/bin/env bash
# Detached wrapper for train_cloud.sh — fire it, close your laptop, walk away.
#
# Starts train_cloud.sh under nohup so the SSH session can disconnect without
# killing training. Output streams to /workspace/train.log on the pod's
# persistent volume disk so it survives a pod stop.
#
# Pre-reqs: same env vars as train_cloud.sh (HF_TOKEN, HF_HUB_REPO, optionally
# NTFY_TOPIC and AUTO_SHUTDOWN). They must be exported in the calling shell —
# typically loaded from /workspace/Draper.ai/.env.train.
#
# Usage on the pod:
#   cd /workspace/Draper.ai
#   set -a; . ./.env.train; set +a
#   bash scripts/train_cloud_bg.sh
#
# Or in one SSH call from your laptop:
#   ssh ... 'cd /workspace/Draper.ai && set -a && . ./.env.train && set +a && bash scripts/train_cloud_bg.sh'
#
# To watch progress later: ssh in, then `tail -f /workspace/train.log`.
# To know it's done: subscribe to your NTFY_TOPIC, or watch HF Hub for the
# new adapter folder.

set -e

REPO_DIR="${REPO_DIR:-/workspace/Draper.ai}"
LOG_FILE="${LOG_FILE:-/workspace/train.log}"

if [[ -z "${HF_TOKEN:-}" || -z "${HF_HUB_REPO:-}" ]]; then
    echo "ERROR: export HF_TOKEN and HF_HUB_REPO first (e.g. via . .env.train)." >&2
    exit 1
fi

cd "$REPO_DIR"

# Truncate the log so we see only this run's output.
: > "$LOG_FILE"

# nohup + & detaches from the parent shell so closing SSH won't SIGHUP us.
# The env -i trick isn't used because we want to inherit the loaded env vars
# (HF_TOKEN etc.); instead we just let nohup/bash carry them forward.
nohup bash scripts/train_cloud.sh > "$LOG_FILE" 2>&1 &
PID=$!
disown "$PID"

# Sleep briefly so the script gets past argv parsing AND the python child has
# spawned, then dump the first few lines so the user sees it actually started.
# We sleep a bit longer (10s) than before so the python PID is reliably visible
# to pgrep — it usually takes 5–8s for `uv run python ... train.py` to start
# after the bash parent fires.
sleep 10

# Try to find the actual python child. This is what eats the GPU and what you
# need to kill to actually stop training. The bash parent (PID $PID) does NOT
# manage the python subprocess as a controlled child — `train_cloud.sh` execs
# `uv run python ...`, which makes the python its own session leader. Killing
# the bash parent leaves the python orphaned but ALIVE.
#
# Footgun encountered in Run #002 (2026-05): user ran `kill $BASH_PID`, bash
# exited cleanly, but training kept running on the GPU. Always kill the python
# PID (or both) when stopping a run.
PYTHON_PID="$(pgrep -f 'python.*train.py' | head -1 || true)"

echo "Started train_cloud.sh in background"
echo "  bash parent PID:   $PID"
if [[ -n "$PYTHON_PID" ]]; then
    echo "  python child PID:  $PYTHON_PID  ← this one holds the GPU"
else
    echo "  python child PID:  (not yet visible — re-run pgrep -f 'python.*train.py' in 30s)"
fi
echo "Log: $LOG_FILE"
echo "Tail it with:  tail -f $LOG_FILE"
echo
echo "To kill the run cleanly, kill BOTH PIDs (python first):"
if [[ -n "$PYTHON_PID" ]]; then
    echo "    kill $PYTHON_PID && kill $PID"
else
    echo "    kill \$(pgrep -f 'python.*train.py') && kill $PID"
fi
echo "Killing only the bash parent ($PID) will NOT stop training."
echo
echo "=== first $(wc -l < "$LOG_FILE") log lines ==="
head -20 "$LOG_FILE" 2>/dev/null || echo "(log empty — script just started)"
echo
echo "Safe to close this SSH session. Training will continue."
