#!/usr/bin/env bash
# Ablation harness: run smoke under 4 (padding_free, use_liger_kernel)
# combinations to isolate the Run #001 collapse cause.
#
#   FF — both off (current safe config). Should pass.
#   TF — padding_free only. Isolates that flag.
#   FT — liger only. Isolates that flag.
#   TT — both on. Closest to Run #001's broken combo.
#
# Per-run output: outputs/ablate/<run>.log
# Final summary: prints loss at steps 10/30/50 + outcome per run.
# Runtime: ~20 min on RTX 3060 (4 × ~5 min).

set -uo pipefail
cd "$(dirname "$0")/.."

ABLATE_DIR=outputs/ablate
mkdir -p "$ABLATE_DIR"

RUNS=("FF" "TF" "FT" "TT")
declare -A PF=(["FF"]=false ["TF"]=true  ["FT"]=false ["TT"]=true)
declare -A LIGER=(["FF"]=false ["TF"]=false ["FT"]=true  ["TT"]=true)

BASE_CONFIG=configs/training.yaml

for run in "${RUNS[@]}"; do
  cfg="$ABLATE_DIR/training_${run}.yaml"
  log="$ABLATE_DIR/${run}.log"
  pf=${PF[$run]}
  liger=${LIGER[$run]}

  echo
  echo "===== ABLATE $run: padding_free=$pf use_liger_kernel=$liger ====="

  sed -E \
    -e "s/^(  padding_free:).*/\1 $pf/" \
    -e "s/^(  use_liger_kernel:).*/\1 $liger/" \
    -e "s/^(  trackio_project:).*/\1 draper-ablate-$run/" \
    "$BASE_CONFIG" > "$cfg"

  uv run python scripts/train.py smoke --config "$cfg" 2>&1 | tee "$log"
  echo "===== $run exit code: ${PIPESTATUS[0]} =====" | tee -a "$log"
done

echo
echo "===== ABLATION SUMMARY ====="
printf "%-4s %-3s %-5s %-12s %-12s %-12s %s\n" "RUN" "PF" "LIGER" "STEP10" "STEP30" "STEP50" "OUTCOME"
for run in "${RUNS[@]}"; do
  log="$ABLATE_DIR/${run}.log"
  pf=${PF[$run]}
  liger=${LIGER[$run]}

  # TRL prints losses as Python dicts: {'loss': '2.947', ...}. Value in quotes.
  extract_step() {
    grep -oE "'loss': '[0-9.eE+-]+'" "$log" | sed -n "${1}p" | grep -oE "[0-9.eE+-]+"
  }
  s10=$(extract_step 10)
  s30=$(extract_step 30)
  s50=$(extract_step 50)

  outcome="ok"
  grep -qE "Traceback|LOSS_COLLAPSE|CUDA out of memory" "$log" && outcome="ERROR"
  for v in "$s10" "$s30" "$s50"; do
    [ -n "$v" ] && awk -v x="$v" 'BEGIN{exit !(x+0 < 0.05)}' && outcome="COLLAPSE"
  done

  printf "%-4s %-3s %-5s %-12s %-12s %-12s %s\n" \
    "$run" "$pf" "$liger" "${s10:-?}" "${s30:-?}" "${s50:-?}" "$outcome"
done
