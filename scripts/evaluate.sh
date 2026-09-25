#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: bash scripts/evaluate.sh CHECKPOINT TEST_RECOGNIZER_OUTPUT_DIR FEATURE_ROOT OUTPUT_DIR [DEVICE]" >&2
  exit 2
fi

checkpoint=$1
trajectory_dir=$2
feature_root=$3
output_dir=$4
device=${5:-cpu}

if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint not found: $checkpoint" >&2
  exit 2
fi

python evaluate_checkpoint.py \
  --checkpoint "$checkpoint" \
  --trajectory-dir "$trajectory_dir" \
  --feature-root "$feature_root" \
  --output-dir "$output_dir" \
  --device "$device"
