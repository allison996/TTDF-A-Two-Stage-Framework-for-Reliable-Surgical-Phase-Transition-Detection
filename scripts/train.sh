#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash scripts/train.sh RECOGNIZER_OUTPUT_ROOT FEATURE_ROOT OUTPUT_DIR [DEVICE]" >&2
  exit 2
fi

recognizer_root=$1
feature_root=$2
output_dir=$3
device=${4:-cuda}
epochs=${TTDF_EPOCHS:-50}
seed=${TTDF_SEED:-0}

for split in train val test; do
  if [[ ! -d "${recognizer_root}/${split}" || ! -d "${feature_root}/${split}" ]]; then
    echo "Missing ${split} directory under recognizer output or feature root" >&2
    exit 2
  fi
done

python train_transition_reliability.py \
  --train-dir "${recognizer_root}/train" \
  --val-dir "${recognizer_root}/val" \
  --test-dir "${recognizer_root}/test" \
  --feature-root "$feature_root" \
  --legality-gate train --dwell-duration 5 --tolerance 30 \
  --left-context 8 --feature-set none \
  --prepost-prob-cues --visual-change-cue \
  --encoder tcn --pooling mean --hidden-dim 32 --num-layers 1 \
  --dropout 0.1 --batch-size 128 --epochs "$epochs" \
  --lr 1e-3 --weight-decay 1e-4 --offset-weight 0 \
  --device "$device" --seed "$seed" --output-dir "$output_dir"
