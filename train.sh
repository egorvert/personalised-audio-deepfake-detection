#!/bin/bash
# Train the WavLM baseline detector. Override knobs via env vars.

set -e

DATA_ROOT="${DATA_ROOT:-asvspoof_dataset/LA/LA}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-2e-4}"
OUTPUT="${OUTPUT:-assets/checkpoints/wavlm_baseline.pt}"

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: data root not found at $DATA_ROOT"
    echo "Set DATA_ROOT or place the ASVspoof2019 LA dataset at asvspoof_dataset/LA/LA/"
    exit 1
fi

mkdir -p assets/checkpoints

python -m vdetect.train_wavlm \
    --data-root "$DATA_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --out "$OUTPUT" \
    --num-workers 4
