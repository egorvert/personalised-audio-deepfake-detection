#!/bin/bash
# Train the WavLM + AASIST fusion head. Backbones stay frozen by default.

DATA_ROOT="${DATA_ROOT:-./asvspoof_dataset/LA/LA}"

python -m vdetect.train_fusion \
    --data-root "${DATA_ROOT}" \
    --epochs 10 \
    --batch-size 8 \
    --lr 2e-4 \
    --wavlm-checkpoint assets/checkpoints/wavlm_baseline.pt \
    --grad-clip 1.0 \
    --out assets/checkpoints/two_stream.pt \
    "$@"
