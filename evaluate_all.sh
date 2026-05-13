#!/bin/bash
# Evaluate WavLM, AASIST and the fusion model on the dev set.

DATA_ROOT="${DATA_ROOT:-./asvspoof_dataset/LA/LA}"

python -m vdetect.evaluate_fusion \
    --data-root "${DATA_ROOT}" \
    --split dev \
    --batch-size 16 \
    --wavlm-weights assets/checkpoints/wavlm_baseline.pt \
    --fusion-weights assets/checkpoints/two_stream.pt \
    --models all \
    "$@"
