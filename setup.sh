#!/bin/bash
# Create a .venv and install the Python deps.

set -e

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Done. Activate with: source .venv/bin/activate"
echo "Then try:           python -m vdetect.train_wavlm --data-root asvspoof_dataset/LA/LA --epochs 10"
