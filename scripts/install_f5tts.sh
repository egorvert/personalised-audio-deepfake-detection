#!/usr/bin/env bash
# Install F5-TTS and pre-download its weights into the HuggingFace cache.
# Run on the deploy target before any deepfake generation. Requires an active
# Python venv and ~2 GB free disk. Idempotent — rerunning is safe.
#
# When running generate_deepfakes.py on Apple Silicon, also set:
# export PYTORCH_ENABLE_MPS_FALLBACK=1
# so unsupported ops fall back to CPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "ERROR: no active Python venv. Run 'source .venv/bin/activate' first." >&2
  exit 1
fi

echo "==> Installing F5-TTS dependencies from requirements-tts.txt"
pip install -r "$REPO_ROOT/requirements-tts.txt"

echo "==> Pre-downloading F5-TTS model weights (SWivid/F5-TTS, ~1.5 GB)"
huggingface-cli download SWivid/F5-TTS

echo "==> Verifying f5_tts import"
python - <<'PY'
import f5_tts  # noqa: F401
print("f5_tts import OK")
PY

echo "==> F5-TTS install complete."
