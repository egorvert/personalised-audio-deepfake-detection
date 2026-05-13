#!/usr/bin/env bash
# PM2 entry wrapper for the FastAPI server. Wrapped because PM2's `script:`
# field doesn't pass `python -m <module>` cleanly, and running the package
# file directly shadows the stdlib `logging` module with vdetect/logging.py.

set -euo pipefail

cd "$(dirname "$0")/.."

exec .venv/bin/python -m vdetect
