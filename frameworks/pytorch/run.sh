#!/usr/bin/env bash
# PyTorch benchmark runner wrapper.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-SmolLM2-135M}"

# Create venv if it doesn't exist.
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "[pytorch] Creating virtual environment..." >&2
    python3 -m venv "$SCRIPT_DIR/.venv"
    "$SCRIPT_DIR/.venv/bin/pip" install --quiet torch transformers >&2
fi

exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/bench.py" "$MODEL"
