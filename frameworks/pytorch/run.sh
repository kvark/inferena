#!/usr/bin/env bash
# PyTorch benchmark runner wrapper.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-SmolLM2-135M}"

# Use venv if it exists, otherwise try system python.
if [ -d "$SCRIPT_DIR/.venv" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    echo "[pytorch] python3 not found" >&2
    exit 1
fi

# Check torch is importable.
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
    echo "[pytorch] torch not installed. Run: pip install -r $ROOT_DIR/requirements-cpu.txt" >&2
    echo "[pytorch] Or create a venv: python3 -m venv .venv && .venv/bin/pip install -r $ROOT_DIR/requirements-cpu.txt" >&2
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/bench.py" "$MODEL"
