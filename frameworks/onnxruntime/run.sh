#!/usr/bin/env bash
# ONNX Runtime benchmark runner for inferena.
# Exports model to ONNX via PyTorch, then benchmarks with onnxruntime.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-SmolLM2-135M}"

# --- Check dependencies ---
if ! python3 -c "import onnxruntime" 2>/dev/null; then
    echo "[onnxruntime] onnxruntime not found. Install via: pip install onnxruntime onnx" >&2
    echo "[onnxruntime] Or install all deps: pip install -r \$ROOT_DIR/requirements.txt" >&2
    exit 1
fi

exec python3 "$SCRIPT_DIR/bench.py" "$MODEL"
