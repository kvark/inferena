#!/usr/bin/env bash
# JAX benchmark runner for inferena.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-SmolLM2-135M}"

# --- Check dependencies ---
if ! python3 -c "import jax" 2>/dev/null; then
    echo "[jax] jax not found. Install via: pip install jax jaxlib" >&2
    exit 1
fi

exec python3 "$SCRIPT_DIR/bench.py" "$MODEL"
