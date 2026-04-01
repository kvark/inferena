#!/usr/bin/env bash
# JAX benchmark runner for inferena.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-SmolLM2-135M}"

# --- Check dependencies ---
python3 -c "import jax" 2>/dev/null || {
    echo "[jax] installing jax..." >&2
    pip install jax jaxlib --quiet 2>&1 >&2
}

exec python3 "$SCRIPT_DIR/bench.py" "$MODEL"
