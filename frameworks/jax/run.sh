#!/usr/bin/env bash
# JAX benchmark runner for inferena.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-SmolLM2-135M}"

# --- Check dependencies ---
if ! python3 -c "import jax" 2>/dev/null; then
    echo "[jax] jax not found. Install via:" >&2
    echo "  pip install jax                       # CPU only" >&2
    echo "  pip install jax[cuda12]               # NVIDIA GPU" >&2
    echo "  pip install jax-metal                  # Apple Metal" >&2
    exit 1
fi

# Log which backend JAX will use.
python3 -c "import jax; print(f'[jax] backend: {jax.default_backend()}, devices: {jax.devices()}')" >&2

exec python3 "$SCRIPT_DIR/bench.py" "$MODEL"
