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

# Log backend and warn if GPU is available but not used.
python3 -c "
import jax, sys
backend = jax.default_backend()
print(f'[jax] backend: {backend}, devices: {jax.devices()}', file=sys.stderr)
if backend == 'cpu':
    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    except ImportError:
        import subprocess, shutil
        has_gpu = subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0 if shutil.which('nvidia-smi') else False
    if not has_gpu and sys.platform == 'darwin':
        has_gpu = True  # macOS always has Metal
    if has_gpu:
        print('[jax] WARNING: GPU detected but JAX is using CPU backend.', file=sys.stderr)
        print('  Install: pip install jax[cuda12]      (NVIDIA)', file=sys.stderr)
        print('       or: pip install jax-metal         (Apple)', file=sys.stderr)
" 2>/dev/null || true

exec python3 "$SCRIPT_DIR/bench.py" "$MODEL"
