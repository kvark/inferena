#!/usr/bin/env bash
# JAX benchmark runner for inferena.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-SmolLM2-135M}"

: "${PYTHON:?must be set by run.sh}"

# --- Check dependencies ---
if ! "$PYTHON" -c "import jax" 2>/dev/null; then
    echo "[jax] jax not found. Install via:" >&2
    echo "  pip install jax                       # CPU only" >&2
    echo "  pip install jax[cuda12]               # NVIDIA GPU" >&2
    echo "  pip install jax-metal                  # Apple Metal" >&2
    exit 1
fi

# Log backend and warn if GPU is available but not used.
"$PYTHON" -c "
import jax, sys
backend = jax.default_backend()
print(f'[jax] backend: {backend}, devices: {jax.devices()}', file=sys.stderr)
if backend == 'cpu':
    has_gpu = False
    is_amd = False
    try:
        import torch
        has_gpu = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
        if hasattr(torch.version, 'hip') and torch.version.hip:
            is_amd = True
    except ImportError:
        import subprocess, shutil
        if shutil.which('nvidia-smi'):
            has_gpu = subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
        elif shutil.which('rocm-smi'):
            has_gpu = True
            is_amd = True
    if not has_gpu and sys.platform == 'darwin':
        has_gpu = True
    if has_gpu:
        print('[jax] WARNING: GPU detected but JAX is using CPU backend.', file=sys.stderr)
        if sys.platform == 'darwin':
            print('  Install: pip install jax-metal         (Apple Metal)', file=sys.stderr)
        elif is_amd:
            print('  Install: pip install jax[rocm]         (AMD ROCm)', file=sys.stderr)
        else:
            print('  Install: pip install jax[cuda12]       (NVIDIA CUDA)', file=sys.stderr)
" 2>/dev/null || true

# Pre-flight: detect if CUDA backend will crash (e.g. Blackwell GPU + old jaxlib PTX).
# JAX's XLA compiler calls abort() on LLVM errors, so we must test in a subprocess.
if "$PYTHON" -c "import jax; exit(0 if jax.default_backend() == 'gpu' else 1)" 2>/dev/null; then
    _JAX_CUDA_OK=$(timeout 15 "$PYTHON" -c "
import jax
x = jax.numpy.ones(1)
_ = (x + x).block_until_ready()
print('ok')
" 2>&1) || _JAX_CUDA_OK="fail"

    if ! echo "$_JAX_CUDA_OK" | grep -q "^ok$"; then
        echo "[jax] CUDA backend crashed (likely GPU too new for installed jaxlib PTX) — falling back to CPU" >&2
        export JAX_PLATFORMS="cpu"
    fi
fi

# Pre-flight: detect if Metal backend is broken (e.g. jax-metal StableHLO version mismatch).
if "$PYTHON" -c "import jax; exit(0 if jax.default_backend() == 'METAL' else 1)" 2>/dev/null; then
    _JAX_METAL_OK=$("$PYTHON" -c "
import jax.numpy as jnp
_ = (jnp.ones(1) + jnp.ones(1)).block_until_ready()
print('ok')
" 2>&1) || _JAX_METAL_OK="fail"

    if ! echo "$_JAX_METAL_OK" | grep -q "^ok$"; then
        echo "[jax] Metal backend broken (likely jax-metal version mismatch) — falling back to CPU" >&2
        export JAX_PLATFORMS="cpu"
    fi
fi

exec "$PYTHON" "$SCRIPT_DIR/bench.py" "$MODEL"
