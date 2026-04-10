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
    echo "[onnxruntime] onnxruntime not found. Install via:" >&2
    echo "  pip install onnxruntime              # CPU only" >&2
    echo "  pip install onnxruntime-gpu           # NVIDIA CUDA" >&2
    echo "  pip install onnxruntime-rocm          # AMD ROCm" >&2
    exit 1
fi

# Warn if GPU is available but only CPU provider is installed.
python3 -c "
import onnxruntime as ort, sys
providers = ort.get_available_providers()
gpu_providers = [p for p in providers if p != 'CPUExecutionProvider']
if gpu_providers:
    print(f'[onnxruntime] GPU providers: {gpu_providers}', file=sys.stderr)
else:
    # Check if a GPU exists but we don't have GPU providers.
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
        print('[onnxruntime] WARNING: GPU detected but no GPU execution provider installed.', file=sys.stderr)
        if sys.platform == 'darwin':
            print('  CoreMLExecutionProvider should be included in onnxruntime >= 1.14 on macOS.', file=sys.stderr)
            print('  Try: pip install --upgrade onnxruntime', file=sys.stderr)
        elif is_amd:
            print('  Install: pip install onnxruntime-rocm (AMD ROCm)', file=sys.stderr)
        else:
            print('  Install: pip install onnxruntime-gpu  (NVIDIA CUDA)', file=sys.stderr)
" 2>/dev/null || true

exec python3 "$SCRIPT_DIR/bench.py" "$MODEL"
