#!/usr/bin/env bash
# PyTorch benchmark runner wrapper.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-SmolLM2-135M}"

: "${PYTHON:?must be set by run.sh}"

# Own the CUDA-library override before importing torch. Strict mode closes the
# environment-variable escape hatch that can force TF32 even when
# torch.set_float32_matmul_precision("highest") is requested. The practical
# default leaves the choice to the explicit PyTorch settings in bench.py.
if [ "${INFERENA_STRICT:-0}" = "1" ]; then
    export NVIDIA_TF32_OVERRIDE=0
    unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
else
    unset NVIDIA_TF32_OVERRIDE
    unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
fi

# Check torch is importable.
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
    echo "[pytorch] torch not installed. Run: pip install -r $ROOT_DIR/requirements-cpu.txt" >&2
    echo "[pytorch] Or create a venv: python3 -m venv .venv && .venv/bin/pip install -r $ROOT_DIR/requirements-cpu.txt" >&2
    exit 1
fi

PREFIX=()
if [ -n "${INFERENA_NSYS:-}" ]; then
    # Diagnostic compilation stays in-process; do not trace forked compiler workers.
    export TORCHINDUCTOR_COMPILE_THREADS=1
    PREFIX=("$INFERENA_NSYS" profile --trace=cuda-sw,nvtx --cuda-graph-trace=node
        --sample=none --cpuctxsw=none --wait=primary "--output=${INFERENA_NSYS_DIR:?}/pytorch")
    printf '%q ' "${PREFIX[@]}" "$PYTHON" "$SCRIPT_DIR/bench.py" "$MODEL" >&2
    echo >&2
fi
exec "${PREFIX[@]}" "$PYTHON" "$SCRIPT_DIR/bench.py" "$MODEL"
