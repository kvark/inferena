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
if [ "${INFERENA_STRICT:-0}" = "1" ] || [[ "${INFERENA_AUTOCAST:-auto}" =~ ^(auto|bf16|fp16)$ ]]; then
    export NVIDIA_TF32_OVERRIDE=0
    unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
else
    unset NVIDIA_TF32_OVERRIDE
    unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
fi

# Resolve the device in the existing interpreter preflight. The actual runner
# is a fresh process, so its public Inductor settings take effect before torch
# imports and constructs the shared kernel benchmarker.
if ! BENCHMARK_DEVICE="$("$PYTHON" -c 'import sys; sys.path.insert(0, sys.argv[1]); from bench import detect_device; print(detect_device())' "$SCRIPT_DIR")"; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
    echo "[pytorch] device probe failed with $PYTHON. Check the requested backend and bash $ROOT_DIR/scripts/setup.sh <wheel-backend>" >&2
    exit 1
fi

# XPU event-based microbenchmarks selected materially different launch configs
# for identical generated programs. Use PyTorch's built-in device profiler for
# autotuning on that backend. This does not profile the final timing interval.
export TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER=1
if [[ "$BENCHMARK_DEVICE" == xpu:* ]]; then
    export TORCHINDUCTOR_USE_TORCH_PROFILER_BENCHMARKER=1
else
    export TORCHINDUCTOR_USE_TORCH_PROFILER_BENCHMARKER=0
fi

PREFIX=()
if [ -n "${INFERENA_NSYS:-}" ]; then
    # Diagnostic compilation stays in-process; do not trace forked compiler workers.
    export TORCHINDUCTOR_COMPILE_THREADS=1
    PREFIX=("$INFERENA_NSYS" profile --trace=cuda-sw,nvtx --cuda-graph-trace=node
        --sample=none --cpuctxsw=none --wait=primary "--output=${INFERENA_NSYS_DIR:?}/pytorch")
    printf '%q ' "${PREFIX[@]+"${PREFIX[@]}"}" "$PYTHON" "$SCRIPT_DIR/budget.py" "$SCRIPT_DIR/bench.py" "$MODEL" >&2
    echo >&2
fi
exec "${PREFIX[@]+"${PREFIX[@]}"}" "$PYTHON" "$SCRIPT_DIR/budget.py" "$SCRIPT_DIR/bench.py" "$MODEL"
