#!/usr/bin/env bash
# Luminal framework benchmark runner wrapper.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-SmolLM2-135M}"

source "$ROOT_DIR/scripts/cargo-rev.sh"
export FRAMEWORK_REV=$(cargo_rev_short luminal "$ROOT_DIR")

# Select GPU backend based on platform.
FEATURES=""
EXE=""
case "$(uname -s)" in
    Linux*|MINGW*|MSYS*|CYGWIN*)
        case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) EXE=.exe ;; esac
        if command -v nvcc &>/dev/null || [ -d /usr/local/cuda ] || [ -n "${CUDA_PATH:-}" ]; then
            FEATURES="--features cuda"
            # Auto-detect max supported compute capability for nvcc.
            if [ -z "${CUDA_COMPUTE_CAP:-}" ]; then
                _max_cc=$(nvcc --list-gpu-code 2>/dev/null | grep -oP 'sm_\K[0-9]+' | sort -n | tail -1)
                _gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
                if [ -n "$_gpu_cc" ] && [ -n "$_max_cc" ] && [ "$_gpu_cc" -gt "$_max_cc" ]; then
                    # Luminal's CUDA graph backend doesn't tolerate forward-compat PTX
                    # (CUDA_ERROR_INVALID_VALUE at execute time). Disable CUDA;
                    # the backend check below will skip unless reference mode
                    # was explicitly requested.
                    echo "[luminal] nvcc max sm_${_max_cc} < GPU sm_${_gpu_cc} — Luminal CUDA graphs need an exact match; CUDA backend unavailable" >&2
                    FEATURES=""
                fi
            fi
        elif command -v rocm-smi &>/dev/null || [ -d /opt/rocm ]; then
            echo "[luminal] AMD ROCm detected but Luminal only supports CUDA and Metal" >&2
        fi
        ;;
    Darwin*)
        FEATURES="--features metal"
        ;;
esac

# The pinned Luminal revision no longer includes the optimized NativeRuntime
# used by earlier CPU results. ReferenceRuntime is a correctness interpreter;
# on this 135M-parameter graph it can consume several GiB and run for minutes
# without completing a single forward pass. Do not let an all-framework suite
# silently fall into that path. Keep it available as an explicit developer
# opt-in for debugging Luminal itself.
if [ -z "$FEATURES" ] && [ "${INFERENA_LUMINAL_REFERENCE:-0}" != "1" ]; then
    echo "[luminal] unsupported: no CUDA or Metal backend is available; CPU ReferenceRuntime is disabled because this full-model benchmark is prohibitively slow" >&2
    echo "[luminal] set INFERENA_LUMINAL_REFERENCE=1 to run the reference backend anyway" >&2
    exit 2
fi

if [ -z "$FEATURES" ]; then
    echo "[luminal] WARNING: using the slow CPU ReferenceRuntime (explicit opt-in)" >&2
fi

echo "[luminal] Building release binary... $FEATURES" >&2
cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" -p inferena-luminal $FEATURES 2>&1 >&2

exec "$ROOT_DIR/target/release/inferena-luminal${EXE}" "$MODEL"
