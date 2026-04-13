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
case "$(uname -s)" in
    Linux*)
        if command -v nvcc &>/dev/null || [ -d /usr/local/cuda ]; then
            FEATURES="--features cuda"
            # Auto-detect max supported compute capability for nvcc.
            if [ -z "${CUDA_COMPUTE_CAP:-}" ]; then
                _max_cc=$(nvcc --list-gpu-code 2>/dev/null | grep -oP 'sm_\K[0-9]+' | sort -n | tail -1)
                _gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
                if [ -n "$_gpu_cc" ] && [ -n "$_max_cc" ] && [ "$_gpu_cc" -gt "$_max_cc" ]; then
                    export CUDA_COMPUTE_CAP="$_max_cc"
                    echo "[luminal] nvcc max sm_${_max_cc}, GPU needs sm_${_gpu_cc} — using forward compat (CUDA_COMPUTE_CAP=${_max_cc})" >&2
                fi
            fi
        elif command -v rocm-smi &>/dev/null || [ -d /opt/rocm ]; then
            echo "[luminal] AMD ROCm detected but Luminal only supports CUDA and Metal — running on CPU" >&2
        fi
        ;;
    Darwin*)
        FEATURES="--features metal"
        ;;
esac

echo "[luminal] Building release binary... $FEATURES" >&2
cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" -p inferena-luminal $FEATURES 2>&1 >&2

exec "$ROOT_DIR/target/release/inferena-luminal" "$MODEL"
