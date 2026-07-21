#!/usr/bin/env bash
# Candle framework benchmark runner wrapper.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-SmolLM2-135M}"

source "$ROOT_DIR/scripts/cargo-rev.sh"
export FRAMEWORK_REV=$(cargo_rev_short candle-core "$ROOT_DIR")

# Select GPU backend based on platform.
FEATURES=""
EXE=""
case "$(uname -s)" in
    Linux*|MINGW*|MSYS*|CYGWIN*)
        case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) EXE=.exe ;; esac
        if command -v nvcc &>/dev/null || [ -d /usr/local/cuda ] || [ -n "${CUDA_PATH:-}" ]; then
            FEATURES="--features cuda"
            # Newer glibc (2.36+) declares GNU-extension rsqrt/rsqrtf as
            # noexcept under the C23 IEC-60559-ext feature set, which nvcc's
            # own crt/math_functions.h redeclares without noexcept — a hard
            # "exception specification is incompatible" error. glibc only
            # enables that feature set when __USE_GNU is set (i.e.
            # _GNU_SOURCE, which nvcc's host-compiler pass defines
            # implicitly), so undefining it for the host-compiler pass avoids
            # the conflicting redeclaration. _GNU_SOURCE also implies
            # _DEFAULT_SOURCE (__USE_MISC), which some kernels rely on for
            # M_SQRT1_2/M_2_SQRTPI — restore that separately so undefining
            # _GNU_SOURCE doesn't collateral-damage those constants.
            # NVCC_APPEND_FLAGS is a native nvcc env var, not something this
            # repo's build script controls, so this is the only lever
            # available short of patching vendored CUDA headers.
            if [ -z "${NVCC_APPEND_FLAGS:-}" ]; then
                export NVCC_APPEND_FLAGS="-Xcompiler -U_GNU_SOURCE -Xcompiler -D_DEFAULT_SOURCE=1"
            fi
            # Auto-detect max supported compute capability for nvcc.
            # Distro nvcc may be older than the installed GPU (e.g. nvcc 12.4 vs Blackwell).
            if [ -z "${CUDA_COMPUTE_CAP:-}" ]; then
                _max_cc=$(nvcc --list-gpu-code 2>/dev/null | grep -oP 'sm_\K[0-9]+' | sort -n | tail -1)
                _gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
                if [ -n "$_gpu_cc" ] && [ -n "$_max_cc" ] && [ "$_gpu_cc" -gt "$_max_cc" ]; then
                    # candle's cudaforge helper auto-appends the "a" (architecture-
                    # specific) suffix to any target >= sm_90, and "a"-suffixed PTX
                    # is not forward-portable across GPU generations (NVIDIA
                    # reserves that guarantee for plain, non-"a" virtual archs).
                    # So falling back to nvcc's raw max (often sm_90 -> sm_90a)
                    # still fails to JIT on a newer/different-generation GPU.
                    # Cap at 80 (Ampere, below the auto-suffix threshold) — the
                    # highest baseline guaranteed to JIT forward-compatibly.
                    if [ "$_max_cc" -ge 90 ]; then
                        _fallback_cc=80
                    else
                        _fallback_cc="$_max_cc"
                    fi
                    export CUDA_COMPUTE_CAP="$_fallback_cc"
                    echo "[candle] nvcc max sm_${_max_cc}, GPU needs sm_${_gpu_cc} — using forward compat (CUDA_COMPUTE_CAP=${_fallback_cc})" >&2
                fi
            fi
        elif command -v rocm-smi &>/dev/null || [ -d /opt/rocm ]; then
            echo "[candle] AMD ROCm detected but Candle only supports CUDA and Metal — running on CPU" >&2
        fi
        ;;
    Darwin*)
        FEATURES="--features metal"
        ;;
esac

echo "[candle] Building release binary... $FEATURES" >&2
cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" -p inferena-candle $FEATURES 2>&1 >&2

exec "$ROOT_DIR/target/release/inferena-candle${EXE}" "$MODEL"
