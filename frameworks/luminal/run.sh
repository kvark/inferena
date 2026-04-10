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
        if command -v nvidia-smi &>/dev/null || [ -d /usr/local/cuda ] || command -v nvcc &>/dev/null; then
            FEATURES="--features cuda"
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
