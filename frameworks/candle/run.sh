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
case "$(uname -s)" in
    Linux*)
        if command -v nvidia-smi &>/dev/null || [ -d /usr/local/cuda ] || command -v nvcc &>/dev/null; then
            FEATURES="--features cuda"
        fi
        ;;
    Darwin*)
        FEATURES="--features metal"
        ;;
esac

echo "[candle] Building release binary... $FEATURES" >&2
cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" -p inferena-candle $FEATURES 2>&1 >&2

exec "$ROOT_DIR/target/release/inferena-candle" "$MODEL"
