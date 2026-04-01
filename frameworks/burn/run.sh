#!/usr/bin/env bash
# Burn framework benchmark runner wrapper.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-SmolLM2-135M}"

# Select wgpu backend based on OS.
if [ -z "${WGPU_BACKEND:-}" ]; then
    case "$(uname -s)" in
        Linux*)  export WGPU_BACKEND=vulkan ;;
        Darwin*) export WGPU_BACKEND=metal ;;
    esac
fi

source "$ROOT_DIR/scripts/cargo-rev.sh"
export FRAMEWORK_REV=$(cargo_rev_short burn "$ROOT_DIR")

echo "[burn] Building release binary..." >&2
cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" -p inferena-burn 2>&1 >&2

exec "$ROOT_DIR/target/release/inferena-burn" "$MODEL"
