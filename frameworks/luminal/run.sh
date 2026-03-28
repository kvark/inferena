#!/usr/bin/env bash
# Luminal framework benchmark runner wrapper.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-SmolLM2-135M}"

echo "[luminal] Building release binary..." >&2
cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" -p infermark-luminal 2>&1 >&2

exec "$ROOT_DIR/target/release/infermark-luminal" "$MODEL"
