#!/usr/bin/env bash
# Meganeura framework benchmark runner wrapper.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-SmolLM2-135M}"

echo "[meganeura] Building release binary..." >&2
cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" -p infermark-meganeura 2>&1 >&2

exec "$ROOT_DIR/target/release/infermark-meganeura" "$MODEL"
