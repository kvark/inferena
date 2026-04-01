#!/usr/bin/env bash
# Meganeura framework benchmark runner wrapper.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-SmolLM2-135M}"

# Extract git revision from Cargo.lock.
source "$ROOT_DIR/scripts/cargo-rev.sh"
export FRAMEWORK_REV=$(cargo_rev_short meganeura "$ROOT_DIR")

echo "[meganeura] Building release binary..." >&2
cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" -p inferena-meganeura 2>&1 >&2

exec "$ROOT_DIR/target/release/inferena-meganeura" "$MODEL"
