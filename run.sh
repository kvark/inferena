#!/usr/bin/env bash
# infermark - ML Framework Inference Benchmark
#
# Usage:
#   ./run.sh                         # All models, all frameworks
#   ./run.sh -m SmolLM2-135M         # Single model
#   ./run.sh -m SmolLM2-135M -f pytorch  # Single model + framework
#   ./run.sh --json                  # Output JSON
#   ./run.sh --help                  # Show help
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

ALL_MODELS="SmolLM2-135M SmolVLA"

# --- Parse arguments ---
MODELS=""
FRAMEWORKS=""
JSON_FLAG=""
DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODELS="$2"
            shift 2
            ;;
        -f|--frameworks)
            FRAMEWORKS="$2"
            shift 2
            ;;
        --json)
            JSON_FLAG="--json"
            shift
            ;;
        --download)
            DOWNLOAD=true
            shift
            ;;
        -h|--help)
            echo "infermark - ML Framework Inference Benchmark"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -m, --model <name>        Model to benchmark (default: all)"
            echo "  -f, --frameworks <list>   Comma-separated frameworks (default: all)"
            echo "  --json                    Output results as JSON"
            echo "  --download                Download model weights before running"
            echo "  -h, --help                Show this help"
            echo ""
            echo "Models: $ALL_MODELS"
            echo "Frameworks: pytorch, burn, luminal, meganeura"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Default: all models.
if [ -z "$MODELS" ]; then
    MODELS="$ALL_MODELS"
fi

# --- Make runner scripts executable ---
chmod +x "$ROOT_DIR"/frameworks/*/run.sh 2>/dev/null || true

# --- Create results directory ---
mkdir -p "$ROOT_DIR/results"

# --- Build the harness ---
echo "Building infermark harness..." >&2
cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" -p infermark-harness 2>&1 >&2

HARNESS="$ROOT_DIR/target/release/infermark"

# --- Run each model ---
for MODEL in $MODELS; do
    echo "" >&2
    echo "==============================" >&2
    echo "  Model: $MODEL" >&2
    echo "==============================" >&2

    # Download if requested.
    if [ "$DOWNLOAD" = true ]; then
        echo "Downloading $MODEL ..." >&2
        bash "$ROOT_DIR/models/download.sh" "$MODEL" || true
    fi

    ARGS=("--model" "$MODEL" "--root" "$ROOT_DIR")

    if [ -n "$FRAMEWORKS" ]; then
        ARGS+=("--frameworks" "$FRAMEWORKS")
    fi

    if [ -n "$JSON_FLAG" ]; then
        ARGS+=("--json")
    fi

    "$HARNESS" "${ARGS[@]}" || true
done
