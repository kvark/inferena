#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

ALL_MODELS="SmolLM2-135M SmolVLA StableDiffusion"

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
            echo "inferena - Inference Arena"
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
            echo "Frameworks: pytorch, mlx, candle, burn, luminal, meganeura, llama-cpp"
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

# --- Create results directory ---
mkdir -p "$ROOT_DIR/results"

# --- Build all Rust crates (harness + framework runners) at once ---
echo "Building all Rust crates..." >&2
cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" --workspace 2>&1 >&2

HARNESS="$ROOT_DIR/target/release/inferena"

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
