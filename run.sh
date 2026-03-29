#!/usr/bin/env bash
# infermark - ML Framework Inference Benchmark
#
# Usage:
#   ./run.sh                         # Run all frameworks, default model
#   ./run.sh -m SmolLM2-135M         # Specify model
#   ./run.sh -f pytorch,burn         # Specify frameworks
#   ./run.sh --json                  # Output JSON
#   ./run.sh --download              # Download model before benchmarking
#   ./run.sh --help                  # Show help
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Parse arguments ---
MODEL="SmolLM2-135M"
FRAMEWORKS=""
JSON_FLAG=""
DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL="$2"
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
            echo "  -m, --model <name>        Model to benchmark (default: SmolLM2-135M)"
            echo "  -f, --frameworks <list>   Comma-separated frameworks (default: all)"
            echo "  --json                    Output results as JSON"
            echo "  --download                Download model weights before running"
            echo "  -h, --help                Show this help"
            echo ""
            echo "Models: SmolLM2-135M, SmolLM2-360M, SmolLM2-1.7B, SmolVLM-256M"
            echo "Frameworks: pytorch, burn, luminal, meganeura"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# --- Download model if requested ---
if [ "$DOWNLOAD" = true ]; then
    echo "Downloading model $MODEL ..." >&2
    bash "$ROOT_DIR/models/download.sh" "$MODEL"
fi

# --- Make runner scripts executable ---
chmod +x "$ROOT_DIR"/frameworks/*/run.sh 2>/dev/null || true

# --- Create results directory ---
mkdir -p "$ROOT_DIR/results"

# --- Build the harness ---
echo "Building infermark harness..." >&2
cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" -p infermark-harness 2>&1 >&2

# --- Run ---
HARNESS="$ROOT_DIR/target/release/infermark"
ARGS=("--model" "$MODEL" "--root" "$ROOT_DIR")

if [ -n "$FRAMEWORKS" ]; then
    ARGS+=("--frameworks" "$FRAMEWORKS")
fi

if [ -n "$JSON_FLAG" ]; then
    ARGS+=("--json")
fi

exec "$HARNESS" "${ARGS[@]}"
