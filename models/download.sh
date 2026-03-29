#!/usr/bin/env bash
# Download model weights from HuggingFace Hub into the shared HF cache.
#
# All framework runners should read from the standard HF cache at
# ~/.cache/huggingface/hub/ (or $HF_HOME). This script pre-populates
# that cache so framework runners don't each need download logic.
#
# Usage: ./download.sh <model_name> [model_name ...]
#
# Requires one of:
#   - huggingface-cli  (pip install huggingface-hub)
#   - curl + manual download
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

declare -A MODEL_MAP=(
    ["SmolLM2-135M"]="HuggingFaceTB/SmolLM2-135M"
    ["SmolLM2-360M"]="HuggingFaceTB/SmolLM2-360M-Instruct"
    ["SmolLM2-1.7B"]="HuggingFaceTB/SmolLM2-1.7B"
    ["SmolVLM-256M"]="HuggingFaceTB/SmolVLM-256M-Instruct"
)

# Files needed by each runner. Only download what's actually used.
MODEL_FILES="config.json model.safetensors tokenizer.json tokenizer_config.json"

download_model() {
    local name="$1"
    local hf_id="${MODEL_MAP[$name]:-}"

    if [ -z "$hf_id" ]; then
        echo "Unknown model: $name" >&2
        echo "Available: ${!MODEL_MAP[*]}" >&2
        return 1
    fi

    echo "Downloading $name ($hf_id) ..." >&2

    if command -v huggingface-cli &>/dev/null; then
        # huggingface-cli downloads to the standard HF cache.
        huggingface-cli download "$hf_id" $MODEL_FILES
    else
        echo "Error: huggingface-cli not found." >&2
        echo "  pip install huggingface-hub" >&2
        return 1
    fi

    echo "Downloaded $name ($hf_id)" >&2
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name> [model_name ...]" >&2
    echo "Available models: ${!MODEL_MAP[*]}" >&2
    exit 1
fi

for model in "$@"; do
    download_model "$model"
done
