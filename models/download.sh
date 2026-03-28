#!/usr/bin/env bash
# Download model weights from HuggingFace Hub.
# Usage: ./download.sh [model_name]
#
# Requires: huggingface-cli (pip install huggingface-hub)
# Or: git-lfs + git clone
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

declare -A MODEL_MAP=(
    ["SmolLM2-135M"]="HuggingFaceTB/SmolLM2-135M"
    ["SmolLM2-360M"]="HuggingFaceTB/SmolLM2-360M-Instruct"
    ["SmolLM2-1.7B"]="HuggingFaceTB/SmolLM2-1.7B"
    ["SmolVLM-256M"]="HuggingFaceTB/SmolVLM-256M-Instruct"
)

download_model() {
    local name="$1"
    local hf_id="${MODEL_MAP[$name]:-}"

    if [ -z "$hf_id" ]; then
        echo "Unknown model: $name" >&2
        echo "Available: ${!MODEL_MAP[*]}" >&2
        return 1
    fi

    local dest="$SCRIPT_DIR/$name"
    if [ -d "$dest" ] && [ -f "$dest/config.json" ]; then
        echo "Model $name already downloaded at $dest" >&2
        return 0
    fi

    echo "Downloading $name ($hf_id) ..." >&2

    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$hf_id" --local-dir "$dest"
    elif command -v git &>/dev/null && git lfs env &>/dev/null 2>&1; then
        git clone "https://huggingface.co/$hf_id" "$dest"
    else
        echo "Error: need either huggingface-cli or git-lfs to download models." >&2
        echo "  pip install huggingface-hub   # recommended" >&2
        return 1
    fi

    echo "Downloaded $name to $dest" >&2
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name> [model_name ...]" >&2
    echo "Available models: ${!MODEL_MAP[*]}" >&2
    exit 1
fi

for model in "$@"; do
    download_model "$model"
done
