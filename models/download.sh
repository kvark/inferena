#!/usr/bin/env bash
# Fetch shared local SmolLM2 checkpoints; retain the optional SmolVLM cache download.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=${PYTHON:-python3}
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <model_name> [model_name ...]" >&2
    exit 2
fi
for MODEL in "$@"; do
    case "$MODEL" in
        SmolLM2-135M|SmolLM2-360M|SmolLM2-1.7B)
            "$PYTHON" "$ROOT/scripts/prepare_models.py" "$MODEL"
            ;;
        SmolVLM-256M)
            "$PYTHON" -c 'from huggingface_hub import snapshot_download; snapshot_download("HuggingFaceTB/SmolVLM-256M-Instruct", allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"])'
            ;;
        SmolVLA|StableDiffusion|ResNet-50|Whisper-tiny)
            echo "$MODEL uses synthetic parameters; no checkpoint download." >&2
            ;;
        *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
    esac
done
