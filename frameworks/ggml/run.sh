#!/usr/bin/env bash
# GGML benchmark runner (llama.cpp + whisper.cpp).
# Uses llama-cpp-python for LLMs, whisper-cli for audio models.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-SmolLM2-135M}"

# --- Model routing ---
case "$MODEL" in
    SmolLM2-135M)
        BACKEND="llama.cpp"
        ;;
    Whisper-tiny)
        BACKEND="whisper.cpp"
        ;;
    *)
        echo "Unknown model: $MODEL. Available: SmolLM2-135M Whisper-tiny" >&2
        exit 1
        ;;
esac

# --- Check dependencies per backend ---
if [ "$BACKEND" = "llama.cpp" ]; then
    if ! python3 -c "import llama_cpp" 2>/dev/null; then
        echo "[ggml] llama-cpp-python not found. Install via: pip install llama-cpp-python" >&2
        exit 1
    fi
elif [ "$BACKEND" = "whisper.cpp" ]; then
    if ! python3 -c "import faster_whisper" 2>/dev/null; then
        echo "[ggml] faster-whisper not found. Install via: pip install faster-whisper" >&2
        exit 1
    fi
fi

# --- Dry-run: stop after validation ---
if [ "${INFERENA_DRY_RUN:-}" = "1" ]; then
    echo "[ggml] dry-run OK: $MODEL (via $BACKEND)" >&2
    exit 0
fi

# --- SmolLM2 via llama.cpp ---
if [ "$BACKEND" = "llama.cpp" ]; then
    MODEL_DIR="$ROOT_DIR/models/$MODEL"
    GGUF_FILE="$MODEL_DIR/model-f32.gguf"

    if [ ! -f "$GGUF_FILE" ]; then
        echo "[ggml] converting safetensors to GGUF..." >&2
        SAFETENSORS="$MODEL_DIR/model.safetensors"
        if [ ! -f "$SAFETENSORS" ]; then
            HF_PATH=$(python3 -c "
from huggingface_hub import snapshot_download
ids = {'SmolLM2-135M': 'HuggingFaceTB/SmolLM2-135M'}
hf_id = ids.get('$MODEL')
if hf_id:
    print(snapshot_download(hf_id, allow_patterns=['*.safetensors', '*.json']))
" 2>/dev/null) || true
            if [ -n "$HF_PATH" ] && [ -f "$HF_PATH/model.safetensors" ]; then
                MODEL_DIR="$HF_PATH"
                echo "[ggml] using HF cache: $MODEL_DIR" >&2
            else
                echo "[ggml] model.safetensors not found at $MODEL_DIR" >&2
                exit 1
            fi
        fi
        mkdir -p "$(dirname "$GGUF_FILE")"
        python3 "$SCRIPT_DIR/convert_to_gguf.py" "$MODEL_DIR" "$GGUF_FILE" 2>&1 >&2
    fi

    exec python3 "$SCRIPT_DIR/bench.py" "$MODEL" "$GGUF_FILE"
fi

# --- Whisper-tiny via whisper.cpp ---
if [ "$BACKEND" = "whisper.cpp" ]; then
    exec python3 "$SCRIPT_DIR/bench_whisper.py"
fi
