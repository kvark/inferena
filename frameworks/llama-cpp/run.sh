#!/usr/bin/env bash
# llama.cpp benchmark runner wrapper (inference/forward only).
# Uses llama-cpp-python bindings for proper logit output.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-SmolLM2-135M}"

LLAMA_CPP_DIR="$SCRIPT_DIR/llama.cpp"
BENCH_SCRIPT="$SCRIPT_DIR/bench.py"

# --- Ensure llama-cpp-python is importable ---
if ! python3 -c "import llama_cpp" 2>/dev/null; then
    echo "[llama-cpp] llama-cpp-python not found. Install via: pip install llama-cpp-python" >&2
    exit 1
fi

# --- Clone llama.cpp for gguf-py library (needed for GGUF conversion) ---
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "[llama-cpp] cloning llama.cpp (for gguf-py)..." >&2
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR" 2>&1 >&2
fi

# --- Find or convert model to GGUF ---
MODEL_DIR="$ROOT_DIR/models/$MODEL"
GGUF_FILE="$MODEL_DIR/model-f32.gguf"

if [ ! -f "$GGUF_FILE" ]; then
    echo "[llama-cpp] converting safetensors to GGUF..." >&2
    SAFETENSORS="$MODEL_DIR/model.safetensors"
    # Try local path, then fall back to HF hub cache.
    if [ ! -f "$SAFETENSORS" ]; then
        HF_PATH=$(python3 -c "
from huggingface_hub import hf_hub_download
ids = {'SmolLM2-135M': 'HuggingFaceTB/SmolLM2-135M'}
hf_id = ids.get('$MODEL')
if hf_id:
    import os
    from huggingface_hub import snapshot_download
    print(snapshot_download(hf_id, allow_patterns=['*.safetensors', '*.json']))
" 2>/dev/null) || true
        if [ -n "$HF_PATH" ] && [ -f "$HF_PATH/model.safetensors" ]; then
            MODEL_DIR="$HF_PATH"
            echo "[llama-cpp] using HF cache: $MODEL_DIR" >&2
        else
            echo "[llama-cpp] model.safetensors not found at $MODEL_DIR" >&2
            exit 1
        fi
    fi
    mkdir -p "$(dirname "$GGUF_FILE")"
    python3 "$SCRIPT_DIR/convert_to_gguf.py" "$MODEL_DIR" "$GGUF_FILE" 2>&1 >&2
fi

# --- Run benchmark via Python wrapper ---
exec python3 "$BENCH_SCRIPT" "$MODEL" "$GGUF_FILE" "$LLAMA_CPP_DIR"
