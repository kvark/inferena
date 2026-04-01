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
    echo "[llama-cpp] installing llama-cpp-python..." >&2
    pip install llama-cpp-python --quiet 2>&1 >&2 || {
        echo "[llama-cpp] failed to install llama-cpp-python" >&2
        exit 1
    }
fi

# --- Clone llama.cpp for the GGUF converter (if model needs conversion) ---
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "[llama-cpp] cloning llama.cpp (for GGUF converter)..." >&2
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR" 2>&1 >&2
fi

# --- Find or convert model to GGUF ---
MODEL_DIR="$ROOT_DIR/models/$MODEL"
GGUF_FILE="$MODEL_DIR/model-f32.gguf"

if [ ! -f "$GGUF_FILE" ]; then
    echo "[llama-cpp] converting safetensors to GGUF..." >&2
    if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
        echo "[llama-cpp] model.safetensors not found at $MODEL_DIR" >&2
        exit 1
    fi
    python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" \
        --outfile "$GGUF_FILE" --outtype f32 2>&1 >&2
fi

# --- Run benchmark via Python wrapper ---
exec python3 "$BENCH_SCRIPT" "$MODEL" "$GGUF_FILE" "$LLAMA_CPP_DIR"
