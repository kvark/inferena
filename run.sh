#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

ALL_MODELS="SmolLM2-135M SmolVLA StableDiffusion ResNet-50 Whisper-tiny"

# --- Parse arguments ---
MODELS=""
FRAMEWORKS=""
JSON_FLAG=""
DOWNLOAD=false
CHECK_ONLY=false
HAS_ARGS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODELS="$2"
            HAS_ARGS=true
            shift 2
            ;;
        -f|--frameworks)
            FRAMEWORKS="$2"
            HAS_ARGS=true
            shift 2
            ;;
        --json)
            JSON_FLAG="--json"
            HAS_ARGS=true
            shift
            ;;
        --download)
            DOWNLOAD=true
            HAS_ARGS=true
            shift
            ;;
        --check)
            CHECK_ONLY=true
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
            echo "  --check                   Check framework availability (don't run benchmarks)"
            echo "  -h, --help                Show this help"
            echo ""
            echo "Models: $ALL_MODELS"
            echo "Frameworks: pytorch, mlx, candle, burn, luminal, meganeura, llama-cpp, onnxruntime, jax"
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

# --- Check framework availability ---
run_check() {
    echo "Framework availability:"
    echo ""

    # Python frameworks
    check_python() {
        local name="$1" mod="$2" extra="${3:-}"
        if python3 -c "import $mod" 2>/dev/null; then
            ver=$(python3 -c "import $mod; print(getattr($mod, '__version__', 'unknown'))" 2>/dev/null)
            echo "  ✓ $name ($ver)$extra"
        else
            echo "  ✗ $name — python3 -c 'import $mod' failed (pip install $mod)"
        fi
    }

    check_python "PyTorch" "torch"
    check_python "torchvision" "torchvision" " (needed for ResNet-50)"
    check_python "transformers" "transformers" " (needed for Whisper-tiny)"
    check_python "optimum" "optimum" " (needed for ONNX Runtime export of SmolLM2/Whisper)"
    check_python "ONNX Runtime" "onnxruntime"
    check_python "JAX" "jax"
    check_python "safetensors" "safetensors" " (needed for JAX, llama.cpp weight loading)"
    check_python "huggingface_hub" "huggingface_hub" " (needed for model downloads)"
    check_python "MLX" "mlx.core" " (macOS only)"

    # llama.cpp
    if python3 -c "import llama_cpp" 2>/dev/null; then
        ver=$(python3 -c "import llama_cpp; print(getattr(llama_cpp, '__version__', 'unknown'))" 2>/dev/null)
        echo "  ✓ llama.cpp ($ver via llama-cpp-python)"
    else
        echo "  ✗ llama.cpp — pip install llama-cpp-python"
    fi

    echo ""

    # Rust frameworks — check if binaries exist or can compile
    RUST_FW="inferena-candle inferena-burn inferena-luminal inferena-meganeura"
    for pkg in $RUST_FW; do
        name="${pkg#inferena-}"
        bin="$ROOT_DIR/target/release/$pkg"
        if [ -f "$bin" ]; then
            echo "  ✓ $name (binary at $bin)"
        elif cargo check -p "$pkg" --manifest-path "$ROOT_DIR/Cargo.toml" 2>/dev/null; then
            echo "  ~ $name (compiles, not yet built — run without --check to build)"
        else
            echo "  ✗ $name — cargo check -p $pkg failed"
        fi
    done

    echo ""

    # GPU backends
    echo "GPU backends:"
    if command -v vulkaninfo &>/dev/null; then
        dev=$(vulkaninfo --summary 2>/dev/null | grep "deviceName" | head -1 | sed 's/.*= //')
        echo "  ✓ Vulkan ($dev)"
    else
        echo "  ✗ Vulkan — vulkaninfo not found"
    fi
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "  ✓ Metal (macOS detected)"
    fi
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        dev=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo "  ✓ CUDA ($dev)"
    else
        echo "  ✗ CUDA — torch.cuda.is_available() is False"
    fi

    echo ""
}

# Always run check when invoked without arguments.
if [ "$CHECK_ONLY" = true ] || [ "$HAS_ARGS" = false ]; then
    run_check
    if [ "$CHECK_ONLY" = true ]; then
        exit 0
    fi
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
