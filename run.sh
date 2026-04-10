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
DRY_RUN=false
UPDATE=false
PLATFORM_OVERRIDE=""
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
        --dry-run)
            DRY_RUN=true
            HAS_ARGS=true
            shift
            ;;
        --update)
            UPDATE=true
            HAS_ARGS=true
            shift
            ;;
        --platform)
            PLATFORM_OVERRIDE="$2"
            shift 2
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
            echo "  --dry-run                 Validate framework+model support without running benchmarks"
            echo "  --update                  Update models/*.md with results after benchmarking"
            echo "  --platform <name>         Override auto-detected platform name (with --update)"
            echo "  -h, --help                Show this help"
            echo ""
            echo "Models: $ALL_MODELS"
            echo "Frameworks: pytorch, candle, burn, luminal, meganeura, ggml, onnxruntime, jax, mlx"
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

# --- Platform detection ---
detect_platform() {
    # GPU name is the most distinctive — try that first.
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null
        return
    fi
    if python3 -c "import torch; assert hasattr(torch.backends,'mps') and torch.backends.mps.is_available()" 2>/dev/null; then
        # macOS with MPS — use chip name
        sysctl -n machdep.cpu.brand_string 2>/dev/null | sed 's/.*\(Apple M[0-9]*[^ ]*\).*/\1/' || echo "Apple Silicon"
        return
    fi
    if command -v vulkaninfo &>/dev/null; then
        local vk_dev
        vk_dev=$(vulkaninfo --summary 2>/dev/null | grep "deviceName" | head -1 | sed 's/.*= //' | xargs)
        if [ -n "$vk_dev" ] && ! echo "$vk_dev" | grep -qi "llvmpipe\|lavapipe\|swrast"; then
            echo "$vk_dev"
            return
        fi
    fi
    # Fall back to CPU model name.
    if [ "$(uname -s)" = "Darwin" ]; then
        sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple $(uname -m)"
    else
        grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | sed 's/.*: //' | sed 's/  */ /g' | sed 's/(R)//g; s/(TM)//g; s/CPU //g' | xargs
    fi
}

# Check if a platform name exists in any model markdown file.
platform_exists_in_md() {
    local platform="$1"
    python3 -c "
import sys, os
sys.path.insert(0, os.path.join('$ROOT_DIR', 'scripts'))
from update_results import find_results_table, group_by_platform, _platforms_match
import glob
for md in glob.glob(os.path.join('$ROOT_DIR', 'models', '*.md')):
    with open(md) as f:
        content = f.read()
    parsed = find_results_table(content)
    if parsed is None:
        continue
    _, _, _, rows, _ = parsed
    for name, _ in group_by_platform(rows):
        if _platforms_match(name, '$platform'):
            print(md)
            sys.exit(0)
sys.exit(1)
" 2>/dev/null
}

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
    check_python "safetensors" "safetensors" " (needed for JAX, llama.cpp weight loading)"
    check_python "huggingface_hub" "huggingface_hub" " (needed for model downloads)"
    if [ "$(uname -s)" = "Darwin" ]; then
        check_python "MLX" "mlx.core"
    fi

    # GPU-aware Python framework checks.
    echo ""
    echo "  GPU-aware frameworks:"

    # ONNX Runtime
    if python3 -c "import onnxruntime" 2>/dev/null; then
        ver=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null)
        gpu=$(python3 -c "
import onnxruntime as ort
providers = [p for p in ort.get_available_providers() if p != 'CPUExecutionProvider']
print(', '.join(providers) if providers else 'CPU only')
" 2>/dev/null)
        echo "    ✓ ONNX Runtime ($ver) — $gpu"
        if [ "$gpu" = "CPU only" ]; then
            echo "      ↳ pip install onnxruntime-gpu (NVIDIA) or onnxruntime-rocm (AMD)"
        fi
    else
        echo "    ✗ ONNX Runtime — pip install onnxruntime"
    fi

    # JAX
    if python3 -c "import jax" 2>/dev/null; then
        ver=$(python3 -c "import jax; print(jax.__version__)" 2>/dev/null)
        backend=$(python3 -c "import jax; print(jax.default_backend())" 2>/dev/null)
        echo "    ✓ JAX ($ver) — backend: $backend"
        if [ "$backend" = "cpu" ]; then
            echo "      ↳ pip install jax[cuda12] (NVIDIA) or jax-metal (Apple)"
        fi
    else
        echo "    ✗ JAX — pip install jax"
    fi

    # GGML backends (llama.cpp, whisper.cpp)
    echo ""
    echo "  GGML backends:"
    if python3 -c "import llama_cpp" 2>/dev/null; then
        ver=$(python3 -c "import llama_cpp; print(getattr(llama_cpp, '__version__', 'unknown'))" 2>/dev/null)
        gpu=$(python3 -c "
from llama_cpp import llama_supports_gpu_offload
print('GPU offload' if llama_supports_gpu_offload() else 'CPU only')
" 2>/dev/null)
        echo "    ✓ llama.cpp ($ver via llama-cpp-python) — $gpu"
        if [ "$gpu" = "CPU only" ]; then
            echo "      ↳ CMAKE_ARGS=\"-DGGML_CUDA=ON\" pip install llama-cpp-python --force-reinstall --no-cache-dir  (NVIDIA)"
            echo "      ↳ CMAKE_ARGS=\"-DGGML_VULKAN=ON\" pip install llama-cpp-python --force-reinstall --no-cache-dir (Vulkan)"
            if [ "$(uname -s)" = "Darwin" ]; then
                echo "      ↳ CMAKE_ARGS=\"-DGGML_METAL=ON\" pip install llama-cpp-python --force-reinstall --no-cache-dir  (Metal)"
            fi
        fi
    else
        echo "    ✗ llama.cpp — pip install llama-cpp-python"
    fi
    if python3 -c "import faster_whisper" 2>/dev/null; then
        ver=$(python3 -c "import faster_whisper; print(faster_whisper.__version__)" 2>/dev/null)
        echo "    ✓ whisper.cpp ($ver via faster-whisper)"
    else
        echo "    ~ whisper.cpp — not installed (pip install faster-whisper)"
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

# --- Resolve platform name for --update ---
PLATFORM=""
if [ "$UPDATE" = true ]; then
    if [ -n "$PLATFORM_OVERRIDE" ]; then
        PLATFORM="$PLATFORM_OVERRIDE"
    else
        PLATFORM=$(detect_platform)
    fi

    echo "" >&2
    echo "Platform: $PLATFORM" >&2

    if platform_exists_in_md "$PLATFORM" >/dev/null 2>&1; then
        echo "  (found in existing results — rows will be replaced)" >&2
    else
        echo "  (not found in existing results — will be added as new)" >&2
        # Interactive prompt: let user confirm or edit the name.
        if [ -t 0 ]; then
            echo "" >&2
            read -r -p "Use '$PLATFORM' as platform name? [Y/n/edit]: " reply </dev/tty
            case "$reply" in
                [nN]*)
                    echo "Aborted." >&2
                    exit 0
                    ;;
                [eE]*)
                    read -r -p "Enter platform name: " PLATFORM </dev/tty
                    ;;
                *)
                    # Accept default
                    ;;
            esac
        fi
    fi
    echo "" >&2
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

    if [ "$DRY_RUN" = true ]; then
        ARGS+=("--dry-run")
    fi

    if [ "$UPDATE" = true ]; then
        # Capture table output for markdown update.
        TABLE_FILE=$(mktemp)
        "$HARNESS" "${ARGS[@]}" | tee "$TABLE_FILE" || true
        python3 "$ROOT_DIR/scripts/update_results.py" \
            --model "$MODEL" --platform "$PLATFORM" --table "$TABLE_FILE" --root "$ROOT_DIR"
        rm -f "$TABLE_FILE"
    else
        "$HARNESS" "${ARGS[@]}" || true
    fi
done
