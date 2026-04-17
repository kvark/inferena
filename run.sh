#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Platform flags (Windows via git bash = MINGW/MSYS) ---
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=true ;;
    *)                    IS_WINDOWS=false ;;
esac
EXE_SUFFIX=""
$IS_WINDOWS && EXE_SUFFIX=".exe"

# --- Pick a Python interpreter ---
# 1. Honor $VIRTUAL_ENV if set (user activated a venv).
# 2. On Windows, prefer `python` — venvs don't ship a `python3` shim, so
#    `python3` falls through to the Store shim, which has none of our packages.
# 3. On Linux/macOS, prefer `python3`.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    if [ -x "$VIRTUAL_ENV/bin/python" ]; then
        PYTHON="$VIRTUAL_ENV/bin/python"
    elif [ -x "$VIRTUAL_ENV/Scripts/python.exe" ]; then
        PYTHON="$VIRTUAL_ENV/Scripts/python.exe"
    fi
fi
if [ -z "${PYTHON:-}" ]; then
    if $IS_WINDOWS; then
        if command -v python &>/dev/null; then PYTHON=python
        elif command -v python3 &>/dev/null; then PYTHON=python3
        else echo "Python not found on PATH (need python or python3)" >&2; exit 1; fi
    else
        if command -v python3 &>/dev/null; then PYTHON=python3
        elif command -v python &>/dev/null; then PYTHON=python
        else echo "Python not found on PATH (need python3 or python)" >&2; exit 1; fi
    fi
fi
export PYTHON

# --- Expose CUDA libraries bundled by pip packages (e.g. nvidia-cublas-cu12) ---
# Linux-only: Windows pip wheels ship CUDA DLLs in site-packages/nvidia/*/bin
# and the runtime loader is PATH, not LD_LIBRARY_PATH. Torch/ORT wheels on
# Windows handle DLL discovery themselves, so this block is a no-op there.
if ! $IS_WINDOWS; then
    NVIDIA_LIBS=$("$PYTHON" -c "
import os, site
for d in site.getsitepackages():
    nv = os.path.join(d, 'nvidia')
    if os.path.isdir(nv):
        for sub in os.listdir(nv):
            lib = os.path.join(nv, sub, 'lib')
            if os.path.isdir(lib):
                print(lib)
" 2>/dev/null | paste -sd: -)
    if [ -n "$NVIDIA_LIBS" ]; then
        export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

# --- Prefer discrete NVIDIA GPU over integrated GPU for Vulkan ---
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
    NVIDIA_ICD=$(find /usr/share/vulkan/icd.d /etc/vulkan/icd.d -name '*nvidia*' 2>/dev/null | head -1 || true)
    if [ -n "$NVIDIA_ICD" ]; then
        export VK_ICD_FILENAMES="$NVIDIA_ICD"
    fi
fi

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
            echo "Frameworks: pytorch, candle, burn, inferi, luminal, meganeura, ggml, onnxruntime, jax, mlx"
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
    if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        "$PYTHON" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null
        return
    fi
    if "$PYTHON" -c "import torch; assert hasattr(torch.backends,'mps') and torch.backends.mps.is_available()" 2>/dev/null; then
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
    elif $IS_WINDOWS; then
        # WMIC is deprecated but still present; fall back to PowerShell.
        local name
        name=$(wmic cpu get name /value 2>/dev/null | tr -d '\r' | sed -n 's/^Name=//p' | head -1)
        if [ -z "$name" ] && command -v powershell &>/dev/null; then
            name=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name" 2>/dev/null | tr -d '\r' | head -1)
        fi
        echo "$name" | sed 's/  */ /g; s/(R)//g; s/(TM)//g; s/CPU //g' | xargs
    else
        grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | sed 's/.*: //' | sed 's/  */ /g' | sed 's/(R)//g; s/(TM)//g; s/CPU //g' | xargs
    fi
}

# Check if a platform name exists in any model markdown file.
platform_exists_in_md() {
    local platform="$1"
    "$PYTHON" -c "
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
        if "$PYTHON" -c "import $mod" 2>/dev/null; then
            ver=$("$PYTHON" -c "import $mod; print(getattr($mod, '__version__', 'unknown'))" 2>/dev/null)
            echo "  ✓ $name ($ver)$extra"
        else
            echo "  ✗ $name — "$PYTHON" -c 'import $mod' failed (pip install $mod)"
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

    # Detect GPU type for targeted install hints.
    GPU_TYPE="none"
    if [ "$(uname -s)" = "Darwin" ]; then
        GPU_TYPE="apple"
    elif command -v nvidia-smi &>/dev/null; then
        GPU_TYPE="nvidia"
    elif command -v rocm-smi &>/dev/null || [ -d /opt/rocm ] || \
         python3 -c "import torch; assert torch.version.hip" 2>/dev/null; then
        GPU_TYPE="amd"
    elif command -v vulkaninfo &>/dev/null; then
        vk_dev=$(vulkaninfo --summary 2>/dev/null | grep "deviceName" | head -1 | sed 's/.*= //' | xargs)
        if [ -n "$vk_dev" ] && ! echo "$vk_dev" | grep -qi "llvmpipe\|lavapipe\|swrast"; then
            GPU_TYPE="vulkan"
        fi
    fi
    if [ "$GPU_TYPE" != "none" ]; then
        echo "  Detected GPU type: $GPU_TYPE"
        echo ""
    fi

    # GPU-aware Python framework checks.
    echo "  GPU-aware frameworks:"

    # ONNX Runtime
    if "$PYTHON" -c "import onnxruntime" 2>/dev/null; then
        ver=$("$PYTHON" -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null)
        gpu=$("$PYTHON" -c "
import onnxruntime as ort
providers = [p for p in ort.get_available_providers() if p != 'CPUExecutionProvider']
print(', '.join(providers) if providers else 'CPU only')
" 2>/dev/null)
        echo "    ✓ ONNX Runtime ($ver) — $gpu"
        if [ "$gpu" = "CPU only" ]; then
            case "$GPU_TYPE" in
                apple)  echo "      ↳ CoreML should be included in onnxruntime >= 1.14; try: pip install --upgrade onnxruntime" ;;
                nvidia) echo "      ↳ pip install onnxruntime-gpu" ;;
                amd)    echo "      ↳ pip install onnxruntime-rocm" ;;
                *)      echo "      ↳ pip install onnxruntime-gpu (NVIDIA) or onnxruntime-rocm (AMD)" ;;
            esac
        fi
    else
        echo "    ✗ ONNX Runtime — pip install onnxruntime"
    fi

    # JAX
    if "$PYTHON" -c "import jax" 2>/dev/null; then
        ver=$("$PYTHON" -c "import jax; print(jax.__version__)" 2>/dev/null)
        backend=$("$PYTHON" -c "import jax; print(jax.default_backend())" 2>/dev/null)
        echo "    ✓ JAX ($ver) — backend: $backend"
        if [ "$backend" = "cpu" ]; then
            case "$GPU_TYPE" in
                apple)  echo "      ↳ pip install jax-metal" ;;
                nvidia) echo "      ↳ pip install jax[cuda12]" ;;
                amd)    echo "      ↳ pip install jax[rocm]" ;;
                *)      echo "      ↳ pip install jax[cuda12] (NVIDIA) or jax-metal (Apple)" ;;
            esac
        fi
    else
        echo "    ✗ JAX — pip install jax"
    fi

    # GGML backends (llama.cpp, whisper.cpp)
    echo ""
    echo "  GGML backends:"
    if "$PYTHON" -c "import llama_cpp" 2>/dev/null; then
        ver=$("$PYTHON" -c "import llama_cpp; print(getattr(llama_cpp, '__version__', 'unknown'))" 2>/dev/null)
        gpu=$("$PYTHON" -c "
from llama_cpp import llama_supports_gpu_offload
print('GPU offload' if llama_supports_gpu_offload() else 'CPU only')
" 2>/dev/null)
        echo "    ✓ llama.cpp ($ver via llama-cpp-python) — $gpu"
        if [ "$gpu" = "CPU only" ]; then
            echo "      ↳ The ggml runner will auto-rebuild with GPU support on first run."
            if $IS_WINDOWS; then
                echo "      ↳ Requires: Vulkan SDK (https://vulkan.lunarg.com) or CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)"
            else
                echo "      ↳ Requires: apt install libvulkan-dev glslc  (Vulkan, preferred)"
                echo "      ↳      or:  apt install nvidia-cuda-toolkit  (CUDA, needs ≥12.8 for Blackwell)"
            fi
        fi
    else
        # Diagnose import failure — could be missing CUDA runtime.
        llama_err=$("$PYTHON" -c "import llama_cpp" 2>&1 || true)
        if echo "$llama_err" | grep -q "libcudart"; then
            echo "    ✗ llama.cpp — installed with CUDA but libcudart not found"
            if $IS_WINDOWS; then
            echo "      ↳ Install CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)"
        else
            echo "      ↳ apt install libcudart12  (or install the full CUDA Toolkit)"
        fi
            echo "      ↳ or: pip install llama-cpp-python --force-reinstall  (to get CPU-only build)"
        elif echo "$llama_err" | grep -q "libcuda\|libnvidia\|libcuBLAS"; then
            echo "    ✗ llama.cpp — installed with CUDA but missing runtime library"
            echo "      ↳ $llama_err"
        else
            echo "    ✗ llama.cpp — pip install llama-cpp-python"
        fi
    fi
    if "$PYTHON" -c "import faster_whisper" 2>/dev/null; then
        ver=$("$PYTHON" -c "import faster_whisper; print(faster_whisper.__version__)" 2>/dev/null)
        echo "    ✓ whisper.cpp ($ver via faster-whisper)"
    else
        echo "    ~ whisper.cpp — not installed (pip install faster-whisper)"
    fi

    echo ""

    # Rust framework GPU support notes.
    echo "  Rust framework GPU support:"
    echo "    candle:    CUDA, Metal          (no Vulkan/ROCm)"
    echo "    burn:      wgpu (Vulkan, Metal, DX12)"
    echo "    inferi:    wgpu (Vulkan, Metal), CUDA"
    echo "    luminal:   CUDA, Metal          (no Vulkan/ROCm)"
    echo "    meganeura: Vulkan, Metal"

    echo ""

    # Rust frameworks — check if binaries exist or can compile
    RUST_FW="inferena-candle inferena-burn inferena-inferi inferena-luminal inferena-meganeura"
    for pkg in $RUST_FW; do
        name="${pkg#inferena-}"
        bin="$ROOT_DIR/target/release/$pkg"
        if [ -f "$bin" ]; then
            echo "  ✓ $name (binary at $bin)"
        elif [ "$name" = "inferi" ] && ! cargo gpu --version &>/dev/null; then
            echo "  ✗ $name — requires cargo-gpu (cargo install cargo-gpu --git https://github.com/Rust-GPU/cargo-gpu)"
        elif cargo check -p "$pkg" --manifest-path "$ROOT_DIR/Cargo.toml" 2>/dev/null; then
            echo "  ~ $name (compiles, not yet built — run without --check to build)"
        else
            echo "  ✗ $name — cargo check -p $pkg failed"
        fi
    done

    echo ""

    # GPU backends
    echo "GPU backends:"
    if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        dev=$("$PYTHON" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        is_rocm=$("$PYTHON" -c "import torch; print('yes' if torch.version.hip else 'no')" 2>/dev/null)
        if [ "$is_rocm" = "yes" ]; then
            echo "  ✓ ROCm/HIP ($dev)"
        else
            echo "  ✓ CUDA ($dev)"
        fi
    else
        echo "  ✗ CUDA/ROCm — torch.cuda.is_available() is False"
    fi
    if command -v vulkaninfo &>/dev/null; then
        if [ -n "${VK_ICD_FILENAMES:-}" ]; then
            dev=$(vulkaninfo --summary 2>/dev/null | grep "deviceName" | head -1 | sed 's/.*= //')
            echo "  ✓ Vulkan ($dev) — forced via VK_ICD_FILENAMES"
        else
            # List all Vulkan devices
            vulkaninfo --summary 2>/dev/null | grep "deviceName" | sed 's/.*= //' | while read -r dev; do
                echo "  ✓ Vulkan ($dev)"
            done
        fi
    else
        echo "  ✗ Vulkan — vulkaninfo not found"
    fi
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "  ✓ Metal (macOS detected)"
    fi
    if command -v nvcc &>/dev/null || [ -d /usr/local/cuda ] || [ -n "${CUDA_PATH:-}" ]; then
        nvcc_ver=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        echo "  ✓ CUDA Toolkit (nvcc ${nvcc_ver:-found}) — needed for candle, luminal"
    elif "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "  ✗ CUDA Toolkit (nvcc) — needed to build candle/luminal with CUDA"
        if $IS_WINDOWS; then
            echo "    ↳ Install CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)"
        else
            echo "    ↳ apt install nvidia-cuda-toolkit"
        fi
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
if cargo gpu --version &>/dev/null; then
    cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" --workspace 2>&1 >&2
else
    echo "  (cargo-gpu not found — skipping inferi; install via: cargo install cargo-gpu --git https://github.com/Rust-GPU/cargo-gpu)" >&2
    cargo build --release --manifest-path "$ROOT_DIR/Cargo.toml" --workspace --exclude inferena-inferi 2>&1 >&2
fi

HARNESS="$ROOT_DIR/target/release/inferena${EXE_SUFFIX}"

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
