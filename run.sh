#!/usr/bin/env bash
set -euo pipefail

# --- Platform flags (Windows via git bash = MINGW/MSYS) ---
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=true ;;
    *)                    IS_WINDOWS=false ;;
esac
EXE_SUFFIX=""
$IS_WINDOWS && EXE_SUFFIX=".exe"

# Use a native path on Windows (e.g. C:/Code/inferena), so values embedded
# into Python/Cargo commands resolve correctly. MSYS `pwd` returns
# `/c/Code/...` which Python cannot interpret as a filesystem path.
if $IS_WINDOWS; then
    ROOT_DIR="$(cd "$(dirname "$0")" && pwd -W)"
    export INFERENA_BASH="$(cygpath -m "$BASH")"
else
    ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

# --- Pick a Python interpreter ---
# 1. Honor explicit $PYTHON, then an activated $VIRTUAL_ENV.
# 2. Prefer the paper environment, then the legacy .venv.
# 3. On Windows, prefer `python` — venvs don't ship a `python3` shim, so
#    `python3` falls through to the Store shim, which has none of our packages.
# 4. On Linux/macOS, prefer `python3`.
if [ -z "${PYTHON:-}" ] && [ -n "${VIRTUAL_ENV:-}" ]; then
    if [ -x "$VIRTUAL_ENV/bin/python" ]; then
        PYTHON="$VIRTUAL_ENV/bin/python"
    elif [ -x "$VIRTUAL_ENV/Scripts/python.exe" ]; then
        PYTHON="$VIRTUAL_ENV/Scripts/python.exe"
    fi
fi
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$ROOT_DIR/.venv-p3hpc/bin/python" ]; then
        PYTHON="$ROOT_DIR/.venv-p3hpc/bin/python"
    elif [ -x "$ROOT_DIR/.venv-p3hpc/Scripts/python.exe" ]; then
        PYTHON="$ROOT_DIR/.venv-p3hpc/Scripts/python.exe"
    elif [ -x "$ROOT_DIR/.venv/bin/python" ]; then
        PYTHON="$ROOT_DIR/.venv/bin/python"
    elif [ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]; then
        PYTHON="$ROOT_DIR/.venv/Scripts/python.exe"
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

# Force UTF-8 I/O. Windows' default cp1252 stdout chokes when libraries
# (e.g. torch.onnx's dynamo exporter) print non-ASCII characters, killing
# the subprocess with UnicodeEncodeError even if the real work succeeded.
export PYTHONIOENCODING="utf-8"
export PYTHONUTF8="1"

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
if [ -z "${VK_ICD_FILENAMES:-}" ] && [[ "${INFERENA_TORCH_BACKEND:-cuda}" == cuda ]]; then
    NVIDIA_ICD=$(find /usr/share/vulkan/icd.d /etc/vulkan/icd.d -name '*nvidia*' 2>/dev/null | head -1 || true)
    if [ -n "$NVIDIA_ICD" ]; then
        export VK_ICD_FILENAMES="$NVIDIA_ICD"
    fi
fi

ALL_MODELS="SmolLM2-135M SmolVLA StableDiffusion ResNet-50 Whisper-tiny"
SUPPORTED_MODELS="$ALL_MODELS SmolLM2-360M SmolLM2-1.7B"

# --- Parse arguments ---
MODELS=""
FRAMEWORKS=""
JSON_FLAG=""
DOWNLOAD=false
CHECK_ONLY=false
DRY_RUN=false
UPDATE=false
PLATFORM_OVERRIDE=""
STRICT=false
INFERENCE_ONLY=false
ALLOW_INTEGRATED_GPU=false
WARMUP_RUNS=5
MEASUREMENT_RUNS=20
PROFILE=false
PROFILE_SAMPLES=3
RESULTS_DIR="$ROOT_DIR/results"
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
        --strict)
            STRICT=true
            HAS_ARGS=true
            shift
            ;;
        --inference-only)
            INFERENCE_ONLY=true
            HAS_ARGS=true
            shift
            ;;
        --allow-integrated-gpu)
            ALLOW_INTEGRATED_GPU=true
            HAS_ARGS=true
            shift
            ;;
        --warmup-runs)
            WARMUP_RUNS="$2"
            HAS_ARGS=true
            shift 2
            ;;
        --measurement-runs)
            MEASUREMENT_RUNS="$2"
            HAS_ARGS=true
            shift 2
            ;;
        --profile)
            PROFILE=true
            HAS_ARGS=true
            shift
            ;;
        --profile-samples)
            PROFILE_SAMPLES="$2"
            HAS_ARGS=true
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            HAS_ARGS=true
            shift 2
            ;;
        -h|--help)
            echo "inferena - Inference Arena"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -m, --model <name|all>    Model to benchmark (default: all)"
            echo "  -f, --frameworks <list>   Comma-separated frameworks (default: all)"
            echo "  --json                    Output results as JSON"
            echo "  --download                Download model weights before running"
            echo "  --check                   Check framework availability (don't run benchmarks)"
            echo "  --dry-run                 Validate framework+model support without running benchmarks"
            echo "  --update                  Update models/*.md with results after benchmarking"
            echo "  --platform <name>         Override auto-detected platform name (with --update)"
            echo "  --strict                  Disable reduced-input fast paths for an f32 control run"
            echo "  --inference-only          Forward-only paired SmolLM2 workloads; no training allocation"
            echo "  --allow-integrated-gpu    Do not prefer a discrete GPU on hybrid systems"
            echo "  --warmup-runs <n>         Untimed runs per measurement (default: 5)"
            echo "  --measurement-runs <n>    Timed samples per measurement (default: 20)"
            echo "  --profile                 Collect separate Meganeura/PyTorch diagnostic profiles"
            echo "  --profile-samples <n>     Timestamp samples per profile (default: 3)"
            echo "  --results-dir <path>      JSON/chart artifact directory (default: results/)"
            echo "  -h, --help                Show this help"
            echo ""
            echo "Models: $SUPPORTED_MODELS (larger SmolLM2 sizes are opt-in)"
            echo "Frameworks: pytorch, candle, burn, inferi, luminal, meganeura, ggml, onnxruntime, max, jax, mlx"
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
elif [ "$MODELS" = "all" ]; then
    MODELS="$ALL_MODELS"
fi

# --- Platform detection ---
_detect_platform_raw() {
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
        # Prefer discrete GPU over integrated (APU) over anything else —
        # `head -1` used to grab GPU0, which on hybrid AMD systems is the iGPU
        # whose RADV deviceName is the parent CPU's product string.
        local vk_dev
        vk_dev=$(vulkaninfo --summary 2>/dev/null | awk '
            /^GPU[0-9]+:/      { type=""; next }
            /deviceType/       { type=$0; next }
            /deviceName/ {
                sub(/.*= /, "")
                if      (type ~ /DISCRETE_GPU/   && !discrete)   discrete=$0
                else if (type ~ /INTEGRATED_GPU/ && !integrated) integrated=$0
                else if (!other)                                 other=$0
            }
            END {
                if      (discrete)   print discrete
                else if (integrated) print integrated
                else if (other)      print other
            }' | xargs)
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

detect_platform() {
    local name
    name=$(_detect_platform_raw)
    if $IS_WINDOWS; then
        echo "$name (Windows)"
    else
        echo "$name"
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
    check_python "MAX" "max.engine" " (Modular inference engine, Linux/macOS only)"
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
         "$PYTHON" -c "import torch; assert torch.version.hip" 2>/dev/null; then
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
        bin="$ROOT_DIR/target/release/$pkg${EXE_SUFFIX}"
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
mkdir -p "$RESULTS_DIR"

# --- Build the Rust crates this run actually needs ---
# Building the whole workspace compiles every framework runner regardless of
# --frameworks. That wastes time on a two-engine run, and it is how the inferi
# shader toolchain reaches builds that never asked for it.
echo "Building Rust crates..." >&2
WORKSPACE_CARGO_ARGS=(
    build
    --release
    --manifest-path "$ROOT_DIR/Cargo.toml"
)
WORKSPACE_LOCK_BACKUP=""
restore_workspace_lockfile() {
    if [ -n "$WORKSPACE_LOCK_BACKUP" ] && [ -f "$WORKSPACE_LOCK_BACKUP" ]; then
        cp "$WORKSPACE_LOCK_BACKUP" "$ROOT_DIR/Cargo.lock"
        rm -f "$WORKSPACE_LOCK_BACKUP"
        WORKSPACE_LOCK_BACKUP=""
    fi
}
if [ -n "${INFERENA_MEGANEURA_PATH:-}" ]; then
    if [ ! -f "$INFERENA_MEGANEURA_PATH/Cargo.toml" ]; then
        echo "INFERENA_MEGANEURA_PATH does not contain Cargo.toml: $INFERENA_MEGANEURA_PATH" >&2
        exit 2
    fi
    INFERENA_MEGANEURA_PATH=$(cd "$INFERENA_MEGANEURA_PATH" && pwd -P)
    export INFERENA_MEGANEURA_PATH
    WORKSPACE_LOCK_BACKUP=$(mktemp)
    cp "$ROOT_DIR/Cargo.lock" "$WORKSPACE_LOCK_BACKUP"
    trap restore_workspace_lockfile EXIT
    WORKSPACE_CARGO_ARGS+=(
        --config
        "patch.\"https://github.com/kvark/meganeura\".meganeura.path=\"$INFERENA_MEGANEURA_PATH\""
    )
else
    WORKSPACE_CARGO_ARGS+=(--locked)
fi
# An empty --frameworks means "all".
want_framework() {
    [ -z "$FRAMEWORKS" ] && return 0
    case ",$FRAMEWORKS," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# The harness itself is always required. Python-based runners (pytorch, ggml,
# onnxruntime, max, jax, mlx) have no crate in this workspace.
WORKSPACE_CARGO_ARGS+=(-p inferena-harness)
if want_framework meganeura; then WORKSPACE_CARGO_ARGS+=(-p inferena-meganeura); fi
if want_framework burn; then WORKSPACE_CARGO_ARGS+=(-p inferena-burn); fi
if want_framework candle; then WORKSPACE_CARGO_ARGS+=(-p inferena-candle); fi
if want_framework luminal; then WORKSPACE_CARGO_ARGS+=(-p inferena-luminal); fi

# Inferi is opt-in even when requested. It pulls khal -> khal-builder -> vortx,
# whose build script installs the rust-gpu toolchain and prompts for consent on
# a missing one. Build scripts have no TTY, so that prompt fails the build
# rather than asking. The consent bypass is a cargo-gpu CLI flag
# (--auto-install-rust-toolchain), not an environment variable, so it cannot be
# supplied from out here: the toolchain has to already be installed.
# `cargo gpu --version` is not a sufficient check, because installing the
# binary and installing the toolchain are separate steps.
if want_framework inferi; then
    if [ "${INFERENA_ENABLE_INFERI:-0}" = "1" ]; then
        if ! cargo gpu --version &>/dev/null; then
            echo "INFERENA_ENABLE_INFERI=1 but cargo-gpu is not on PATH." >&2
            echo "  cargo install cargo-gpu --version 0.10.0-alpha.1 && cargo gpu install" >&2
            exit 2
        fi
        WORKSPACE_CARGO_ARGS+=(-p inferena-inferi)
    else
        echo "  (skipping inferi: needs the rust-gpu toolchain; run" >&2
        echo "   'cargo install cargo-gpu --version 0.10.0-alpha.1 && cargo gpu install'," >&2
        echo "   then set INFERENA_ENABLE_INFERI=1)" >&2
    fi
fi

cargo "${WORKSPACE_CARGO_ARGS[@]}" >&2
restore_workspace_lockfile
trap - EXIT

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

    ARGS=(
        "--model" "$MODEL"
        "--root" "$ROOT_DIR"
        "--results-dir" "$RESULTS_DIR"
        "--warmup-runs" "$WARMUP_RUNS"
        "--measurement-runs" "$MEASUREMENT_RUNS"
    )

    if [ "$STRICT" = true ]; then
        ARGS+=("--strict")
    fi
    if [ "$INFERENCE_ONLY" = true ]; then
        ARGS+=("--inference-only")
    fi
    if [ "$ALLOW_INTEGRATED_GPU" = true ]; then
        ARGS+=("--allow-integrated-gpu")
    fi

    if [ "$PROFILE" = true ]; then
        ARGS+=("--profile" "--profile-samples" "$PROFILE_SAMPLES")
    fi

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
        "$PYTHON" "$ROOT_DIR/scripts/update_results.py" \
            --model "$MODEL" --platform "$PLATFORM" --table "$TABLE_FILE" --root "$ROOT_DIR"
        rm -f "$TABLE_FILE"
    else
        "$HARNESS" "${ARGS[@]}" || true
    fi
done

# --- Per-platform SVG chart ---
# Always emitted (not gated on --update) so every run leaves a visual summary
# of what the machine just produced. Uses the override if one was supplied,
# otherwise defers to the script's own gpu_name extraction.
CHART_PLATFORM="${PLATFORM:-}"
if [ -z "$CHART_PLATFORM" ] && [ "$DRY_RUN" != true ]; then
    CHART_PLATFORM=$(detect_platform)
fi
if [ "$DRY_RUN" != true ]; then
    "$PYTHON" "$ROOT_DIR/scripts/generate_chart.py" \
        --results-dir "$RESULTS_DIR" \
        ${CHART_PLATFORM:+--platform "$CHART_PLATFORM"} || true
fi
