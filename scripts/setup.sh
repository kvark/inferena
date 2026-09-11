#!/usr/bin/env bash
# Create an isolated comparison environment, downloading Python when needed.
set -euo pipefail
case "$(uname -s)" in
    MINGW*|MSYS*) ROOT=$(cd "$(dirname "$0")/.." && pwd -W) ;;
    *) ROOT=$(cd "$(dirname "$0")/.." && pwd) ;;
esac
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: bash scripts/setup.sh <cu130|xpu|cpu|mps|rocm7.2> [new-venv-path] [--no-max-autotune]" >&2
    exit 2
fi
command -v uv >/dev/null || {
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 2
}
BACKEND=$1
shift
case "$BACKEND" in
    cu130|xpu|cpu|mps|rocm7.2) ;;
    *) echo "Unknown wheel backend: $BACKEND" >&2; exit 2 ;;
esac
ENV_DIR=
NO_MAX_AUTOTUNE=0
for ARG in "$@"; do
    case "$ARG" in
        --no-max-autotune) NO_MAX_AUTOTUNE=1 ;;
        -*) echo "Unknown setup option: $ARG" >&2; exit 2 ;;
        *)
            if [ -n "$ENV_DIR" ]; then
                echo "Only one new environment path may be provided" >&2
                exit 2
            fi
            ENV_DIR=$ARG
            ;;
    esac
done
ENV_DIR=${ENV_DIR:-"$ROOT/.venv-p3hpc"}
if [ -e "$ENV_DIR" ]; then
    echo "Environment already exists: $ENV_DIR. Choose a new path; nothing was replaced." >&2
    exit 2
fi
read -r PYTHON_VERSION < "$ROOT/.python-version"
uv venv --managed-python --python "$PYTHON_VERSION" "$ENV_DIR"
ENV_PYTHON="$ENV_DIR/bin/python"
if [ -f "$ENV_DIR/Scripts/python.exe" ]; then
    ENV_PYTHON="$ENV_DIR/Scripts/python.exe"
fi
TORCH_ARGS=()
if [ "$BACKEND" != mps ]; then
    TORCH_ARGS=(--torch-backend "$BACKEND")
fi
if [ "$BACKEND" = rocm7.2 ]; then
    # The ROCm torch wheel depends on a newly published triton-rocm wheel.
    # Expose and refresh the vendor index explicitly instead of trusting a
    # possibly stale implicit --torch-backend index entry.
    TORCH_ARGS+=(
        --extra-index-url "https://download.pytorch.org/whl/$BACKEND"
        --refresh-package triton-rocm
    )
fi
REQUIREMENTS="$ROOT/requirements-p3hpc.txt"
if [ -f "$ENV_DIR/Scripts/python.exe" ] && [ "$BACKEND" = cu130 ]; then
    REQUIREMENTS="$ROOT/requirements-p3hpc-cu130-windows.txt"
fi
uv pip install --python "$ENV_PYTHON" "${TORCH_ARGS[@]}" -r "$REQUIREMENTS"
case "$BACKEND" in cu130) PROBE_BACKEND=cuda ;; rocm*) PROBE_BACKEND=rocm ;; *) PROBE_BACKEND=$BACKEND ;; esac
PROBE_ARGS=(--backend "$PROBE_BACKEND")
if [ "$NO_MAX_AUTOTUNE" -eq 1 ]; then
    PROBE_ARGS+=(--no-max-autotune)
fi
"$ENV_PYTHON" "$ROOT/scripts/check_environment.py" "${PROBE_ARGS[@]}"
