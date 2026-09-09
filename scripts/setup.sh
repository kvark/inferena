#!/usr/bin/env bash
# Create an isolated comparison environment, downloading Python when needed.
set -euo pipefail
case "$(uname -s)" in
    MINGW*|MSYS*) ROOT=$(cd "$(dirname "$0")/.." && pwd -W) ;;
    *) ROOT=$(cd "$(dirname "$0")/.." && pwd) ;;
esac
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: bash scripts/setup.sh <cu130|xpu|cpu|mps|rocm7.2> [new-venv-path]" >&2
    exit 2
fi
command -v uv >/dev/null || {
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 2
}
BACKEND=$1
case "$BACKEND" in
    cu130|xpu|cpu|mps|rocm7.2) ;;
    *) echo "Unknown wheel backend: $BACKEND" >&2; exit 2 ;;
esac
ENV_DIR=${2:-"$ROOT/.venv-p3hpc"}
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
REQUIREMENTS="$ROOT/requirements-p3hpc.txt"
if [ -f "$ENV_DIR/Scripts/python.exe" ] && [ "$BACKEND" = cu130 ]; then
    REQUIREMENTS="$ROOT/requirements-p3hpc-cu130-windows.txt"
fi
uv pip install --python "$ENV_PYTHON" "${TORCH_ARGS[@]}" -r "$REQUIREMENTS"
case "$BACKEND" in cu130) PROBE_BACKEND=cuda ;; rocm*) PROBE_BACKEND=rocm ;; *) PROBE_BACKEND=$BACKEND ;; esac
"$ENV_PYTHON" "$ROOT/scripts/check_environment.py" --backend "$PROBE_BACKEND"
