#!/usr/bin/env bash
# Create an isolated comparison environment, downloading Python when needed.
set -euo pipefail
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) ROOT=$(cd "$(dirname "$0")/.." && pwd -W) ;;
    *) ROOT=$(cd "$(dirname "$0")/.." && pwd) ;;
esac
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
uv pip install --python "$ENV_PYTHON" "${TORCH_ARGS[@]}" -r "$ROOT/requirements-p3hpc.txt"
"$ENV_PYTHON" - "$ROOT" <<'PY'
from pathlib import Path
import platform
import sys
import torch
sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
from p3hpc import check_torch_identity
check_torch_identity(torch.__version__, torch.version.git_version, torch.__version__)
print(f"Ready: Python {platform.python_version()}, torch {torch.__version__}, source {torch.version.git_version}")
print(f"Collector: {sys.executable} scripts/p3hpc.py --help")
PY
