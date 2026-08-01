#!/usr/bin/env bash
# Meganeura framework benchmark runner wrapper.
# Usage: ./run.sh <model_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-SmolLM2-135M}"

# The practical default enables eligible f16-input/f32-accumulate forward
# kernels. `--strict` disables reduced-input paths for a controlled f32 run.
if [ "${INFERENA_STRICT:-0}" = "1" ]; then
    unset MEGANEURA_COOP_F16
    export MEGANEURA_FLASH_FWD_COOP=0
    export MEGANEURA_FLASH_BWD_COOP=0
else
    export MEGANEURA_COOP_F16=1
    export MEGANEURA_FLASH_FWD_COOP=1
    # Backward remains f32: IEEE-f16 rounding of small derivative operands
    # caused a historical Whisper gradient failure. Full-precision autodiff
    # preserves the gradient gate while forward cooperative attention remains
    # enabled.
    export MEGANEURA_FLASH_BWD_COOP=0
fi
if [ -n "${INFERENA_PROFILE_DIR:-}" ]; then
    # Blade allocates timestamp query pools when the first GPU context is
    # created, so profiling must be enabled before launching the runner.
    export MEGANEURA_GPU_TIMING=1
fi

echo "[meganeura] Building release binary..." >&2
CARGO_ARGS=(
    build
    --release
    --manifest-path "$ROOT_DIR/Cargo.toml"
    -p inferena-meganeura
)
LOCK_BACKUP=""
restore_lockfile() {
    if [ -n "$LOCK_BACKUP" ] && [ -f "$LOCK_BACKUP" ]; then
        cp "$LOCK_BACKUP" "$ROOT_DIR/Cargo.lock"
        rm -f "$LOCK_BACKUP"
        LOCK_BACKUP=""
    fi
}
if [ -n "${INFERENA_MEGANEURA_PATH:-}" ]; then
    if [ ! -f "$INFERENA_MEGANEURA_PATH/Cargo.toml" ]; then
        echo "INFERENA_MEGANEURA_PATH does not contain Cargo.toml: $INFERENA_MEGANEURA_PATH" >&2
        exit 2
    fi
    # A command-line path patch makes Cargo remove the pinned git source from
    # Cargo.lock. Preserve the caller's exact lockfile (including any existing
    # edits) so a local benchmark does not dirty Inferena.
    LOCK_BACKUP=$(mktemp)
    cp "$ROOT_DIR/Cargo.lock" "$LOCK_BACKUP"
    trap restore_lockfile EXIT
    CARGO_ARGS+=(
        --config
        "patch.\"https://github.com/kvark/meganeura\".meganeura.path=\"$INFERENA_MEGANEURA_PATH\""
    )
    FRAMEWORK_REV=$(git -C "$INFERENA_MEGANEURA_PATH" rev-parse --short HEAD 2>/dev/null || echo local)
    if ! git -C "$INFERENA_MEGANEURA_PATH" diff --quiet --ignore-submodules HEAD 2>/dev/null; then
        FRAMEWORK_REV="${FRAMEWORK_REV}-dirty"
    fi
    export FRAMEWORK_REV
else
    # Extract the pinned git revision from Cargo.lock.
    source "$ROOT_DIR/scripts/cargo-rev.sh"
    export FRAMEWORK_REV=$(cargo_rev_short meganeura "$ROOT_DIR")
fi
cargo "${CARGO_ARGS[@]}" >&2
restore_lockfile
trap - EXIT

case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) EXE=.exe ;; *) EXE= ;; esac
exec "$ROOT_DIR/target/release/inferena-meganeura${EXE}" "$MODEL"
