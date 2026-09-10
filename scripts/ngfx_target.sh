#!/usr/bin/env bash
set -euo pipefail
# Keep the target in the launcher's bounded scope; save complete validation output.
capture_dir=$1
runner=$2
model=$3
exec "$runner" "$model" > "$capture_dir/runner.json" 2> "$capture_dir/runner.log"
