#!/usr/bin/env bash
# Extract a crate's git revision from Cargo.lock.
# Usage: source scripts/cargo-rev.sh
#   cargo_rev <crate_name>  ->  prints the full git SHA
#   cargo_rev_short <crate_name>  ->  prints the first 7 chars

cargo_rev() {
    local crate="$1"
    local root="${2:-.}"
    grep -A5 "name = \"$crate\"" "$root/Cargo.lock" \
        | sed -n 's/.*#\([a-f0-9]*\)".*/\1/p' \
        | head -1
}

cargo_rev_short() {
    local full
    full=$(cargo_rev "$@")
    echo "${full:0:7}"
}
