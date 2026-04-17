#!/usr/bin/env python3
"""Lightweight format check for models/*.md and the site include plumbing.

Fast pre-publish gate: catches broken tables and stale index.md includes
without recompiling anything.

Checks:
- Every models/*.md has a parseable results table with the expected header.
- Every data row has the expected column count.
- Every framework cell contains a markdown link (or is a bare cell like ✗).
- The tab includes in index.md map to existing models/*.md files (and vice
  versa), matching the slug transform that pages.yml applies at build time.

Usage: python3 scripts/check_results_md.py [--root .]
Exit 0 on success, 1 on any validation failure.
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from update_results import find_results_table  # noqa: E402


EXPECTED_HEADER_CELLS = [
    "Platform",
    "Framework",
    "Compile (s)",
    "Inference (ms)",
    "Latency (ms)",
    "Training (ms)",
    "Loss",
]
COLUMNS = len(EXPECTED_HEADER_CELLS)


def split_row(line):
    """Split `| a | b | c |` into ['a', 'b', 'c'] with cells trimmed."""
    cells = [c.strip() for c in line.split("|")]
    # Drop the empty strings before the first | and after the last |.
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def check_md(path):
    """Return a list of error strings for this file (empty = OK)."""
    errors = []
    with open(path) as f:
        content = f.read()
    parsed = find_results_table(content)
    if parsed is None:
        return [f"{path}: no results table found"]
    _, header, _, rows, _ = parsed
    header_cells = split_row(header)
    if header_cells != EXPECTED_HEADER_CELLS:
        errors.append(
            f"{path}: header mismatch\n  got:      {header_cells}\n  expected: {EXPECTED_HEADER_CELLS}"
        )
    for i, row in enumerate(rows):
        cells = split_row(row)
        if len(cells) != COLUMNS:
            errors.append(
                f"{path}: row {i} has {len(cells)} cells, expected {COLUMNS}: {row.strip()}"
            )
            continue
        framework_cell = cells[1]
        # Framework cell must contain a markdown link [...](...) — the empty
        # value "" is allowed only for continuation rows, but continuation
        # rows also have an empty platform cell, so the framework column is
        # always populated.
        if not re.search(r"\[[^\]]+\]\([^)]+\)", framework_cell):
            errors.append(
                f"{path}: row {i} framework cell missing markdown link: {framework_cell!r}"
            )
    return errors


def slug(model_name):
    """Mirror the slug transform done in pages.yml when generating _includes."""
    return model_name.lower()


def check_index_coverage(root):
    """Confirm index.md's tab includes line up with the models/ directory."""
    errors = []
    index_path = os.path.join(root, "index.md")
    if not os.path.isfile(index_path):
        return [f"{index_path}: missing"]
    with open(index_path) as f:
        index = f.read()
    included = set(re.findall(r"{%\s*include\s+([^\s%]+)\.md\s*%}", index))
    model_files = sorted(glob.glob(os.path.join(root, "models", "*.md")))
    model_slugs = {slug(os.path.basename(p)[:-3]) for p in model_files}
    missing_in_index = model_slugs - included
    extra_in_index = included - model_slugs
    for s in sorted(missing_in_index):
        errors.append(f"index.md: no tab for models/{s}.md (would orphan it)")
    for s in sorted(extra_in_index):
        errors.append(f"index.md: includes {s}.md but models/{s}.md is missing")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    all_errors = []
    md_files = sorted(glob.glob(os.path.join(args.root, "models", "*.md")))
    if not md_files:
        all_errors.append(f"{args.root}/models/: no markdown files")
    for path in md_files:
        all_errors.extend(check_md(path))
    all_errors.extend(check_index_coverage(args.root))

    if all_errors:
        for e in all_errors:
            print(e, file=sys.stderr)
        print(f"\n{len(all_errors)} error(s); {len(md_files)} file(s) checked", file=sys.stderr)
        sys.exit(1)
    print(f"OK — {len(md_files)} file(s) checked", file=sys.stderr)


if __name__ == "__main__":
    main()
