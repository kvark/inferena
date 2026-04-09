#!/usr/bin/env python3
"""Update model markdown files with new benchmark results.

Reads the harness's markdown table output (stdin or --table), identifies the
platform section, and merges the new rows into the corresponding
models/<model>.md file — replacing existing rows for that platform or
appending a new platform section.

Usage:
    inferena --model SmolLM2-135M | python3 scripts/update_results.py --model SmolLM2-135M --platform "Apple M3"
    python3 scripts/update_results.py --model SmolLM2-135M --platform "Apple M3" --table results.md
"""

import argparse
import os
import re
import sys


def parse_table_rows(lines):
    """Parse markdown table lines into a list of row dicts.

    Returns [(platform_or_empty, rest_of_row_text), ...] — one per data row.
    The header and separator rows are returned separately.
    """
    header = None
    separator = None
    rows = []
    for line in lines:
        line = line.rstrip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        # cells[0] and cells[-1] are empty from leading/trailing |
        if header is None:
            header = line
            continue
        if separator is None:
            separator = line
            continue
        # Data row
        rows.append(line)
    return header, separator, rows


def extract_platform(row):
    """Extract the platform name from a table row's first data cell.

    Returns the platform name (possibly empty for continuation rows).
    """
    cells = row.split("|")
    if len(cells) < 3:
        return ""
    return cells[1].strip()


def group_by_platform(rows):
    """Group rows into (platform_name, [rows]) pairs.

    Continuation rows (empty first cell) belong to the preceding platform.
    """
    groups = []  # [(platform, [row_lines])]
    current_platform = None
    current_rows = []

    for row in rows:
        platform = extract_platform(row)
        if platform:
            if current_platform is not None:
                groups.append((current_platform, current_rows))
            current_platform = platform
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_platform is not None:
        groups.append((current_platform, current_rows))

    return groups


def find_results_table(content):
    """Find the results table in markdown content.

    Returns (before, header, separator, rows_text, after) where:
    - before: content before the table (including ## Results heading)
    - header, separator: the table header lines
    - rows_text: all data rows as a string
    - after: content after the table (starting at the next non-table line)
    """
    lines = content.split("\n")
    table_start = None
    table_end = None

    # Find the first markdown table (| ... | pattern)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("| Platform"):
            table_start = i
            break

    if table_start is None:
        return None

    # Find end of table (first non-table line after the table)
    table_end = table_start
    for i in range(table_start, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("|"):
            table_end = i
        elif stripped == "":
            # Blank lines within table area are OK
            # Check if next non-blank line is still a table row
            continue
        else:
            break
    table_end += 1  # exclusive

    before = "\n".join(lines[:table_start])
    header = lines[table_start]
    separator = lines[table_start + 1] if table_start + 1 < table_end else ""
    data_rows = []
    for i in range(table_start + 2, table_end):
        stripped = lines[i].strip()
        if stripped.startswith("|"):
            data_rows.append(lines[i])
    after = "\n".join(lines[table_end:])

    return before, header, separator, data_rows, after


_FILLER_WORDS = {"processor", "graphics", "with", "the", "cpu", "gpu", "integrated"}


def _normalize_platform(name):
    """Normalize platform name for fuzzy matching.

    Strips filler words, trademarks, and extra whitespace.
    """
    s = name.lower()
    s = re.sub(r'\(r\)|\(tm\)', '', s)
    words = [w for w in s.split() if w not in _FILLER_WORDS]
    return " ".join(words)


def _platforms_match(a, b):
    """Check if two platform names refer to the same machine."""
    if a.lower() == b.lower():
        return True
    na, nb = _normalize_platform(a), _normalize_platform(b)
    if na == nb:
        return True
    # Substring match on normalized names
    if na in nb or nb in na:
        return True
    # Word overlap: match if 2+ significant words overlap
    wa, wb = set(na.split()), set(nb.split())
    overlap = wa & wb - {"@"}
    return len(overlap) >= 2


def merge_results(md_path, new_rows, platform_name):
    """Merge new benchmark rows into an existing model markdown file.

    If the platform already exists in the table, replace its rows.
    If it's new, append the platform section at the end of the table.
    """
    with open(md_path, "r") as f:
        content = f.read()

    parsed = find_results_table(content)
    if parsed is None:
        print(f"Warning: no results table found in {md_path}", file=sys.stderr)
        return False

    before, header, separator, existing_rows, after = parsed

    # Group existing rows by platform
    existing_groups = group_by_platform(existing_rows)

    # Group new rows by platform (should be just one platform)
    new_groups = group_by_platform(new_rows)
    if not new_groups:
        print(f"Warning: no data rows in new results", file=sys.stderr)
        return False

    # The new rows use the gpu_name from the harness. We need to match/replace
    # based on the user-provided platform_name.
    # Rewrite the first cell of the first new row to use platform_name.
    new_platform_rows = []
    for i, row in enumerate(new_rows):
        if i == 0:
            # Replace the first cell with the platform name
            cells = row.split("|")
            if len(cells) >= 3:
                cells[1] = f" {platform_name} "
                row = "|".join(cells)
        else:
            # Continuation rows: ensure first cell is empty
            cells = row.split("|")
            if len(cells) >= 3:
                cells[1] = " "
                row = "|".join(cells)
        new_platform_rows.append(row)

    # Check if platform already exists (fuzzy match)
    replaced = False
    updated_groups = []
    for existing_name, existing_group_rows in existing_groups:
        if _platforms_match(existing_name, platform_name):
            # Replace this platform's rows
            updated_groups.append((platform_name, new_platform_rows))
            replaced = True
            print(f"Replaced platform '{existing_name}' with '{platform_name}' ({len(new_platform_rows)} rows)", file=sys.stderr)
        else:
            updated_groups.append((existing_name, existing_group_rows))

    if not replaced:
        updated_groups.append((platform_name, new_platform_rows))
        print(f"Added new platform '{platform_name}' ({len(new_platform_rows)} rows)", file=sys.stderr)

    # Rebuild the table
    all_rows = []
    for _name, rows in updated_groups:
        all_rows.extend(rows)

    new_content = before + "\n"
    new_content += header + "\n"
    new_content += separator + "\n"
    new_content += "\n".join(all_rows) + "\n"
    new_content += after

    with open(md_path, "w") as f:
        f.write(new_content)

    return True


def main():
    parser = argparse.ArgumentParser(description="Update model markdown with benchmark results")
    parser.add_argument("--model", required=True, help="Model name (e.g. SmolLM2-135M)")
    parser.add_argument("--platform", required=True, help="Platform name for the results")
    parser.add_argument("--table", help="File with markdown table (default: stdin)")
    parser.add_argument("--root", help="Project root directory")
    args = parser.parse_args()

    # Determine root
    root = args.root
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    md_path = os.path.join(root, "models", f"{args.model}.md")
    if not os.path.isfile(md_path):
        print(f"Error: {md_path} not found", file=sys.stderr)
        sys.exit(1)

    # Read new table rows
    if args.table:
        with open(args.table) as f:
            input_lines = f.readlines()
    else:
        input_lines = sys.stdin.readlines()

    # Parse the input table, extract just the data rows (skip header/separator)
    _header, _sep, new_rows = parse_table_rows(input_lines)
    if not new_rows:
        print(f"No table rows found in input", file=sys.stderr)
        sys.exit(1)

    if merge_results(md_path, new_rows, args.platform):
        print(f"Updated {md_path}", file=sys.stderr)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
