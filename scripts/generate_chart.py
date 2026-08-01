#!/usr/bin/env python3
"""Render an SVG bar chart of per-framework inference times for the current
platform, one group per model. Reads results/<model>_summary.json files
written by the harness and emits results/<platform>.svg.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from xml.sax.saxutils import escape


FRAMEWORK_COLORS = {
    "pytorch":     "#ee4c2c",
    "mlx":         "#0a7cff",
    "candle":      "#f57c00",
    "burn":        "#b71c1c",
    "luminal":     "#8d6e63",
    "meganeura":   "#2ecc71",
    "inferi":      "#9c27b0",
    "ggml":        "#607d8b",
    "onnxruntime": "#0078d4",
    "max":         "#00a67e",
    "jax":         "#7b1fa2",
}
DEFAULT_COLOR = "#888888"


def load_summaries(results_dir):
    """Return [(model, [{framework, inference_ms, gpu_name}, ...]), ...]
    sorted by model name. Only 'ok' outcomes with inference_ms>0 are kept.
    """
    data = []
    pattern = os.path.join(results_dir, "*_summary.json")
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        model = base[: -len("_summary.json")]
        try:
            with open(path) as f:
                outcomes = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"warning: skipping {path}: {e}", file=sys.stderr)
            continue
        rows = []
        for o in outcomes:
            if o.get("status") != "ok":
                continue
            validation = o.get("validation", {})
            if (
                validation.get("comparison_performed")
                and validation.get("forward_valid") is not True
            ):
                continue
            ms = o.get("timings", {}).get("inference_ms", 0.0)
            if ms <= 0:
                continue
            rows.append({
                "framework": o["framework"],
                "inference_ms": float(ms),
                "gpu_name": o.get("gpu_name", ""),
            })
        if rows:
            data.append((model, rows))
    return data


def pick_platform(data, override):
    if override:
        return override
    for _model, rows in data:
        for r in rows:
            name = r.get("gpu_name")
            if name and name.lower() != "cpu":
                return name
    # All-CPU runs (or empty). Fall back to first gpu_name or "unknown".
    for _model, rows in data:
        for r in rows:
            if r.get("gpu_name"):
                return r["gpu_name"]
    return "unknown"


def filename_from_platform(platform):
    # Strip forbidden path characters but keep spaces for readability.
    return re.sub(r"[\\/:*?\"<>|]", "_", platform).strip() + ".svg"


def render_svg(platform, data):
    """Grouped-bars SVG, styled after amd-inference-light.svg:
    - Warm neutral palette (#111/#6b635a/#e5e0d4/#b4b2a9), white background.
    - Title + subtitle at top; horizontal legend below them.
    - Uniform framework slots per model group, so bars line up across groups.
      Missing (model, framework) pairs show a dashed "—" placeholder.
    - Log Y-axis with decade + half-decade gridlines.
    - Rounded bars with value labels above them.
    - Accessibility: role, aria-label, <title>, <desc>.
    """
    data = [(m, rows) for m, rows in data if rows]
    if not data:
        return None

    # Stable framework order (first-seen across models) — drives slot order
    # AND legend order.
    fw_order = []
    seen = set()
    for _m, rows in data:
        for r in rows:
            if r["framework"] not in seen:
                seen.add(r["framework"])
                fw_order.append(r["framework"])

    fw_index = {fw: i for i, fw in enumerate(fw_order)}
    by_model = {m: {r["framework"]: r for r in rows} for m, rows in data}

    # Layout — mirrors amd-inference-light.svg proportions.
    LEFT_PAD = 32
    RIGHT_PAD = 32
    TITLE_Y = 32
    SUBTITLE_Y = 54
    LEGEND_Y = 74
    LEGEND_SW = 10
    LEGEND_GAP_X = 20            # gap between legend items
    Y_AXIS_W = 52
    CHART_TOP = 110
    CHART_H = 230
    CHART_BOTTOM = CHART_TOP + CHART_H
    X_LABEL_Y = CHART_BOTTOM + 22
    BOTTOM_PAD = 28
    BAR_W = 20
    BAR_GAP = 2                  # intra-group — squished
    GROUP_GAP = 28

    chart_left = LEFT_PAD + Y_AXIS_W

    # Each model group's width varies with how many frameworks produced a
    # result for that model — empty (framework, model) slots are dropped
    # entirely so no dashed placeholders remain.
    slot_w = BAR_W + BAR_GAP
    group_widths = [len(rows) * slot_w - BAR_GAP for _m, rows in data]
    chart_area_w = sum(group_widths) + GROUP_GAP * (len(data) - 1)
    chart_right = chart_left + chart_area_w

    total_w = chart_right + RIGHT_PAD
    total_h = X_LABEL_Y + BOTTOM_PAD

    # Log bounds snapped to decades.
    all_vals = [r["inference_ms"] for _m, rows in data for r in rows]
    lo, hi = min(all_vals), max(all_vals)
    y_min = 10 ** math.floor(math.log10(lo))
    y_max = 10 ** math.ceil(math.log10(hi))
    if y_max == y_min:
        y_max = y_min * 10
    log_min = math.log10(y_min)
    log_max = math.log10(y_max)

    def y_for(ms):
        frac = (math.log10(ms) - log_min) / (log_max - log_min)
        return CHART_BOTTOM - frac * CHART_H

    aria = (
        f"Inference time on {platform}, log scale, milliseconds per forward pass, "
        f"grouped by model with one bar per framework."
    )

    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" '
        f'role="img" aria-label="{escape(aria)}" '
        f"font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif\">"
    )
    out.append(
        f"<title>Inference time on {escape(platform)} by framework and workload</title>"
    )
    out.append(
        "<desc>Grouped bar chart, log scale, milliseconds per forward pass. "
        "Dashed placeholders mark framework/model combinations that did not "
        "produce a successful result.</desc>"
    )
    out.append(f'<rect width="{total_w}" height="{total_h}" fill="#ffffff"/>')

    # Title + subtitle.
    out.append(
        f'<text x="{LEFT_PAD}" y="{TITLE_Y}" font-size="16" font-weight="500" '
        f'fill="#111">Inference time, {escape(platform)} — lower is better</text>'
    )
    out.append(
        f'<text x="{LEFT_PAD}" y="{SUBTITLE_Y}" font-size="13" fill="#6b635a">'
        f"Milliseconds per forward pass, log scale.</text>"
    )

    # Legend — horizontal row just above the chart.
    out.append(
        f'<g transform="translate({LEFT_PAD}, {LEGEND_Y})" font-size="12" '
        f'fill="#6b635a">'
    )
    lx = 0
    for fw in fw_order:
        color = FRAMEWORK_COLORS.get(fw, DEFAULT_COLOR)
        out.append(
            f'<rect x="{lx}" y="0" width="{LEGEND_SW}" height="{LEGEND_SW}" '
            f'rx="2" fill="{color}"/>'
        )
        out.append(f'<text x="{lx + LEGEND_SW + 6}" y="9">{escape(fw)}</text>')
        # Rough width budget: swatch + 6px + label width (~6.5 per char) + item gap.
        lx += LEGEND_SW + 6 + int(len(fw) * 6.8) + LEGEND_GAP_X
    out.append("</g>")

    # Gridlines + y-axis labels at decades and half-decades.
    out.append('<g stroke="#e5e0d4" stroke-width="1">')
    tick_labels = []
    decade = y_min
    while decade <= y_max + 1e-9:
        yp = y_for(decade)
        out.append(
            f'<line x1="{chart_left}" y1="{yp:.1f}" x2="{chart_right}" '
            f'y2="{yp:.1f}"/>'
        )
        tick_labels.append((yp, f"{decade:g} ms"))
        half = decade * (10 ** 0.5)
        if half < y_max - 1e-9:
            yh = y_for(half)
            out.append(
                f'<line x1="{chart_left}" y1="{yh:.1f}" x2="{chart_right}" '
                f'y2="{yh:.1f}" opacity="0.55"/>'
            )
            tick_labels.append((yh, f"{round(half):g}"))
        decade *= 10
    out.append("</g>")

    out.append('<g font-size="11" fill="#6b635a" text-anchor="end">')
    for y, label in tick_labels:
        out.append(f'<text x="{chart_left - 8}" y="{y + 4:.1f}">{label}</text>')
    out.append("</g>")

    # Y-axis line.
    out.append(
        f'<line x1="{chart_left}" y1="{CHART_TOP}" x2="{chart_left}" '
        f'y2="{CHART_BOTTOM}" stroke="#b4b2a9" stroke-width="1"/>'
    )

    # Bars, value labels, model labels. Only frameworks that produced a
    # successful result for this model get a column — empty slots dropped.
    group_start = chart_left
    for (model, rows), group_w in zip(data, group_widths):
        # Keep legend ordering consistent across groups.
        rows = sorted(rows, key=lambda r: fw_index[r["framework"]])
        x = group_start
        for r in rows:
            fw = r["framework"]
            ms = r["inference_ms"]
            color = FRAMEWORK_COLORS.get(fw, DEFAULT_COLOR)
            top = y_for(ms)
            h = CHART_BOTTOM - top
            cx = x + BAR_W / 2
            out.append(
                f'<rect x="{x}" y="{top:.1f}" width="{BAR_W}" '
                f'height="{h:.1f}" rx="2" fill="{color}">'
                f"<title>{escape(fw)} — {ms:.1f} ms</title></rect>"
            )
            value = f"{ms:.1f}" if ms < 10 else f"{ms:.0f}"
            out.append(
                f'<text x="{cx:.1f}" y="{top - 4:.1f}" font-size="10" '
                f'fill="#111" text-anchor="middle" font-weight="500">{value}</text>'
            )
            x += slot_w
        group_mid = group_start + group_w / 2
        out.append(
            f'<text x="{group_mid:.1f}" y="{X_LABEL_Y}" text-anchor="middle" '
            f'font-size="12" font-weight="500" fill="#111">{escape(model)}</text>'
        )
        group_start += group_w + GROUP_GAP

    out.append("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results",
                    help="Directory containing <model>_summary.json files")
    ap.add_argument("--platform", default="",
                    help="Override platform name (defaults to gpu_name from results)")
    ap.add_argument("--output", default="",
                    help="Override output SVG path (defaults to <results-dir>/<platform>.svg)")
    args = ap.parse_args()

    data = load_summaries(args.results_dir)
    if not data:
        print(f"no summary JSON files with successful results in {args.results_dir}",
              file=sys.stderr)
        return 1

    platform = pick_platform(data, args.platform)
    output = args.output or os.path.join(args.results_dir, filename_from_platform(platform))
    svg = render_svg(platform, data)
    if svg is None:
        print("no non-empty model groups to render", file=sys.stderr)
        return 1
    with open(output, "w") as f:
        f.write(svg)
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
