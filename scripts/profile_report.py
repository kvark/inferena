#!/usr/bin/env python3
"""Summarize or compare Inferena/Meganeura structured GPU profiles."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def load_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    profile = artifact.get("profile", artifact)
    if "dispatches" not in profile or "measurement" not in profile:
        raise ValueError(f"{path} is not a structured Meganeura profile")
    return artifact, profile


def title(artifact: dict[str, Any], profile: dict[str, Any], path: Path) -> str:
    model = artifact.get("model", path.stem)
    mode = artifact.get("mode", "session")
    precision = artifact.get("precision", "unspecified precision")
    revision = artifact.get("framework_rev", "unknown revision")
    device = profile.get("device", {}).get("device_name", "unknown device")
    return f"{model} / {mode} / {precision} / {revision} / {device}"


def summarize(path: Path, artifact: dict[str, Any], profile: dict[str, Any], top: int) -> None:
    print(f"\n{title(artifact, profile, path)}")
    print("=" * len(title(artifact, profile, path)))

    measurement = profile["measurement"]
    plan = profile.get("plan", {})
    baseline = measurement.get("unprofiled_median_ms")
    ratio = measurement.get("instrumentation_wall_ratio")
    baseline_text = f"{baseline:.3f} ms" if baseline is not None else "not supplied"
    ratio_text = f"{ratio:.2f}x" if ratio is not None else "n/a"
    print(
        f"normal={baseline_text}; profiled={measurement['profiled_wall_median_ms']:.3f} ms; "
        f"GPU pass intervals={measurement['gpu_total_median_ms']:.3f} ms; instrumentation={ratio_text}"
    )
    print(f"Timing contract: {profile.get('timing_contract', 'unspecified in this artifact')}")
    print("Pass intervals are not kernel-only time; wall minus their sum is not CPU overhead.")
    print(
        f"dispatches={plan.get('dispatch_count', '?')} "
        f"(forward={plan.get('forward_dispatch_count', '?')}, "
        f"backward={plan.get('backward_dispatch_count', '?')}); "
        f"barrier groups={plan.get('barrier_group_count', '?')}"
    )

    print("\nDominant families")
    families = sorted(
        profile.get("families", []),
        key=lambda family: family["share_of_dispatch_median_sum_pct"],
        reverse=True,
    )
    for family in families:
        print(
            f"  {family['share_of_dispatch_median_sum_pct']:6.2f}%  "
            f"{family['dispatch_median_sum_ms']:9.3f} ms  "
            f"{family['dispatch_count']:4d}x  "
            f"{family['phase']}/{family['family']}"
        )

    print(f"\nTop {min(top, len(profile['dispatches']))} dispatches")
    dispatches = sorted(profile["dispatches"], key=lambda dispatch: dispatch["median_ms"], reverse=True)
    for dispatch in dispatches[:top]:
        print(
            f"  {dispatch['median_ms']:9.3f} ms  "
            f"{dispatch['share_of_dispatch_median_sum_pct']:6.2f}%  "
            f"#{dispatch['index']:04d} {dispatch['phase']}/{dispatch['family']}  "
            f"{dispatch['pipeline']}  {dispatch['label']}"
        )


def aggregate_dispatches(profile: dict[str, Any]) -> dict[tuple[str, str, str], tuple[int, float]]:
    aggregate: dict[tuple[str, str, str], list[float]] = collections.defaultdict(
        lambda: [0.0, 0.0]
    )
    for dispatch in profile["dispatches"]:
        key = (dispatch["phase"], dispatch["pipeline"], dispatch["label"])
        aggregate[key][0] += 1
        aggregate[key][1] += dispatch["median_ms"]
    return {key: (int(value[0]), value[1]) for key, value in aggregate.items()}


def compare(
    left_path: Path,
    left_artifact: dict[str, Any],
    left: dict[str, Any],
    right_path: Path,
    right_artifact: dict[str, Any],
    right: dict[str, Any],
    top: int,
) -> None:
    left_dispatches = aggregate_dispatches(left)
    right_dispatches = aggregate_dispatches(right)
    rows = []
    for key in left_dispatches.keys() | right_dispatches.keys():
        left_count, left_ms = left_dispatches.get(key, (0, 0.0))
        right_count, right_ms = right_dispatches.get(key, (0, 0.0))
        rows.append((right_ms - left_ms, key, left_count, left_ms, right_count, right_ms))
    rows.sort(reverse=True)

    print(f"\nLargest dispatch-group regressions: {left_path.name} -> {right_path.name}")
    print(f"  left:  {title(left_artifact, left, left_path)}")
    print(f"  right: {title(right_artifact, right, right_path)}")
    for delta, key, left_count, left_ms, right_count, right_ms in rows[:top]:
        ratio = right_ms / left_ms if left_ms > 0.0 else float("inf")
        phase, pipeline, label = key
        print(
            f"  {delta:+9.3f} ms  {ratio:7.2f}x  "
            f"{left_count}x/{left_ms:.3f} -> {right_count}x/{right_ms:.3f}  "
            f"{phase}  {pipeline}  {label}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", type=Path, nargs="+", help="one profile, or two to compare")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    if len(args.profiles) > 2:
        parser.error("provide one profile to summarize or two profiles to compare")
    if args.top < 1:
        parser.error("--top must be positive")

    loaded = [(path, *load_artifact(path)) for path in args.profiles]
    for path, artifact, profile in loaded:
        summarize(path, artifact, profile, args.top)
    if len(loaded) == 2:
        compare(*loaded[0], *loaded[1], args.top)


if __name__ == "__main__":
    main()
