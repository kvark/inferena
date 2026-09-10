#!/usr/bin/env python3
"""Report actual plan-buffer bindings and CPU preparation from a memory study."""

import json
from pathlib import Path
import sys

from study_results import compare


def report(root):
    manifest = json.loads((root / "study.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("incomplete study")
    rows = []
    for run in manifest["runs"]:
        path = Path(run["destination"])
        result = json.loads((path / "runner.json").read_text())
        baseline = json.loads((path.parent / "untuned/runner.json").read_text())
        compare(baseline, result)
        trace = [json.loads(line) for line in (path / "compilation.jsonl").read_text().splitlines()]
        builds = sorted((row for row in trace if row["stage"] == "inferena_build_session"),
                        key=lambda row: row["start_ns"])
        phases = ("inference", "latency")
        if not run["model"].startswith("SmolLM2-") or len(builds) != len(phases):
            raise ValueError("this report expects the two sequential SmolLM2 inference plans")
        sessions = {}
        for phase, build in zip(phases, builds):
            bindings = {}
            start, end = build["start_ns"], build["start_ns"] + build["duration_ns"]
            for row in trace:
                if row["stage"] != "blade_allocation_binding" or not start <= row["start_ns"] < end:
                    continue
                fields = row["fields"]
                if not json.loads(fields["name"]).startswith("buf_"):
                    continue
                key = (fields["requested_memory"], int(fields["memory_type"]), fields["properties"])
                group = bindings.setdefault(key, {"count": 0, "requested_bytes": 0, "block_bytes": 0})
                group["count"] += 1
                for size in ("requested_bytes", "block_bytes"):
                    group[size] += int(fields[size])
            if sum(group["count"] for group in bindings.values()) != result["memory"]["phases"][phase]["allocation_count"]:
                raise ValueError("plan allocation count and traced bindings disagree")
            sessions[phase] = [{"requested_memory": key[0], "memory_type": key[1], "properties": key[2], **value}
                               for key, value in sorted(bindings.items())]
        setup = {}
        for stage in ("checkpoint_file_load", "inferena_build_session", "parameter_preparation"):
            setup[stage + "_s"] = [row["duration_ns"] / 1e9 for row in trace if row["stage"] == stage]
        rows.append({"model": run["model"], "variant": run["variant"], "destination": str(path),
                     "plan_bindings": sessions, "setup": setup,
                     "device": result["memory"]["device"], "timings": result["timings"],
                     "recorded_outputs_exact": baseline["outputs"] == result["outputs"]})
    print(json.dumps({"source": manifest["source"], "runs": rows,
                      "notes": "Bindings are actual selected memory types; block sizes include suballocation rounding, "
                               "not unused pool capacity. Sequential plans must not be summed as a resident peak. "
                               "Binding-span durations are not allocation timings. CPU parameter preparation includes "
                               "conversion, transpose, copies and waits. Traced timings are diagnostic only."}, indent=2))


if __name__ == "__main__":
    report(Path(sys.argv[1]))
