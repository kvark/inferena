"""Summarize complete replicate groups, including Android correctness-only references."""

import argparse
import json
from pathlib import Path
import statistics


def campaign_rows(directory):
    manifest = json.loads((directory / "campaign.json").read_text())
    android = manifest["protocol"] == "p3hpc-android-campaign-v1"
    if not android and manifest["protocol"] != "p3hpc-paired-campaign-v18":
        raise ValueError("unsupported campaign protocol")
    args = manifest["args"]
    requested = args["replicates"] if args["collect"] else 1
    phases = ("inference", "latency") if args["inference_only"] else ("inference", "latency", "training")
    conditions = [(None, None)] if android else [(c["mode"], c["graph_replay"])
                   for c in manifest["reference_conditions"]["selected"]]
    rows = []
    for model in args["models"]:
        for precision in args["precisions"]:
            for mode, graphs in conditions:
                runs = [r for r in manifest["runs"] if r["model"] == model and r["precision"] == precision
                        and (android or (r["mode"], r["graphs"]) == (mode, graphs))]
                for phase in phases:
                    valid = [r for r in runs if r.get("phases", {}).get(phase, {}).get("status") == "valid"]
                    complete = (len(runs) == len(valid) == requested
                                and {r["replicate"] for r in runs} == set(range(1, requested + 1))
                                and manifest["status"] in ("complete", "complete-with-failures"))
                    if not android and phase == "training" and args["collect"]:
                        complete &= manifest.get("replicated_gradient_validation", {}).get("status") == "pass"
                    row = dict(platform="Quest 3S" if android else args["gpu"], model=model, precision=precision,
                               phase=phase, condition="native / CPU f64 oracle" if android else f"{mode}/graph{int(graphs)}",
                               coverage=f"{len(valid)}/{requested}", status="valid" if complete else "incomplete-or-failed",
                               native_ms=None, reference_ms=None, speedup=None,
                               source=manifest["source"], campaign=str(directory))
                    # Qualification has no publication samples; CPU correctness
                    # references can never contribute a timing or speedup ratio.
                    if complete and args["collect"]:
                        native = [json.loads((directory / r["path"] / f"{model}_meganeura.json").read_text()) for r in runs]
                        row["native_ms"] = statistics.median(statistics.median(n["timing_samples_ms"][phase]) for n in native)
                        if not android and args["backend"] != "cpu":
                            reference = [json.loads((directory / r["path"] / f"{model}_pytorch.json").read_text()) for r in runs]
                            row["reference_ms"] = statistics.median(statistics.median(n["timing_samples_ms"][phase]) for n in reference)
                            row["speedup"] = row["reference_ms"] / row["native_ms"]
                    rows.append(row)
    return rows


def render(rows):
    lines = ["| Platform | Model | Precision | Phase | Reference | Valid processes | Meganeura ms | PyTorch ms | Speedup |",
             "|---|---|---|---|---|---|---|---|---|"]
    number = lambda value: "—" if value is None else f"{value:.3f}"
    for r in rows:
        lines.append("| " + " | ".join([r["platform"], r["model"], r["precision"], r["phase"], r["condition"],
            r["coverage"], number(r["native_ms"]), number(r["reference_ms"]), number(r["speedup"])]) + " |")
    lines.append("\nTimes are medians of process medians and require every declared replicate to pass. "
                 "Qualification-only campaigns have no publication timings. Host CPU oracles have no speedup ratio.\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", type=Path, nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = [row for campaign in args.campaigns for row in campaign_rows(campaign)]
    if len({r["source"] for r in rows}) > 1:
        parser.error("cohort campaigns must have one common Inferena revision")
    print(json.dumps(rows, indent=2) if args.json else render(rows))


if __name__ == "__main__":
    main()
