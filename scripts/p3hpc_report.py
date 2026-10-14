"""Summarize complete replicate groups, including Android correctness-only references."""

import argparse
import json
from pathlib import Path
import statistics


def campaign_rows(directory):
    manifest = json.loads((directory / "campaign.json").read_text())
    android_protocols = {"p3hpc-android-campaign-v1": "p3hpc-paired-campaign-v18",
                         "p3hpc-android-campaign-v2": "p3hpc-paired-campaign-v21",
                         "p3hpc-android-campaign-v3": "p3hpc-paired-campaign-v22",
                         "p3hpc-android-campaign-v4": "p3hpc-paired-campaign-v23",
                         "p3hpc-android-campaign-v5": "p3hpc-paired-campaign-v24"}
    android = manifest["protocol"] in android_protocols
    paired_protocol = android_protocols.get(manifest["protocol"], manifest["protocol"])
    if paired_protocol not in {f"p3hpc-paired-campaign-v{v}" for v in (18, 19, 20, 21, 22, 23, 24)}:
        raise ValueError("unsupported campaign protocol")
    if android and manifest.get("paired_protocol", paired_protocol) != paired_protocol:
        raise ValueError("Android and paired campaign protocols disagree")
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
                        validation = manifest.get("replicated_gradient_validation", {})
                        groups = [g for g in validation.get("groups", [])
                                  if (g["model"], g["precision"], g["mode"], g["graph_replay"])
                                  == (model, precision, mode, graphs)]
                        if "groups" in validation or paired_protocol in (
                                "p3hpc-paired-campaign-v21", "p3hpc-paired-campaign-v22", "p3hpc-paired-campaign-v23", "p3hpc-paired-campaign-v24"):
                            complete &= len(groups) == 1 and groups[0]["status"] == "pass"
                        else:
                            # Older aggregate-only receipts can prove all groups
                            # passed, but cannot rescue a group from a global fail.
                            complete &= validation.get("status") == "pass"
                    row = dict(platform="Quest 3S" if android else args["gpu"], model=model, precision=precision,
                               phase=phase, condition="native / CPU f64 oracle" if android else f"{mode}/graph{int(graphs)}",
                               coverage=f"{len(valid)}/{requested}", status="valid" if complete else "incomplete-or-failed",
                               native_ms=None, reference_ms=None, speedup=None,
                               source=manifest["source"], protocol=paired_protocol,
                               campaign_protocol=manifest["protocol"], campaign=str(directory))
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


def cohort_rows(campaigns):
    rows = [row for campaign in campaigns for row in campaign_rows(campaign)]
    if len({(r["source"], r["protocol"]) for r in rows}) > 1:
        raise ValueError("cohort campaigns must have one common Inferena revision and paired protocol")
    return rows


def render(rows):
    lines = ["| Platform | Model | Precision | Phase | Reference | Valid processes | Meganeura ms | PyTorch ms | Speedup |",
             "|---|---|---|---|---|---|---|---|---|"]
    number = lambda value: "—" if value is None else f"{value:.3f}"
    for r in rows:
        lines.append("| " + " | ".join([r["platform"], r["model"], r["precision"], r["phase"], r["condition"],
            r["coverage"], number(r["native_ms"]), number(r["reference_ms"]), number(r["speedup"])]) + " |")
    lines.append("\nTimes are medians of process medians and require every declared replicate to pass; "
                 "training also requires its gradient group gate. "
                 "Qualification-only campaigns have no publication timings. Host CPU oracles have no speedup ratio.\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", type=Path, nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        rows = cohort_rows(args.campaigns)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(rows, indent=2) if args.json else render(rows))


if __name__ == "__main__":
    main()
