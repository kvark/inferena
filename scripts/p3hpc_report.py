"""Summarize complete replicate groups, including Android correctness-only references."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def archive_path(directory, *parts):
    """Read relative receipt paths from either Windows or POSIX collectors."""
    return directory.joinpath(*(part.replace("\\", "/") for part in parts))


def read_record(directory, run, engine):
    path = archive_path(directory, run["path"], f"{run['model']}_{engine}.json")
    return json.loads(path.read_text()) if path.exists() else {}


def native_passes(record, phase, replicated):
    validation = record.get("validation", {})
    if validation.get("forward_valid") is not True:
        return False
    if phase != "training" or validation.get("training_valid") is True:
        return True
    if replicated:
        from p3hpc import _replicated_candidate
        try:
            return _replicated_candidate(record)
        except (KeyError, TypeError, ValueError):
            pass
    return False


def phase_evidence(directory, run, phase, native, reference, *, replicated=False):
    """Separate engine qualification from the eligibility of their timing ratio.

    A completed native process alone is not an independent correctness proof.
    Reuse the collector's checked comparison, including its diagnostic oracle,
    but never substitute diagnostic PyTorch timings into the primary condition.
    """
    paired = run.get("phases", {}).get(phase, {}).get("status")
    if paired == "valid":
        return "valid", "valid", native
    report = reference.get("execution", {}).get("graph_replay", {}).get("phases", {}).get(phase, {})
    reference_status = report.get("status", paired if paired in ("failed", "not-attempted") else "unqualified")
    if reference_status in ("captured-and-validated", "validated-uncaptured"):
        reference_status = "valid"
        if reference.get("precision", {}).get("comparison_class") == "accelerated":
            from precision import check_precision_qualification
            try:
                quality = check_precision_qualification(reference)
                if quality.get(phase) != "pass":
                    reference_status = "failed"
            except (KeyError, TypeError, ValueError):
                reference_status = "unqualified"
    validation = native.get("validation", {})
    # For accelerated runs, cross-engine validation compares the two original
    # F32 snapshots, independently of the measured PyTorch candidate. A failed
    # strict candidate cannot supply this independent oracle.
    independent = (native.get("precision", {}).get("comparison_class") == "accelerated"
                   and (reference.get("correctness_reference") or {}).get("precision") == "strict-f32")
    if phase != "training" and reference.get("precision", {}).get("comparison_class") == "strict-f32":
        # The paired protocol anchors both native forward sessions to the
        # qualified F32 inference result. A later training/replay failure does
        # not remove that already-completed oracle.
        inference = reference.get("execution", {}).get("graph_replay", {}).get("phases", {}).get("inference", {})
        independent |= inference.get("status") in ("captured-and-validated", "validated-uncaptured")
    checked = (validation.get("comparison_performed") is True
               and validation.get("reference_framework") == "pytorch")
    valid = native_passes(native, phase, replicated)
    native_status = "valid" if checked and valid and (independent or reference_status == "valid") else "unqualified"
    if checked and (independent or reference_status == "valid") and not valid:
        native_status = "failed"
    receipt = run.get("native_oracle")
    if receipt:
        folder = archive_path(directory, run["path"])
        recorded = json.loads(archive_path(folder, receipt["path"]).read_text())
        if (recorded != {k: v for k, v in receipt.items() if k != "path"}
                or receipt["policy"] != "preserved-f32-oracle-v1"
                or receipt["timing_substituted"] is not False
                or hashlib.sha256((folder / f"{run['model']}_meganeura.json").read_bytes()).hexdigest()
                    != receipt["native_sha256"]
                or hashlib.sha256(archive_path(folder, receipt["oracle"]).read_bytes()).hexdigest() != receipt["oracle_sha256"]):
            raise ValueError("independent oracle receipt does not belong to these records")
        validation = receipt["validation"]
        native = dict(native, validation=validation)
        valid = (validation.get("comparison_performed") is True
                 and native_passes(native, phase, replicated))
        native_status = "valid" if valid else "failed"
    diagnostic = run.get("eager_diagnostic", {})
    if native_status != "valid" and diagnostic.get("phases", {}).get(phase, {}).get("status") == "valid":
        path = archive_path(directory, run["path"], diagnostic["path"], "comparison.json")
        if path.exists():
            records = json.loads(path.read_text())
            evidence = next((r for r in records if r.get("framework") == "meganeura"), {})
            # An oracle receipt must belong to these exact measured outputs and
            # this runner; matching a model name is insufficient.
            without_validation = lambda r: {k: v for k, v in r.items() if k != "validation"}
            if without_validation(evidence) == without_validation(native):
                return "valid", reference_status, evidence
    return native_status, reference_status, native


def measured_median(records, phase, expected_count=20):
    samples = [record.get("timing_samples_ms", {}).get(phase) for record in records]
    if not samples or any(not isinstance(values, list) or len(values) != expected_count
                          or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                                 for v in values) for values in samples):
        return None
    return statistics.median(statistics.median(values) for values in samples)


def campaign_rows(directory):
    manifest = json.loads((directory / "campaign.json").read_text())
    android_protocols = {"p3hpc-android-campaign-v1": "p3hpc-paired-campaign-v18",
                         "p3hpc-android-campaign-v2": "p3hpc-paired-campaign-v21",
                         "p3hpc-android-campaign-v3": "p3hpc-paired-campaign-v22",
                         "p3hpc-android-campaign-v4": "p3hpc-paired-campaign-v23",
                         "p3hpc-android-campaign-v5": "p3hpc-paired-campaign-v24",
                         "p3hpc-android-campaign-v6": "p3hpc-paired-campaign-v25",
                         "p3hpc-android-campaign-v7": "p3hpc-paired-campaign-v26",
                         "p3hpc-android-campaign-v8": "p3hpc-paired-campaign-v27",
                         "p3hpc-android-campaign-v9": "p3hpc-paired-campaign-v28",
                         "p3hpc-android-campaign-v10": "p3hpc-paired-campaign-v28",
                         "p3hpc-android-campaign-v11": "p3hpc-paired-campaign-v29",
                         "p3hpc-android-campaign-v12": "p3hpc-paired-campaign-v30",
                         "p3hpc-android-campaign-v13": "p3hpc-paired-campaign-v31"}
    android = manifest["protocol"] in android_protocols
    paired_protocol = android_protocols.get(manifest["protocol"], manifest["protocol"])
    if paired_protocol not in {f"p3hpc-paired-campaign-v{v}" for v in range(18, 32)}:
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
                                "p3hpc-paired-campaign-v21", "p3hpc-paired-campaign-v22", "p3hpc-paired-campaign-v23",
                                "p3hpc-paired-campaign-v24", "p3hpc-paired-campaign-v25", "p3hpc-paired-campaign-v26", "p3hpc-paired-campaign-v27", "p3hpc-paired-campaign-v28", "p3hpc-paired-campaign-v29", "p3hpc-paired-campaign-v30", "p3hpc-paired-campaign-v31"):
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
                    native = [read_record(directory, r, "meganeura") for r in runs]
                    reference = [read_record(directory, r, "pytorch") for r in runs]
                    evidence = ([phase_evidence(directory, r, phase, n, p, replicated=args["collect"])
                                 for r, n, p in zip(runs, native, reference)] if not android
                                else [("valid" if r in valid else "unqualified", "correctness-only", n)
                                      for r, n in zip(runs, native)])
                    intact = (len(runs) == requested
                              and {r["replicate"] for r in runs} == set(range(1, requested + 1))
                              and manifest["status"] in ("complete", "complete-with-failures"))
                    native_ok = intact and all(e[0] == "valid" for e in evidence)
                    reference_ok = intact and all(e[1] == "valid" for e in evidence)
                    if phase == "training" and args["collect"] and not android:
                        if len(valid) == requested:
                            native_ok &= complete
                        elif native_ok:
                            from p3hpc import validate_replicated_gradients
                            pairs = [{"meganeura": e[2]} for e in evidence]
                            try:
                                native_ok = validate_replicated_gradients(
                                    {(precision, model, mode, graphs): pairs}, requested)["status"] == "pass"
                            except (KeyError, TypeError, ValueError):
                                native_ok = False
                    row.update(native_coverage=f"{sum(e[0] == 'valid' for e in evidence)}/{requested}",
                               reference_coverage=f"{sum(e[1] == 'valid' for e in evidence)}/{requested}",
                               native_status="valid" if native_ok else "qualification-incomplete-or-failed",
                               reference_status="correctness-only" if android or args["backend"] == "cpu" else
                                   "valid" if reference_ok else "qualification-incomplete-or-failed",
                               native_measured_ms=None)
                    # Raw completed measurements remain visible, explicitly
                    # unqualified. They never contribute a speedup ratio.
                    if intact and args["collect"]:
                        row["native_measured_ms"] = measured_median(native, phase)
                        if native_ok:
                            row["native_ms"] = row["native_measured_ms"]
                        if reference_ok and not android and args["backend"] != "cpu":
                            row["reference_ms"] = measured_median(reference, phase)
                        if complete and row["native_ms"] is not None and row["reference_ms"] is not None:
                            row["speedup"] = row["reference_ms"] / row["native_ms"]
                    rows.append(row)
    return rows


def cohort_rows(campaigns):
    rows = [row for campaign in campaigns for row in campaign_rows(campaign)]
    if len({(r["source"], r["protocol"]) for r in rows}) > 1:
        raise ValueError("cohort campaigns must have one common Inferena revision and paired protocol")
    return rows


def render(rows):
    lines = ["| Platform | Model | Precision | Phase | Reference | Qualified pairs | Meganeura ms | Native qualification | PyTorch ms | Reference qualification | Speedup |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    number = lambda value: "—" if value is None else f"{value:.3f}"
    for r in rows:
        lines.append("| " + " | ".join([r["platform"], r["model"], r["precision"], r["phase"], r["condition"],
            r["coverage"], number(r["native_ms"] if r["native_ms"] is not None else r["native_measured_ms"]),
            r["native_status"], number(r["reference_ms"]), r["reference_status"], number(r["speedup"])]) + " |")
    lines.append("\nTimes are medians of process medians. Native measurements with incomplete/failed qualification are descriptive only; "
                 "qualified training also requires its gradient group gate. Speedups require every paired replicate to pass. "
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
