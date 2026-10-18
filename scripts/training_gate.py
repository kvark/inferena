"""Frozen accelerated training workloads and checks; no framework dependency."""

import hashlib
import json
import math
from pathlib import Path
import statistics

FIXTURE = Path(__file__).resolve().parents[1] / "data/smollm2-real-text-v1.json"
FIXTURE_SHA256 = "cf9d122497a1075119d2b3dca3c9173d69edf4e51c9ae6b5eb8236f8440f29e9"
REAL_TEXT = "smollm2-real-text-v2"
SMOLLM2_HIDDEN_SIZES = {"SmolLM2-135M": 576, "SmolLM2-360M": 960, "SmolLM2-1.7B": 2048}
# Verified at each revision in models/smollm2-revisions.json: the tokenizer bytes
# are identical, so the original frozen token batches apply to all three models.
TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
REPEATED = "repeated-synthetic-v1"
CURVE_POLICY = "loss-curve-v2"
CURVE_RTOL, CURVE_ATOL = 0.02, 1e-6


def receipt_matches(actual, expected):
    """Compare recomputed diagnostics across JSON/libm roundoff, not gate outcomes."""
    if isinstance(expected, float):
        return (isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(receipt_matches(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(receipt_matches(a, e) for a, e in zip(actual, expected)))
    return actual == expected


def load_fixture():
    # Freeze the token batches and their provenance. Its embedded gate describes
    # the original study; current acceptance is versioned here and in receipts.
    data = FIXTURE.read_bytes()
    if hashlib.sha256(data).hexdigest() != FIXTURE_SHA256:
        raise ValueError("real-text training fixture differs from the frozen workload")
    return json.loads(data)


def workload_for(model, *, repeated=False):
    if model in SMOLLM2_HIDDEN_SIZES and not repeated:
        return {"policy": REAL_TEXT, "steps": 256, "fixture_sha256": FIXTURE_SHA256,
                "model": model, "parameter_elements": SMOLLM2_HIDDEN_SIZES[model],
                "tokenizer_sha256": TOKENIZER_SHA256,
                "learning_rate": 0.0010000000474974513, "checkpoint_every": 64,
                "scope": "varying real-text batches; fixed training probe and held-out text"}
    return {"policy": REPEATED, "steps": 32, "learning_rate": 0.001,
            "scope": "repeated synthetic benchmark inputs; numerical diagnostic, not real-data learning"}


def curve_metrics(actual, reference):
    if not actual or len(actual) != len(reference):
        raise ValueError("loss curves must have the same nonzero length")
    if not all(math.isfinite(value) and value >= 0 for value in [*actual, *reference]):
        raise ValueError("non-finite or negative loss in convergence check")
    errors = [abs(a - r) for a, r in zip(actual, reference, strict=True)]
    relative_l2 = math.sqrt(sum(error * error for error in errors)) / max(
        math.sqrt(sum(value * value for value in reference)), CURVE_ATOL)
    mismatches = sum(error > CURVE_ATOL + CURVE_RTOL * abs(r)
                     for error, r in zip(errors, reference, strict=True))
    return {"status": "pass" if not mismatches else "failed",
            "policy": CURVE_POLICY, "steps": len(actual), "rtol": CURVE_RTOL,
            "atol": CURVE_ATOL, "relative_l2_role": "diagnostic",
            "relative_l2": relative_l2, "max_abs_error": max(errors), "mismatched_steps": mismatches}


def check_trajectory(trajectory, workload):
    steps = workload["steps"]
    if (trajectory["completed_updates"] != steps or trajectory["skipped_steps"] != 0
            or len(trajectory["losses"]) != steps or trajectory["optimizer"] != "SGD"
            or trajectory["learning_rate"] != workload["learning_rate"]):
        raise ValueError("incomplete or mismatched SGD trajectory")
    curve_metrics(trajectory["losses"], trajectory["losses"])
    attempts = trajectory["scale_attempts"]
    if len(attempts) != steps:
        raise ValueError("missing optimizer update attempts")
    for loss, batch in zip(trajectory["losses"], attempts, strict=True):
        if (not 1 <= len(batch) <= 8 or batch[-1]["updated"] is not True
                or any(attempt["updated"] is not False for attempt in batch[:-1])
                or batch[-1]["loss"] != loss
                or any(not math.isfinite(attempt["loss"]) or attempt["loss"] < 0
                       or not math.isfinite(attempt["scale"]) or attempt["scale"] <= 0 for attempt in batch)):
            raise ValueError("missing, skipped or invalid optimizer update")
    if trajectory["overflow_attempts"] != sum(len(batch) - 1 for batch in attempts):
        raise ValueError("wrong overflow attempt count")
    if workload["policy"] != REAL_TEXT:
        return None
    checkpoints = trajectory["checkpoints"]
    if [row["step"] for row in checkpoints] != [0, 64, 128, 192, 256]:
        raise ValueError("missing real-text evaluation checkpoints")
    for row in checkpoints:
        for key in ("probe_losses", "heldout_losses"):
            if len(row[key]) != 4 or any(not math.isfinite(x) or x < 0 for x in row[key]):
                raise ValueError("invalid real-text evaluation losses")
    parameter = trajectory["parameter_check"]
    if (parameter["name"] != "model.norm.weight"
            or len(parameter["initial"]) != workload["parameter_elements"]
            or len(parameter["final"]) != workload["parameter_elements"]
            or any(not math.isfinite(x) for x in [*parameter["initial"], *parameter["final"]])
            or parameter["changed"] != (parameter["initial"] != parameter["final"])):
        raise ValueError("invalid parameter update evidence")
    ratios = {key: statistics.mean(checkpoints[-1][key]) / max(statistics.mean(checkpoints[0][key]), CURVE_ATOL)
              for key in ("probe_losses", "heldout_losses")}
    passed = parameter["changed"] and ratios["probe_losses"] <= 0.95 and ratios["heldout_losses"] <= 1.02
    return {"status": "pass" if passed else "failed", "probe_loss_ratio": ratios["probe_losses"],
            "heldout_loss_ratio": ratios["heldout_losses"], "probe_ratio_limit": 0.95,
            "heldout_ratio_limit": 1.02, "parameter_changed": parameter["changed"]}


def assess_trajectories(actual, reference, workload):
    learning = {"candidate": check_trajectory(actual, workload), "reference": check_trajectory(reference, workload)}
    comparisons = {"training": curve_metrics(actual["losses"], reference["losses"])}
    if workload["policy"] == REAL_TEXT:
        if actual["parameter_check"]["initial"] != reference["parameter_check"]["initial"]:
            raise ValueError("candidate and reference started from different parameters")
        for key in ("probe_losses", "heldout_losses"):
            comparisons[key] = curve_metrics([x for row in actual["checkpoints"] for x in row[key]],
                                             [x for row in reference["checkpoints"] for x in row[key]])
    passed = all(row["status"] == "pass" for row in comparisons.values()) and all(
        row is None or row["status"] == "pass" for row in learning.values())
    return {"status": "pass" if passed else "failed", "comparisons": comparisons, "learning": learning}
