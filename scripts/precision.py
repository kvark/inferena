"""Validate accelerated arithmetic evidence without importing PyTorch."""

import math

from training_gate import CURVE_POLICY, assess_trajectories, receipt_matches, workload_for
from quality import check_language_model_report


def check_arithmetic_policy(record, expected):
    """The measured recipe must equal the declaration made before collection."""
    precision = record["precision"]
    for key in ("autocast_requested", "autocast", "autocast_selection", "model_guard", "f32_linear_modules",
                "phase_autocast", "model_guard_phases"):
        if key not in precision or precision[key] != expected[key]:
            raise ValueError(f"executed arithmetic differs from the declared policy: {key}")


def check_forward_quality(forward, *, language_model=False, shape=None, require_pass=True):
    if forward["rtol"] != 0.02 or forward["atol"] != 1e-6 or forward["elements"] < 1:
        raise ValueError("wrong accelerated forward gate")
    for metric in ("rms", "max_abs"):
        error, bound = forward[f"{metric}_error"], forward[f"{metric}_bound"]
        if (not math.isfinite(error) or not math.isfinite(bound) or error < 0 or bound < 0
                or not language_model and error > bound):
            raise ValueError("failed accelerated forward gate claims success")
    if language_model:
        report = check_language_model_report(forward["language_model"], require_pass=require_pass)
        if (forward.get("raw_logits_role") != "diagnostic"
                or forward["elements"] != report["tokens"] * report["vocabulary"]
                or shape is not None and (report["tokens"] != math.prod(shape[:-1])
                                          or report["vocabulary"] != shape[-1])):
            raise ValueError("language-model quality covers the wrong output")
    elif "language_model" in forward:
        raise ValueError("non-language output requires full-tensor fidelity")


def check_precision_qualification(record, *, diagnostic=False):
    precision, execution = record["precision"], record["execution"]
    dtype = precision.get("autocast")
    if precision["comparison_class"] != "accelerated" or dtype not in (None, "bf16", "fp16"):
        raise ValueError("wrong accelerated arithmetic policy")
    if dtype is not None and (precision.get("loss") != "f32"
            or precision.get("torch_float32_matmul_precision") != "highest"
            or precision.get("cuda_matmul_allow_tf32", False) or precision.get("cudnn_allow_tf32", False)
            or execution["gradient_scaler"]["enabled"] != (dtype == "fp16")):
        raise ValueError("wrong AMP arithmetic or scaling policy")
    if dtype is None:
        expected = "highest" if record["backend"].startswith("CPU") else "high"
        if (precision.get("torch_float32_matmul_precision") != expected
                or record["backend"].startswith("CUDA") and (
                    precision.get("cuda_matmul_allow_tf32") is not True
                    or precision.get("cudnn_allow_tf32") is not True)):
            raise ValueError("wrong accelerated f32/TF32 arithmetic policy")
    gate = execution["precision_qualification"]
    if gate["status"] not in ("pass", "partial") or gate["reference"] != "eager strict f32":
        raise ValueError("missing accelerated precision qualification")
    language_model = record["model"].startswith("SmolLM2-")
    shape = record["outputs"]["output_shape"]
    check_forward_quality(gate["forward"], language_model=language_model, shape=shape)
    latency_status = "pass"
    if language_model:
        latency = gate["latency_forward"]
        if latency.get("status") == "failed":
            latency_status = "failed"
            failure = latency["failure"]
            if (failure.get("kind") != "numerical" or not failure.get("message")
                    or record["timings"].get("latency_ms") is not None
                    or record["timing_samples_ms"].get("latency") is not None
                    or record["status"] != "partial" or execution.get("latency_precision_failure") != latency):
                raise ValueError("failed accelerated latency supplies timing or lacks its failure receipt")
            if "metrics" in failure:
                check_forward_quality(failure["metrics"], language_model=True, shape=[1, 1, shape[-1]], require_pass=False)
                if failure["metrics"]["language_model"]["status"] != "failed":
                    raise ValueError("latency failure disagrees with quality evidence")
        else:
            check_forward_quality(latency, language_model=True, shape=[1, 1, shape[-1]])
    training = record["protocol"]["training_requested"]
    request = record["protocol"].get("training_gate")
    if request not in ("auto", "repeated-synthetic"):
        raise ValueError("missing or unknown training gate request")
    repeated = request == "repeated-synthetic"
    if repeated and (not diagnostic or record["protocol"].get("diagnostic") is not True):
        raise ValueError("repeated-batch diagnostic cannot replace the collection training gate")
    if training:
        curve = gate["training"]
        workload = workload_for(record["model"], repeated=repeated)
        if curve["workload"] != workload:
            raise ValueError("wrong accelerated training workload")
        failed_trajectory = curve.get("failure")
        if failed_trajectory is not None:
            trajectory = failed_trajectory["trajectory"]
            if (curve["status"] != "failed" or curve["policy"] != CURVE_POLICY
                    or curve["requested_updates"] != workload["steps"]
                    or failed_trajectory["kind"] != "numerical"
                    or trajectory["arithmetic"] not in ("strict-f32", dtype or "accelerated-f32")
                    or not 0 <= trajectory["completed_updates"] <= workload["steps"]
                    or len(trajectory["losses"]) != trajectory["completed_updates"]
                    or any(not math.isfinite(x) or x < 0 for x in trajectory["losses"])):
                raise ValueError("unclassified accelerated trajectory failure")
        else:
            # Recompute all gates from the raw evidence. In particular a passing
            # training curve cannot mask a failed fixed probe or learning check.
            reference = curve["reference_trajectory"]
            expected = assess_trajectories(curve, reference, workload)
            if (curve["reference_losses"] != reference["losses"]
                    or any(not receipt_matches(curve[key], value) for key, value in expected.items())
                    or any(not receipt_matches(curve[key], value) for key, value in expected["comparisons"]["training"].items()
                           if key != "status")):
                raise ValueError("training gate disagrees with recorded trajectories")
        if curve["status"] == "failed" and (record["timings"].get("training_ms") is not None
                or record["timing_samples_ms"].get("training") is not None
                or record["status"] != "partial" or execution.get("training_precision_failure") != curve):
            raise ValueError("failed accelerated training supplies timing or lacks its failure receipt")
    elif gate["training"] is not None:
        raise ValueError("unrequested accelerated training gate")
    training_status = gate["training"]["status"] if training else None
    if gate["status"] != ("partial" if "failed" in (training_status, latency_status) else "pass"):
        raise ValueError("inconsistent precision gate status")
    oracle = record["correctness_reference"]
    outputs = oracle["outputs"]
    if (oracle["precision"] != "strict-f32" or oracle["training_requested"] != training
            or oracle.get("prepared_before_candidate_compilation") is not True
            or outputs["output_shape"] != record["outputs"]["output_shape"]
            or len(outputs["logits_sample"]) != len(record["outputs"]["logits_sample"])
            or not math.isfinite(outputs["loss"])
            or any(not math.isfinite(x) for x in outputs["logits_sample"])
            or training and (not outputs["gradient_norms"] or outputs["grad_norm"] is None
                or not math.isfinite(outputs["grad_norm"])
                or any(not math.isfinite(x) for x in outputs["gradient_norms"].values()))):
        raise ValueError("invalid strict-f32 oracle for the accelerated reference")
    return {"inference": "pass", "latency": latency_status, "training": training_status}
