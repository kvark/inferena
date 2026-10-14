"""Validate accelerated arithmetic evidence without importing PyTorch."""

import math

from training_gate import CURVE_POLICY, assess_trajectories, workload_for
from quality import check_language_model_report


def check_forward_quality(forward, *, language_model=False, shape=None):
    if forward["rtol"] != 0.02 or forward["atol"] != 1e-6 or forward["elements"] < 1:
        raise ValueError("wrong accelerated forward gate")
    for metric in ("rms", "max_abs"):
        error, bound = forward[f"{metric}_error"], forward[f"{metric}_bound"]
        if (not math.isfinite(error) or not math.isfinite(bound) or error < 0 or bound < 0
                or not language_model and error > bound):
            raise ValueError("failed accelerated forward gate claims success")
    if language_model:
        report = check_language_model_report(forward["language_model"])
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
    if language_model:
        check_forward_quality(gate["latency_forward"], language_model=True, shape=[1, 1, shape[-1]])
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
                    or any(curve[key] != value for key, value in expected.items())
                    or any(curve[key] != value for key, value in expected["comparisons"]["training"].items()
                           if key != "status")):
                raise ValueError("training gate disagrees with recorded trajectories")
        if curve["status"] == "failed" and (record["timings"].get("training_ms") is not None
                or record["timing_samples_ms"].get("training") is not None
                or record["status"] != "partial" or execution.get("training_precision_failure") != curve):
            raise ValueError("failed accelerated training supplies timing or lacks its failure receipt")
        if gate["status"] != ("pass" if curve["status"] == "pass" else "partial"):
            raise ValueError("inconsistent precision gate status")
    elif gate["training"] is not None or gate["status"] != "pass":
        raise ValueError("unrequested accelerated training gate")
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
    return {"inference": "pass", "latency": "pass", "training": gate["training"]["status"] if training else None}
