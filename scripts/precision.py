"""Validate the accelerated profile's AMP evidence without importing PyTorch."""

import math


def check_amp_qualification(record):
    dtype = record["precision"].get("autocast")
    if dtype not in ("bf16", "fp16"):
        raise ValueError("AMP requires an explicit bf16/fp16 dtype")
    precision, execution = record["precision"], record["execution"]
    if (precision["comparison_class"] != "accelerated" or precision.get("loss") != "f32"
            or precision.get("torch_float32_matmul_precision") != "highest"
            or precision.get("cuda_matmul_allow_tf32", False) or precision.get("cudnn_allow_tf32", False)
            or execution["gradient_scaler"]["enabled"] != (dtype == "fp16")):
        raise ValueError("wrong AMP arithmetic or scaling policy")
    gate = execution["amp_qualification"]
    if gate["status"] not in ("pass", "partial") or gate["reference"] != "eager strict f32":
        raise ValueError("missing AMP precision qualification")
    forward = gate["forward"]
    if forward["rtol"] != 0.02 or forward["atol"] != 1e-6 or forward["elements"] < 1:
        raise ValueError("wrong AMP forward gate")
    for metric in ("rms", "max_abs"):
        error, bound = forward[f"{metric}_error"], forward[f"{metric}_bound"]
        if not math.isfinite(error) or not math.isfinite(bound) or error < 0 or error > bound:
            raise ValueError("failed AMP forward gate claims success")
    training = record["protocol"]["training_requested"]
    if training:
        curve = gate["training"]
        metrics = curve.get("metrics", curve)
        failed_trajectory = curve.get("failure")
        if failed_trajectory is not None:
            trajectory = failed_trajectory["trajectory"]
            if (curve["status"] != "failed" or curve["policy"] != "loss-curve-v1" or curve["requested_updates"] != 32
                    or failed_trajectory["kind"] != "numerical" or trajectory["arithmetic"] not in ("strict-f32", dtype)
                    or not 0 <= trajectory["completed_updates"] < 32
                    or len(trajectory["losses"]) != trajectory["completed_updates"]
                    or any(not math.isfinite(x) for x in trajectory["losses"])):
                raise ValueError("unclassified AMP trajectory failure")
        elif (curve["status"] not in ("pass", "failed") or metrics["steps"] != 32
                or curve["completed_updates"] != 32 or curve["skipped_steps"]
                or metrics["policy"] != "loss-curve-v1" or metrics["rtol"] != 0.02
                or metrics["atol"] != 1e-6 or metrics["relative_l2_limit"] != 0.01
                or any(len(curve[key]) != 32 or any(not math.isfinite(x) for x in curve[key])
                       for key in ("losses", "reference_losses"))):
            raise ValueError("wrong AMP training curve gate")
        if curve["status"] == "pass" and (not math.isfinite(metrics["relative_l2"])
                or metrics["relative_l2"] > 0.01 or metrics["mismatched_steps"]):
            raise ValueError("failed AMP curve claims success")
        if curve["status"] == "failed" and record["timings"].get("training_ms") is not None:
            raise ValueError("failed AMP training supplies a timing")
    elif gate["training"] is not None:
        raise ValueError("unrequested AMP training gate")
    oracle = record["correctness_reference"]
    outputs = oracle["outputs"]
    if (oracle["precision"] != "strict-f32" or oracle["training_requested"] != training
            or oracle.get("prepared_before_amp_compilation") is not True
            or outputs["output_shape"] != record["outputs"]["output_shape"]
            or len(outputs["logits_sample"]) != len(record["outputs"]["logits_sample"])
            or not math.isfinite(outputs["loss"])
            or any(not math.isfinite(x) for x in outputs["logits_sample"])
            or training and (not outputs["gradient_norms"] or outputs["grad_norm"] is None
                or not math.isfinite(outputs["grad_norm"])
                or any(not math.isfinite(x) for x in outputs["gradient_norms"].values()))):
        raise ValueError("invalid strict-f32 oracle for the AMP reference")
    return {"inference": "pass", "latency": "pass", "training": gate["training"]["status"] if training else None}
