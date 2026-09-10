#!/usr/bin/env python3
"""Check experiment pairs with Inferena's existing cross-engine numerical gates."""

import json
import math
from pathlib import Path
import statistics
import sys


def relative_l2(reference, other):
    if len(reference) != len(other) or not reference:
        raise ValueError("missing or mismatched samples")
    if not all(math.isfinite(value) for value in (*reference, *other)):
        raise ValueError("non-finite numerical result")
    error = sum((a - b) ** 2 for a, b in zip(reference, other))
    scale = sum(a * a for a in reference)
    return math.sqrt(error / scale) if scale else (math.inf if error else 0.0)


def compare(reference, other):
    """The <1% forward / <5% gradient-norm gates in harness::compare_result.

    These sampled cross-engine gates do not replace full PyTorch replay checks
    or the tuner's private full-output / sampled-f64 candidate qualification.
    """
    for key in ("model", "gpu_name"):
        if reference[key] != other[key]:
            raise ValueError(f"mismatched {key}")
    if reference["precision"]["comparison_class"] != other["precision"]["comparison_class"]:
        raise ValueError("mismatched precision")
    a, b = reference["outputs"], other["outputs"]
    if not a["output_shape"] or a["output_shape"] != b["output_shape"]:
        raise ValueError("mismatched output shapes")
    loss = abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), abs(b["loss"]), 1e-12)
    output = relative_l2(a["logits_sample"], b["logits_sample"])
    if not (loss < 0.01 and output < 0.01):
        raise ValueError(f"forward gate failed: loss={loss}, output={output}")
    metrics = {"loss_relative": loss, "output_relative_l2": output}
    training = reference["timings"].get("training_ms") is not None
    if training != (other["timings"].get("training_ms") is not None):
        raise ValueError("mismatched training contract")
    if training:
        norms_a, norms_b = a["gradient_norms"], b["gradient_norms"]
        if not norms_a or norms_a.keys() != norms_b.keys():
            raise ValueError("missing or mismatched parameter gradients")
        total = abs(a["grad_norm"] - b["grad_norm"]) / max(abs(a["grad_norm"]), 1e-12)
        gradients = relative_l2(list(norms_a.values()), [norms_b[key] for key in norms_a])
        if not (total < 0.05 and gradients < 0.05):
            raise ValueError(f"gradient gate failed: total={total}, parameters={gradients}")
        metrics.update(total_gradient_relative=total, parameter_gradient_relative_l2=gradients)
    return metrics


def compilation_report(root):
    manifest = json.loads((root / "study.json").read_text())
    if manifest["status"] != "complete" or any(row["status"] != "complete" for row in manifest["runs"]):
        raise ValueError("incomplete study")
    grouped = {}
    baselines = {}
    worst = {}
    for row in manifest["runs"]:
        path = Path(row["destination"])
        result = json.loads((path / "runner.json").read_text())
        if row["engine"] == "meganeura":
            reference = json.loads((path.parent / f"pytorch-{row['cache_state']}" / "runner.json").read_text())
            for key, value in compare(reference, result).items():
                worst[key] = max(worst.get(key, 0), value)
            baseline = baselines.setdefault(row["model"], result["outputs"])
            if baseline != result["outputs"]:
                raise ValueError("native numerical results changed across repetitions/cache states")
        else:
            phases = result["execution"]["cuda_graphs"]["phases"]
            if not phases or any(phase["status"] != "captured-and-validated" for phase in phases.values()):
                raise ValueError("missing validated CUDA graphs")
        key = (row["model"], row["cache_state"], row["engine"])
        group = grouped.setdefault(key, {"preparation_s": [], "stages": {}})
        group["preparation_s"].append(result["timings"]["compile_s"])
        stages = {}
        if (path / "compilation.jsonl").exists():
            for line in (path / "compilation.jsonl").read_text().splitlines():
                span = json.loads(line)
                stages.setdefault(span["stage"], []).append(span["duration_ns"] / 1e3)
        if (path / "compilation.json").exists():
            trace = json.loads((path / "compilation.json").read_text())
            if trace["status"] != "complete" or trace["pending_loads"]:
                raise ValueError("incomplete compiler trace")
            for unit in trace["compilations"]:
                name = "triton_cache_hit" if unit["cache_hit"] else "triton_cold"
                stages.setdefault(name, []).append(unit["total_us"])
                if not unit["cache_hit"]:
                    for name, elapsed in unit["times_us"]["lowering_stages"]:
                        stages.setdefault(name, []).append(elapsed)
                    stages.setdefault("triton_ir_initialization", []).append(unit["times_us"]["ir_initialization"])
            for unit in trace["kernel_loads"]:
                stages.setdefault("cuda_load_binary", []).append(unit["duration_ns"] / 1e3)
        for stage, values in stages.items():
            group["stages"].setdefault(stage, []).append(
                {"count": len(values), "median_us": statistics.median(values), "sum_ms": sum(values) / 1e3})
    print(json.dumps({"source": manifest["source"], "cross_engine_worst": worst,
                      "native_outputs_repeat_exactly": True,
                      "groups": [{"key": key, **value} for key, value in grouped.items()]}, indent=2))


def tuning_report(root):
    manifest = json.loads((root / "study.json").read_text())
    if manifest["status"] != "complete" or any(row["status"] != "complete" for row in manifest["runs"]):
        raise ValueError("incomplete study")
    groups, outputs, token_hashes = {}, {}, {}
    reference_variant = manifest["args"].get("baseline", "untuned")
    for row in manifest["runs"]:
        path = Path(row["destination"])
        result = json.loads((path / "runner.json").read_text())
        baseline = json.loads((path.parent / reference_variant / "runner.json").read_text())
        compare(baseline, result)
        token = result.get("environment", {}).get("stateless_validation")
        if token is not None:
            if not (0 <= token["prefill_prefix_relative_l2"] < 0.01):
                raise ValueError("stateless/prefill full-vector gate failed")
            token_hashes.setdefault(row["model"], []).append(token["logits_hash"])
        outputs.setdefault(row["model"], []).append(result["outputs"])
        groups.setdefault((row["model"], row["variant"]), []).append((baseline, result))
    report = []
    for (model, variant), pairs in groups.items():
        phases = {}
        for key in pairs[0][0]["timings"]:
            if pairs[0][0]["timings"][key] is None:
                continue
            baseline = [a["timings"][key] for a, _ in pairs]
            candidate = [b["timings"][key] for _, b in pairs]
            gain = [a - b for a, b in zip(baseline, candidate)]
            median_gain = statistics.median(gain)
            noise = 2 * statistics.median(abs(value - median_gain) for value in gain)
            phases[key] = {"baseline_median": statistics.median(baseline),
                           "candidate_median": statistics.median(candidate),
                           "paired_gain_median": median_gain, "paired_gain_2mad": noise,
                           "clears_5percent_plus_noise": median_gain > 0.05 * statistics.median(baseline) + noise}
        report.append({"model": model, "variant": variant, "process_pairs": len(pairs), "phases": phases})
    print(json.dumps({"source": manifest["source"], "groups": report,
                      "stateless_hashes_exact": {model: len(set(values)) == 1 for model, values in token_hashes.items()},
                      "recorded_outputs_exact": {model: all(value == values[0] for value in values)
                                                 for model, values in outputs.items()}}, indent=2))


if __name__ == "__main__":
    root = Path(sys.argv[1])
    manifest = json.loads((root / "study.json").read_text())
    if "variant" in manifest["runs"][0]:
        tuning_report(root)
    else:
        compilation_report(root)
