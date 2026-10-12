#!/usr/bin/env python3
"""Optional dtype diagnostics for the accelerated profile, outside paired collection."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

from collection import write_json
from p3hpc import (SUPPORTED_MODELS, check_torch_identity, gpu_matches, input_hashes, positive_seconds,
                  cpu_frequency_policies, reference_sdpa_policy, PYTHON_VERSION)
from study import ROOT, begin, environment, run_json
from precision import check_amp_qualification


def check_record(record, model, precision, mode, gpu, *, sdpa=None):
    execution, protocol = record["execution"], record["protocol"]
    if sdpa is not None and execution["sdpa_policy"] != sdpa:
        raise ValueError("requested attention policy did not execute")
    if record["model"] != model or record["status"] not in ("ok", "partial") or not gpu_matches(gpu, record["gpu_name"]):
        raise ValueError("unexpected model/device or incomplete AMP result")
    if record["precision"]["comparison_class"] != "accelerated" or protocol["name"] != "inferena-graph-replay-v9":
        raise ValueError("requested AMP arithmetic did not execute")
    if (record["precision"]["autocast"] != precision
            or record["precision"]["torch_float32_matmul_precision"] != "highest"
            or record["precision"].get("cuda_matmul_allow_tf32", False)
            or record["precision"].get("cudnn_allow_tf32", False)):
        raise ValueError("AMP's remaining f32 arithmetic changed")
    if (execution["requested_mode"] != mode or execution["compiled"] != (mode != "eager")
            or execution["graph_replay"]["requested"] or not execution["compile_budget_enforced"]):
        raise ValueError("requested uncaptured compiler policy did not execute")
    check_amp_qualification(record)
    gate = execution["amp_qualification"]
    training_valid = gate["training"]["status"] == "pass"
    if training_valid and (gate["training"]["steps"] != 32
            or gate["training"]["completed_updates"] != 32 or gate["training"]["skipped_steps"]
            or gate["training"]["policy"] != "loss-curve-v1" or gate["training"]["rtol"] != 0.02
            or gate["training"]["atol"] != 1e-6 or gate["training"]["relative_l2_limit"] != 0.01):
        raise ValueError("AMP output/loss-curve gate did not pass")
    for phase in ("inference", "training", "latency"):
        if phase == "training" and not training_valid:
            if (record["timing_samples_ms"][phase] is not None or record["timings"]["training_ms"] is not None
                    or record["status"] != "partial" or "amp_training_failure" not in execution):
                raise ValueError("failed AMP training must not supply timings")
            continue
        if (execution["graph_replay"]["phases"][phase]["status"] != "validated-uncaptured"
                or protocol["warmup"][phase]["runs"] < 5 or protocol["warmup"][phase]["seconds"] < 10
                or len(record["timing_samples_ms"][phase]) != 20):
            raise ValueError(f"{phase} is missing qualification or matched timing samples")
        if not all(math.isfinite(value) and value > 0 for value in record["timing_samples_ms"][phase]):
            raise ValueError(f"{phase} has invalid timings")
    if execution["gradient_scaler"]["enabled"] != (precision == "fp16"):
        raise ValueError("wrong gradient-scaling policy")
    return {"inference": "pass", "latency": "pass", "training": "pass" if training_valid else "failed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cuda", "rocm", "xpu", "mps", "cpu"))
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=["Whisper-tiny"])
    parser.add_argument("--precisions", nargs="+", choices=("bf16", "fp16"), default=["bf16", "fp16"])
    parser.add_argument("--mode", choices=("default", "eager"), default="default")
    parser.add_argument("--sdpa", choices=("auto", "math", "efficient"),
                        help="cohort default: math on XPU, auto elsewhere")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--compile-seconds", type=positive_seconds, default=120)
    parser.add_argument("--run-seconds", type=positive_seconds, default=1200)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 1 or len(set(args.models)) != len(args.models) or len(set(args.precisions)) != len(args.precisions):
        parser.error("positive replicates and unique models/precisions are required")
    import platform
    import torch
    if platform.python_version() != PYTHON_VERSION:
        parser.error(f"use pinned Python {PYTHON_VERSION}")
    check_torch_identity(torch.__version__, torch.version.git_version, torch.__version__)
    hashes = input_hashes(args.models)
    destination = args.results_dir.resolve()
    manifest = begin(destination, "accelerated-dtype-diagnostic-v1", args)
    manifest.update(input_sha256=hashes, cpu_frequency_policies=cpu_frequency_policies(),
                    reference_sdpa_policy={model: reference_sdpa_policy(args, model) for model in args.models},
                    timing_class="supplementary AMP, compiled uncaptured" if args.mode == "default" else "supplementary AMP, eager uncaptured",
                    publication_ready=False, scope="per-model precision qualification and timing; real-data learning is a separate study")
    path = destination / "campaign.json"
    write_json(path, manifest)
    try:
        columns = []
        for replicate in range(args.replicates):
            for model in args.models:
                choices = args.precisions[replicate % len(args.precisions):] + args.precisions[:replicate % len(args.precisions)]
                for precision in choices:
                    folder = destination / f"r{replicate + 1}" / model / precision
                    env = environment(args.backend)
                    env.update(INFERENA_AUTOCAST=precision, INFERENA_STRICT="0", INFERENA_TORCH_MODE=args.mode,
                               INFERENA_GRAPH_REPLAY="0", INFERENA_SDPA=reference_sdpa_policy(args, model),
                               INFERENA_WARMUP_RUNS="5", INFERENA_MEASUREMENT_RUNS="20",
                               INFERENA_COMPILE_SECONDS=str(args.compile_seconds),
                               INFERENA_PREPARATION_REPORT=str(folder / "preparation.json"),
                               INFERENA_AMP_REPORT=str(folder / "qualification.json"))
                    run = {"path": str(folder.relative_to(destination)), "model": model, "precision": precision,
                           "replicate": replicate + 1, "status": "running"}
                    manifest["runs"].append(run)
                    write_json(path, manifest)
                    print(run["path"], flush=True)
                    record = run_json([sys.executable, str(ROOT / "frameworks/pytorch/budget.py"),
                                       str(ROOT / "frameworks/pytorch/bench.py"), model], folder, env, args.run_seconds)
                    phases = check_record(record, model, precision, args.mode, args.gpu,
                                          sdpa=reference_sdpa_policy(args, model))
                    if record["backend"].split()[0].lower() != args.backend:
                        raise ValueError("requested PyTorch backend did not execute")
                    run.update(status="pass" if all(value == "pass" for value in phases.values()) else "partial",
                               phases=phases, timings=record["timings"])
                    columns.append({"model": model, "gpu": record["gpu_name"], "reference": f"amp-{precision}",
                                    "mode": args.mode, "graph_replay": False, "replicate": replicate + 1,
                                    "training_gate": phases["training"],
                                    **record["timings"]})
                    write_json(path, manifest)
        if input_hashes(args.models) != hashes:
            raise ValueError("input checkpoint changed during collection")
        manifest["status"] = "pass" if all(run["status"] == "pass" for run in manifest["runs"]) else "partial"
        with (destination / "columns.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(columns[0]))
            writer.writeheader()
            writer.writerows(columns)
    except BaseException as error:
        if manifest["runs"] and manifest["runs"][-1]["status"] == "running":
            manifest["runs"][-1].update(status="failed", error=str(error))
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(path, manifest)
    print(json.dumps({"status": manifest["status"], "campaign": str(path)}))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
