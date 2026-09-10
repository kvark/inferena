#!/usr/bin/env python3
"""Full-model tuning/implementation ablation, with unchanged numerical gates."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from p3hpc import ROOT, SUPPORTED_MODELS, input_hashes, select_native_device
from study_results import compare


def main():
    specializations = {"fixed-params": "1", "fixed-native-div": "native-div",
                       "fixed-k32": "k32", "fixed-native-div-k32": "native-div-k32"}
    parameter_memory = {"device-params": "1", "device-params-buddy": "device-buddy",
                        "device-params-reuse": "device-buddy", "device-params-tiled": "device-buddy"}
    widths = {f"{prefix}gemv{width}": str(width) for prefix in ("", "unpacked-") for width in (32, 64, 128)}
    layouts = ("interleaved", "unpacked", "unpacked-interleaved", *widths)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS,
                        default=["SmolLM2-135M", "ResNet-50", "Whisper-tiny"])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--measurement-runs", type=int, default=20)
    parser.add_argument("--profile", action="store_true", help="separate serialized GPU pass diagnostics")
    parser.add_argument("--host-trace", action="store_true", help="separate Linux thread-CPU/frequency diagnostic")
    parser.add_argument("--cpu", type=int, help="Linux process-local CPU affinity control (not a system setting)")
    parser.add_argument("--baseline", default="untuned", help="reference variant present in every replicate")
    parser.add_argument("--precision", choices=("strict", "accelerated"), default="strict")
    parser.add_argument("--variants", nargs="+",
                        choices=("untuned", "default", "expanded", "shared-freelist",
                                 *parameter_memory, *specializations, *layouts),
                        default=["untuned", "default", "expanded"])
    parser.add_argument("--expanded-scratch-mib", type=int, default=64)
    parser.add_argument("--transpose-tile", type=int, default=16,
                        help="CPU tile for the device-params-tiled arm only")
    parser.add_argument("--fresh-driver-cache", action="store_true",
                        help="use a new private driver disk cache for each process")
    parser.add_argument("--trace-setup", action="store_true",
                        help="separate CPU compilation and parameter-preparation spans (diagnostic)")
    parser.add_argument("--stream-weights", action="store_true",
                        help="bound SmolLM2 checkpoint residency to one stored tensor plus conversion")
    parser.add_argument("--resident", action="store_true",
                        help="kernel studies: device-buddy parameters, reused uploads, CPU transpose tile 16")
    args = parser.parse_args()
    if min(args.replicates, args.warmup_runs, args.measurement_runs) < 1 or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        parser.error("positive replicate count and committed source required")
    if args.baseline not in args.variants or len(set(args.variants)) != len(args.variants) or min(args.expanded_scratch_mib, args.transpose_tile) < 1:
        parser.error("distinct variants including the baseline, and positive scratch budget required")
    if args.resident and any(variant in parameter_memory or variant == "shared-freelist" for variant in args.variants):
        parser.error("--resident is a common kernel-study setup, not a parameter-placement arm")
    if args.cpu is not None and (not hasattr(os, "sched_getaffinity") or args.cpu not in os.sched_getaffinity(0)):
        parser.error("--cpu must be a currently allowed Linux logical CPU")
    args.output = args.output.resolve()
    if args.output == ROOT or ROOT in args.output.parents:
        parser.error("keep artifacts outside the checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    runner = str(ROOT / "target/release/inferena-meganeura")
    device = select_native_device(json.loads(subprocess.check_output([runner, "--list-devices"], text=True)), None)
    expanded = args.output / "expanded-options.json"
    options = {"scope": "All", "max_classes": 128, "max_scratch_bytes": args.expanded_scratch_mib * 1024 * 1024,
               "staging": "Download", "staging_reuse": "SameSize", "max_time": {"secs": 60, "nanos": 0},
               "warmup_runs": 1, "sample_pairs": 6, "dispatches_per_sample": 16, "min_improvement": 0.05}
    with expanded.open("x") as output:
        json.dump(options, output, indent=2)
    manifest = {"source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "args": vars(args), "gpu": device, "inputs": input_hashes(args.models),
                "runs": [], "status": "incomplete"}
    try:
        for replicate in range(args.replicates):
            for model in args.models:
                results = {}
                start = replicate % len(args.variants)
                order = args.variants if (replicate // len(args.variants)) % 2 == 0 else args.variants[::-1]
                variants = order[start:] + order[:start]
                for variant in variants:
                    destination = args.output / f"r{replicate + 1}" / model / variant
                    destination.mkdir(parents=True)
                    env = {key: value for key, value in os.environ.items()
                           if not key.startswith(("INFERENA_", "MEGANEURA_"))}
                    env.update({"INFERENA_STRICT": str(int(args.precision == "strict")),
                                "INFERENA_INFERENCE_ONLY": str(int(model.startswith("SmolLM2-"))),
                                "INFERENA_WARMUP_RUNS": str(args.warmup_runs),
                                "INFERENA_MEASUREMENT_RUNS": str(args.measurement_runs),
                                "INFERENA_REQUIRE_LOCAL_WEIGHTS": "1", "MEGANEURA_DEVICE_ID": str(device["device_id"]),
                                "INFERENA_STREAM_WEIGHTS": str(int(args.stream_weights)),
                                "MEGANEURA_TUNE": str(int(variant in ("default", "expanded"))),
                                "MEGANEURA_SPECIALIZE_CONV": specializations.get(variant, "0"),
                                "MEGANEURA_DEVICE_PARAMETERS": parameter_memory.get(variant, "device-buddy" if args.resident else "0"),
                                "MEGANEURA_REUSE_UPLOAD": str(int(args.resident or variant in ("device-params-reuse", "device-params-tiled"))),
                                "MEGANEURA_TRANSPOSE_TILE": str(16 if args.resident else args.transpose_tile if variant == "device-params-tiled" else 0),
                                "MEGANEURA_INTERLEAVE_COLUMNS": str(int(variant in ("interleaved", "unpacked-interleaved"))),
                                "MEGANEURA_GREEDY_PACK_SWIGLU": str(int(not variant.startswith("unpacked"))),
                                "MEGANEURA_GEMV_THREADS": widths.get(variant, "256"),
                                "BLADE_SHARED_TRANSIENT": str(int(variant == "shared-freelist")),
                                "RUST_LOG": "warn,meganeura::runtime::tuning=info"})
                    if args.trace_setup:
                        env["INFERENA_COMPILE_TRACE"] = str(destination / "compilation.jsonl")
                    if args.host_trace:
                        env["INFERENA_HOST_TRACE"] = str(destination / "host")
                    if args.profile:
                        env["INFERENA_PROFILE_DIR"] = str(destination / "profiles")
                        env["MEGANEURA_GPU_TIMING"] = "1"
                    if args.fresh_driver_cache:
                        cache = destination / "driver-cache"
                        cache.mkdir()
                        env.update(__GL_SHADER_DISK_CACHE="1", __GL_SHADER_DISK_CACHE_PATH=str(cache))
                    if variant in ("default", "expanded"):
                        env["INFERENA_TUNE_REPORT"] = str(destination)
                    if variant == "expanded":
                        env["INFERENA_TUNE_OPTIONS"] = str(expanded)
                    print(f"r{replicate + 1} {model} {variant}", flush=True)
                    row = {"model": model, "variant": variant, "destination": str(destination), "status": "incomplete"}
                    manifest["runs"].append(row)
                    start = time.monotonic()
                    with (destination / "runner.json").open("x") as output, (destination / "runner.log").open("x") as log:
                        command = [runner, model]
                        if args.cpu is not None:
                            command = ["taskset", "--cpu-list", str(args.cpu), *command]
                        subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=log, check=True)
                    row["process_elapsed_s"] = time.monotonic() - start
                    results[variant] = json.loads((destination / "runner.json").read_text())
                    row["status"] = "complete"
                for variant, result in results.items():
                    compare(results[args.baseline], result)
        if input_hashes(args.models) != manifest["inputs"]:
            raise ValueError("inputs changed during the study")
        manifest["status"] = "complete"
    finally:
        with (args.output / "study.json").open("x") as output:
            json.dump(manifest, output, indent=2, default=str)
            output.write("\n")


if __name__ == "__main__":
    main()
