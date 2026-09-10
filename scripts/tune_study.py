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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS,
                        default=["SmolLM2-135M", "ResNet-50", "Whisper-tiny"])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--precision", choices=("strict", "accelerated"), default="strict")
    parser.add_argument("--variants", nargs="+",
                        choices=("untuned", "default", "expanded", "device-params", *specializations),
                        default=["untuned", "default", "expanded"])
    parser.add_argument("--expanded-scratch-mib", type=int, default=64)
    parser.add_argument("--fresh-driver-cache", action="store_true",
                        help="use a new private driver disk cache for each process")
    parser.add_argument("--trace-setup", action="store_true",
                        help="separate CPU compilation and parameter-preparation spans (diagnostic)")
    args = parser.parse_args()
    if args.replicates < 1 or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        parser.error("positive replicate count and committed source required")
    if "untuned" not in args.variants or len(set(args.variants)) != len(args.variants) or args.expanded_scratch_mib < 1:
        parser.error("distinct variants including untuned, and positive scratch budget required")
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
                                "INFERENA_WARMUP_RUNS": "5", "INFERENA_MEASUREMENT_RUNS": "20",
                                "INFERENA_REQUIRE_LOCAL_WEIGHTS": "1", "MEGANEURA_DEVICE_ID": str(device["device_id"]),
                                "MEGANEURA_TUNE": str(int(variant in ("default", "expanded"))),
                                "MEGANEURA_SPECIALIZE_CONV": specializations.get(variant, "0"),
                                "MEGANEURA_DEVICE_PARAMETERS": str(int(variant == "device-params")),
                                "RUST_LOG": "warn,meganeura::runtime::tuning=info"})
                    if args.trace_setup:
                        env["INFERENA_COMPILE_TRACE"] = str(destination / "compilation.jsonl")
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
                        subprocess.run([runner, model], cwd=ROOT, env=env, stdout=output, stderr=log, check=True)
                    row["process_elapsed_s"] = time.monotonic() - start
                    results[variant] = json.loads((destination / "runner.json").read_text())
                    row["status"] = "complete"
                for variant, result in results.items():
                    compare(results["untuned"], result)
        if input_hashes(args.models) != manifest["inputs"]:
            raise ValueError("inputs changed during the study")
        manifest["status"] = "complete"
    finally:
        with (args.output / "study.json").open("x") as output:
            json.dump(manifest, output, indent=2, default=str)
            output.write("\n")


if __name__ == "__main__":
    main()
