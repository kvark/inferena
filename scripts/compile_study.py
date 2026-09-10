#!/usr/bin/env python3
"""Paired fresh/reused-cache compilation diagnostics; artifacts stay outside Git."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from p3hpc import ROOT, MODELS, gpu_matches, input_hashes, select_native_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        default=["SmolLM2-135M", "ResNet-50", "Whisper-tiny"])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--mode", choices=("default", "max-autotune"), default="default")
    parser.add_argument("--precision", choices=("strict", "accelerated"), default="strict")
    parser.add_argument("--untraced", action="store_true")
    parser.add_argument("--gpu", help="optional assertion when selecting the one physical GPU")
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("replicates must be positive")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        parser.error("commit source changes before measurement")
    args.output = args.output.resolve()
    if args.output == ROOT or ROOT in args.output.parents:
        parser.error("use a new directory outside the checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    devices = json.loads(subprocess.check_output(
        [str(ROOT / "target/release/inferena-meganeura"), "--list-devices"], text=True))
    device = select_native_device(devices, args.gpu)
    manifest = {"source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "args": vars(args), "gpu": device, "inputs": input_hashes(args.models), "runs": [], "status": "incomplete"}
    try:
        for replicate in range(args.replicates):
            for model in args.models:
                base = args.output / f"r{replicate + 1}" / model
                base.mkdir(parents=True)
                engines = ("meganeura", "pytorch") if replicate % 2 == 0 else ("pytorch", "meganeura")
                for state in ("fresh", "reused"):
                    for engine in engines:
                        destination = base / f"{engine}-{state}"
                        cache = base / f"{engine}-cache"
                        env = dict(os.environ)
                        if engine == "pytorch":
                            command = [sys.executable, str(ROOT / "scripts/compile_torch.py"),
                                       "--output", str(destination), "--cache", str(cache), "--model", model,
                                       "--mode", args.mode, "--precision", args.precision]
                            if state == "reused": command.append("--reuse-cache")
                            if model == "SmolLM2-135M": command.append("--inference-only")
                            if args.untraced: command.append("--untraced")
                        else:
                            destination.mkdir()
                            cache.mkdir(exist_ok=state == "reused")
                            env.update({"INFERENA_STRICT": str(int(args.precision == "strict")),
                                        "INFERENA_INFERENCE_ONLY": str(int(model == "SmolLM2-135M")),
                                        "INFERENA_WARMUP_RUNS": "5", "INFERENA_MEASUREMENT_RUNS": "3",
                                        "INFERENA_REQUIRE_LOCAL_WEIGHTS": "1", "MEGANEURA_DEVICE_ID": str(device["device_id"]),
                                        "__GL_SHADER_DISK_CACHE": "1", "__GL_SHADER_DISK_CACHE_PATH": str(cache)})
                            if not args.untraced:
                                env["INFERENA_COMPILE_TRACE"] = str(destination / "compilation.jsonl")
                            command = [str(ROOT / "target/release/inferena-meganeura"), model]
                        print(f"r{replicate + 1} {model} {engine} {state}", flush=True)
                        row = {"engine": engine, "model": model, "cache_state": state,
                               "destination": str(destination), "command": command, "status": "incomplete"}
                        manifest["runs"].append(row)
                        with (base / f"{engine}-{state}.log").open("x") as log:
                            if engine == "meganeura":
                                with (destination / "runner.json").open("x") as output:
                                    subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=log, check=True)
                            else:
                                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                        result = json.loads((destination / "runner.json").read_text())
                        if not result["outputs"]:
                            raise ValueError("missing numerical results")
                        if not gpu_matches(device["name"], result["gpu_name"]):
                            raise ValueError("engines did not select the same GPU")
                        row["status"] = "complete"
        if input_hashes(args.models) != manifest["inputs"]:
            raise ValueError("inputs changed during the study")
        manifest["status"] = "complete"
    finally:
        with (args.output / "study.json").open("x") as output:
            json.dump(manifest, output, indent=2, default=str)
            output.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
