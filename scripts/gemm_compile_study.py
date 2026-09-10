#!/usr/bin/env python3
"""Matched-domain f32 GEMM compiler study; no kernel-speed claim."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from p3hpc import ROOT, select_native_device, gpu_matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--warm-compiler", action="store_true",
                        help="compile/load the opposite tile first; record its cost separately")
    args = parser.parse_args()
    if args.replicates < 1 or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        parser.error("positive repetitions and committed source required")
    args.output = args.output.resolve()
    if args.output == ROOT or ROOT in args.output.parents:
        parser.error("keep generated data outside Git")
    args.output.mkdir(parents=True, exist_ok=False)
    device = select_native_device(json.loads(subprocess.check_output(
        [str(ROOT / "target/release/inferena-meganeura"), "--list-devices"], text=True)), None)
    manifest = {"source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "device": device, "warm_compiler": args.warm_compiler,
                "runs": [], "status": "incomplete"}
    try:
        for replicate in range(args.replicates):
            for k in (576, 1536):
                for tile in (32, 64):
                    base = args.output / f"r{replicate + 1}-128-576-{k}-tile{tile}"
                    base.mkdir()
                    for state in ("fresh", "reused"):
                        engines = ("meganeura", "triton") if replicate % 2 == 0 else ("triton", "meganeura")
                        for engine in engines:
                            cache = base / f"{engine}-cache"
                            cache.mkdir(exist_ok=state == "reused")
                            env = {key: value for key, value in os.environ.items()
                                   if not key.startswith(("INFERENA_", "MEGANEURA_"))}
                            env.update(MEGANEURA_DEVICE_ID=str(device["device_id"]), NVIDIA_TF32_OVERRIDE="0",
                                       __GL_SHADER_DISK_CACHE="1", __GL_SHADER_DISK_CACHE_PATH=str(cache),
                                       TRITON_CACHE_DIR=str(cache / "triton"), CUDA_CACHE_PATH=str(cache / "cuda"))
                            if args.warm_compiler:
                                env["INFERENA_GEMM_WARM_COMPILER"] = "1"
                            output = base / f"{engine}-{state}.json"
                            if engine == "meganeura":
                                env["INFERENA_COMPILE_TRACE"] = str(base / f"{engine}-{state}.jsonl")
                                command = [str(ROOT / "target/release/compile_gemm")]
                            else:
                                command = [sys.executable, str(ROOT / "scripts/compile_gemm_torch.py")]
                            command += ["128", "576", str(k), str(tile)]
                            print(base.name,engine,state,flush=True)
                            row = {"engine": engine, "state": state, "output": str(output), "status": "incomplete"}
                            manifest["runs"].append(row)
                            with output.open("x") as stream, output.with_suffix(".log").open("x") as log:
                                subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=log, check=True)
                            result = json.loads(output.read_text())
                            if not gpu_matches(device["name"], result["gpu"]):
                                raise ValueError("wrong GPU")
                            # Keep rejected rows; never turn failed qualification into a speed claim.
                            row["qualification_failures"] = sum(item["failures"] for item in result["validation"])
                            row["status"] = "complete"
        manifest["status"] = "complete"
    finally:
        with (args.output / "study.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    main()
