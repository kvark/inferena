#!/usr/bin/env python3
"""Capture a validated CUDA/Vulkan pair; diagnostic timings are not paper samples."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

from p3hpc import ROOT, SUPPORTED_MODELS, check_pair, input_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="ResNet-50")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--precision", choices=("strict", "accelerated"), default="strict")
    parser.add_argument("--mode", choices=("default", "max-autotune", "eager"), default="max-autotune")
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--nsys", default=os.environ.get("NSYS") or shutil.which("nsys"), help="Nsight Systems executable (or NSYS/PATH)")
    args = parser.parse_args()
    if not args.nsys:
        parser.error("put Nsight Systems on PATH, or pass --nsys /path/to/nsys")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        parser.error("commit source changes before capture")
    if any(key.startswith("MEGANEURA_") or key in (
        "INFERENA_PROFILE_DIR", "INFERENA_MEGANEURA_PATH", "INFERENA_DRY_RUN", "CARGO_TARGET_DIR",
    ) for key in os.environ):
        parser.error("remove local dependency and profiling overrides; capture the normal grouped schedule")
    destination = args.results_dir.resolve()
    if destination == ROOT or ROOT in destination.parents:
        parser.error("use a new directory outside the checkout")
    hashes = input_hashes([args.model])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    version = subprocess.check_output([args.nsys, "--version"], text=True).strip()
    subprocess.run(["cargo", "build", "--release", "--locked", "-p", "inferena-harness", "-p", "inferena-meganeura"], cwd=ROOT, check=True)
    destination.mkdir(parents=True, exist_ok=False)
    command = [
        args.nsys, "profile", "--trace=cuda,vulkan,nvtx,osrt", "--cuda-graph-trace=node",
        "--vulkan-gpu-workload=individual", "--sample=none", "--cpuctxsw=none",
        f"--output={destination / 'timeline'}",
        "bash", str(ROOT / "run.sh"), "-m", args.model, "-f", "pytorch,meganeura",
        "--warmup-runs", "5", "--measurement-runs", "3", "--results-dir", str(destination),
    ]
    if args.precision == "strict":
        command.append("--strict")
    if args.inference_only:
        command.append("--inference-only")
    env = dict(os.environ, PYTHON=sys.executable, INFERENA_NSYS="1", INFERENA_TORCH_BACKEND="cuda",
               INFERENA_TORCH_MODE=args.mode, INFERENA_CUDA_GRAPHS=str(int(not args.no_graphs)),
               INFERENA_REQUIRE_LOCAL_WEIGHTS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TORCH_LOGS="graph_breaks,recompiles,perf_hints")
    env.pop("VIRTUAL_ENV", None)
    manifest = {"source": revision, "nsys": version, "command": command, "sha256": hashes,
                "args": {**vars(args), "results_dir": str(destination)},
                "status": "in-progress", "purpose": "diagnostic, not benchmark or barrier-cost measurement"}

    def save():
        (destination / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        with (destination / "runner.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        args.backend = "cuda"
        records = json.loads((destination / f"{args.model}_summary.json").read_text())
        check_pair(records, args, args.mode, not args.no_graphs, 3, revision, diagnostic=True)
        if input_hashes([args.model]) != hashes:
            raise ValueError("inputs changed during capture")
        database = destination / "timeline.sqlite"
        subprocess.run([args.nsys, "export", "--type=sqlite", f"--output={database}",
                        str(destination / "timeline.nsys-rep")], check=True)
        with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as connection:
            tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            counts = {name: connection.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
                      for name in tables if name.startswith(("CUPTI_ACTIVITY_KIND_", "VULKAN_", "NVTX_"))}
        manifest["event_counts"] = counts
        if not counts.get("CUPTI_ACTIVITY_KIND_KERNEL") or not counts.get("NVTX_EVENTS"):
            raise ValueError("missing CUDA GPU events or host phase markers; inspect profiler diagnostics")
        if not any(count for name, count in counts.items() if name.startswith("VULKAN_") and "GPU" in name):
            raise ValueError("missing Vulkan GPU events; API-only capture is insufficient")
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "incomplete"
        manifest["error"] = str(error)
        raise
    finally:
        save()
    print(f"Open {destination / 'timeline.nsys-rep'} in Nsight Systems")


if __name__ == "__main__":
    main()
