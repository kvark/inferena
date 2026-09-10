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

from p3hpc import ROOT, SUPPORTED_MODELS, check_pair, input_hashes, runner_bash


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
    parser.add_argument("--stream-weights", action="store_true",
                        help="bound checkpoint host residency; PyTorch needs accelerate==1.15.0")
    parser.add_argument("--device-parameters", choices=("0", "1", "device-buddy"), default="0",
                        help="experiment-only native parameter placement")
    parser.add_argument("--reuse-upload", action="store_true", help="experiment-only bounded upload cache")
    parser.add_argument("--nsys", default=os.environ.get("NSYS") or shutil.which("nsys"), help="Nsight Systems executable (or NSYS/PATH)")
    args = parser.parse_args()
    if not args.nsys:
        parser.error("put Nsight Systems on PATH, or pass --nsys /path/to/nsys")
    args.nsys = shutil.which(args.nsys) or str(Path(args.nsys).resolve())
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
        runner_bash(), (ROOT / "run.sh").as_posix(), "-m", args.model, "-f", "pytorch,meganeura",
        "--warmup-runs", "5", "--measurement-runs", "3", "--results-dir", str(destination),
    ]
    if args.precision == "strict":
        command.append("--strict")
    if args.inference_only:
        command.append("--inference-only")
    env = dict(os.environ, PYTHON=Path(sys.executable).as_posix(), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               INFERENA_BASH=command[0], INFERENA_NSYS=args.nsys,
               INFERENA_NSYS_DIR=str(destination), INFERENA_TORCH_BACKEND="cuda",
               INFERENA_TORCH_MODE=args.mode, INFERENA_CUDA_GRAPHS=str(int(not args.no_graphs)),
               INFERENA_STREAM_WEIGHTS=str(int(args.stream_weights)),
               MEGANEURA_DEVICE_PARAMETERS=args.device_parameters,
               MEGANEURA_REUSE_UPLOAD=str(int(args.reuse_upload)),
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
        manifest["event_counts"] = {}
        for engine, gpu_table in (("pytorch", "CUPTI_ACTIVITY_KIND_KERNEL"), ("meganeura", "VULKAN_WORKLOAD")):
            database = destination / f"{engine}.sqlite"
            with (destination / f"{engine}-export.log").open("w") as log:
                subprocess.run([args.nsys, "export", "--type=sqlite", f"--output={database}",
                                str(destination / f"{engine}.nsys-rep")], stdout=log, stderr=subprocess.STDOUT, check=True)
            with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as connection:
                tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                counts = {name: connection.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
                          for name in tables if name.startswith(("CUPTI_ACTIVITY_KIND_", "VULKAN_", "NVTX_"))}
                manifest["event_counts"][engine] = counts
                if not counts.get(gpu_table) or not counts.get("NVTX_EVENTS"):
                    raise ValueError(f"{engine}: missing GPU events or host markers; API-only capture is insufficient")
                for phase in ("inference", "latency") if args.inference_only else ("inference", "latency", "training"):
                    ranges = connection.execute(
                        "SELECT start, end FROM NVTX_EVENTS LEFT JOIN StringIds ON textId=StringIds.id "
                        "WHERE coalesce(text, value)=?", (f"{engine}/{phase}/measure",),
                    ).fetchall()
                    if len(ranges) != 1 or ranges[0][1] is None:
                        raise ValueError(f"{engine}/{phase}: missing or incomplete host measurement range")
                    events = connection.execute(f'SELECT count(*) FROM "{gpu_table}" WHERE start>=? AND end<=?', ranges[0]).fetchone()[0]
                    if not events:
                        raise ValueError(f"{engine}/{phase}: no GPU events in the measurement range")
                    counts[f"{phase}_gpu_events"] = events
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "incomplete"
        manifest["error"] = str(error)
        raise
    finally:
        save()
    print(f"Open pytorch.nsys-rep and meganeura.nsys-rep in {destination}")


if __name__ == "__main__":
    main()
