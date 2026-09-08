#!/usr/bin/env python3
"""Qualification-first, paired P3HPC collection. Generated evidence stays outside Git."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny")
PHASES = ("inference", "latency", "training")


def conditions(backend):
    if backend == "cuda":
        return [("default", False), ("default", True), ("max-autotune", True)]
    if backend == "rocm":
        return [("default", False), ("max-autotune", False)]
    return [("eager", False)]


def check_pair(records, args, mode, graphs, count, revision):
    by_engine = {record["framework"]: record for record in records}
    if len(records) != 2 or set(by_engine) != {"pytorch", "meganeura"}:
        raise ValueError("both engine records are required")
    for engine, record in by_engine.items():
        if record["status"] != "ok":
            raise ValueError(f"{engine} failed: {record.get('error', record.get('reason'))}")
        validation = record["validation"]
        if not all(validation.get(key) is True for key in (
            "comparison_performed", "forward_valid", "training_valid",
        )) or validation.get("reference_framework") != "pytorch":
            raise ValueError(f"{engine} failed the matched forward/backward gate: {validation}")
        if not revision.startswith(record["benchmark_rev"]):
            raise ValueError("source changed during collection")
        if record["protocol"]["warmup_runs"] != 5:
            raise ValueError("unexpected warmup count")
        for phase in PHASES:
            samples = record["timing_samples_ms"][phase]
            if len(samples) != count or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError(f"{engine} has invalid {phase} samples")
    pt, mg = by_engine["pytorch"], by_engine["meganeura"]
    if pt["torch_version"] != args.torch_version:
        raise ValueError("PyTorch version differs from the declared version")
    if pt["backend"].split()[0].lower() != args.backend:
        raise ValueError(f"unexpected reference backend: {pt['backend']}")
    if args.gpu.casefold() not in mg["gpu_name"].casefold():
        raise ValueError(f"unexpected Meganeura GPU: {mg['gpu_name']}")
    if args.backend in ("cuda", "rocm") and args.gpu.casefold() not in pt["gpu_name"].casefold():
        raise ValueError(f"unexpected PyTorch GPU: {pt['gpu_name']}")
    execution = pt["execution"]
    if execution["requested_mode"] != mode or execution["compiled"] != (mode != "eager"):
        raise ValueError("requested compiler mode did not execute")
    if execution["cuda_graphs"]["requested"] != graphs:
        raise ValueError("unexpected graph configuration")
    for phase in PHASES:
        report = execution["cuda_graphs"]["phases"][phase]
        expected = "captured-and-validated" if graphs else "not-requested"
        if report["status"] != expected:
            raise ValueError(f"{phase} did not execute the requested capture mode")
    return by_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cuda", "rocm", "mps", "cpu"), required=True)
    parser.add_argument("--gpu", required=True, help="expected GPU-name substring")
    parser.add_argument("--torch-version", required=True, help="exact version including vendor suffix")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--precisions", nargs="+", choices=("strict", "accelerated"), default=["strict", "accelerated"])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--collect", action="store_true", help="measure only after all requested qualification pairs pass")
    args = parser.parse_args()
    if args.replicates < 3:
        parser.error("collection requires at least three fresh processes per condition")
    if len(set(args.models)) != len(args.models) or len(set(args.precisions)) != len(args.precisions):
        parser.error("duplicate models or precision classes")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        parser.error("commit source changes before qualification/collection")
    overrides = [key for key in os.environ if key.startswith("MEGANEURA_") or key in (
        "INFERENA_MEGANEURA_PATH", "INFERENA_PROFILE_DIR", "INFERENA_DRY_RUN",
        "TORCH_LOGS", "TORCH_TRACE", "CARGO_TARGET_DIR",
    )]
    if overrides:
        parser.error(f"remove experimental/profiling overrides: {', '.join(overrides)}")
    destination = args.results_dir.resolve()
    if destination == ROOT or ROOT in destination.parents:
        parser.error("use a new results directory outside the checkout")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if importlib.metadata.version("torch") != args.torch_version:
        parser.error("this Python interpreter has a different PyTorch version")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    lockfile = ROOT / "Cargo.lock"
    model_files = []
    if "SmolLM2-135M" in args.models:
        model_files = [ROOT / "models/SmolLM2-135M" / name for name in ("config.json", "model.safetensors")]
        if not all(path.is_file() for path in model_files):
            parser.error("prepare local SmolLM2-135M config.json/model.safetensors before collection")
    hashes = {}
    for path in [lockfile, *model_files]:
        with path.open("rb") as source:
            hashes[str(path.relative_to(ROOT))] = hashlib.file_digest(source, "sha256").hexdigest()
    dependency = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["dependencies"]["meganeura"]
    destination.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHON=sys.executable, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    manifest = {
        "protocol": "p3hpc-paired-campaign-v1", "source": revision,
        "meganeura": dependency, "python": sys.version, "packages": packages,
        "args": {**vars(args), "results_dir": str(destination)}, "sha256": hashes,
        "device_selection": {key: env[key] for key in (
            "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "VK_ICD_FILENAMES",
            "MESA_VK_DEVICE_SELECT", "HSA_OVERRIDE_GFX_VERSION",
        ) if key in env},
        "runs": [], "status": "in-progress",
    }

    def save():
        (destination / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        # Finish compilation before the first paired process; later wrapper
        # builds are locked no-ops, outside the timed engine calls.
        subprocess.run(["cargo", "build", "--release", "--locked", "-p", "inferena-harness", "-p", "inferena-meganeura"], cwd=ROOT, check=True)
        stages = [("qualification", 1, 1)]
        if args.collect:
            stages.append(("measurement", args.replicates, 20))
        sequence = 0
        for stage, replicates, count in stages:
            for replicate in range(replicates):
                for precision in args.precisions:
                    for model in args.models:
                        configs = conditions(args.backend)
                        offset = replicate % len(configs)
                        for mode, graphs in configs[offset:] + configs[:offset]:
                            label = f"{mode}-graph{int(graphs)}"
                            folder = destination / stage / f"r{replicate + 1}" / precision / model / label
                            folder.mkdir(parents=True)
                            order = "pytorch,meganeura" if sequence % 2 == 0 else "meganeura,pytorch"
                            sequence += 1
                            command = ["bash", str(ROOT / "run.sh"), "-m", model, "-f", order,
                                       "--warmup-runs", "5", "--measurement-runs", str(count),
                                       "--results-dir", str(folder)]
                            if precision == "strict":
                                command.append("--strict")
                            run = {"path": str(folder.relative_to(destination)), "command": command,
                                   "mode": mode, "graphs": graphs, "status": "running"}
                            manifest["runs"].append(run)
                            save()
                            print(run["path"], flush=True)
                            with (folder / "runner.log").open("w") as log:
                                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                        env=dict(env, INFERENA_TORCH_MODE=mode, INFERENA_CUDA_GRAPHS=str(int(graphs))))
                            run["returncode"] = result.returncode
                            if result.returncode:
                                raise RuntimeError(f"runner failed; inspect {folder / 'runner.log'}")
                            records = json.loads((folder / f"{model}_summary.json").read_text())
                            pair = check_pair(records, args, mode, graphs, count, revision)
                            if not dependency["rev"].startswith(pair["meganeura"]["framework_rev"]):
                                raise ValueError("Meganeura dependency revision changed")
                            run["status"] = "valid"
                            save()
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "incomplete"
        manifest["error"] = str(error)
        raise
    finally:
        save()
    print(f"Complete: {destination / 'campaign.json'}")


if __name__ == "__main__":
    main()
