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
SMOLLM2_REVISIONS = json.loads((ROOT / "models/smollm2-revisions.json").read_text())
SUPPORTED_MODELS = (*MODELS, "SmolLM2-360M", "SmolLM2-1.7B")
PHASES = ("inference", "latency", "training")
PYTHON_VERSION = (ROOT / ".python-version").read_text().strip()
TORCH_VERSION = "2.13.0"
TORCH_REVISION = "cf30153c4c131c8164ee7798e5022d810682e2cb"


def check_torch_identity(version, revision, declared_version):
    if version != declared_version or version.split("+")[0] != TORCH_VERSION:
        raise ValueError(f"PyTorch must be {TORCH_VERSION} with the declared vendor suffix")
    if revision != TORCH_REVISION:
        raise ValueError(f"PyTorch source must be {TORCH_REVISION}, got {revision!r}")


def conditions(backend):
    if backend == "cuda":
        return [("default", False), ("default", True), ("max-autotune", True)]
    if backend in ("rocm", "xpu"):
        return [("default", False), ("max-autotune", False)]
    return [("eager", False)]


def check_pair(records, args, mode, graphs, count, revision, diagnostic=False):
    phases = PHASES[:2] if args.inference_only else PHASES
    by_engine = {record["framework"]: record for record in records}
    if len(records) != 2 or set(by_engine) != {"pytorch", "meganeura"}:
        raise ValueError("both engine records are required")
    for engine, record in by_engine.items():
        if record["status"] != "ok":
            raise ValueError(f"{engine} failed: {record.get('error', record.get('reason'))}")
        validation = record["validation"]
        gates = ("comparison_performed", "forward_valid")
        if not args.inference_only:
            gates += ("training_valid",)
        if not all(validation.get(key) is True for key in gates) or validation.get("reference_framework") != "pytorch":
            raise ValueError(f"{engine} failed the requested numerical gates: {validation}")
        if record["protocol"]["training_requested"] != (not args.inference_only):
            raise ValueError("unexpected training scope")
        if args.inference_only and (validation.get("training_valid") is not None or record["timings"].get("training_ms") is not None):
            raise ValueError("inference-only run claims training results")
        if record["protocol"].get("diagnostic") is not diagnostic:
            raise ValueError("diagnostic and benchmark samples must not be mixed")
        if not revision.startswith(record["benchmark_rev"]):
            raise ValueError("source changed during collection")
        if record["protocol"]["warmup_runs"] != 5:
            raise ValueError("unexpected warmup count")
        for phase in phases:
            samples = record["timing_samples_ms"][phase]
            if len(samples) != count or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError(f"{engine} has invalid {phase} samples")
    pt, mg = by_engine["pytorch"], by_engine["meganeura"]
    check_torch_identity(pt["torch_version"], pt["environment"].get("torch_git_version"), args.torch_version)
    if pt["environment"]["python_version"] != PYTHON_VERSION:
        raise ValueError(f"use the campaign's Python {PYTHON_VERSION}")
    if pt["backend"].split()[0].lower() != args.backend:
        raise ValueError(f"unexpected reference backend: {pt['backend']}")
    if args.gpu.casefold() not in mg["gpu_name"].casefold():
        raise ValueError(f"unexpected Meganeura GPU: {mg['gpu_name']}")
    if args.backend in ("cuda", "rocm", "xpu") and args.gpu.casefold() not in pt["gpu_name"].casefold():
        raise ValueError(f"unexpected PyTorch GPU: {pt['gpu_name']}")
    execution = pt["execution"]
    if execution["requested_mode"] != mode or execution["compiled"] != (mode != "eager"):
        raise ValueError("requested compiler mode did not execute")
    if execution["cuda_graphs"]["requested"] != graphs:
        raise ValueError("unexpected graph configuration")
    for phase in phases:
        report = execution["cuda_graphs"]["phases"][phase]
        expected = "captured-and-validated" if graphs else "not-requested"
        if report["status"] != expected:
            raise ValueError(f"{phase} did not execute the requested capture mode")
    return by_engine


def input_hashes(models):
    """Refuse unpinned checkpoints, including accidentally mixed base/Instruct files."""
    hashes = {"Cargo.lock": hashlib.sha256((ROOT / "Cargo.lock").read_bytes()).hexdigest()}
    for model in models:
        if model not in SMOLLM2_REVISIONS:
            continue
        directory = ROOT / "models" / model
        source = json.loads((directory / "source.json").read_text())
        if source["repo"] != f"HuggingFaceTB/{model}" or source["revision"] != SMOLLM2_REVISIONS[model]:
            raise ValueError(f"{model}: use scripts/prepare_models.py for the pinned checkpoint")
        for name in ("config.json", "model.safetensors"):
            path = directory / name
            with path.open("rb") as data:
                digest = hashlib.file_digest(data, "sha256").hexdigest()
            if source["sha256"][name] != digest:
                raise ValueError(f"{model}/{name}: checkpoint changed since preparation")
            hashes[str(path.relative_to(ROOT))] = digest
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cuda", "rocm", "xpu", "mps", "cpu"), required=True)
    parser.add_argument("--gpu", required=True, help="expected GPU-name substring")
    parser.add_argument("--torch-version", required=True, help="exact version including vendor suffix")
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=list(MODELS))
    parser.add_argument("--inference-only", action="store_true", help="prefill and stateless one-token SmolLM2; no training")
    parser.add_argument("--allow-integrated-gpu", action="store_true")
    parser.add_argument("--precisions", nargs="+", choices=("strict", "accelerated"), default=["strict", "accelerated"])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--collect", action="store_true", help="measure only after all requested qualification pairs pass")
    args = parser.parse_args()
    if ".".join(map(str, sys.version_info[:3])) != PYTHON_VERSION:
        parser.error(f"use Python {PYTHON_VERSION}: bash scripts/setup.sh <wheel-backend>")
    if args.inference_only and any(model not in SMOLLM2_REVISIONS for model in args.models):
        parser.error("--inference-only currently supports SmolLM2 workloads")
    if args.replicates < 3:
        parser.error("collection requires at least three fresh processes per condition")
    if len(set(args.models)) != len(args.models) or len(set(args.precisions)) != len(args.precisions):
        parser.error("duplicate models or precision classes")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        parser.error("commit source changes before qualification/collection")
    overrides = [key for key in os.environ if key.startswith("MEGANEURA_") or key in (
        "INFERENA_MEGANEURA_PATH", "INFERENA_PROFILE_DIR", "INFERENA_DRY_RUN", "INFERENA_NSYS",
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
    import torch
    try:
        check_torch_identity(torch.__version__, torch.version.git_version, args.torch_version)
    except ValueError as error:
        parser.error(str(error))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    try:
        hashes = input_hashes(args.models)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"prepare local pinned models first: {error}")
    dependency = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["dependencies"]["meganeura"]
    destination.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHON=sys.executable, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               INFERENA_REQUIRE_LOCAL_WEIGHTS="1", INFERENA_TORCH_BACKEND=args.backend)
    env.pop("VIRTUAL_ENV", None)
    manifest = {
        "protocol": "p3hpc-paired-campaign-v3", "source": revision,
        "model_revisions": {name: SMOLLM2_REVISIONS[name] for name in args.models if name in SMOLLM2_REVISIONS},
        "meganeura": dependency, "python": sys.version, "packages": packages,
        "torch": {"version": torch.__version__, "git_version": torch.version.git_version,
                  "build_config": torch.__config__.show()},
        "args": {**vars(args), "results_dir": str(destination)}, "sha256": hashes,
        "device_selection": {key: env[key] for key in (
            "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "VK_ICD_FILENAMES",
            "MESA_VK_DEVICE_SELECT", "HSA_OVERRIDE_GFX_VERSION",
            "ONEAPI_DEVICE_SELECTOR", "ZE_AFFINITY_MASK", "SYCL_CACHE_PERSISTENT",
        ) if key in env},
        "runs": [], "status": "in-progress",
    }

    def save():
        (destination / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    print(f"Campaign: {destination}", flush=True)
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
                            if args.inference_only:
                                command.append("--inference-only")
                            if args.allow_integrated_gpu:
                                command.append("--allow-integrated-gpu")
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
        if input_hashes(args.models) != hashes:
            raise ValueError("input changed during collection")
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "incomplete"
        manifest["error"] = str(error)
        if manifest["runs"] and manifest["runs"][-1]["status"] == "running":
            manifest["runs"][-1]["status"] = "failed"
        raise
    finally:
        save()
    print(f"Complete: {destination / 'campaign.json'}")


if __name__ == "__main__":
    main()
