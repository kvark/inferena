#!/usr/bin/env python3
"""Qualification-first, paired P3HPC collection. Generated evidence stays outside Git."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
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


def gpu_matches(expected, actual):
    def normalize(name):
        return " ".join(re.sub(r"\((?:tm|r)\)|[™®]", "", name.casefold()).split())
    return normalize(expected) in normalize(actual)


def select_native_device(devices, expected):
    matches = [device for device in devices if device["available"]
               and not device["software_emulated"]
               and (expected is None or gpu_matches(expected, device["name"]))]
    if len(matches) != 1:
        names = ", ".join(device["name"] for device in devices)
        raise ValueError(f"need one matching native GPU, got {len(matches)} for {expected!r}: {names}; "
                         "select the intended device with --gpu and backend/loader visibility controls")
    return matches[0]


def runner_bash():
    """Use Git Bash on native Windows, never the unrelated WSL launcher."""
    shell = os.environ.get("INFERENA_BASH")
    if not shell and sys.platform == "win32":
        git = shutil.which("git")
        if git:
            shell = next((str(parent / "bin/bash.exe") for parent in Path(git).parents
                          if (parent / "bin/bash.exe").is_file()), None)
    shell = shell or shutil.which("bash")
    if not shell:
        raise RuntimeError("Bash not found; install Git for Windows or set INFERENA_BASH")
    if sys.platform == "win32":
        system = subprocess.check_output([shell, "-c", "uname -s"], text=True, encoding="utf-8").strip()
        if not system.startswith(("MINGW", "MSYS")):
            raise RuntimeError("native Windows needs Git Bash, not WSL; set INFERENA_BASH to Git's bash.exe")
    return shell


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
    if not gpu_matches(args.gpu, mg["gpu_name"]):
        raise ValueError(f"unexpected Meganeura GPU: {mg['gpu_name']}")
    if args.backend in ("cuda", "rocm", "xpu") and not gpu_matches(args.gpu, pt["gpu_name"]):
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


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, help="new directory; default: ../inferena-results/<host>-<UTC>-<source>")
    parser.add_argument("--backend", choices=("cuda", "rocm", "xpu", "mps", "cpu"), help="default: installed GPU backend; no CPU fallback")
    parser.add_argument("--gpu", help="expected GPU-name substring; default: reference GPU")
    parser.add_argument("--torch-version", help="exact vendor build; default: installed build, with common source/version checks")
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=list(MODELS))
    parser.add_argument("--inference-only", action="store_true", help="prefill and stateless one-token SmolLM2; no training")
    # The collector selects a particular device; legacy discrete-GPU preferences
    # must not override it. Keep accepting the old opt-in spelling.
    parser.add_argument("--allow-integrated-gpu", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--precisions", nargs="+", choices=("strict", "accelerated"), default=["strict", "accelerated"])
    parser.add_argument("--replicates", type=int, default=3)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--collect", dest="collect", action="store_true", default=True, help="qualify then measure (default)")
    stage.add_argument("--qualify-only", dest="collect", action="store_false", help="run correctness gates without publication samples")
    parser.add_argument("--offline", action="store_true", help="require all pinned models to be prepared already")
    return parser


def main():
    parser = create_parser()
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
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    args.torch_version = args.torch_version or importlib.metadata.version("torch")
    if importlib.metadata.version("torch") != args.torch_version:
        parser.error("this Python interpreter has a different PyTorch version")
    import torch
    try:
        check_torch_identity(torch.__version__, torch.version.git_version, args.torch_version)
    except ValueError as error:
        parser.error(str(error))
    probe_env = dict(os.environ)
    if args.backend:
        probe_env["INFERENA_TORCH_BACKEND"] = args.backend
    try:
        # Device queries/XPU qualification can create GPU contexts. Let this
        # process exit before collection so the controller holds no GPU memory.
        reference = json.loads(subprocess.check_output([
            sys.executable, "-c", "import json; from bench import detect_device, device_name, backend_name; "
            "d = detect_device(); print(json.dumps({'backend': backend_name(d).split()[0].lower(), 'gpu': device_name(d)}))",
        ], cwd=ROOT / "frameworks/pytorch", env=probe_env, text=True, encoding="utf-8"))
    except subprocess.CalledProcessError as error:
        parser.error(str(error))
    backend = reference["backend"]
    if backend == "cpu" and args.backend != "cpu":
        parser.error("no working GPU backend; fix setup or explicitly request --backend cpu")
    args.backend = backend
    if backend in ("cuda", "rocm", "xpu"):
        reference_gpu = reference["gpu"]
        args.gpu = args.gpu or reference_gpu
        if not gpu_matches(args.gpu, reference_gpu):
            parser.error(f"reference selected {reference_gpu}, not {args.gpu}; check device visibility")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())
    args.results_dir = args.results_dir or ROOT.parent / "inferena-results" / f"{host}-{stamp}-{revision[:8]}"
    destination = args.results_dir.resolve()
    if destination == ROOT or ROOT in destination.parents:
        parser.error("use a new results directory outside the checkout")
    if destination.exists():
        parser.error("results directory already exists; nothing was replaced")
    try:
        if not args.offline:
            from prepare_models import prepare_model
            for model in args.models:
                if model in SMOLLM2_REVISIONS and not (ROOT / "models" / model).exists():
                    prepare_model(model)
        hashes = input_hashes(args.models)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"prepare local pinned models first: {error}")
    dependency = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["dependencies"]["meganeura"]
    destination.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHON=Path(sys.executable).as_posix(), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               INFERENA_BASH=runner_bash(), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
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
            "VK_DRIVER_FILES", "VK_LOADER_DRIVERS_SELECT", "VK_LOADER_DRIVERS_DISABLE",
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
        subprocess.run(["cargo", "build", "--release", "--locked", "-p", "inferena-harness", "-p", "inferena-meganeura"],
                       cwd=ROOT, env=dict(os.environ, CARGO_BUILD_JOBS=os.environ.get("CARGO_BUILD_JOBS", "1")), check=True)
        executable = ROOT / "target/release" / ("inferena-meganeura.exe" if sys.platform == "win32" else "inferena-meganeura")
        devices = json.loads(subprocess.check_output([str(executable), "--list-devices"], cwd=ROOT, env=env, text=True, encoding="utf-8"))
        native = select_native_device(devices, args.gpu)
        args.gpu = args.gpu or native["name"]
        env["MEGANEURA_DEVICE_ID"] = str(native["device_id"])
        manifest["args"]["gpu"] = args.gpu
        manifest["native_device"] = native
        manifest["device_selection"]["MEGANEURA_DEVICE_ID"] = env["MEGANEURA_DEVICE_ID"]
        save()
        print(f"Reference: {args.backend}, torch {args.torch_version}; native: {native['name']}", flush=True)
        print(f"Models: {', '.join(args.models)}; precision: {', '.join(args.precisions)}; "
              f"{'qualification then collection' if args.collect else 'qualification only'}", flush=True)
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
                            command = [env["INFERENA_BASH"], (ROOT / "run.sh").as_posix(), "-m", model, "-f", order,
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
                            if pair["meganeura"]["environment"].get("gpu_device_id") != native["device_id"]:
                                raise ValueError("Meganeura ran on a different device than the preflight selected")
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
