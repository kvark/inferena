#!/usr/bin/env python3
"""Validated paired P3HPC collection. Generated evidence stays outside Git."""

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
import statistics
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny")
SMOLLM2_PINS = json.loads((ROOT / "models/smollm2-revisions.json").read_text())
SMOLLM2_REVISIONS = {name: pin["revision"] for name, pin in SMOLLM2_PINS.items()}
SUPPORTED_MODELS = (*MODELS, "SmolLM2-360M", "SmolLM2-1.7B")
PHASES = ("inference", "latency", "training")
PYTHON_VERSION = (ROOT / ".python-version").read_text().strip()
TORCH_VERSION = "2.13.0"
TORCH_REVISION = "cf30153c4c131c8164ee7798e5022d810682e2cb"
GRADIENT_LIMIT = 0.05
ACCELERATED_SAMPLE_LIMIT = 0.10
TUNE_SECONDS = 60.0
TUNE_SCRATCH_BYTES = 1024**3
COMPILE_SECONDS = 120.0
WARMUP_SECONDS = 2.0
SYNTHETIC_PARAMETER_INIT = "name-index-uniform-v1"
SDPA_BACKENDS = {"MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "OVERRIDEABLE"}


class NumericalMismatch(ValueError):
    pass


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


def conditions(backend, max_autotune=False, graph_ablation=False, no_graphs=False, eager=False):
    replay = backend in ("cuda", "rocm", "xpu") and not no_graphs
    mode = "eager" if eager else "default"
    configs = [(mode, replay)]
    if graph_ablation and replay:
        configs.insert(0, (mode, False))
    if max_autotune:
        configs.append(("max-autotune", replay))
    return configs


def reference_sdpa_policy(args, model):
    if override := getattr(args, "sdpa", None):
        return override
    return "math" if args.backend == "xpu" else "auto"


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return seconds


def archive_results(destination):
    archive = destination.parent / "latest.tgz"
    with tempfile.TemporaryDirectory(prefix=".p3hpc-archive-", dir=destination.parent) as temporary:
        staged = shutil.make_archive(str(Path(temporary) / "latest"), "gztar",
                                     root_dir=destination.parent, base_dir=destination.name)
        Path(staged).replace(archive)
    return archive


def _replicated_candidate(record):
    validation = record["validation"]
    return (
        record["precision"]["reduced_precision_allowed"]
        and validation.get("forward_valid") is True
        and validation.get("gradients_available") is True
        and math.isfinite(validation.get("total_gradient_relative_error", math.inf))
        and math.isfinite(validation.get("parameter_gradient_relative_l2_error", math.inf))
        and validation["total_gradient_relative_error"] < ACCELERATED_SAMPLE_LIMIT
        and validation["parameter_gradient_relative_l2_error"] < ACCELERATED_SAMPLE_LIMIT
    )


def duration_seconds(duration):
    return duration["secs"] + duration["nanos"] / 1e9


def check_native_search(session, seconds):
    search = session.get("search")
    if not search:
        raise ValueError("native session did not use calibrated construction")
    options = search["options"]
    tuning = options["tuning"]
    memory = session["memory_budget"]
    if (duration_seconds(options["max_time"]) != seconds
            or options["max_graphs"] != 16 or options["max_programs"] != 64
            or options["warmup_runs"] != 2
            or duration_seconds(options["warmup_time"]) != 0.25
            or memory["plan_fraction_of_available"] != 0.75
            or options["max_plan_bytes"] != (memory["device_budget_bytes"] - memory["device_usage_bytes"]) // 4 * 3
            or options["max_plan_bytes"] <= 0
            or tuning["scope"] != "All" or tuning["max_classes"] != 2 * sys.maxsize + 1
            or duration_seconds(tuning["max_time"]) != seconds
            or tuning["max_scratch_bytes"] != TUNE_SCRATCH_BYTES):
        raise ValueError("native construction limits differ from the declared policy")
    trials = search["trials"]
    if not trials or not 0 <= search["selected"] < len(trials):
        raise ValueError("native construction has no selected program")
    selected = trials[search["selected"]]
    if selected["outcome"]["qualified"] is not True or not selected["kernel_tuning"]:
        raise ValueError("native selected program was not tuned and qualified")
    for trial in trials:
        kernel = trial["kernel_tuning"]
        if kernel is None:
            continue
        if (kernel["options"]["scope"] != "All"
                or kernel["options"]["max_classes"] != tuning["max_classes"]
                or kernel["options"]["max_scratch_bytes"] != TUNE_SCRATCH_BYTES
                or not 0 <= duration_seconds(kernel["options"]["max_time"]) <= seconds
                or kernel["class_limit_reached"]
                or not 0 <= kernel["visited_classes"] <= kernel["eligible_classes"]
                or kernel["visited_classes"] < kernel["eligible_classes"] and not kernel["time_budget_exhausted"]):
            raise ValueError("native kernel search stopped outside the declared limits")
    qualification = session["qualification"]
    if (qualification["policy"] != "fixed-full-tensor-v4"
            or qualification["rtol"] != 1e-4 or qualification["atol"] != 1e-6
            or qualification["accelerated_gradient_rtol"] != 0.01
            or qualification["qualified_calls"] < 2 or qualification["output_elements"] <= 0
            or session["mode"] == "Training" and qualification["gradient_elements"] <= 0):
        raise ValueError("native construction lacks full-output/gradient qualification")


def check_pair(records, args, mode, graphs, count, revision, diagnostic=False,
               replicated=False, precision=None, phases=None, oracle=False):
    requested_phases = PHASES[:2] if args.inference_only else PHASES
    phases = requested_phases if phases is None else tuple(phases)
    if set(phases) - set(requested_phases):
        raise ValueError("unrequested phase")
    by_engine = {record["framework"]: record for record in records}
    if len(records) != 2 or set(by_engine) != {"pytorch", "meganeura"}:
        raise ValueError("both engine records are required")
    for engine, record in by_engine.items():
        if record["status"] not in ("ok", "partial"):
            raise ValueError(f"{engine} failed: {record.get('error', record.get('reason'))}")
        if record["status"] == "partial" and (
                engine != "pytorch" or record.get("execution", {}).get("failure", {}).get("kind")
                not in ("numerical", "capture")):
            raise ValueError("partial result lacks a classified failure")
    if len({record["model"] for record in records}) != 1:
        raise ValueError("engine records have different models")
    for engine, record in by_engine.items():
        validation = record["validation"]
        if record["protocol"]["training_requested"] != (not args.inference_only):
            raise ValueError("unexpected training scope")
        if args.inference_only and (validation.get("training_valid") is not None or record["timings"].get("training_ms") is not None):
            raise ValueError("inference-only run claims training results")
        if record["protocol"].get("diagnostic") is not (diagnostic or oracle and engine == "pytorch"):
            raise ValueError("diagnostic and benchmark samples must not be mixed")
        if not revision.startswith(record["benchmark_rev"]):
            raise ValueError("source changed during collection")
        if record["protocol"]["warmup_runs"] != 5:
            raise ValueError("unexpected warmup count")
        protocol = record["protocol"]
        expected = "inferena-paper-v3" if engine == "meganeura" else "inferena-graph-replay-v7"
        if protocol.get("name") != expected or protocol.get("warmup_seconds") != WARMUP_SECONDS:
            raise ValueError("runner did not declare the workload warmup policy")
        if protocol.get("synthetic_parameter_init") != SYNTHETIC_PARAMETER_INIT:
            raise ValueError("runner did not declare the synthetic parameter policy")
        for phase in phases:
            warmup = protocol.get("warmup", {}).get(phase, {})
            seconds = warmup.get("seconds", 0)
            if (warmup.get("runs", 0) < 5 or not math.isfinite(seconds)
                    or seconds < WARMUP_SECONDS):
                raise ValueError(f"{engine} {phase} did not complete its workload warmup")
        for phase in phases:
            samples = record["timing_samples_ms"][phase]
            expected_count = 1 if oracle and engine == "pytorch" else count
            if len(samples) != expected_count or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError(f"{engine} has invalid {phase} samples")
    pt, mg = by_engine["pytorch"], by_engine["meganeura"]
    check_torch_identity(pt["torch_version"], pt["environment"].get("torch_git_version"), args.torch_version)
    if pt["environment"]["python_version"] != PYTHON_VERSION:
        raise ValueError(f"use the campaign's Python {PYTHON_VERSION}")
    if pt["backend"].split()[0].lower() != args.backend:
        raise ValueError(f"unexpected reference backend: {pt['backend']}")
    triton_backend = {"cuda": "nvidia", "rocm": "amd", "xpu": "intel"}.get(args.backend)
    if triton_backend and pt["environment"].get("triton_backend") != triton_backend:
        raise ValueError("compiler driver differs from the requested reference backend")
    if not gpu_matches(args.gpu, mg["gpu_name"]):
        raise ValueError(f"unexpected Meganeura GPU: {mg['gpu_name']}")
    if args.backend in ("cuda", "rocm", "xpu") and not gpu_matches(args.gpu, pt["gpu_name"]):
        raise ValueError(f"unexpected PyTorch GPU: {pt['gpu_name']}")
    expected_class = ("strict-f32" if precision == "strict" else "reduced-input-f32-accumulate")
    if precision is not None and any(record["precision"]["comparison_class"] != expected_class
                                     for record in (mg, pt)):
        raise ValueError("requested arithmetic contract did not execute")
    if (mg.get("optimizer", {}).get("measured_construction") is not True
            or mg["optimizer"].get("mode") != "egglog-outlined"):
        raise ValueError("Meganeura calibrated egglog construction must be enabled")
    strict = not mg["precision"]["reduced_precision_allowed"]
    if strict and (mg["precision"].get("cooperative_matrix_policy") != "NativeF32"
                   or mg["precision"].get("native_f32_cooperative_matrix_permitted") is not True
                   or mg["precision"].get("f16_cooperative_matrix_permitted") is not False):
        raise ValueError("strict arithmetic must permit native-f32, but not f16-input, cooperative tiles")
    sessions = mg["optimizer"].get("sessions", [])
    modes = [session["mode"] for session in sessions]
    if (modes.count("Training") != int(not args.inference_only) or modes.count("Inference") not in (1, 2)
            or set(modes) - {"Training", "Inference"}):
        raise ValueError("missing native session preparation evidence")
    for session in sessions:
        if session["cooperative_matrix_policy"] != ("NativeF32" if strict else "Auto"):
            raise ValueError("native session used the wrong cooperative policy")
        check_native_search(session, getattr(args, "tune_seconds", TUNE_SECONDS))
    execution = pt["execution"]
    sdpa = "math" if oracle else reference_sdpa_policy(args, pt["model"])
    if execution["sdpa_policy"] != sdpa:
        raise ValueError("reference attention policy differs from the declared backend configuration")
    expected_sdpa = {"auto": SDPA_BACKENDS, "math": {"MATH"}, "efficient": {"EFFICIENT_ATTENTION"}}[sdpa]
    if set(execution["sdpa_enabled_backends"]) != expected_sdpa:
        raise ValueError("active attention backends differ from the declared policy")
    sdpa_compile = "eager" if mode == "eager" else "compiled"
    if execution.get("sdpa_compile") != sdpa_compile:
        raise ValueError("reference SDPA compilation differs from the declared model/backend policy")
    if args.backend in ("cuda", "rocm", "xpu") and execution.get("stream_policy") != "single dedicated preparation/run stream":
        raise ValueError("preparation and execution must share the declared stream policy")
    if execution["requested_mode"] != mode or execution["compiled"] != (mode != "eager"):
        raise ValueError("requested compiler mode did not execute")
    if execution["compile_budget_seconds"] != getattr(args, "compile_seconds", COMPILE_SECONDS):
        raise ValueError("reference compilation budget changed")
    if execution.get("compile_budget_enforced") is not True:
        raise ValueError("reference compilation watchdog was not active")
    if mode != "eager":
        options = execution["compiler_options"]
        if (options["max_autotune"] != (mode == "max-autotune")
                or options["coordinate_descent_tuning"] != (mode == "max-autotune")
                or options["max_autotune_gemm"] or options["max_autotune_pointwise"]
                or options["triton.cudagraphs"]):
            raise ValueError("reference search/replay options differ from the declared policy")
    if execution["graph_replay"]["requested"] != graphs:
        raise ValueError("unexpected graph configuration")
    for phase in phases:
        report = execution["graph_replay"]["phases"][phase]
        expected = "captured-and-validated" if graphs else "validated-uncaptured"
        if report["status"] != expected:
            raise ValueError(f"{phase} did not execute the requested capture mode")
        if graphs:
            expected_api = "torch.xpu.XPUGraph" if args.backend == "xpu" else "torch.cuda.CUDAGraph"
            if report["api"] != expected_api:
                raise ValueError("requested replay backend did not execute")
        validation = report["validation"]
        repeats = 8 if phase == "training" and pt["precision"]["reduced_precision_allowed"] else 2
        if (validation.get("policy") != "fixed-full-tensor-v4"
                or validation.get("output_metric") != "per-tensor RMS and maximum absolute error"
                or validation.get("rtol") != 1e-4 or validation.get("atol") != 1e-6
                or validation.get("accelerated_gradient_rtol") != 0.01
                or validation.get("uncaptured_calls") != repeats + 1
                or validation.get("uncaptured_repeats") != repeats
                or validation.get("consecutive_replays") != (2 if graphs else 0)):
            raise ValueError(f"{phase} did not use the declared qualification policy")
    for engine, record in by_engine.items():
        validation = record["validation"]
        valid = all(validation.get(key) is True for key in ("comparison_performed", "forward_valid"))
        if "training" in phases:
            valid = valid and (validation.get("training_valid") is True
                              or replicated and _replicated_candidate(record))
        if phases and (not valid or validation.get("reference_framework") != "pytorch"):
            raise NumericalMismatch(f"{engine} failed the requested numerical gates: {validation}")
    return by_engine


def assess_phases(records, args, mode, graphs, count, revision, preparation, *, precision, oracle=False):
    """Classify numerical failures, but let missing receipts and execution faults stop collection."""
    pair = {record["framework"]: record for record in records}
    if len(records) != 2 or set(pair) != {"pytorch", "meganeura"}:
        raise ValueError("both engine records are required")
    native, reference = pair["meganeura"], pair["pytorch"]
    if native["status"] != "ok":
        raise ValueError(f"meganeura failed: {native.get('error', native.get('reason'))}")
    failure = reference.get("execution", {}).get("failure") or preparation.get("qualification_failure")
    if reference["status"] != "ok" and (
            not failure or failure.get("kind") not in ("numerical", "capture")
            or failure.get("phase") not in PHASES):
        raise ValueError(f"unclassified pytorch failure: {reference.get('error', reference.get('reason'))}")
    phases = PHASES[:2] if args.inference_only else PHASES
    outcomes = {}
    if reference["status"] == "error":
        # An inference failure has no usable result. The sidecar is written
        # only by the numerical/capture gate, never by an arbitrary exception.
        if failure["phase"] != "inference":
            raise ValueError("unexpected error after a completed inference phase")
        return {phase: {"status": "failed" if phase == "inference" else "not-attempted",
                        "failure": failure} for phase in phases}, pair
    check_pair(records, args, mode, graphs, count, revision, phases=(), precision=precision, oracle=oracle)
    for phase in phases:
        report = reference["execution"]["graph_replay"]["phases"][phase]
        if report["status"] in ("failed", "not-attempted"):
            if not failure or reference["status"] != "partial":
                raise ValueError("missing phase without a classified failure")
            if (report["status"] == "failed") != (phase == failure["phase"]):
                raise ValueError("failed phase disagrees with the failure receipt")
            if reference["timing_samples_ms"].get(phase) is not None:
                raise ValueError("unqualified phase claims timing samples")
            outcomes[phase] = {"status": report["status"], "failure": failure}
            continue
        try:
            check_pair(records, args, mode, graphs, count, revision, phases=(phase,),
                       precision=precision, replicated=args.collect and not oracle, oracle=oracle)
            outcomes[phase] = {"status": "valid"}
        except NumericalMismatch as error:
            outcomes[phase] = {"status": "mismatch", "error": str(error)}
    return outcomes, pair


def eager_diagnostic(command, env, folder, model, args, count, revision, precision):
    """One fresh, uncaptured reference; never rewrite the failed primary pair."""
    diagnostic = folder / "diagnostic-eager"
    diagnostic.mkdir()
    command = list(command)
    for flag, value in (("-f", "pytorch"), ("--measurement-runs", "1"), ("--results-dir", str(diagnostic))):
        command[command.index(flag) + 1] = value
    environment = dict(env, INFERENA_TORCH_MODE="eager", INFERENA_GRAPH_REPLAY="0", INFERENA_SDPA="math",
                       INFERENA_REFERENCE_DIAGNOSTIC="1",
                       INFERENA_PREPARATION_REPORT=str(diagnostic / "torch-preparation.json"))
    with (diagnostic / "runner.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    native_path = folder / f"{model}_meganeura.json"
    reference_path = diagnostic / f"{model}_pytorch.json"
    reference = json.loads(reference_path.read_text())
    preparation_path = diagnostic / "torch-preparation.json"
    preparation = json.loads(preparation_path.read_text()) if preparation_path.exists() else {}
    if result.returncode and reference["status"] != "error":
        raise RuntimeError(f"eager diagnostic runner failed; inspect {diagnostic / 'runner.log'}")
    records = [json.loads(native_path.read_text()), reference]
    if reference["status"] in ("ok", "partial"):
        harness = ROOT / "target/release" / ("inferena.exe" if sys.platform == "win32" else "inferena")
        records = json.loads(subprocess.check_output(
            [str(harness), "--compare-results", str(native_path), str(reference_path)],
            cwd=ROOT, text=True, encoding="utf-8"))
        (diagnostic / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    phases, _ = assess_phases(records, args, "eager", False, count, revision, preparation,
                              precision=precision, oracle=True)
    return {"path": "diagnostic-eager", "command": command, "returncode": result.returncode,
            "mode": "eager", "graph_replay": False, "sdpa": "math", "diagnostic": True,
            "status": "valid" if all(p["status"] == "valid" for p in phases.values()) else "failed",
            "phases": phases, "timing_substituted": False}


def validate_replicated_gradients(groups, replicates):
    reports = []
    for (precision, model, mode, graphs), pairs in groups.items():
        identity = {"precision": precision, "model": model, "mode": mode, "graph_replay": graphs}
        if len(pairs) != replicates:
            reports.append({**identity, "status": "incomplete", "valid_replicates": len(pairs),
                            "required_replicates": replicates})
            continue
        validations = [pair["meganeura"]["validation"] for pair in pairs]
        metrics = {
            "parameter_gradient_relative_l2": [
                item["parameter_gradient_relative_l2_error"] for item in validations
            ],
            "total_gradient_relative": [
                item["total_gradient_relative_error"] for item in validations
            ],
        }
        sample_limit = ACCELERATED_SAMPLE_LIMIT if precision == "accelerated" else GRADIENT_LIMIT
        accepted = all(
            all(math.isfinite(value) and value < sample_limit for value in values)
            and statistics.median(values) < GRADIENT_LIMIT
            for values in metrics.values()
        )
        reports.append({
            "status": "pass" if accepted else "fail",
            **identity,
            "sample_limit": sample_limit, "median_limit": GRADIENT_LIMIT,
            "metrics": {
                name: {"samples": values, "median": statistics.median(values)}
                for name, values in metrics.items()
            },
        })
    return {
        "policy": "replicated-gradient-median-v1",
        "status": "pass" if all(report["status"] == "pass" for report in reports) else "fail",
        "groups": reports,
    }


def input_hashes(models):
    """Refuse unpinned checkpoints, including accidentally mixed base/Instruct files."""
    hashes = {"Cargo.lock": hashlib.sha256((ROOT / "Cargo.lock").read_bytes()).hexdigest()}
    for model in models:
        if model not in SMOLLM2_REVISIONS:
            continue
        directory = ROOT / "models" / model
        source = json.loads((directory / "source.json").read_text())
        pin = SMOLLM2_PINS[model]
        if (source.get("repo") != f"HuggingFaceTB/{model}"
                or source.get("revision") != pin["revision"]
                or source.get("sha256") != pin["sha256"]):
            raise ValueError(f"{model}: use scripts/prepare_models.py for the pinned checkpoint")
        for name in ("config.json", "model.safetensors"):
            path = directory / name
            with path.open("rb") as data:
                digest = hashlib.file_digest(data, "sha256").hexdigest()
            if pin["sha256"][name] != digest:
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
    search = parser.add_mutually_exclusive_group()
    search.add_argument("--max-autotune", action="store_true", default=False,
                        help="add optional PyTorch search under the same compilation deadline")
    search.add_argument("--no-max-autotune", dest="max_autotune", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--graph-ablation", action="store_true", help="also measure the uncaptured reference")
    parser.add_argument("--no-graphs", action="store_true", help="explicitly omit whole-phase replay; recorded as an override")
    parser.add_argument("--eager", action="store_true", help="explicit eager reference; never an automatic compiler fallback")
    parser.add_argument("--sdpa", choices=("auto", "math", "efficient"),
                        help="override PyTorch attention for all models; default: math on XPU, auto otherwise")
    parser.add_argument("--tune-seconds", type=positive_seconds, default=TUNE_SECONDS,
                        help="Meganeura soft construction deadline per session (default: 60 s), including initialization/qualification")
    parser.add_argument("--compile-seconds", type=positive_seconds, default=COMPILE_SECONDS,
                        help="PyTorch compilation/first-specialization deadline (default: 120 s)")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--collect", dest="collect", action="store_true", default=True, help="validated measurement (default)")
    stage.add_argument("--qualify-only", dest="collect", action="store_false", help="run correctness gates without publication samples")
    parser.add_argument("--offline", action="store_true", help="require all pinned models to be prepared already")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.eager and args.max_autotune:
        parser.error("--eager and --max-autotune select different reference policies")
    if ".".join(map(str, sys.version_info[:3])) != PYTHON_VERSION:
        parser.error(f"use Python {PYTHON_VERSION}: bash scripts/setup.sh <wheel-backend>")
    if args.inference_only and any(model not in SMOLLM2_REVISIONS for model in args.models):
        parser.error("--inference-only currently supports SmolLM2 workloads")
    if args.collect and args.replicates < 3:
        parser.error("collection requires at least three fresh processes per condition")
    if len(set(args.models)) != len(args.models) or len(set(args.precisions)) != len(args.precisions):
        parser.error("duplicate models or precision classes")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        parser.error("commit source changes before qualification/collection")
    overrides = [key for key in os.environ if key.startswith("MEGANEURA_") or key in (
        "INFERENA_MEGANEURA_PATH", "INFERENA_PROFILE_DIR", "INFERENA_DRY_RUN", "INFERENA_NSYS",
        "TORCH_LOGS", "TORCH_TRACE", "CARGO_TARGET_DIR",
        "INFERENA_TORCH_MODE", "INFERENA_GRAPH_REPLAY", "INFERENA_CUDA_GRAPHS",
        "INFERENA_TUNE_SECONDS", "INFERENA_COMPILE_SECONDS", "INFERENA_PREPARATION_REPORT", "INFERENA_BUDGET_ENFORCED",
        "INFERENA_SDPA", "INFERENA_REFERENCE_DIAGNOSTIC",
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
                if model in SMOLLM2_REVISIONS:
                    prepare_model(model)
        hashes = input_hashes(args.models)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"prepare local pinned models first: {error}")
    dependency = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["dependencies"]["meganeura"]
    declared_conditions = conditions(args.backend, max_autotune=True, graph_ablation=True)
    selected_conditions = conditions(args.backend, args.max_autotune, args.graph_ablation, args.no_graphs, args.eager)
    omitted_conditions = [config for config in declared_conditions if config not in selected_conditions]
    sdpa = {model: reference_sdpa_policy(args, model) for model in args.models}
    destination.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHON=Path(sys.executable).as_posix(), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               INFERENA_BASH=runner_bash(), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               INFERENA_REQUIRE_LOCAL_WEIGHTS="1", INFERENA_TORCH_BACKEND=args.backend)
    env.pop("VIRTUAL_ENV", None)
    manifest = {
        "protocol": "p3hpc-paired-campaign-v14", "source": revision,
        "synthetic_parameter_init": SYNTHETIC_PARAMETER_INIT,
        "model_revisions": {name: SMOLLM2_REVISIONS[name] for name in args.models if name in SMOLLM2_REVISIONS},
        "meganeura": dependency, "python": sys.version, "packages": packages,
        "torch": {"version": torch.__version__, "git_version": torch.version.git_version,
                  "build_config": torch.__config__.show()},
        "args": {**vars(args), "results_dir": str(destination)}, "sha256": hashes,
        "native_policy": {"measured_construction": True, "optimizer": "egglog-outlined",
                          "scope": "All", "class_limit": None,
                          "max_scratch_bytes": TUNE_SCRATCH_BYTES,
                          "kernel_seconds_per_program": args.tune_seconds,
                          "max_graphs": 16, "max_programs": 64, "plan_fraction_of_available": 0.75,
                          "warmup_pairs": 2, "warmup_seconds": 0.25,
                          "qualification": "fixed-full-tensor-v4",
                          "search_seconds_per_session": args.tune_seconds, "strict_coop": "NativeF32"},
        "reference_compile_seconds": args.compile_seconds,
        "warmup": {"minimum_runs": 5, "minimum_seconds": WARMUP_SECONDS},
        "reference_sdpa_policy": sdpa,
        "reference_sdpa_compile": {
            model: "eager" if args.eager else "compiled"
            for model in args.models
        },
        "failure_policy": {"recoverable": ["numerical", "capture", "cross-engine-mismatch"],
                           "eager_diagnostic_attempts": 1, "eager_sdpa": "math",
                           "timing_substitution": False, "retry_failed_primary": False},
        "reference_conditions": {
            "declared": [{"mode": mode, "graph_replay": graphs}
                         for mode, graphs in declared_conditions],
            "selected": [{"mode": mode, "graph_replay": graphs}
                         for mode, graphs in selected_conditions],
            "omitted": [{"mode": mode, "graph_replay": graphs,
                         "reason": "not selected; omission alone is not a backend failure"}
                        for mode, graphs in omitted_conditions],
            "coverage": "explicit-reference-override" if args.no_graphs or args.eager or args.sdpa is not None else "primary",
        },
        "device_selection": {key: env[key] for key in (
            "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "VK_ICD_FILENAMES",
            "VK_DRIVER_FILES", "VK_LOADER_DRIVERS_SELECT", "VK_LOADER_DRIVERS_DISABLE",
            "MESA_VK_DEVICE_SELECT", "HSA_OVERRIDE_GFX_VERSION", "ROCR_VISIBLE_DEVICES",
            "ONEAPI_DEVICE_SELECTOR", "ZE_AFFINITY_MASK", "SYCL_CACHE_PERSISTENT",
        ) if key in env},
        "runtime_overrides": {key: env[key] for key in (
            "HSA_OVERRIDE_GFX_VERSION", "HSA_ENABLE_SDMA", "AMD_SERIALIZE_KERNEL",
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
              f"{'validated collection' if args.collect else 'qualification only'}", flush=True)
        stages = ([('measurement', args.replicates, 20)] if args.collect
                  else [('qualification', 1, 1)])
        measurement_pairs = {(precision, model, mode, graphs): []
                             for precision in args.precisions for model in args.models
                             for mode, graphs in selected_conditions}
        sequence = 0
        for stage, replicates, count in stages:
            for replicate in range(replicates):
                for precision in args.precisions:
                    for model in args.models:
                        configs = selected_conditions
                        offset = replicate % len(configs)
                        for mode, graphs in configs[offset:] + configs[:offset]:
                            label = f"{mode}-graph{int(graphs)}"
                            folder = destination / stage / f"r{replicate + 1}" / precision / model / label
                            folder.mkdir(parents=True)
                            order = "pytorch,meganeura" if sequence % 2 == 0 else "meganeura,pytorch"
                            sequence += 1
                            command = [env["INFERENA_BASH"], (ROOT / "run.sh").as_posix(), "-m", model, "-f", order,
                                       "--platform", args.gpu,
                                       "--warmup-runs", "5", "--measurement-runs", str(count),
                                       "--results-dir", str(folder)]
                            if precision == "strict":
                                command.append("--strict")
                            if args.inference_only:
                                command.append("--inference-only")
                            if args.allow_integrated_gpu:
                                command.append("--allow-integrated-gpu")
                            run = {"path": str(folder.relative_to(destination)), "command": command,
                                   "mode": mode, "graphs": graphs, "model": model,
                                   "precision": precision, "replicate": replicate + 1,
                                   "preparation_policy": "native-tuned/reference-" + mode, "status": "running"}
                            manifest["runs"].append(run)
                            save()
                            print(run["path"], flush=True)
                            run_env = dict(env, INFERENA_TORCH_MODE=mode,
                                           INFERENA_SDPA=sdpa[model], INFERENA_GRAPH_REPLAY=str(int(graphs)),
                                           MEGANEURA_TUNE="1", INFERENA_TUNE_SECONDS=str(args.tune_seconds),
                                           INFERENA_COMPILE_SECONDS=str(args.compile_seconds),
                                           INFERENA_PREPARATION_REPORT=str(folder / "torch-preparation.json"))
                            with (folder / "runner.log").open("w", encoding="utf-8") as log:
                                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                        env=run_env)
                            run["returncode"] = result.returncode
                            if result.returncode:
                                raise RuntimeError(f"runner failed; inspect {folder / 'runner.log'}")
                            records = json.loads((folder / f"{model}_summary.json").read_text())
                            preparation_path = folder / "torch-preparation.json"
                            preparation = json.loads(preparation_path.read_text()) if preparation_path.exists() else {}
                            phases, pair = assess_phases(records, args, mode, graphs, count, revision,
                                                       preparation, precision=precision)
                            if not dependency["rev"].startswith(pair["meganeura"]["framework_rev"]):
                                raise ValueError("Meganeura dependency revision changed")
                            if pair["meganeura"]["environment"].get("gpu_device_id") != native["device_id"]:
                                raise ValueError("Meganeura ran on a different device than the preflight selected")
                            run["phases"] = phases
                            run["status"] = ("valid" if all(p["status"] == "valid" for p in phases.values())
                                             else "partial" if any(p["status"] == "valid" for p in phases.values())
                                             else "failed")
                            save()
                            if run["status"] != "valid" and mode != "eager":
                                run["eager_diagnostic"] = {"path": "diagnostic-eager", "status": "running"}
                                save()
                                run["eager_diagnostic"] = eager_diagnostic(
                                    command, run_env, folder, model, args, count, revision, precision)
                            if stage == "measurement" and phases.get("training", {}).get("status") == "valid":
                                key = (precision, model, mode, graphs)
                                measurement_pairs.setdefault(key, []).append(pair)
                            save()
        if args.collect and not args.inference_only:
            manifest["replicated_gradient_validation"] = validate_replicated_gradients(
                measurement_pairs, args.replicates
            )
            save()
        if input_hashes(args.models) != hashes:
            raise ValueError("input changed during collection")
        manifest["phase_coverage"] = {
            phase: {"planned": len(manifest["runs"]),
                    "valid": sum(run["phases"][phase]["status"] == "valid" for run in manifest["runs"])}
            for phase in (PHASES[:2] if args.inference_only else PHASES)
        }
        valid = (all(run["status"] == "valid" for run in manifest["runs"])
                 and manifest.get("replicated_gradient_validation", {}).get("status", "pass") == "pass")
        manifest["status"] = "complete" if valid else "complete-with-failures"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "incomplete"
        manifest["error"] = str(error)
        if manifest["runs"] and manifest["runs"][-1]["status"] == "running":
            manifest["runs"][-1]["status"] = "failed"
        if manifest["runs"]:
            diagnostic = manifest["runs"][-1].get("eager_diagnostic", {})
            if diagnostic.get("status") == "running":
                diagnostic.update(status="incomplete", error=str(error))
        raise
    finally:
        save()
        try:
            print(f"Archive: {archive_results(destination)}", flush=True)
        except OSError as error:
            print(f"Archive not updated: {error}; results remain in {destination}", file=sys.stderr)
    print(f"{manifest['status']}: {destination / 'campaign.json'}")
    if manifest["status"] != "complete":
        sys.exit(1)


if __name__ == "__main__":
    main()
