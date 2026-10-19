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
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib

from collection import keep_awake, run_recorded, write_json
from precision import check_arithmetic_policy, check_precision_qualification
from quality import LM_POLICY, MEAN_KL_LIMIT, TOKEN_KL_LIMIT, check_language_model_report
from training_gate import CURVE_POLICY, FIXTURE, FIXTURE_SHA256, load_fixture, workload_for

ROOT = Path(__file__).resolve().parents[1]
PYTORCH_PROTOCOL = (ROOT / "frameworks/pytorch/PROTOCOL").read_text().strip()
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
TUNE_SECONDS = 30.0
MAX_SUBMISSION_CHUNKS = 64
TUNE_SCRATCH_BYTES = 1024**3
KERNEL_MEMO_POLICY = "campaign-kernel-memo-v1"
COMPILE_SECONDS = 120.0
RUN_SECONDS = 1200.0
WARMUP_SECONDS = 10.0
SYNTHETIC_PARAMETER_INIT = "name-index-uniform-v1"
SDPA_BACKENDS = {"MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "OVERRIDEABLE"}


class NumericalMismatch(ValueError):
    pass


def reference_kernel_benchmarking(backend, mode):
    if mode == "eager":
        return None
    profiler = backend == "xpu"
    return {"implementation": "TorchProfilerBenchmarker" if profiler else "InductorBenchmarker",
            "use_experimental_benchmarker": True, "use_torch_profiler_benchmarker": profiler}


def cpu_frequency_policies():
    """Read Linux policy settings, not transient sampled clock frequencies."""
    policies = {}
    for directory in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        settings = {}
        for name in ("related_cpus", "scaling_driver", "scaling_governor",
                     "energy_performance_preference", "scaling_min_freq", "scaling_max_freq"):
            try:
                settings[name] = (directory / name).read_text().strip()
            except OSError:
                continue
        if settings:
            policies[directory.name] = settings
    return policies or None


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
    """Package results and validation receipts; keep bulky evidence locally."""
    result_suffixes = {".json", ".jsonl", ".md", ".log", ".txt", ".csv", ".tsv"}

    def results_only(member):
        if member.isdir():
            return None if Path(member.name).name == "blobs" else member
        if member.isfile() and Path(member.name).suffix.lower() in result_suffixes:
            return member
        return None

    archive = destination.parent / "latest.tgz"
    with tempfile.TemporaryDirectory(prefix=".p3hpc-archive-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "latest.tgz"
        with tarfile.open(staged, "w:gz", compresslevel=1) as bundle:
            bundle.add(destination, arcname=destination.name, filter=results_only)
        staged.replace(archive)
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


def check_native_kernel_search(kernel, tuning, seconds):
    options = kernel["options"]
    cutoff = kernel.get("share_cutoff", False)
    if cutoff:
        prediction = kernel.get("predicted_plan_ms")
        profile = kernel.get("machine_profile")
        bandwidth = profile.get("bandwidth") if isinstance(profile, dict) else None
        if (cutoff is not True or not isinstance(bandwidth, (int, float))
                or not math.isfinite(bandwidth) or bandwidth <= 0
                or not isinstance(prediction, (int, float))
                or not math.isfinite(prediction) or prediction <= 0
                or kernel["visited_classes"] >= kernel["eligible_classes"]):
            raise ValueError("native kernel share cutoff lacks a calibrated plan prediction")
    if (options.get("strategy") != "Race" or options["scope"] != "All"
            or options["max_classes"] != tuning["max_classes"]
            or options["min_improvement"] != tuning["min_improvement"]
            or options["max_scratch_bytes"] != TUNE_SCRATCH_BYTES
            or not 0 <= duration_seconds(options["max_time"]) <= seconds
            or kernel["class_limit_reached"]
            or not 0 <= kernel["visited_classes"] <= kernel["eligible_classes"]
            or (kernel["visited_classes"] < kernel["eligible_classes"]
                and not kernel["time_budget_exhausted"] and not cutoff)):
        raise ValueError("native kernel search stopped outside the declared limits")


def check_native_search(session, seconds, max_submission_chunks, kernel_memo=False, android=False,
                        language_model=False, output_shape=None):
    search = session.get("search")
    if not search:
        raise ValueError("native session did not use calibrated construction")
    options = search["options"]
    tuning = options["tuning"]
    memory = session["memory_budget"]
    available = memory["device_budget_bytes"] - memory["device_usage_bytes"]
    if memory.get("source", "vulkan") == "android-memavailable-capped-3gib":
        if not android or available > 0 or memory["system_available_bytes"] <= 0:
            raise ValueError("invalid Android shared-memory search budget")
        available = min(memory["system_available_bytes"], 3 * 1024**3)
        if memory["available_bytes"] != available:
            raise ValueError("Android search budget differs from the declared cap")
    elif memory.get("source", "vulkan") != "vulkan":
        raise ValueError("unknown native memory budget source")
    if (duration_seconds(options["max_time"]) != seconds
            or options["max_graphs"] != 16 or options["max_programs"] != 64
            or options["max_submission_chunks"] != max_submission_chunks
            or options["patience"] is not None
            or options["min_improvement"] != 0.01
            or options["warmup_runs"] != 2
            or options.get("qualification_runs") != 1
            or options.get("final_qualification_runs") != 2
            or duration_seconds(options["warmup_time"]) != 0.25
            or memory["plan_fraction_of_available"] != 0.75
            or options["max_plan_bytes"] != available // 4 * 3
            or options["max_plan_bytes"] <= 0
            or tuning.get("strategy") != "Race"
            or tuning["scope"] != "All" or tuning["max_classes"] != 2 * sys.maxsize + 1
            or tuning["min_improvement"] != 0.05
            or duration_seconds(tuning["max_time"]) != seconds
            or tuning["max_scratch_bytes"] != TUNE_SCRATCH_BYTES):
        raise ValueError("native construction limits differ from the declared policy")
    trials = search["trials"]
    if not trials or not 0 <= search["selected"] < len(trials):
        raise ValueError("native construction has no selected program")
    selected = trials[search["selected"]]
    if selected["outcome"]["qualified"] is not True or selected["qualifications"] < 1:
        raise ValueError("native selected program was not qualified")
    if selected["kernel_tuning"] is None:
        # Search preserves an ordinary, qualified baseline when tuning loses
        # or the budget runs out. Only the initial trial can be that fallback.
        outcome = selected["outcome"]
        if (search["selected"] != 0 or outcome.get("decision") != "KeepBaseline"
                or any(outcome.get(key) != 0 for key in ("initial", "candidate", "selected"))):
            raise ValueError("native untuned selection is not the qualified initial fallback")
    for trial in trials:
        if trial["outcome"]["qualified"] is True and trial["qualifications"] < 1:
            raise ValueError("native candidate lacks qualification")
        kernel = trial["kernel_tuning"]
        if kernel is None:
            continue
        check_native_kernel_search(kernel, tuning, seconds)
    if search.get("final_qualifications") != 2 or duration_seconds(
            search.get("final_qualification_time", {"secs": 0, "nanos": 0})) <= 0:
        raise ValueError("native selected program lacks two final qualification executions")
    qualification = session["qualification"]
    if (qualification["policy"] != "selected-program-repeatability-v2"
            or qualification["reference"] != "same execution plan"
            or qualification["rtol"] != 1e-4 or qualification["atol"] != 1e-6
            or qualification["accelerated_gradient_rtol"] != 0.01
            or qualification["qualified_calls"] < 2 or qualification["output_elements"] <= 0
            or qualification.get("program_checks", 0) < 2
            or qualification.get("repeatability_checks", 0) < 1
            or session["mode"] == "Training" and qualification["gradient_elements"] <= 0):
        raise ValueError("native construction lacks full-output/gradient qualification")
    arithmetic = qualification.get("arithmetic_reference", {})
    strict = session["cooperative_matrix_policy"] == "NativeF32"
    semantic = language_model and not strict and session["mode"] == "Inference"
    if (arithmetic.get("policy") != "native-f32-full-tensor-v2"
            or arithmetic.get("status") != "pass"
            or arithmetic.get("reference") != "original graph; untuned native strict-f32"
            or arithmetic.get("prepared_before_precision_conversion") is not True
            or arithmetic.get("output_policy") != (LM_POLICY if semantic else "tensor-fidelity-v1")
            or arithmetic.get("output_rtol") != (None if semantic else 1e-4 if strict else 0.02)
            or arithmetic.get("gradient_rtol") != (1e-4 if strict else 0.01)
            or arithmetic.get("atol") != 1e-6
            or arithmetic.get("checked_calls") != qualification["qualified_calls"]
            or not math.isfinite(arithmetic.get("preparation_seconds", math.nan))
            or arithmetic["preparation_seconds"] <= 0
            or session.get("search_input_policy") != "refresh before each step, outside timing"):
        raise ValueError("native construction lacks an original-graph F32 arithmetic check or matched inputs")
    if semantic:
        report = check_language_model_report(arithmetic["language_model"])
        if (report["tokens"] * report["vocabulary"] != qualification["output_elements"]
                or output_shape is not None and (report["tokens"] != math.prod(output_shape[:-1])
                                                 or report["vocabulary"] != output_shape[-1])):
            raise ValueError("native language-model quality covers the wrong output")
        raw = arithmetic["raw_logits"]
        if (raw["role"] != "diagnostic" or raw["elements"] != qualification["output_elements"]
                or any(not math.isfinite(raw[key]) or raw[key] < 0 for key in (
                    "max_abs_error", "max_abs_reference", "rms_error", "rms_reference"))):
            raise ValueError("invalid native raw-logit diagnostics")
    elif arithmetic.get("language_model") is not None:
        raise ValueError("unexpected language-model quality policy")
    memo = session["kernel_memo"]
    reports = [trial["kernel_tuning"] for trial in trials if trial["kernel_tuning"] is not None]
    # TuneReport uses serde's skip_serializing_if for an empty reuse list.
    # Absence means zero visits, but the aggregate receipt must still agree.
    reused_visits = sum(len(report.get("reused_classes", [])) for report in reports)
    if (memo["policy"] != KERNEL_MEMO_POLICY or memo["enabled"] is not kernel_memo
            or memo["load"] not in (("miss", "hit", "invalid") if kernel_memo else ("disabled",))
            or not isinstance(memo["loaded_classes"], int) or memo["loaded_classes"] < 0
            or memo["load"] != "hit" and memo["loaded_classes"] != 0
            or memo["reused_class_visits"] != reused_visits
            or memo["new_comparisons"] != sum(len(report["outcomes"]) for report in reports)):
        raise ValueError("native kernel-memo receipt differs from executed search")
    if kernel_memo:
        if (not isinstance(memo["key"], str) or not re.fullmatch(r"[0-9a-f]{64}", memo["key"])
                or not isinstance(memo["saved_classes"], int) or memo["saved_classes"] < 0):
            raise ValueError("native kernel memo was not saved")
    elif memo["key"] is not None or memo["saved_classes"] is not None:
        raise ValueError("disabled native kernel memo has persistent state")


def check_native_reference(record):
    """Tie the cross-engine oracle to the snapshots used by native qualification."""
    sessions = record["optimizer"]["sessions"]
    inference = next(s for s in sessions if s["mode"] == "Inference")
    expected = dict(inference["qualification"]["arithmetic_reference"]["reference_outputs"],
                    output_shape=record["outputs"]["output_shape"])
    training = record["protocol"]["training_requested"]
    if training:
        session = next(s for s in sessions if s["mode"] == "Training")
        source = session["qualification"]["arithmetic_reference"]["reference_outputs"]
        expected.update({key: source[key] for key in ("loss", "grad_norm", "gradient_norms")})
    oracle = record["correctness_reference"]
    if (oracle.get("precision") != "strict-f32" or oracle.get("training_requested") is not training
            or oracle.get("prepared_before_candidate_compilation") is not True
            or oracle.get("outputs") != expected
            or not math.isfinite(expected["loss"])
            or len(expected["logits_sample"]) != len(record["outputs"]["logits_sample"])
            or any(not math.isfinite(x) for x in expected["logits_sample"])
            or training and (not expected["gradient_norms"] or not math.isfinite(expected["grad_norm"])
                or any(not math.isfinite(x) for x in expected["gradient_norms"].values()))):
        raise ValueError("native F32 correctness reference differs from its qualification snapshots")


def check_native_record(mg, args, count, revision, precision, *, diagnostic=False):
    """Validate native receipts even when PyTorch cannot produce a result."""
    if mg["status"] != "ok" or not revision.startswith(mg["benchmark_rev"]):
        raise ValueError("native execution or source identity failed")
    protocol = mg["protocol"]
    if (protocol["training_requested"] != (not args.inference_only)
            or protocol.get("diagnostic") is not diagnostic
            or protocol.get("name") != "inferena-paper-v6"
            or protocol.get("synthetic_parameter_init") != SYNTHETIC_PARAMETER_INIT
            or protocol.get("warmup_runs") != 5
            or protocol.get("warmup_seconds") != WARMUP_SECONDS
            or not gpu_matches(args.gpu, mg["gpu_name"])):
        raise ValueError("native workload, warmup policy, or device changed")
    if precision is not None and mg["precision"]["comparison_class"] != ("strict-f32" if precision == "strict" else "accelerated"):
        raise ValueError("native arithmetic contract changed")
    for phase in PHASES[:2] if args.inference_only else PHASES:
        warmup = protocol.get("warmup", {}).get(phase, {})
        samples = mg["timing_samples_ms"].get(phase, [])
        seconds = warmup.get("seconds", 0)
        if (warmup.get("runs", 0) < 5 or not math.isfinite(seconds) or seconds < WARMUP_SECONDS
                or len(samples) != count or any(not math.isfinite(x) or x <= 0 for x in samples)):
            raise ValueError(f"native {phase} has incomplete warmup or measurements")
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
    language_model = mg["model"].startswith("SmolLM2-")
    if language_model and modes.count("Inference") != 2:
        raise ValueError("language-model prefill and latency both require qualification")
    inference_index = 0
    for session in sessions:
        if session["cooperative_matrix_policy"] != ("NativeF32" if strict else "Auto"):
            raise ValueError("native session used the wrong cooperative policy")
        check_native_search(session, getattr(args, "tune_seconds", TUNE_SECONDS),
                            getattr(args, "max_submission_chunks", MAX_SUBMISSION_CHUNKS),
                            getattr(args, "kernel_memo", False), language_model=language_model,
                            output_shape=(mg["outputs"]["output_shape"] if inference_index == 0
                            else [1, 1, mg["outputs"]["output_shape"][-1]]) if language_model else None)
        inference_index += session["mode"] == "Inference"
    if not strict:
        check_native_reference(mg)


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
        expected = "inferena-paper-v6" if engine == "meganeura" else PYTORCH_PROTOCOL
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
    expected_class = ("strict-f32" if precision == "strict" else "accelerated")
    if precision is not None and any(record["precision"]["comparison_class"] != expected_class
                                     for record in (mg, pt)):
        raise ValueError("requested arithmetic contract did not execute")
    dtype = pt["precision"].get("autocast")
    if hasattr(args, "autocast_policies"):
        policy = args.autocast_policies[pt["model"]]
        if precision == "strict":
            policy = dict(policy, autocast_requested="off", autocast=None,
                          model_guard=None, f32_linear_modules=[], model_guard_phases=[],
                          phase_autocast={phase: None for phase in PHASES})
        check_arithmetic_policy(pt, policy)
    if pt["precision"]["comparison_class"] == "accelerated":
        check_precision_qualification(pt)
        if pt["correctness_reference"]["outputs"]["output_shape"] != mg["outputs"]["output_shape"]:
            raise ValueError("f32 oracle shape differs from the native output")
    check_native_record(mg, args, count, revision, precision, diagnostic=diagnostic)
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
    if execution.get("kernel_benchmarking") != reference_kernel_benchmarking(args.backend, mode):
        raise ValueError("reference kernel benchmarker differs from the declared backend policy")
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
        if phase == "training" and dtype and report.get("post_execution") != "eager GradScaler housekeeping, included in timing":
            raise ValueError("AMP training must include scaler housekeeping after each execution")
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
        check_native_record(native, args, count, revision, precision)
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
            phase_failure = report.get("failure", failure)
            if (phase_failure.get("kind") not in ("numerical", "capture")
                    or (report["status"] == "failed") != (phase == phase_failure.get("phase"))):
                raise ValueError("failed phase disagrees with the failure receipt")
            if reference["timing_samples_ms"].get(phase) is not None:
                raise ValueError("unqualified phase claims timing samples")
            outcomes[phase] = {"status": report["status"], "failure": phase_failure}
            continue
        try:
            check_pair(records, args, mode, graphs, count, revision, phases=(phase,),
                       precision=precision, replicated=args.collect and not oracle, oracle=oracle)
            outcomes[phase] = {"status": "valid"}
        except NumericalMismatch as error:
            outcomes[phase] = {"status": "mismatch", "error": str(error)}
    return outcomes, pair


def native_oracle_comparison(folder, model, args):
    """Use the pre-compilation F32 snapshot without replacing failed timings."""
    path = folder / "torch-f32-oracle.json"
    if not path.exists():
        return None
    oracle = json.loads(path.read_text())
    check_torch_identity(oracle["torch_version"], oracle["torch_git_version"], args.torch_version)
    if (oracle["backend"].split()[0].lower() != args.backend
            or oracle["synthetic_parameter_init"] != SYNTHETIC_PARAMETER_INIT):
        raise ValueError("independent oracle environment differs from the campaign")
    if (oracle.get("correctness_reference") is None
            and oracle.get("unavailable_reason") == "non-finite original F32 snapshot"):
        return None
    native_path = folder / f"{model}_meganeura.json"
    harness = ROOT / "target/release" / ("inferena.exe" if sys.platform == "win32" else "inferena")
    validation = json.loads(subprocess.check_output(
        [str(harness), "--compare-oracle", str(native_path), str(path)], cwd=ROOT, text=True))
    receipt = {"policy": "preserved-f32-oracle-v1", "oracle": path.name,
               "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
               "oracle_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "validation": validation, "timing_substituted": False}
    write_json(folder / "native-oracle-comparison.json", receipt)
    return {"path": "native-oracle-comparison.json", **receipt}


def eager_diagnostic(command, env, folder, model, args, count, revision, precision):
    """One fresh, uncaptured reference; never rewrite the failed primary pair."""
    diagnostic = folder / "diagnostic-eager"
    diagnostic.mkdir()
    command = list(command)
    for flag, value in (("-f", "pytorch"), ("--measurement-runs", "1"), ("--results-dir", str(diagnostic))):
        command[command.index(flag) + 1] = value
    # The collector already hashed target/release/inferena. A pytorch-only
    # run.sh otherwise rebuilds the harness alone and replaces that binary.
    command.append("--no-build")
    environment = dict(env, INFERENA_TORCH_MODE="eager", INFERENA_GRAPH_REPLAY="0", INFERENA_SDPA="math",
                       INFERENA_REFERENCE_DIAGNOSTIC="1",
                       INFERENA_PREPARATION_REPORT=str(diagnostic / "torch-preparation.json"))
    with (diagnostic / "runner.log").open("w", encoding="utf-8") as log:
        result = run_recorded(command, report_path=diagnostic / "watchdog.json", seconds=args.run_seconds,
                              cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
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
        write_json(diagnostic / "comparison.json", records)
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
    if any(model in SMOLLM2_REVISIONS for model in models):
        load_fixture()
        hashes[str(FIXTURE.relative_to(ROOT))] = FIXTURE_SHA256
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


def runner_hashes():
    suffix = ".exe" if sys.platform == "win32" else ""
    hashes = {}
    for name in ("inferena", "inferena-meganeura"):
        path = ROOT / "target/release" / (name + suffix)
        try:
            with path.open("rb") as data:
                hashes[name] = hashlib.file_digest(data, "sha256").hexdigest()
        except FileNotFoundError:
            hashes[name] = None
    return hashes


def verify_runner_binaries(provenance, receipt, stage):
    actual = runner_hashes()
    expected = {name: item["sha256"] for name, item in provenance["binaries"].items()}
    receipt[stage] = {"checked_at": datetime.now(timezone.utc).isoformat(),
                      "sha256": actual, "status": "pass" if actual == expected else "fail"}
    if actual != expected:
        changed = ", ".join(name for name in expected if actual.get(name) != expected[name])
        raise ValueError(f"runner binary changed {stage}: {changed}; preserve this campaign and inspect its build logs")


def build_runners(destination):
    # run.sh builds both crates together, but the native wrapper builds only
    # its own crate. Cargo can select different cached executables for these
    # feature unions even when both invocations report a completely fresh build.
    common = ["cargo", "build", "--release", "--manifest-path", str(ROOT / "Cargo.toml")]
    commands = [common + ["--locked", "-p", "inferena-harness", "-p", "inferena-meganeura"],
                common + ["-p", "inferena-meganeura", "--locked"]]
    build_env = dict(os.environ, CARGO_BUILD_JOBS=os.environ.get("CARGO_BUILD_JOBS", "1"))
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=build_env, check=True)
    directory = destination / "binaries"
    directory.mkdir()
    provenance = {"policy": "locked-runner-binaries-v1", "build_commands": commands, "binaries": {}}
    suffix = ".exe" if sys.platform == "win32" else ""
    for name, digest in runner_hashes().items():
        if digest is None:
            raise FileNotFoundError(f"build did not produce {name}")
        archived = directory / (name + suffix)
        shutil.copy2(ROOT / "target/release" / archived.name, archived)
        with archived.open("rb") as data:
            if hashlib.file_digest(data, "sha256").hexdigest() != digest:
                raise ValueError(f"{name} changed while archiving")
        provenance["binaries"][name] = {"sha256": digest, "path": str(archived.relative_to(destination)),
                                         "size_bytes": archived.stat().st_size}
    verify_runner_binaries(provenance, provenance, "after_archive")
    return provenance


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, help="new directory; default: ../inferena-results/<host>-<UTC>-<source>")
    parser.add_argument("--backend", choices=("cuda", "rocm", "xpu", "mps", "cpu", "android"), help="default: installed GPU backend; no CPU fallback")
    parser.add_argument("--gpu", help="expected GPU-name substring; default: reference GPU")
    parser.add_argument("--torch-version", help="exact vendor build; default: installed build, with common source/version checks")
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=list(MODELS))
    parser.add_argument("--inference-only", action="store_true", help="prefill and stateless one-token SmolLM2; no training")
    # The collector selects a particular device; legacy discrete-GPU preferences
    # must not override it. Keep accepting the old opt-in spelling.
    parser.add_argument("--allow-integrated-gpu", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--precisions", nargs="+", choices=("strict", "accelerated"), default=["strict", "accelerated"])
    parser.add_argument("--autocast", choices=("auto", "off", "bf16", "fp16"), default="auto",
                        help="accelerated PyTorch recipe; auto uses the declared model/backend policy; strict stays f32")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--no-kernel-memo", dest="kernel_memo", action="store_false", default=True,
                        help="repeat private kernel tuning independently in every native process")
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
                        help="Meganeura soft construction deadline per session (default: %(default)g s), including initialization/qualification")
    parser.add_argument("--max-submission-chunks", type=int, choices=(1, 2, 4, 8, 16, 32, 64),
                        default=MAX_SUBMISSION_CHUNKS,
                        help="search powers of two up to this bound on every GPU (default: %(default)s); 1 disables chunking")
    parser.add_argument("--compile-seconds", type=positive_seconds, default=COMPILE_SECONDS,
                        help="PyTorch compilation/first-specialization deadline (default: 120 s)")
    parser.add_argument("--run-seconds", type=positive_seconds, default=RUN_SECONDS,
                        help="wall-clock watchdog for each paired runner and its children (default: %(default)g s); aborts without retry")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--collect", dest="collect", action="store_true", default=True, help="validated measurement (default)")
    stage.add_argument("--qualify-only", dest="collect", action="store_false", help="run correctness gates without publication samples")
    parser.add_argument("--offline", action="store_true", help="require all pinned models to be prepared already")
    android = parser.add_argument_group("Android / Quest 3S")
    android.add_argument("--adb-serial", help="explicit USB serial or wireless IP:port")
    android.add_argument("--android-ndk", type=Path, help="Android NDK r27 directory; defaults to ANDROID_NDK_HOME")
    android.add_argument("--android-timeout", type=positive_seconds, default=1800,
                         help="hard deadline for each device or CPU process (seconds)")
    return parser


def collect(wake_policy):
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
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
        "TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER", "TORCHINDUCTOR_USE_TORCH_PROFILER_BENCHMARKER",
        "INFERENA_TORCH_MODE", "INFERENA_GRAPH_REPLAY", "INFERENA_CUDA_GRAPHS",
        "INFERENA_TUNE_SECONDS", "INFERENA_COMPILE_SECONDS", "INFERENA_PREPARATION_REPORT", "INFERENA_BUDGET_ENFORCED",
        "INFERENA_MAX_SUBMISSION_CHUNKS", "INFERENA_WARMUP_SECONDS",
        "INFERENA_KERNEL_MEMO_DIR",
        "INFERENA_SDPA", "INFERENA_REFERENCE_DIAGNOSTIC", "INFERENA_TENSOR_EXPORT", "INFERENA_MEGANEURA_PRECISION",
        "INFERENA_AUTOCAST", "INFERENA_PRECISION_REPORT", "INFERENA_TRAINING_GATE",
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
    if args.backend == "android":
        from p3hpc_android import collect
        sys.exit(collect(args, parser, packages, started_at, started))
    if args.adb_serial or args.android_ndk:
        parser.error("Android options require --backend android")
    probe_env = dict(os.environ)
    probe_env["INFERENA_AUTOCAST"] = args.autocast
    probe_env["INFERENA_POLICY_MODELS"] = json.dumps(args.models)
    if args.backend:
        probe_env["INFERENA_TORCH_BACKEND"] = args.backend
    try:
        # Device queries/XPU qualification can create GPU contexts. Let this
        # process exit before collection so the controller holds no GPU memory.
        reference = json.loads(subprocess.check_output([
            sys.executable, "-c", "import json, os; from bench import detect_device, device_name, backend_name; "
            "from amp import precision_policy; d = detect_device(); "
            "print(json.dumps({'backend': backend_name(d).split()[0].lower(), 'gpu': device_name(d), "
            "'autocast_policies': {m: precision_policy(d, os.environ['INFERENA_AUTOCAST'], m) "
            "for m in json.loads(os.environ['INFERENA_POLICY_MODELS'])}}))",
        ], cwd=ROOT / "frameworks/pytorch", env=probe_env, text=True, encoding="utf-8"))
    except subprocess.CalledProcessError as error:
        parser.error(str(error))
    backend = reference["backend"]
    if backend == "cpu" and args.backend != "cpu":
        parser.error("no working GPU backend; fix setup or explicitly request --backend cpu")
    args.backend = backend
    if backend == "cpu" and args.autocast not in ("auto", "off"):
        parser.error("the graphics-only CPU correctness oracle requires f32; use --autocast off")
    args.autocast_policies = reference["autocast_policies"]
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
               INFERENA_REQUIRE_LOCAL_WEIGHTS="1", INFERENA_TORCH_BACKEND=args.backend,
               INFERENA_WARMUP_SECONDS=str(WARMUP_SECONDS))
    env.pop("VIRTUAL_ENV", None)
    manifest = {
        "protocol": "p3hpc-paired-campaign-v33", "source": revision,
        "started_at": started_at,
        "host": {"name": socket.gethostname(), "os": platform.system(),
                 "kernel": platform.release(), "platform": platform.platform(), "arch": platform.machine(),
                 "cpu_frequency_policies": cpu_frequency_policies()},
        "driver_shader_cache": {
            "initial_state": "uncontrolled; independent of the native kernel memo",
            "environment": {key: os.environ.get(key) for key in (
                "MESA_SHADER_CACHE_DIR", "MESA_SHADER_CACHE_DISABLE",
                "MESA_SHADER_CACHE_SINGLE_FILE", "__GL_SHADER_DISK_CACHE",
                "__GL_SHADER_DISK_CACHE_PATH", "__GL_SHADER_DISK_CACHE_SIZE")}},
        "synthetic_parameter_init": SYNTHETIC_PARAMETER_INIT,
        "precision_policy": {"strict": "f32", "accelerated": {
            "autocast_requested": args.autocast, "pytorch_by_model": args.autocast_policies,
            "native": "existing qualified reduced-input paths; f32 storage",
            "correctness_reference": "compare both original F32 references; independently qualify each accelerated output",
            "language_model_forward": {"policy": LM_POLICY, "mean_kl_limit": MEAN_KL_LIMIT,
                                       "token_kl_limit": TOKEN_KL_LIMIT, "loss_rtol": 0.02, "loss_atol": 1e-6,
                                       "raw_logits": "diagnostic", "phases": ["prefill", "latency"]},
            "training_curve": {"policy": CURVE_POLICY, "per_loss_rtol": 0.02, "atol": 1e-6,
                               "relative_l2": "diagnostic"},
            "pytorch_training_gates": {model: workload_for(model) for model in args.models}}},
        "model_revisions": {name: SMOLLM2_REVISIONS[name] for name in args.models if name in SMOLLM2_REVISIONS},
        "meganeura": dependency, "python": sys.version, "packages": packages,
        "torch": {"version": torch.__version__, "git_version": torch.version.git_version,
                  "build_config": torch.__config__.show()},
        "args": {**vars(args), "results_dir": str(destination)}, "sha256": hashes,
        "native_policy": {"measured_construction": True, "optimizer": "egglog-outlined",
                          "max_submission_chunks": args.max_submission_chunks,
                          "program_min_improvement": 0.01, "kernel_min_improvement": 0.05,
                          "scope": "All", "class_limit": None,
                          "kernel_strategy": "Race", "negligible_plan_share_cutoff": 0.001,
                          "max_scratch_bytes": TUNE_SCRATCH_BYTES,
                          "kernel_seconds_per_program": args.tune_seconds,
                          "max_graphs": 16, "max_programs": 64, "plan_fraction_of_available": 0.75,
                          "patience": None,
                          "kernel_memo": {"enabled": args.kernel_memo, "policy": KERNEL_MEMO_POLICY,
                                          "scope": "campaign/model/precision", "selected_program_reuse": False},
                          "warmup_pairs": 2, "warmup_seconds": 0.25,
                          "qualification": "selected-program-repeatability-v2",
                          "qualification_runs": 1,
                          "final_qualification_runs": 2,
                          "arithmetic_reference": "native-f32-full-tensor-v2; preparation included in compile_s, outside search budget",
                          "search_input_policy": "refresh before each step, outside timing",
                          "search_seconds_per_session": args.tune_seconds, "strict_coop": "NativeF32"},
        "reference_compile_seconds": args.compile_seconds,
        "run_watchdog_seconds": args.run_seconds,
        "warmup": {"minimum_runs": 5, "minimum_seconds": WARMUP_SECONDS},
        "reference_sdpa_policy": sdpa,
        "reference_kernel_benchmarking": {
            mode: reference_kernel_benchmarking(args.backend, mode)
            for mode, _ in selected_conditions
        },
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
        "keep_awake": wake_policy,
        "runs": [], "status": "in-progress",
    }

    def save():
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_json(destination / "campaign.json", manifest)

    save()
    print(f"Campaign: {destination}", flush=True)
    try:
        # Prepare both wrapper build recipes before probing or archiving the
        # executables. Later builds remain outside the timed engine calls.
        build_started = time.monotonic()
        manifest["runner_binaries"] = build_runners(destination)
        manifest["runner_build_seconds"] = time.monotonic() - build_started
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
                            run = {"path": folder.relative_to(destination).as_posix(), "command": command,
                                   "mode": mode, "graphs": graphs, "model": model,
                                   "precision": precision, "replicate": replicate + 1,
                                   "watchdog": "watchdog.json",
                                   "preparation_policy": "native-searched/reference-" + mode, "status": "running"}
                            manifest["runs"].append(run)
                            run["binary_checks"] = {}
                            verify_runner_binaries(manifest["runner_binaries"], run["binary_checks"], "before")
                            save()
                            print(run["path"], flush=True)
                            run_env = dict(env, INFERENA_TORCH_MODE=mode,
                                           INFERENA_AUTOCAST=args.autocast if precision == "accelerated" else "off",
                                           INFERENA_SDPA=sdpa[model], INFERENA_GRAPH_REPLAY=str(int(graphs)),
                                           MEGANEURA_TUNE="1", INFERENA_TUNE_SECONDS=str(args.tune_seconds),
                                           INFERENA_MAX_SUBMISSION_CHUNKS=str(args.max_submission_chunks),
                                           INFERENA_COMPILE_SECONDS=str(args.compile_seconds),
                                           INFERENA_PREPARATION_REPORT=str(folder / "torch-preparation.json"))
                            if args.kernel_memo:
                                memo_dir = destination / "native-kernel-memos" / precision / model
                                run_env["INFERENA_KERNEL_MEMO_DIR"] = str(memo_dir)
                                run["kernel_memo_directory"] = str(memo_dir.relative_to(destination))
                            with (folder / "runner.log").open("w", encoding="utf-8") as log:
                                run["started_at"] = datetime.now(timezone.utc).isoformat()
                                save()
                                run_started = time.monotonic()
                                try:
                                    result = run_recorded(command, report_path=folder / "watchdog.json",
                                                          seconds=args.run_seconds, cwd=ROOT, stdout=log,
                                                          stderr=subprocess.STDOUT, env=run_env)
                                finally:
                                    run["elapsed_seconds"] = time.monotonic() - run_started
                                    verify_runner_binaries(manifest["runner_binaries"], run["binary_checks"], "after")
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
                            if precision == "accelerated":
                                run["native_oracle"] = native_oracle_comparison(folder, model, args)
                            run["status"] = ("valid" if all(p["status"] == "valid" for p in phases.values())
                                             else "partial" if any(p["status"] == "valid" for p in phases.values())
                                             else "failed")
                            save()
                            if run["status"] != "valid" and mode != "eager":
                                run["eager_diagnostic"] = {"path": "diagnostic-eager", "status": "running"}
                                save()
                                try:
                                    run["eager_diagnostic"] = eager_diagnostic(
                                        command, run_env, folder, model, args, count, revision, precision)
                                finally:
                                    verify_runner_binaries(manifest["runner_binaries"], run["binary_checks"], "after_diagnostic")
                            if stage == "measurement" and phases.get("training", {}).get("status") == "valid":
                                key = (precision, model, mode, graphs)
                                measurement_pairs.setdefault(key, []).append(pair)
                            save()
        if args.collect and not args.inference_only:
            manifest["replicated_gradient_validation"] = validate_replicated_gradients(
                measurement_pairs, args.replicates
            )
            save()
        verify_runner_binaries(manifest["runner_binaries"], manifest, "final_binary_check")
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
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        try:
            print(f"Archive: {archive_results(destination)}", flush=True)
        except OSError as error:
            print(f"Archive not updated: {error}; results remain in {destination}", file=sys.stderr)
    print(f"{manifest['status']}: {destination / 'campaign.json'}")
    if manifest["status"] != "complete":
        sys.exit(1)


def main():
    with keep_awake() as wake_policy:
        collect(wake_policy)


if __name__ == "__main__":
    main()
