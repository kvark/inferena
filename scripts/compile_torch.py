#!/usr/bin/env python3
"""Compilation-stage diagnostic on the unchanged Inferena workload/validation path."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from p3hpc import ROOT, MODELS, PYTHON_VERSION, check_torch_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--mode", choices=("default", "max-autotune"), default="default")
    parser.add_argument("--precision", choices=("strict", "accelerated"), default="strict")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--untraced", action="store_true", help="same serial preparation without timing callbacks")
    args = parser.parse_args()
    if platform.python_version() != PYTHON_VERSION:
        parser.error(f"requires Python {PYTHON_VERSION}")
    if args.inference_only and args.model != "SmolLM2-135M":
        parser.error("inference-only is supported for SmolLM2")
    args.output = args.output.resolve()
    args.cache = args.cache.resolve()
    if ROOT == args.output or ROOT in args.output.parents or ROOT == args.cache or ROOT in args.cache.parents:
        parser.error("keep generated data and compiler caches outside Git")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.reuse_cache:
        if not args.cache.is_dir() or not any(args.cache.iterdir()):
            parser.error("reuse requires a populated experiment cache")
    else:
        args.cache.mkdir(parents=True, exist_ok=False)
    os.environ.update({
        "TORCHINDUCTOR_CACHE_DIR": str(args.cache / "inductor"),
        "TRITON_CACHE_DIR": str(args.cache / "triton"),
        "CUDA_CACHE_PATH": str(args.cache / "cuda"),
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "INFERENA_TORCH_BACKEND": "cuda", "INFERENA_TORCH_MODE": args.mode,
        "INFERENA_CUDA_GRAPHS": "1", "INFERENA_STRICT": str(int(args.precision == "strict")),
        "INFERENA_INFERENCE_ONLY": str(int(args.inference_only)),
        "INFERENA_WARMUP_RUNS": "5", "INFERENA_MEASUREMENT_RUNS": "3",
        "INFERENA_REQUIRE_LOCAL_WEIGHTS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    os.environ.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    if args.precision == "strict":
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    else:
        os.environ.pop("NVIDIA_TF32_OVERRIDE", None)

    import torch
    import triton
    from triton import knobs
    check_torch_identity(torch.__version__, torch.version.git_version, torch.__version__)
    sys.path.insert(0, str(ROOT / "frameworks/pytorch"))
    import bench

    # Only cache ownership changes. Graph capture and all validation remain intact.
    bench.clear_compile_cache = lambda: torch._dynamo.reset()
    compilations = []
    loads = []
    starts = {}

    def compiled(*, src, metadata, metadata_group, times, cache_hit):
        compilations.append({"name": src.name, "metadata": metadata, "cache_hit": cache_hit,
                             "times_us": asdict(times), "total_us": times.total,
                             "artifacts": metadata_group})

    def load_start(module, function, name, metadata_group, key):
        starts[key] = time.perf_counter_ns()

    def load_end(module, function, name, metadata_group, key):
        elapsed = time.perf_counter_ns() - starts.pop(key)
        loads.append({"name": name, "hash": key, "duration_ns": elapsed})

    if not args.untraced:
        if knobs.compilation.listener is not None:
            raise RuntimeError("another compilation listener is already installed")
        knobs.compilation.listener = compiled
        knobs.runtime.kernel_load_start_hook.add(load_start)
        knobs.runtime.kernel_load_end_hook.add(load_end)
    record = {
        "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": torch.__version__, "torch_source": torch.version.git_version, "triton": triton.__version__,
        "python": platform.python_version(), "args": vars(args), "status": "incomplete",
        "scope": "serial compilation diagnostic; not publication preparation/performance samples",
    }
    start = time.perf_counter_ns()
    try:
        with (args.output / "runner.json").open("x") as output, redirect_stdout(output):
            bench.bench(args.model, bench.MODEL_REGISTRY[args.model])
        record["status"] = "complete"
    finally:
        record.update({"runner_ns": time.perf_counter_ns() - start, "compilations": compilations,
                       "kernel_loads": loads, "pending_loads": list(starts),
                       "torch_compile_metrics_s": torch._dynamo.utils.compilation_time_metrics})
        with (args.output / "compilation.json").open("x") as output:
            json.dump(record, output, indent=2, default=str)
            output.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
