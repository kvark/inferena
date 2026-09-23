#!/usr/bin/env python3
"""Exercise the installed reference backend before expensive model qualification."""

import argparse
import os
import platform
import sys

# Supervise before importing torch so the controller stays small.
if __name__ == "__main__" and os.environ.get("INFERENA_BUDGET_ENFORCED") != "1":
    from pathlib import Path
    budget = Path(__file__).resolve().parents[1] / "frameworks/pytorch/budget.py"
    os.execv(sys.executable, [sys.executable, str(budget), *sys.argv])

import torch

from p3hpc import ROOT, PYTHON_VERSION, check_torch_identity, conditions, runner_bash

sys.path.insert(0, str(ROOT / "frameworks/pytorch"))
from bench import attention_context, clear_compile_cache, detect_device, device_name, select_compiler_backend
from budget import compilation_budget
from execution import capture_phase, compare_tensors, graph_backend, synchronize
from contextlib import nullcontext


def check_backend(backend, max_autotune=False):
    os.environ["INFERENA_TORCH_BACKEND"] = backend
    device = detect_device()  # Explicit requests cannot fall back to CPU.
    select_compiler_backend(device)
    torch.manual_seed(7)
    torch.set_float32_matmul_precision("highest")
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.SiLU()).to(device)
    inputs = torch.ones(8, 16, device=device)
    expected = model(inputs)
    gradients = torch.autograd.grad(expected.square().mean(), tuple(model.parameters()))
    expected = expected.detach()
    api = graph_backend(device) if backend in ("cuda", "rocm", "xpu") else None
    stream = api.Stream(device=device) if api is not None else None
    if stream is not None:
        stream.wait_stream(api.current_stream(device))
    attention = attention_context(device)
    with clear_compile_cache(), attention, api.stream(stream) if stream is not None else nullcontext():
        for mode, graphs in conditions(backend, max_autotune):
            candidate = model if mode == "eager" else torch.compile(model, fullgraph=True, options={
                "max_autotune": mode == "max-autotune", "triton.cudagraphs": False,
            })

            def training():
                output = candidate(inputs)
                loss = output.square().mean()
                loss.backward()
                return output, loss

            model.zero_grad(set_to_none=True)
            with compilation_budget():
                training()
                synchronize(device)
            model.zero_grad(set_to_none=True)
            call = capture_phase(training, model, stream=stream, device=device)[0] if graphs else training
            output, loss = call()
            synchronize(device)
            compare_tensors(output, expected.cpu())
            for parameter, reference in zip(model.parameters(), gradients, strict=True):
                compare_tensors(parameter.grad, reference.cpu(), gradient=True)
            print(f"PASS: {device_name(device)} / {mode} / graphs={graphs}, forward + backward", flush=True)
            del output, loss, call, candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "rocm", "xpu", "mps", "cpu"), required=True)
    search = parser.add_mutually_exclusive_group()
    search.add_argument("--max-autotune", action="store_true", default=False)
    search.add_argument("--no-max-autotune", dest="max_autotune", action="store_false", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if platform.python_version() != PYTHON_VERSION:
        parser.error(f"use Python {PYTHON_VERSION}")
    check_torch_identity(torch.__version__, torch.version.git_version, torch.__version__)
    print(f"Python {platform.python_version()}, torch {torch.__version__}, source {torch.version.git_version}", flush=True)
    print(f"Runner shell: {runner_bash()}", flush=True)
    check_backend(args.backend, args.max_autotune)
    print("Environment probe passed; run paired model qualification before collecting data.")


if __name__ == "__main__":
    main()
