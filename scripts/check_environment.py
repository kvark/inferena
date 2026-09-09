#!/usr/bin/env python3
"""Exercise the installed reference backend before expensive model qualification."""

import argparse
import os
import platform
import sys

import torch

from p3hpc import ROOT, PYTHON_VERSION, check_torch_identity, conditions, runner_bash

sys.path.insert(0, str(ROOT / "frameworks/pytorch"))
from bench import clear_compile_cache, detect_device, device_name
from execution import capture_phase, synchronize


def check_backend(backend):
    os.environ["INFERENA_TORCH_BACKEND"] = backend
    device = detect_device()  # Explicit requests cannot fall back to CPU.
    torch.manual_seed(7)
    torch.set_float32_matmul_precision("highest")
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.SiLU()).to(device)
    inputs = torch.ones(8, 16, device=device)
    expected = model(inputs)
    gradients = torch.autograd.grad(expected.square().mean(), tuple(model.parameters()))
    expected = expected.detach()
    with clear_compile_cache():
        for mode, graphs in conditions(backend):
            candidate = model if mode == "eager" else torch.compile(model, fullgraph=True, options={
                "max_autotune": mode == "max-autotune", "triton.cudagraphs": False,
            })

            def training():
                output = candidate(inputs)
                loss = output.square().mean()
                loss.backward()
                return output, loss

            model.zero_grad(set_to_none=True)
            call = capture_phase(training, model)[0] if graphs else training
            output, loss = call()
            synchronize(device)
            torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-6)
            for parameter, reference in zip(model.parameters(), gradients, strict=True):
                torch.testing.assert_close(parameter.grad, reference, rtol=1e-4, atol=1e-6)
            print(f"PASS: {device_name(device)} / {mode} / graphs={graphs}, forward + backward", flush=True)
            del output, loss, call, candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "rocm", "xpu", "mps", "cpu"), required=True)
    args = parser.parse_args()
    if platform.python_version() != PYTHON_VERSION:
        parser.error(f"use Python {PYTHON_VERSION}")
    check_torch_identity(torch.__version__, torch.version.git_version, torch.__version__)
    print(f"Python {platform.python_version()}, torch {torch.__version__}, source {torch.version.git_version}", flush=True)
    print(f"Runner shell: {runner_bash()}", flush=True)
    check_backend(args.backend)
    print("Environment probe passed; run paired model qualification before collecting data.")


if __name__ == "__main__":
    main()
