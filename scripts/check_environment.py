#!/usr/bin/env python3
"""Check the requested PyTorch backend with a tiny forward/backward workload."""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import platform
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frameworks/pytorch"))
from bench import clear_compile_cache, compiler_options, detect_device, device_name, select_compiler_backend
from execution import capture_phase, compare_tensors, graph_backend, synchronize


def check_backend(backend, mode, replay=None):
    os.environ["INFERENA_TORCH_BACKEND"] = backend
    device = detect_device()
    select_compiler_backend(device)
    torch.manual_seed(7)
    torch.set_float32_matmul_precision("highest")
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.SiLU()).to(device)
    inputs = torch.ones(8, 16, device=device)
    expected = model(inputs)
    gradients = torch.autograd.grad(expected.square().mean(), tuple(model.parameters()))
    expected = expected.detach().cpu()
    gradients = [gradient.cpu() for gradient in gradients]
    api = graph_backend(device) if backend in ("cuda", "rocm", "xpu") else None
    stream = api.Stream(device=device) if api is not None else None
    if stream is not None:
        stream.wait_stream(api.current_stream(device))
    options = compiler_options(mode)
    capture = api is not None if replay is None else replay
    with clear_compile_cache(), api.stream(stream) if stream is not None else nullcontext():
        candidate = model if options is None else torch.compile(model, options=options)

        def training():
            output = candidate(inputs)
            loss = output.square().mean()
            loss.backward()
            return output, loss

        call, _ = capture_phase(training, model, stream=stream, device=device, capture=capture)
        model.zero_grad(set_to_none=not capture)
        output, _ = call()
        synchronize(device)
        compare_tensors(output, expected)
        for parameter, reference in zip(model.parameters(), gradients, strict=True):
            compare_tensors(parameter.grad, reference, gradient=True)
    print(f"PASS: {device_name(device)} / {mode} / replay={capture}, forward + backward")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "rocm", "xpu", "mps", "cpu"), required=True)
    parser.add_argument("--mode", default=os.environ.get("INFERENA_TORCH_MODE", "default"))
    parser.add_argument("--replay", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    print(f"Python {platform.python_version()}, torch {torch.__version__}, source {torch.version.git_version}", flush=True)
    check_backend(args.backend, args.mode, args.replay)
    print("Environment probe passed. Each real workload is still qualified before timing.")


if __name__ == "__main__":
    main()
