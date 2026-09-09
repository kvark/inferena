"""Whole-phase CUDA replay with untimed, full-tensor qualification."""

import time
import os
from contextlib import nullcontext
from pathlib import Path

import torch


def nsys_range(name):
    return torch.cuda.nvtx.range(name) if "INFERENA_NSYS" in os.environ else nullcontext()


def synchronize(device):
    backend = torch.device(device).type
    if backend in ("cuda", "xpu"):
        getattr(torch, backend).synchronize(device)
    elif backend == "mps":
        torch.mps.synchronize()


def profile_phase(fn, path, samples, before=None, device="cuda"):
    """Separate diagnostic timeline; these durations are never benchmark samples."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    backend = torch.device(device).type
    if backend == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif backend == "xpu":
        activities.append(torch.profiler.ProfilerActivity.XPU)
    wall_ms = []
    with torch.profiler.profile(activities=activities) as trace:
        for _ in range(samples):
            if before is not None:
                before()
            synchronize(device)
            with torch.profiler.record_function("inferena.phase"):
                start = time.perf_counter()
                fn()
                synchronize(device)
                wall_ms.append((time.perf_counter() - start) * 1000)
    trace.export_chrome_trace(str(path))
    return {
        "trace": str(path),
        "instrumented_wall_ms": wall_ms,
        "activities": [activity.name for activity in activities],
        "scope": "diagnostic host/device timeline, not an additive CPU/GPU cost split",
    }


class CapturedPhase:
    def __init__(self, fn, model, graph, outputs):
        # The callable may be the sole owner of captured inputs/parameters.
        self.fn = fn
        self.model = model
        self.graph = graph
        self.outputs = outputs

    def __call__(self):
        self.graph.replay()
        return self.outputs


def capture_phase(fn, model=None):
    """Return a replay callable retaining its graph, outputs and gradient storage.

    fn returns a tensor or a tuple of tensors. Inputs and parameters must keep
    their addresses. Training is forward/loss/backward, without an optimizer.
    Capture errors and validation failures propagate; never time a fallback.
    """
    start = time.perf_counter()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            if model is not None:
                model.zero_grad(set_to_none=True)
            outputs = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    def tensors(values):
        return (values,) if isinstance(values, torch.Tensor) else values

    expected = tuple(t.detach().cpu().clone() for t in tensors(outputs))
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    } if model is not None else {}
    del outputs

    if model is not None:
        # Allocate gradients during capture; backward then overwrites them on
        # every replay. Do not detach or reset those tensors between replays.
        model.zero_grad(set_to_none=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = fn()

    replay = CapturedPhase(fn, model, graph, outputs)
    capture_s = time.perf_counter() - start
    validation_start = time.perf_counter()

    def check(label, actual, reference):
        actual = actual.detach().cpu()
        if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
            raise ValueError(f"CUDA graph {label}: non-finite values")
        torch.testing.assert_close(
            actual, reference, rtol=1e-4, atol=1e-6,
            msg=lambda message: f"CUDA graph {label}: {message}",
        )

    for _ in range(2):
        actual = tensors(replay())
        torch.cuda.synchronize()
        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            check(f"output {index}", value, reference)
        if model is not None:
            actual_gradients = {
                name: parameter.grad
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            if actual_gradients.keys() != gradients.keys():
                raise ValueError("CUDA graph changed the set of parameter gradients")
            for name, reference in gradients.items():
                check(f"gradient {name}", actual_gradients[name], reference)

    return replay, {
        "status": "captured-and-validated",
        "scope": "forward + loss + backward" if model is not None else "forward",
        "capture_s": capture_s,
        "validation_s": time.perf_counter() - validation_start,
        "validation": {
            "outputs": "all elements",
            "gradient_tensors": len(gradients),
            "gradients": "all elements of every participating parameter",
            "consecutive_replays": 2,
            "rtol": 1e-4,
            "atol": 1e-6,
        },
    }
