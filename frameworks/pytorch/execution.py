"""Whole-phase CUDA replay with untimed, full-tensor qualification."""

import time
import os
import math
from contextlib import nullcontext
from pathlib import Path

import torch

REPLAY_RTOL = 1e-4
REPLAY_ATOL = 1e-6
REPLAY_POLICY = "fixed-full-gradient-v3"
ACCELERATED_GRADIENT_REPEATS = 8
ACCELERATED_GRADIENT_RTOL = 0.01


def compare_tensors(actual, reference, *, gradient=False, reduced_precision=False):
    """Compare full tensors and return metrics for whole-gradient checks."""
    actual = actual.detach().cpu()
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("tensor shape or dtype changed")
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("non-finite values")
    if not actual.numel():
        raise ValueError("empty qualification tensor")
    error = actual - reference
    def rms(value):
        return float(torch.linalg.vector_norm(value, dtype=torch.float64) / math.sqrt(value.numel()))

    report = {
        "elements": actual.numel(),
        "max_abs_error": float(error.abs().max()),
        "rms_error": rms(error),
        "max_abs_reference": float(reference.abs().max()),
        "rms_reference": rms(reference),
    }
    report["max_abs_bound"] = REPLAY_ATOL + REPLAY_RTOL * report["max_abs_reference"]
    report["rms_bound"] = REPLAY_ATOL + REPLAY_RTOL * report["rms_reference"]
    if not all(math.isfinite(value) for value in report.values()):
        raise ValueError(f"non-finite error metric: {report}")
    if gradient:
        # Cancellation makes elementwise relative gradient errors misleading.
        # Strict execution retains the tight per-parameter gate. Accelerated
        # reductions are allowed to be nondeterministic and are checked as one
        # complete gradient below, without fitting bounds to observed repeats.
        if not reduced_precision and (report["max_abs_error"] > report["max_abs_bound"]
                                      or report["rms_error"] > report["rms_bound"]):
            raise ValueError(f"gradient repeatability exceeds fixed bounds: {report}")
    else:
        torch.testing.assert_close(actual, reference, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
    return report


def check_gradient_set(rows, *, reduced_precision=False):
    """Bound sparse and diffuse error across the complete gradient."""
    elements = sum(row["elements"] for row in rows)
    if not elements:
        raise ValueError("training qualification has no participating gradients")
    rtol = ACCELERATED_GRADIENT_RTOL if reduced_precision else REPLAY_RTOL
    report = {
        "elements": elements,
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "max_abs_reference": max(row["max_abs_reference"] for row in rows),
        "rms_error": math.sqrt(
            sum(row["rms_error"] ** 2 * row["elements"] for row in rows) / elements
        ),
        "rms_reference": math.sqrt(
            sum(row["rms_reference"] ** 2 * row["elements"] for row in rows) / elements
        ),
        "rtol": rtol,
    }
    report["max_abs_bound"] = REPLAY_ATOL + rtol * report["max_abs_reference"]
    report["rms_bound"] = REPLAY_ATOL + rtol * report["rms_reference"]
    if (report["max_abs_error"] > report["max_abs_bound"]
            or report["rms_error"] > report["rms_bound"]):
        raise ValueError(f"full-gradient error exceeds fixed bounds: {report}")
    return report


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


def capture_phase(fn, model=None, stream=None, reduced_precision=False):
    """Return a replay callable retaining its graph, outputs and gradient storage.

    fn returns a tensor or a tuple of tensors. Inputs and parameters must keep
    their addresses. Training is forward/loss/backward, without an optimizer.
    Capture errors and validation failures propagate; never time a fallback.
    Reuse the preparation stream when compilation retains autograd nodes.
    """
    start = time.perf_counter()
    stream = stream if stream is not None else torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def tensors(values):
        return (values,) if isinstance(values, torch.Tensor) else values

    def parameter_gradients():
        return {name: parameter.grad for name, parameter in model.named_parameters()
                if parameter.grad is not None} if model is not None else {}

    totals = {"uncaptured": [], "replays": []}

    def check(values, stage):
        rows = {}
        actual_gradients = parameter_gradients()
        if actual_gradients.keys() != gradients.keys():
            raise ValueError(f"{stage}: changed the set of parameter gradients")
        pairs = [(f"output {index}", value, reference, False)
                 for index, (value, reference) in enumerate(zip(tensors(values), expected, strict=True))]
        pairs.extend((f"gradient {name}", actual_gradients[name], reference, True)
                     for name, reference in gradients.items())
        for label, actual, reference, gradient in pairs:
            try:
                rows[label] = compare_tensors(
                    actual, reference, gradient=gradient,
                    reduced_precision=reduced_precision and gradient,
                )
            except (ValueError, AssertionError) as error:
                raise ValueError(f"{stage} {label}: {error}") from error
        if model is not None:
            totals["replays" if stage == "CUDA graph replay" else "uncaptured"].append(
                check_gradient_set(
                    [row for label, row in rows.items() if label.startswith("gradient ")],
                    reduced_precision=reduced_precision,
                )
            )
        return rows

    ordinary = []
    ordinary_validation_s = 0.0
    ordinary_repeats = ACCELERATED_GRADIENT_REPEATS if reduced_precision and model is not None else 2
    with torch.cuda.stream(stream):
        for index in range(ordinary_repeats + 1):
            if model is not None:
                model.zero_grad(set_to_none=True)
            outputs = fn()
            torch.cuda.synchronize()
            validation_start = time.perf_counter()
            if index == 0:
                expected = tuple(t.detach().cpu().clone() for t in tensors(outputs))
                gradients = {name: grad.detach().cpu().clone() for name, grad in parameter_gradients().items()}
            else:
                ordinary.append(check(outputs, "uncaptured repeat"))
            ordinary_validation_s += time.perf_counter() - validation_start
            del outputs
    torch.cuda.current_stream().wait_stream(stream)

    if model is not None:
        # Allocate gradients during capture; backward then overwrites them on
        # every replay. Do not detach or reset those tensors between replays.
        model.zero_grad(set_to_none=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = fn()

    replay = CapturedPhase(fn, model, graph, outputs)
    capture_s = time.perf_counter() - start - ordinary_validation_s
    validation_start = time.perf_counter()

    replays = []
    for _ in range(2):
        actual = replay()
        torch.cuda.synchronize()
        replays.append(check(actual, "CUDA graph replay"))

    return replay, {
        "status": "captured-and-validated",
        "scope": "forward + loss + backward" if model is not None else "forward",
        "capture_s": capture_s,
        "validation_s": ordinary_validation_s + time.perf_counter() - validation_start,
        "validation": {
            "policy": REPLAY_POLICY,
            "outputs": "all elements",
            "gradient_tensors": len(gradients),
            "gradients": "all elements of every participating parameter",
            "gradient_metric": "whole-gradient RMS and maximum absolute error",
            "uncaptured_calls": ordinary_repeats + 1,
            "uncaptured_repeats": ordinary_repeats,
            "consecutive_replays": 2,
            "rtol": REPLAY_RTOL,
            "atol": REPLAY_ATOL,
            "accelerated_gradient_rtol": ACCELERATED_GRADIENT_RTOL,
            "full_gradient": totals,
            "uncaptured": ordinary,
            "replays": replays,
        },
    }
