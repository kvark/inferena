"""Whole-phase CUDA/HIP/XPU replay with untimed, full-tensor qualification."""

import time
import os
import math
from contextlib import nullcontext
from pathlib import Path

import torch

REPLAY_RTOL = 1e-4
REPLAY_ATOL = 1e-6
REPLAY_POLICY = "fixed-full-tensor-v4"
ACCELERATED_GRADIENT_REPEATS = 8
ACCELERATED_GRADIENT_RTOL = 0.01


class QualificationError(ValueError):
    """A completed numerical check failed; execution errors are not swallowed."""

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = {"kind": "numerical", "message": message, **details}


def compare_tensors(actual, reference, *, gradient=False, reduced_precision=False):
    """Compare full tensors and return metrics for whole-gradient checks."""
    actual = actual.detach().cpu()
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise QualificationError("tensor shape or dtype changed")
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise QualificationError("non-finite values")
    if not actual.numel():
        raise QualificationError("empty qualification tensor")
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
        raise QualificationError("non-finite error metric")
    if not gradient:
        report["elementwise_mismatches"] = int(
            (error.abs() > REPLAY_ATOL + REPLAY_RTOL * reference.abs()).sum())
    # Cancellation affects outputs as well as gradients. Bound sparse error
    # by the tensor maximum and diffuse error by RMS, without fitting either
    # bound to repeats. Every output/loss retains the tight gate in both modes;
    # only accelerated gradients use the separate complete-gradient gate.
    if not (gradient and reduced_precision) and (
            report["max_abs_error"] > report["max_abs_bound"]
            or report["rms_error"] > report["rms_bound"]):
        raise QualificationError(f"tensor repeatability exceeds fixed bounds: {report}", metrics=report)
    return report


def check_gradient_set(rows, *, reduced_precision=False):
    """Bound sparse and diffuse error across the complete gradient."""
    elements = sum(row["elements"] for row in rows)
    if not elements:
        raise QualificationError("training qualification has no participating gradients")
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
        raise QualificationError(f"full-gradient error exceeds fixed bounds: {report}", metrics=report)
    return report


def nsys_range(name):
    return torch.cuda.nvtx.range(name) if "INFERENA_NSYS" in os.environ else nullcontext()


def synchronize(device):
    backend = torch.device(device).type
    if backend in ("cuda", "xpu"):
        getattr(torch, backend).synchronize(device)
    elif backend == "mps":
        torch.mps.synchronize()


def graph_backend(device):
    backend = torch.device(device).type
    if backend not in ("cuda", "xpu"):
        raise ValueError(f"no explicit whole-phase replay API for {backend}")
    return getattr(torch, backend)


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


def capture_phase(fn, model=None, stream=None, reduced_precision=False, device="cuda", capture=True):
    """Qualify ordinary calls, optionally capture and qualify graph replay.

    fn returns a tensor or a tuple of tensors. Inputs and parameters must keep
    their addresses. Training is forward/loss/backward, without an optimizer.
    Capture errors and validation failures propagate; never time a fallback.
    Reuse the preparation stream when compilation retains autograd nodes.
    """
    start = time.perf_counter()
    api = graph_backend(device) if torch.device(device).type in ("cuda", "xpu") else None
    backend = torch.device(device).type
    if capture and api is None:
        raise ValueError(f"no explicit whole-phase replay API for {backend}")
    if api is not None:
        stream = stream if stream is not None else api.Stream(device=device)
        stream.wait_stream(api.current_stream(device))

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
            raise QualificationError("changed the set of parameter gradients", stage=stage)
        if len(tensors(values)) != len(expected):
            raise QualificationError("changed the number of outputs", stage=stage)
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
            except QualificationError as error:
                error.details.update(stage=stage, tensor=label)
                raise
        if model is not None:
            try:
                total = check_gradient_set(
                    [row for label, row in rows.items() if label.startswith("gradient ")],
                    reduced_precision=reduced_precision,
                )
            except QualificationError as error:
                error.details.update(stage=stage, tensor="full gradient")
                raise
            totals["replays" if stage == "graph replay" else "uncaptured"].append(total)
        return rows

    ordinary = []
    ordinary_validation_s = 0.0
    ordinary_repeats = ACCELERATED_GRADIENT_REPEATS if reduced_precision and model is not None else 2
    with api.stream(stream) if api is not None else nullcontext():
        for index in range(ordinary_repeats + 1):
            if model is not None:
                model.zero_grad(set_to_none=True)
            outputs = fn()
            synchronize(device)
            validation_start = time.perf_counter()
            if index == 0:
                expected = tuple(t.detach().cpu().clone() for t in tensors(outputs))
                gradients = {name: grad.detach().cpu().clone() for name, grad in parameter_gradients().items()}
            else:
                ordinary.append(check(outputs, f"uncaptured repeat {index}"))
            ordinary_validation_s += time.perf_counter() - validation_start
            del outputs
    if api is not None:
        api.current_stream(device).wait_stream(stream)

    replay = fn
    replays = []
    capture_s = 0.0
    replay_validation_s = 0.0
    if capture:
        if model is not None:
            # Capture allocates stable gradients that replay overwrites.
            model.zero_grad(set_to_none=True)
        graph_type = api.CUDAGraph if backend == "cuda" else api.XPUGraph
        graph = graph_type()
        try:
            with api.graph(graph, stream=stream):
                outputs = fn()
        except RuntimeError as error:
            # Only a rejected capture with a still-healthy device is recoverable.
            # OOM, device loss, execution faults and unrelated exceptions escape.
            message = str(error).lower()
            if isinstance(error, torch.OutOfMemoryError) or not any(text in message for text in (
                    "operation not permitted when stream is capturing", "not permitted when stream is capturing",
                    "not supported during graph capture", "not supported during capture")):
                raise
            synchronize(device)
            raise QualificationError(str(error), kind="capture", stage="graph capture") from error
        replay = CapturedPhase(fn, model, graph, outputs)
        capture_s = time.perf_counter() - start - ordinary_validation_s
        validation_start = time.perf_counter()
        for _ in range(2):
            actual = replay()
            synchronize(device)
            replays.append(check(actual, "graph replay"))
        replay_validation_s = time.perf_counter() - validation_start

    return replay, {
        "status": "captured-and-validated" if capture else "validated-uncaptured",
        "api": f"torch.{backend}.{graph_type.__name__}" if capture else None,
        "scope": "forward + loss + backward" if model is not None else "forward",
        "capture_s": capture_s,
        "preparation_s": time.perf_counter() - start,
        "validation_s": ordinary_validation_s + replay_validation_s,
        "validation": {
            "policy": REPLAY_POLICY,
            "outputs": "all elements",
            "output_metric": "per-tensor RMS and maximum absolute error",
            "gradient_tensors": len(gradients),
            "gradients": "all elements of every participating parameter",
            "gradient_metric": "whole-gradient RMS and maximum absolute error",
            "uncaptured_calls": ordinary_repeats + 1,
            "uncaptured_repeats": ordinary_repeats,
            "consecutive_replays": len(replays),
            "rtol": REPLAY_RTOL,
            "atol": REPLAY_ATOL,
            "accelerated_gradient_rtol": ACCELERATED_GRADIENT_RTOL,
            "full_gradient": totals,
            "uncaptured": ordinary,
            "replays": replays,
        },
    }
