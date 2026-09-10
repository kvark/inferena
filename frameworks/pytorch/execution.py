"""Whole-phase CUDA replay with untimed, full-tensor qualification."""

import time
import os
import math
from contextlib import nullcontext
from pathlib import Path

import torch

REPLAY_RTOL = 1e-4
REPLAY_ATOL = 1e-6
REPLAY_POLICY = "bounded-repeat-noise-v2"
NOISE_SAMPLES = 8
NOISE_MARGIN = 2.0
GRADIENT_RMS_CEILING = 0.01


def compare_tensors(actual, reference, *, gradient=False, noise=None, calibrating=False):
    """Compare full tensors; only uncaptured calibration may set noise bounds."""
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
        # Require both a worst-element and a full-vector error bound, separately
        # for every parameter; no pooling across tensors or discarded elements.
        if noise is not None:
            report["max_abs_bound"] += NOISE_MARGIN * noise["max_abs_error"]
            report["rms_bound"] += NOISE_MARGIN * noise["rms_error"]
        if not calibrating and (report["max_abs_error"] > report["max_abs_bound"]
                                or report["rms_error"] > report["rms_bound"]):
            raise ValueError(f"gradient repeatability exceeds fixed bounds: {report}")
    else:
        torch.testing.assert_close(actual, reference, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
    return report


def check_gradient_ceiling(rows):
    """A noisy reference must not grant an unbounded replay allowance."""
    elements = sum(row["elements"] for row in rows)
    if not elements:
        raise ValueError("training qualification has no participating gradients")
    error = math.sqrt(sum(row["rms_error"] ** 2 * row["elements"] for row in rows) / elements)
    reference = math.sqrt(sum(row["rms_reference"] ** 2 * row["elements"] for row in rows) / elements)
    bound = REPLAY_ATOL + GRADIENT_RMS_CEILING * reference
    if error > bound:
        raise ValueError(f"full-gradient RMS error {error} exceeds independent ceiling {bound}")
    return {"rms_error": error, "rms_reference": reference, "rms_bound": bound}


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

    noise = {}
    totals = {"uncaptured": [], "replays": []}

    def check(values, stage, calibrating=False):
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
                rows[label] = compare_tensors(actual, reference, gradient=gradient,
                                             noise=noise.get(label), calibrating=calibrating and gradient)
            except (ValueError, AssertionError) as error:
                raise ValueError(f"{stage} {label}: {error}") from error
        if model is not None:
            totals["replays" if stage == "CUDA graph replay" else "uncaptured"].append(
                check_gradient_ceiling([row for label, row in rows.items() if label.startswith("gradient ")]))
        return rows

    ordinary = []
    ordinary_validation_s = 0.0
    calibration_samples = NOISE_SAMPLES if reduced_precision and model is not None else 1
    with torch.cuda.stream(stream):
        for index in range(calibration_samples + 2):
            if model is not None:
                model.zero_grad(set_to_none=True)
            outputs = fn()
            torch.cuda.synchronize()
            validation_start = time.perf_counter()
            if index == 0:
                expected = tuple(t.detach().cpu().clone() for t in tensors(outputs))
                gradients = {name: grad.detach().cpu().clone() for name, grad in parameter_gradients().items()}
            else:
                calibrating = index < calibration_samples
                rows = check(outputs, "uncaptured repeat", calibrating=calibrating)
                ordinary.append(rows)
                if calibrating:
                    for label, row in rows.items():
                        if label.startswith("gradient "):
                            previous = noise.get(label, {"max_abs_error": 0.0, "rms_error": 0.0})
                            noise[label] = {key: max(row[key], previous[key]) for key in previous}
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
            "gradient_metric": "per-tensor RMS and maximum absolute error",
            "uncaptured_repeats": calibration_samples + 2,
            "calibration_samples": calibration_samples,
            "uncaptured_holdouts": 2,
            "consecutive_replays": 2,
            "rtol": REPLAY_RTOL,
            "atol": REPLAY_ATOL,
            "noise_margin": NOISE_MARGIN,
            "noise": noise,
            "gradient_rms_ceiling": GRADIENT_RMS_CEILING,
            "full_gradient": totals,
            "uncaptured": ordinary,
            "replays": replays,
        },
    }
