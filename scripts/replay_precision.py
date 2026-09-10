"""Measure diffusion repeat noise, not qualification or publication timings."""

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frameworks/pytorch"))
import bench
import execution
import torch
from p3hpc import check_torch_identity

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--precision", choices=("strict", "accelerated"), required=True)
parser.add_argument("--deterministic", action="store_true")
args = parser.parse_args()
check_torch_identity(torch.__version__, torch.version.git_version, torch.__version__)
os.environ.update(INFERENA_TORCH_BACKEND="cuda", INFERENA_TORCH_MODE="default",
                  INFERENA_CUDA_GRAPHS="1", INFERENA_STRICT=str(int(args.precision == "strict")),
                  INFERENA_WARMUP_RUNS="5", INFERENA_MEASUREMENT_RUNS="1",
                  INFERENA_REQUIRE_LOCAL_WEIGHTS="1")
os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
if args.precision == "strict":
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
if args.deterministic:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
destination = Path(tempfile.mkdtemp(prefix="inferena-precision-noise-", dir=ROOT.parent))
print(f"Diagnostic: {destination}", flush=True)
report = {"kind": "noise-diagnostic-NOT-qualification", "args": vars(args),
          "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
          "torch_source": torch.version.git_version, "uncaptured": [], "replays": []}
original_compare = execution.compare_tensors
original_capture = bench.capture_phase


def measure(actual, reference, *, gradient=False):
    if not gradient:
        return original_compare(actual, reference)
    actual = actual.detach().cpu()
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    delta = actual - reference
    def rms(value):
        return float(torch.linalg.vector_norm(value, dtype=torch.float64) / math.sqrt(value.numel()))
    # Intentionally report the entire error range, without a gradient gate.
    return {"max_abs_error": float(delta.abs().max()), "rms_error": rms(delta),
            "max_abs_reference": float(reference.abs().max()), "rms_reference": rms(reference)}


def checked_capture(fn, model=None, stream=None):
    if model is None:
        return original_capture(fn, model, stream)
    def gradients():
        return {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
    def compare(actual):
        assert actual.keys() == reference.keys()
        return {name: measure(value, reference[name], gradient=True) for name, value in actual.items()}
    for index in range(16):
        model.zero_grad(set_to_none=True)
        output, loss = fn()
        torch.cuda.synchronize()
        actual = gradients()
        if index == 0:
            reference = actual
            expected = output.detach().cpu().clone(), loss.detach().cpu().clone()
        measure(output, expected[0])
        measure(loss, expected[1])
        report["uncaptured"].append(compare(actual))
        del output, loss
    replay, capture_report = original_capture(fn, model, stream)
    for _ in range(16):
        output, loss = replay()
        torch.cuda.synchronize()
        measure(output, expected[0])
        measure(loss, expected[1])
        report["replays"].append(compare(gradients()))
    report["capture_report"] = capture_report
    return replay, capture_report


execution.compare_tensors = measure
bench.capture_phase = checked_capture
try:
    with (destination / "diagnostic-run.json").open("w") as output, redirect_stdout(output):
        bench.bench("StableDiffusion", bench.MODEL_REGISTRY["StableDiffusion"])
    report["status"] = "measured-not-qualified"
finally:
    (destination / "noise.json").write_text(json.dumps(report, indent=2) + "\n")
    for stage in ("uncaptured", "replays"):
        for error, scale in (("max_abs_error", "max_abs_reference"), ("rms_error", "rms_reference")):
            worst = max(((row[error] / (1e-6 + row[scale]), name, row)
                         for repeat in report[stage] for name, row in repeat.items()), default=None)
            print(stage, error, worst, flush=True)
