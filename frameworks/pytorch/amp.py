"""Explicit AMP reference policy and fixed, untimed numerical gates."""

from contextlib import nullcontext
import math
import os

import torch

from execution import QualificationError

CURVE_STEPS = 32
CURVE_LEARNING_RATE = 0.001
CURVE_POLICY = "loss-curve-v1"
CURVE_RTOL = 0.02
CURVE_L2_LIMIT = 0.01
CURVE_ATOL = 1e-6
FORWARD_RTOL = 0.02
MAX_SCALE_ATTEMPTS = 8


def optimizer_update(optimizer, scaler, loss_call):
    """Complete one update on one batch; retain every overflow/scale attempt.

    An overflow leaves weights unchanged. Repeat that batch after GradScaler
    lowers the scale, rather than silently dropping it from a matched curve.
    """
    updates = []
    attempts = []
    hook = optimizer.register_step_post_hook(lambda *_: updates.append(True))
    try:
        for _ in range(MAX_SCALE_ATTEMPTS):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_call()
            value = float(loss.detach())
            if not math.isfinite(value) or value < 0:
                raise QualificationError("non-finite or negative loss before optimizer update")
            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            attempts.append({"loss": value, "scale": scale, "updated": bool(updates)})
            if updates:
                return value, attempts
        raise QualificationError("gradient scaling did not produce a finite update", attempts=attempts)
    finally:
        hook.remove()


def compare_loss_curves(actual, reference):
    """Fixed bounds, declared before running either trajectory; no fitted envelope."""
    if not actual or len(actual) != len(reference):
        raise QualificationError("loss curves must have the same nonzero length")
    if not all(math.isfinite(value) and value >= 0 for value in [*actual, *reference]):
        raise QualificationError("non-finite or negative loss in convergence check")
    errors = [abs(a - r) for a, r in zip(actual, reference, strict=True)]
    bounds = [CURVE_ATOL + CURVE_RTOL * abs(r) for r in reference]
    relative_l2 = math.sqrt(sum(error * error for error in errors)) / max(
        math.sqrt(sum(value * value for value in reference)), CURVE_ATOL)
    report = {
        "policy": CURVE_POLICY, "steps": len(actual), "rtol": CURVE_RTOL,
        "atol": CURVE_ATOL, "relative_l2_limit": CURVE_L2_LIMIT,
        "relative_l2": relative_l2, "max_abs_error": max(errors),
        "mismatched_steps": sum(error > bound for error, bound in zip(errors, bounds, strict=True)),
    }
    if report["mismatched_steps"] or relative_l2 > CURVE_L2_LIMIT:
        raise QualificationError("loss curve exceeds fixed bounds", metrics=report)
    return {"status": "pass", **report}


class _GradientSink(torch.optim.Optimizer):
    """Let GradScaler check/unscale gradients without updating any parameter."""

    def __init__(self, parameters):
        super().__init__(parameters, {})
        self.calls = 0

    def step(self, closure=None):
        assert closure is None
        self.calls += 1


class AmpPolicy:
    def __init__(self, device, name=None):
        self.device = torch.device(device).type
        self.name = os.environ.get("INFERENA_AUTOCAST", "") if name is None else name
        if self.name not in ("", "bf16", "fp16"):
            raise ValueError("INFERENA_AUTOCAST must be bf16 or fp16")
        if self.name and os.environ.get("INFERENA_STRICT") == "1":
            raise ValueError("AMP is a separate reference class; incompatible with INFERENA_STRICT=1")
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.name)
        if self.dtype is not None and not torch.amp.autocast_mode.is_autocast_available(self.device):
            raise ValueError(f"autocast is unavailable for {self.device}")

    def context(self):
        return torch.autocast(self.device, dtype=self.dtype) if self.dtype is not None else nullcontext()

    def scaler(self):
        return torch.amp.GradScaler(self.device, enabled=self.name == "fp16")

    def metadata(self):
        return {
            "comparison_class": f"amp-{self.name}", "autocast": self.name,
            "tensor_storage": "f32 parameters; autocast activations",
            "matmul_inputs": self.name, "convolution_inputs": self.name,
            "accumulation": "backend autocast policy", "output": "operator-dependent",
            "loss": "f32", "reduced_precision_allowed": True,
            "gradient_scaling": "torch.amp.GradScaler" if self.name == "fp16" else "disabled (bf16)",
            "remaining_f32_matmul": "highest; TF32 disabled",
        }


class BenchmarkBackward:
    """Time AMP housekeeping, while preserving the benchmark's no-update scope."""

    def __init__(self, model, policy):
        self.scaler = policy.scaler()
        self.sink = _GradientSink(model.parameters())

    def __call__(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.sink)  # public unscale + finite check; step changes no weights
        self.scaler.update()

    def metadata(self):
        return {"enabled": self.scaler.is_enabled(), "state": self.scaler.state_dict(),
                "scope": "scale, backward, unscale, finite check, scale update; no optimizer update"}


def qualify(model, candidate, forward, loss_fn, inputs, policy, *, training=True):
    """Compare the actual execution candidate with eager f32, restoring all state.

    The fixed-input trajectory is a precision gate, not evidence of learning on
    real data. scripts/convergence.py runs the separate real-data experiment.
    """
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state()
    device_api = getattr(torch, policy.device, None)
    gpu_rng = (device_api.get_rng_state_all() if policy.device in ("cuda", "xpu") else None)

    def restore():
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        if gpu_rng is not None:
            device_api.set_rng_state_all(gpu_rng)

    def trajectory(current, autocast):
        restore()
        optimizer = torch.optim.SGD(current.parameters(), lr=CURVE_LEARNING_RATE, foreach=False)
        scaler = policy.scaler() if autocast else torch.amp.GradScaler(policy.device, enabled=False)
        losses = []
        attempts = []
        def loss_call():
            with policy.context() if autocast else nullcontext():
                return loss_fn(forward(current, inputs), inputs)
        for _ in range(CURVE_STEPS):
            value, step_attempts = optimizer_update(optimizer, scaler, loss_call)
            losses.append(value)
            attempts.append(step_attempts)
        return losses, attempts, scaler.state_dict()

    try:
        with torch.no_grad():
            reference = forward(model, inputs).detach().float().cpu()
            with policy.context():
                actual = forward(candidate, inputs).detach().float().cpu()
        if actual.shape != reference.shape or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
            raise QualificationError("AMP forward shape/non-finite failure")
        error = (actual - reference).double()
        reference = reference.double()
        forward_report = {
            "elements": actual.numel(), "rtol": FORWARD_RTOL, "atol": CURVE_ATOL,
            "rms_error": float(error.square().mean().sqrt()),
            "rms_bound": CURVE_ATOL + FORWARD_RTOL * float(reference.square().mean().sqrt()),
            "max_abs_error": float(error.abs().max()),
            "max_abs_bound": CURVE_ATOL + FORWARD_RTOL * float(reference.abs().max()),
        }
        if (forward_report["rms_error"] > forward_report["rms_bound"]
                or forward_report["max_abs_error"] > forward_report["max_abs_bound"]):
            raise QualificationError("AMP forward exceeds fixed bounds", metrics=forward_report)
        report = {"status": "pass", "forward": forward_report, "training": None,
                  "reference": "eager strict f32", "candidate": "requested execution mode",
                  "scope": "untimed full-output and fixed-input SGD trajectory; state restored"}
        if training:
            reference_losses, _, _ = trajectory(model, False)
            actual_losses, attempts, scaler_state = trajectory(candidate, True)
            trajectory_report = {"optimizer": "SGD", "learning_rate": CURVE_LEARNING_RATE,
                                 "reference_losses": reference_losses, "losses": actual_losses,
                                 "skipped_steps": 0, "completed_updates": CURVE_STEPS,
                                 "scale_attempts": attempts, "scaler_state": scaler_state,
                                 "overflow_attempts": sum(len(step) - 1 for step in attempts)}
            try:
                report["training"] = {**compare_loss_curves(actual_losses, reference_losses), **trajectory_report}
            except QualificationError as error:
                error.details["trajectory"] = trajectory_report
                raise
        return report
    finally:
        restore()
