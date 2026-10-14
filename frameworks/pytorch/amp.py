"""Accelerated arithmetic policy and fixed, untimed numerical/learning gates."""

from contextlib import contextmanager, nullcontext
import math
import os
from pathlib import Path
import sys

import torch

from execution import QualificationError
from model_precision import AUTOCAST_SELECTION, SMOLLM2_LAYERS, guard_modules, model_guard, preferred_autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from training_gate import (REAL_TEXT, CURVE_POLICY, CURVE_ATOL,
                           assess_trajectories, curve_metrics, load_fixture, workload_for)
from quality import language_model_report

FORWARD_RTOL = 0.02
MAX_SCALE_ATTEMPTS = 8


def resolve_autocast(device, requested="auto", *, strict=False, model_name=None):
    """One declared model/backend recipe before execution; never retry a dtype."""
    if requested not in ("auto", "off", "", "bf16", "fp16"):
        raise ValueError("autocast must be auto, off, bf16 or fp16")
    if strict:
        if requested in ("bf16", "fp16"):
            raise ValueError("strict mode forbids autocast")
        return ""
    if requested != "auto":
        return "" if requested == "off" else requested
    backend = torch.device(device).type
    preferred = preferred_autocast(model_name)
    if backend in ("cuda", "xpu", "mps") and preferred is not None:
        return preferred
    if backend in ("cuda", "xpu"):
        with getattr(torch, backend).device(device):
            return "bf16" if getattr(torch, backend).is_bf16_supported(including_emulation=False) else "fp16"
    # CPU is the graphics-only correctness oracle, not a mixed-precision GPU reference.
    return "fp16" if backend == "mps" else ""


def precision_policy(device, requested, model_name, *, strict=False):
    """Resolve and declare every autocast exception without loading weights."""
    dtype = resolve_autocast(device, requested, strict=strict, model_name=model_name)
    guard = model_guard(model_name, requested, dtype)
    paths = guard_modules(SMOLLM2_LAYERS) if guard is not None else []
    return dict(autocast_requested=requested, autocast=dtype or None,
                autocast_selection=AUTOCAST_SELECTION, model_guard=guard, f32_linear_modules=paths)


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
    try:
        report = curve_metrics(actual, reference)
    except ValueError as error:
        raise QualificationError(str(error)) from error
    if report["status"] != "pass":
        raise QualificationError("loss curve exceeds fixed bounds", metrics=report)
    return report


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
            raise ValueError("strict mode forbids autocast")
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.name)
        if self.dtype is not None and not torch.amp.autocast_mode.is_autocast_available(self.device):
            raise ValueError(f"autocast is unavailable for {self.device}")

    def context(self):
        return torch.autocast(self.device, dtype=self.dtype) if self.dtype is not None else nullcontext()

    def scaler(self):
        return torch.amp.GradScaler(self.device, enabled=self.name == "fp16")

    def metadata(self):
        return {
            "comparison_class": "accelerated", "autocast": self.name,
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
        self.model = model
        self.scaler = policy.scaler()
        self.sink = _GradientSink(model.parameters())

    def __call__(self, loss):
        self.capture_backward(loss)
        self.finish()

    def capture_backward(self, loss):
        """The capture-safe portion; scale's tensor address stays stable."""
        self.scaler.scale(loss).backward()

    def finish(self):
        """Eager housekeeping after every ordinary call or replay, inside timing."""
        self.scaler.step(self.sink)  # public unscale + finite check; step changes no weights
        self.scaler.update()

    def prepare(self, call):
        """Settle FP16 scaling on the restored timing input before its baseline.

        Real-data qualification has its own scaler and different gradients.
        No parameter update occurs here; GradScaler steps only the gradient sink.
        """
        attempts = []
        for _ in range(MAX_SCALE_ATTEMPTS):
            self.model.zero_grad(set_to_none=True)
            previous_calls = self.sink.calls
            scale = self.scaler.get_scale()
            _, loss = call()
            value = float(loss.detach())
            if not math.isfinite(value) or value < 0:
                raise QualificationError("non-finite loss while preparing benchmark scaling")
            self.finish()
            finite = self.sink.calls > previous_calls
            attempts.append({"loss": value, "scale": scale, "finite_gradients": finite})
            if finite:
                self.model.zero_grad(set_to_none=True)
                return {"status": "pass", "scope": "timing input; no parameter updates", "attempts": attempts,
                        "scaler_state": self.scaler.state_dict()}
        raise QualificationError("benchmark scaling did not produce finite gradients", attempts=attempts)

    def metadata(self):
        return {"enabled": self.scaler.is_enabled(), "state": self.scaler.state_dict(),
                "scope": ("scale, backward, unscale, finite check, scale update; no optimizer update"
                          if self.scaler.is_enabled() else "backward; scaling disabled; no optimizer update")}


@contextmanager
def preserved_state(model, policy):
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state()
    api = getattr(torch, policy.device, None)
    gpu_rng = api.get_rng_state_all() if policy.device in ("cuda", "xpu") else None
    mps_rng = api.get_rng_state() if policy.device == "mps" else None
    try:
        yield
    finally:
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        if gpu_rng is not None:
            api.set_rng_state_all(gpu_rng)
        if mps_rng is not None:
            api.set_rng_state(mps_rng)


@contextmanager
def strict_arithmetic():
    """The f32 oracle is independent of the requested TF32/AMP policy."""
    precision = torch.get_float32_matmul_precision()
    matmul = torch.backends.cuda.matmul.allow_tf32
    convolution = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = convolution
        torch.set_float32_matmul_precision(precision)


def training_plan(model_name, inputs, device, *, repeated=False):
    workload = workload_for(model_name, repeated=repeated)
    if workload["policy"] != REAL_TEXT:
        return {"workload": workload, "train": [inputs] * workload["steps"]}
    spec = load_fixture()

    def batch(tokens):
        # Match the benchmark's keys, shapes, strides and dtypes so data changes
        # do not introduce an unbudgeted compiler specialization.
        return {**inputs, "input_ids": torch.tensor([tokens[:-1]], dtype=torch.long, device=device),
                "labels": torch.tensor([tokens[1:]], dtype=torch.long, device=device)}

    return {"workload": workload, **{key: [batch(tokens) for tokens in spec[key]]
                                     for key in ("train", "probe", "heldout")}}


def trajectory(model, forward, loss_fn, inputs, policy, *, autocast, plan=None):
    plan = plan or training_plan("", inputs, policy.device)
    workload = plan["workload"]
    optimizer = torch.optim.SGD(model.parameters(), lr=workload["learning_rate"], foreach=False)
    scaler = policy.scaler() if autocast else torch.amp.GradScaler(policy.device, enabled=False)
    record = {"optimizer": "SGD", "learning_rate": workload["learning_rate"], "losses": [],
              "completed_updates": 0, "skipped_steps": 0, "scale_attempts": [], "overflow_attempts": 0}
    real_text = workload["policy"] == REAL_TEXT
    if real_text:
        if any(isinstance(module, torch.nn.Dropout) and module.p for module in model.modules()):
            raise ValueError("frozen real-text gate requires zero dropout")
        parameter = next(p for name, p in model.named_parameters()
                         if name.removeprefix("_orig_mod.") == "model.norm.weight")
        initial = parameter.detach().cpu().clone()
        record.update(checkpoints=[], parameter_check={"name": "model.norm.weight", "initial": initial.tolist()})

    def loss_call(batch):
        with policy.context() if autocast else nullcontext():
            return loss_fn(forward(model, batch), batch)

    try:
        for step in range(workload["steps"] + 1):
            if real_text and step % workload["checkpoint_every"] == 0:
                with torch.no_grad():
                    checkpoint = {"step": step, **{f"{key}_losses": [float(loss_call(batch)) for batch in plan[key]]
                                                  for key in ("probe", "heldout")}}
                if any(not math.isfinite(x) or x < 0 for key in ("probe_losses", "heldout_losses") for x in checkpoint[key]):
                    raise QualificationError("non-finite or negative evaluation loss")
                record["checkpoints"].append(checkpoint)
                print(f"[pytorch] {'candidate' if autocast else 'f32 reference'} real-text SGD: {step}/{workload['steps']}",
                      file=sys.stderr, flush=True)
            if step == workload["steps"]:
                break
            value, attempts = optimizer_update(optimizer, scaler, lambda: loss_call(plan["train"][step]))
            record["losses"].append(value)
            record["scale_attempts"].append(attempts)
            record["overflow_attempts"] += len(attempts) - 1
            record["completed_updates"] = step + 1
        if real_text:
            final = parameter.detach().cpu()
            if not torch.isfinite(final).all():
                raise QualificationError("non-finite parameter after real-text SGD")
            record["parameter_check"].update(final=final.tolist(), changed=not torch.equal(initial, final))
    except QualificationError as error:
        error.details.update(trajectory={"arithmetic": (policy.name or "accelerated-f32") if autocast else "strict-f32",
                                        **record})
        raise
    record["scaler_state"] = scaler.state_dict()
    return record


def prepare_reference(model, forward, loss_fn, inputs, policy, *, training=True, snapshot=None, plan=None,
                      latency_inputs=None):
    """Compute the f32 oracle before any candidate compilation/execution can affect it.

    Reuse the parameters and restore them in place; do not require a second
    GPU copy of a large model. Only the output tensor and small receipts survive.
    """
    reference = {}
    with preserved_state(model, policy), strict_arithmetic():
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training and snapshot is not None):
            output = forward(model, inputs)
            loss = loss_fn(output, inputs)
            reference["output"] = output.detach().float().cpu().clone()
            if snapshot is not None:
                if training:
                    loss.backward()
                reference["correctness_reference"] = snapshot(model, output, loss, training)
        del output, loss
        if latency_inputs is not None:
            with torch.no_grad():
                reference["latency_output"] = forward(model, latency_inputs).detach().float().cpu().clone()
        if training:
            try:
                reference["trajectory"] = trajectory(model, forward, loss_fn, inputs, policy, autocast=False, plan=plan)
            except QualificationError as error:
                reference["training_failure"] = error.details
    return reference


def forward_quality(actual, reference, labels=None):
    """Full numerical diagnostics, with a task-quality gate for LM logits."""
    actual, reference = actual.detach().float().cpu(), reference.detach().float().cpu()
    if actual.shape != reference.shape or not actual.numel() or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise QualificationError("accelerated forward shape/non-finite failure")
    # NumPy avoids a large multithreaded CPU pool for these untimed readbacks.
    import numpy as np
    a, r = actual.numpy(), reference.numpy()
    error = a.astype(np.float64) - r
    report = {"elements": actual.numel(), "rtol": FORWARD_RTOL, "atol": CURVE_ATOL,
              "rms_error": float(np.sqrt(np.mean(error * error))),
              "rms_bound": CURVE_ATOL + FORWARD_RTOL * float(np.sqrt(np.mean(np.square(r, dtype=np.float64)))),
              "max_abs_error": float(np.abs(error).max()),
              "max_abs_bound": CURVE_ATOL + FORWARD_RTOL * float(np.abs(r).max())}
    if labels is None:
        if report["rms_error"] > report["rms_bound"] or report["max_abs_error"] > report["max_abs_bound"]:
            raise QualificationError("accelerated forward exceeds fixed bounds", metrics=report)
        return report
    if actual.ndim < 2 or actual.shape[:-1] != labels.shape:
        raise QualificationError("language-model target shape changed")
    targets = labels.detach().cpu().numpy().reshape(-1)
    vocabulary = actual.shape[-1]
    if vocabulary < 2 or np.any(targets < 0) or np.any(targets >= vocabulary):
        raise QualificationError("language-model targets are out of range")
    kl, losses, reference_losses = [], [], []
    for av, rv, target in zip(a.reshape(-1, vocabulary), r.reshape(-1, vocabulary), targets, strict=True):
        av, rv = av.astype(np.float64), rv.astype(np.float64)
        av -= av.max()
        rv -= rv.max()
        al, rl = av - np.log(np.exp(av).sum()), rv - np.log(np.exp(rv).sum())
        kl.append(max(0., float(np.sum(np.exp(rl) * (rl - al)))))
        losses.append(float(-al[target]))
        reference_losses.append(float(-rl[target]))
    quality = language_model_report(kl, losses, reference_losses, vocabulary)
    report.update(raw_logits_role="diagnostic", language_model=quality)
    if quality["status"] != "pass":
        raise QualificationError("language-model forward exceeds fixed quality bounds", metrics=report)
    return report


def qualify(model, candidate, forward, loss_fn, inputs, policy, *, training=True, snapshot=None, reference=None, plan=None,
            language_model=False, latency_inputs=None):
    """Qualify the requested candidate and restore parameters, buffers and RNG."""
    if reference is None:
        reference = prepare_reference(model, forward, loss_fn, inputs, policy, training=training, snapshot=snapshot, plan=plan,
                                      latency_inputs=latency_inputs)
    plan = plan or training_plan("", inputs, policy.device)
    prepared = reference
    reference = prepared["output"]
    with preserved_state(model, policy):
        with torch.no_grad(), policy.context():
            actual = forward(candidate, inputs).detach().float().cpu()
        forward_report = forward_quality(actual, reference, inputs["labels"] if language_model else None)
        report = {"status": "pass", "forward": forward_report, "training": None,
                  "reference": "eager strict f32", "candidate": "requested execution mode",
                  "scope": "untimed full-output and declared SGD workload; state restored"}
        if language_model:
            if latency_inputs is None or "latency_output" not in prepared:
                raise QualificationError("missing language-model latency reference")
            try:
                with torch.no_grad(), policy.context():
                    actual_latency = forward(candidate, latency_inputs)
                report["latency_forward"] = forward_quality(actual_latency, prepared["latency_output"], latency_inputs["labels"])
            except QualificationError as error:
                report["status"] = "partial"
                report["latency_forward"] = {"status": "failed", "failure": error.details}
        if "correctness_reference" in prepared:
            report["correctness_reference"] = prepared["correctness_reference"]
        if training:
            try:
                if "training_failure" in prepared:
                    failure = prepared["training_failure"]
                    raise QualificationError(failure["message"], **{key: value for key, value in failure.items() if key != "message"})
                actual = trajectory(candidate, forward, loss_fn, inputs, policy, autocast=True, plan=plan)
            except QualificationError as error:
                report["status"] = "partial"
                report["training"] = {"status": "failed", "policy": CURVE_POLICY,
                                      "workload": plan["workload"], "requested_updates": plan["workload"]["steps"],
                                      "failure": error.details}
                return report
            reference = prepared["trajectory"]
            assessment = assess_trajectories(actual, reference, plan["workload"])
            report["training"] = {**assessment["comparisons"]["training"], **actual, **assessment,
                                  "workload": plan["workload"], "reference_losses": reference["losses"],
                                  "reference_trajectory": reference}
            if assessment["status"] != "pass":
                report["status"] = "partial"
        return report
