"""Precision boundaries, scaling underflow, state restoration and curve failures."""

import os
import unittest
from unittest.mock import patch

import torch

from amp import AmpPolicy, BenchmarkBackward, compare_loss_curves, qualify, optimizer_update, precision_policy, resolve_autocast
from execution import QualificationError, compare_tensors


class AmpTest(unittest.TestCase):
    def test_non_language_latency_error_is_excluded_without_losing_other_phases(self):
        from contextlib import redirect_stdout
        import copy
        import io
        import json
        from bench import _bench
        from precision import check_precision_qualification

        class Model(torch.nn.Module):
            def __init__(self, kind):
                super().__init__()
                self.kind = kind
                self.projection = torch.nn.Linear(3, 2)
                with torch.no_grad():
                    self.projection.weight.fill_(.125)
                    self.projection.bias.fill_(.25)

            def forward(self, images=None, noisy_actions=None, timestep=None, vlm_kv=None):
                values = images.mean((-2, -1)) if self.kind == "resnet" else noisy_actions
                output = self.projection(values)
                if values.shape[-2] == 1 and torch.is_autocast_enabled("cpu"):
                    output = output + 1
                return output

        def measure(fn, *_, before=None):
            if before is not None:
                before()
            return fn(), [1.0] * 20, {"runs": 5, "seconds": 10.0}

        env = {"INFERENA_AUTOCAST": "bf16", "INFERENA_STRICT": "0",
               "INFERENA_TORCH_MODE": "eager", "INFERENA_GRAPH_REPLAY": "0"}
        for name, kind, inputs in [
                ("ResNet-50", "resnet", dict(images=torch.ones(4, 3, 2, 2), labels=torch.zeros(4, dtype=torch.long))),
                ("SmolVLA", "smolvla", dict(noisy_actions=torch.ones(1, 4, 3),
                                          timestep=torch.zeros(1, 1, 3), vlm_kv=torch.ones(1, 16, 3)))]:
            with self.subTest(model=name), patch.dict(os.environ, env), \
                    patch("bench.load_model", return_value=Model(kind)), \
                    patch("bench.prepare_inputs", return_value=inputs), \
                    patch("bench._measure_call", side_effect=measure):
                output = io.StringIO()
                with redirect_stdout(output):
                    _bench(name, {"type": kind}, "cpu", None)
                record = json.loads(output.getvalue())
                self.assertEqual(record["status"], "partial")
                self.assertEqual(check_precision_qualification(record),
                                 {"inference": "pass", "latency": "failed", "training": "pass"})
                self.assertIsNone(record["timings"]["latency_ms"])
                self.assertIsNone(record["timing_samples_ms"]["latency"])
                self.assertEqual(len(record["timing_samples_ms"]["inference"]), 20)
                self.assertEqual(len(record["timing_samples_ms"]["training"]), 20)
                forged = copy.deepcopy(record)
                forged["timings"]["latency_ms"] = 1.0
                with self.assertRaisesRegex(ValueError, "latency supplies timing"):
                    check_precision_qualification(forged)
                forged = copy.deepcopy(record)
                gate = forged["execution"]["precision_qualification"]
                gate["latency_forward"] = copy.deepcopy(gate["forward"])
                with self.assertRaisesRegex(ValueError, "wrong output shape"):
                    check_precision_qualification(forged)
                del gate["latency_forward"]
                with self.assertRaises(KeyError):
                    check_precision_qualification(forged)

    def test_bad_latency_keeps_qualified_prefill_and_training(self):
        from contextlib import redirect_stdout
        from types import SimpleNamespace
        import copy
        import io
        import json
        from bench import _bench
        from precision import check_precision_qualification

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(4, 4)
                self.projection = torch.nn.Linear(4, 4)

            def forward(self, input_ids, attention_mask, use_cache=False):
                logits = self.projection(self.embedding(input_ids))
                if input_ids.shape[1] == 1 and torch.is_autocast_enabled("cpu"):
                    logits = logits * 100
                return SimpleNamespace(logits=logits)

        def inputs(model_type, model, dev, seq_len=8):
            ids = (torch.arange(seq_len) % 4).reshape(1, -1)
            return dict(input_ids=ids, labels=(ids + 1) % 4, attention_mask=torch.ones_like(ids))

        def measure(fn, *_, before=None):
            if before is not None:
                before()
            return fn(), [1.0] * 20, {"runs": 5, "seconds": 10.0}

        torch.manual_seed(17)
        model = Model()
        env = {"INFERENA_AUTOCAST": "bf16", "INFERENA_STRICT": "0",
               "INFERENA_TORCH_MODE": "eager", "INFERENA_GRAPH_REPLAY": "0"}
        with patch.dict(os.environ, env), patch("bench.load_model", return_value=model), \
                patch("bench.prepare_inputs", side_effect=inputs), patch("bench._measure_call", side_effect=measure):
            output = io.StringIO()
            with redirect_stdout(output):
                _bench("SmolLM2-test", {"type": "causal_lm"}, "cpu", None)
        record = json.loads(output.getvalue())
        self.assertEqual(record["status"], "partial")
        self.assertEqual(check_precision_qualification(record),
                         {"inference": "pass", "latency": "failed", "training": "pass"})
        self.assertEqual(record["execution"]["failure"]["phase"], "latency")
        self.assertIsNone(record["timings"]["latency_ms"])
        self.assertIsNone(record["timing_samples_ms"]["latency"])
        self.assertEqual(len(record["timing_samples_ms"]["training"]), 20)
        forged = copy.deepcopy(record)
        forged["timings"]["latency_ms"] = 1.0
        with self.assertRaisesRegex(ValueError, "latency supplies timing"):
            check_precision_qualification(forged)

    def test_language_model_gate_is_shift_invariant_and_checks_each_token(self):
        from amp import forward_quality
        reference = torch.tensor([0.9, 0.1]).log().repeat(1, 512, 1)
        labels = torch.zeros((1, 512), dtype=torch.int64)
        report = forward_quality(reference + 1.0, reference, labels)
        self.assertEqual(report["language_model"]["status"], "pass")
        self.assertGreater(report["max_abs_error"], report["max_abs_bound"])
        with self.assertRaises(QualificationError):
            forward_quality(reference + 1.0, reference)
        for candidate in (torch.zeros_like(reference), -reference, reference.flip(-1)):
            with self.assertRaises(QualificationError):
                forward_quality(candidate, reference, labels)
        sparse = reference.clone()
        sparse[0, 17] = 0
        with self.assertRaises(QualificationError) as error:
            forward_quality(sparse, reference, labels)
        quality = error.exception.details["metrics"]["language_model"]
        self.assertLess(quality["mean_kl"], quality["mean_kl_limit"])
        self.assertLess(quality["loss_error"], quality["loss_bound"])
        self.assertGreater(quality["max_kl"], quality["token_kl_limit"])

    def test_classes_are_explicit(self):
        with patch.dict(os.environ, {"INFERENA_STRICT": "0"}):
            self.assertEqual(AmpPolicy("cpu", "bf16").metadata()["comparison_class"], "accelerated")
            with self.assertRaises(ValueError):
                AmpPolicy("cpu", "f16")
        with patch.dict(os.environ, {"INFERENA_STRICT": "1"}):
            with self.assertRaisesRegex(ValueError, "strict mode forbids"):
                AmpPolicy("cpu", "fp16")

    def test_capability_policy_and_strict_boundary(self):
        from contextlib import nullcontext
        for backend in ("cuda", "xpu"):
            for supported in (False, True):
                with patch.object(getattr(torch, backend), "device", return_value=nullcontext()), \
                        patch.object(getattr(torch, backend), "is_bf16_supported", return_value=supported) as probe:
                    self.assertEqual(resolve_autocast(backend), "bf16" if supported else "fp16")
                    probe.assert_called_once_with(including_emulation=False)
        self.assertEqual(resolve_autocast("cpu"), "")
        self.assertEqual(resolve_autocast("mps"), "fp16")
        self.assertEqual(resolve_autocast("cuda", strict=True), "")
        self.assertEqual(resolve_autocast("cuda", "off"), "")
        with self.assertRaisesRegex(ValueError, "strict"):
            resolve_autocast("cuda", "bf16", strict=True)

    def test_model_recipe_is_fixed_before_execution_and_explicit_requests_win(self):
        from contextlib import ExitStack, nullcontext
        for device in ("cuda", "xpu", "mps"):
            with self.subTest(device=device), ExitStack() as stack:
                if device in ("cuda", "xpu"):
                    stack.enter_context(patch.object(getattr(torch, device), "device", return_value=nullcontext()))
                    stack.enter_context(patch.object(getattr(torch, device), "is_bf16_supported", return_value=True))
                policy = precision_policy(device, "auto", "SmolLM2-135M")
                default = "fp16" if device == "mps" else "bf16"
                self.assertEqual(policy["phase_autocast"], dict(inference=default, training=default, latency=None))
                self.assertIsNone(policy["model_guard"])
                self.assertEqual(policy["model_guard_phases"], [])
                self.assertEqual(policy["f32_linear_modules"], [])
                for requested, dtype in (("bf16", "bf16"), ("fp16", "fp16"), ("off", None)):
                    explicit = precision_policy(device, requested, "SmolLM2-135M")
                    self.assertEqual(explicit["autocast"], dtype)
                    self.assertIsNone(explicit["model_guard"])
                    self.assertEqual(explicit["f32_linear_modules"], [])
                    self.assertEqual(set(explicit["phase_autocast"].values()), {dtype})
                strict = precision_policy(device, "off", "SmolLM2-135M", strict=True)
                self.assertIsNone(strict["autocast"])
                self.assertIsNone(strict["model_guard"])
        cpu = precision_policy("cpu", "auto", "SmolLM2-135M")
        self.assertIsNone(cpu["autocast"])
        self.assertEqual(cpu["f32_linear_modules"], [])

    def test_latency_qualification_uses_its_declared_dtype(self):
        model = torch.nn.Linear(4, 4)
        with torch.no_grad():
            model.weight.fill_(0.125)
            model.bias.zero_()
        calls = []
        def forward(current, batch):
            calls.append((batch["x"].shape[1], torch.get_autocast_dtype("cpu")
                          if torch.is_autocast_enabled("cpu") else None))
            return current(batch["x"])
        def batch(tokens):
            return dict(x=torch.ones(1, tokens, 4), labels=torch.zeros(1, tokens, dtype=torch.long))
        def loss(output, inputs):
            return torch.nn.functional.cross_entropy(output.float().reshape(-1, 4), inputs["labels"].reshape(-1))
        for name, expected in (("fp16", torch.float16), ("", None)):
            calls.clear()
            with self.subTest(latency=name or "f32"), patch.dict(os.environ, {"INFERENA_STRICT": "0"}):
                result = qualify(model, model, forward, loss, batch(2), AmpPolicy("cpu", "bf16"),
                                 training=False, language_model=True, latency_inputs=batch(1),
                                 latency_policy=AmpPolicy("cpu", name))
            self.assertEqual(result["status"], "pass")
            self.assertIn((2, torch.bfloat16), calls)
            self.assertIn((1, expected), calls)
            self.assertNotIn((1, torch.bfloat16), calls)

    def test_scaled_backward_recovers_small_gradients_without_updating_weights(self):
        with patch.dict(os.environ, {"INFERENA_STRICT": "0"}):
            policy = AmpPolicy("cpu", "fp16")
        model = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.125)
        initial = model.weight.detach().clone()
        x = torch.full((2, 4), 0.0001)
        model(x).square().mean().backward()
        expected = model.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        with policy.context():
            loss = model(x).float().square().mean()
        loss.backward()
        self.assertTrue(torch.count_nonzero(model.weight.grad) == 0)
        backward = BenchmarkBackward(model, policy)
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            with policy.context():
                loss = model(x).float().square().mean()
            backward(loss)
            torch.testing.assert_close(model.weight.grad, expected, rtol=0.002, atol=0)
            torch.testing.assert_close(model.weight, initial, rtol=0, atol=0)
        self.assertEqual(backward.sink.calls, 2)

    def test_curve_rejects_missing_nonfinite_and_divergent_trajectories(self):
        for actual, reference in (([], []), ([1], [1, 2]), ([float("nan")], [1]),
                                  ([float("inf")], [1]), ([-1], [1]),
                                  ([1.0, 0.9, 0.8], [1.0, 0.7, 0.5])):
            with self.subTest(actual=actual), self.assertRaises(QualificationError):
                compare_loss_curves(actual, reference)
        self.assertEqual(compare_loss_curves([1.001, 0.499], [1.0, 0.5])["status"], "pass")

    def test_qualification_restores_parameters_and_rng(self):
        with patch.dict(os.environ, {"INFERENA_STRICT": "0"}):
            policy = AmpPolicy("cpu", "bf16")
        torch.manual_seed(17)
        model = torch.nn.Linear(4, 2)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        inputs = torch.ones(2, 4)
        report = qualify(model, model, lambda m, x: m(x), lambda y, _: y.float().square().mean(), inputs, policy)
        self.assertEqual(report["training"]["steps"], 32)
        self.assertLess(report["training"]["reference_losses"][-1], report["training"]["reference_losses"][0])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with self.assertRaises(QualificationError):
            qualify(model, lambda x: model(x) + 1, lambda m, x: m(x),
                    lambda y, _: y.float().square().mean(), inputs, policy)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_nonfinite_training_retains_forward_and_restores_state(self):
        with patch.dict(os.environ, {"INFERENA_STRICT": "0"}):
            policy = AmpPolicy("cpu", "bf16")
        model = torch.nn.Linear(4, 2)
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        inputs = torch.ones(2, 4)
        with patch("amp.optimizer_update", side_effect=QualificationError("non-finite loss")):
            report = qualify(model, model, lambda m, x: m(x), lambda y, _: y.float().square().mean(), inputs, policy)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["training"]["failure"]["trajectory"]["arithmetic"], "strict-f32")
        self.assertEqual(report["training"]["failure"]["trajectory"]["completed_updates"], 0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_bfloat16_repeatability_checks_all_elements(self):
        reference = torch.ones(16, dtype=torch.bfloat16)
        self.assertEqual(compare_tensors(reference, reference)["elements"], 16)
        changed = reference.clone()
        changed[-1] += 1
        with self.assertRaises(QualificationError):
            compare_tensors(changed, reference)

    def test_benchmark_scaler_settles_before_repeatability_without_parameter_updates(self):
        policy = AmpPolicy("cpu", "fp16")
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        initial = model.weight.detach().clone()
        backward = BenchmarkBackward(model, policy)
        def call():
            with policy.context():
                output = model(torch.tensor([[2.0]]))
                loss = output.float().square().mean()
            backward.capture_backward(loss)
            return output, loss
        report = backward.prepare(call)
        self.assertEqual(report["status"], "pass")
        self.assertGreater(len(report["attempts"]), 1)
        self.assertTrue(report["attempts"][-1]["finite_gradients"])
        self.assertTrue(all(not attempt["finite_gradients"] for attempt in report["attempts"][:-1]))
        torch.testing.assert_close(model.weight, initial, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)

    def test_real_text_qualification_restores_state_and_uses_diverse_batches(self):
        from amp import training_plan
        from training_gate import SMOLLM2_HIDDEN_SIZES
        class Model(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.norm = torch.nn.LayerNorm(width, bias=False)
                self.seen = set()
            def forward(self, batch):
                self.seen.add(tuple(batch["input_ids"].flatten().tolist()))
                return self.model.norm.weight
        policy = AmpPolicy("cpu", "")
        inputs = {"input_ids": torch.arange(128).reshape(1, -1), "labels": torch.arange(1, 129).reshape(1, -1)}
        for name, width in SMOLLM2_HIDDEN_SIZES.items():
            with self.subTest(model=name):
                model = Model(width)
                initial = model.model.norm.weight.detach().clone()
                plan = training_plan(name, inputs, "cpu")
                rng = torch.get_rng_state().clone()
                report = qualify(model, model, lambda m, batch: m(batch), lambda output, _: output.square().sum(),
                                 inputs, policy, plan=plan)
                self.assertEqual(report["training"]["status"], "pass")
                self.assertEqual(report["training"]["completed_updates"], 256)
                self.assertGreater(len(model.seen), 185)
                torch.testing.assert_close(model.model.norm.weight, initial, rtol=0, atol=0)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() or torch.xpu.is_available(), "CUDA/HIP/XPU required")
    def test_compiled_amp_replay_keeps_scaling_live_and_does_not_update_parameters(self):
        from execution import capture_phase, graph_backend, synchronize
        from bench import select_compiler_backend
        device = "cuda" if torch.cuda.is_available() else "xpu"
        select_compiler_backend(device)
        api = graph_backend(device)
        for dtype in ("bf16", "fp16"):
            with self.subTest(dtype=dtype), patch.dict(os.environ, {"INFERENA_STRICT": "0"}):
                torch.manual_seed(19)
                model = torch.nn.Linear(32, 16).to(device)
                inputs = torch.full((8, 32), 0.001, device=device)
                with torch.no_grad():
                    model.weight.fill_(0.01)
                    model.bias.zero_()
                initial = [p.detach().clone() for p in model.parameters()]
                policy = AmpPolicy(device, dtype)
                backward = BenchmarkBackward(model, policy)
                compiled = torch.compile(model, options={"triton.cudagraphs": False})
                stream = api.Stream(device=device)
                stream.wait_stream(api.current_stream(device))

                def training():
                    with policy.context():
                        output = compiled(inputs)
                        loss = output.float().square().mean()
                    backward.capture_backward(loss)
                    return output, loss

                with api.stream(stream):
                    replay, report = capture_phase(training, compiled, stream=stream, device=device,
                                                   reduced_precision=True, after=backward.finish)
                    self.assertEqual(report["status"], "captured-and-validated")
                    stable = [p.grad for p in model.parameters()]
                    expected = [g.detach().clone() for g in stable]
                    calls = backward.sink.calls
                    if dtype == "fp16":
                        state = backward.scaler.state_dict()
                        state.update(growth_interval=2, _growth_tracker=0)
                        backward.scaler.load_state_dict(state)
                        before_scale = backward.scaler.get_scale()
                    for _ in range(4):
                        replay()
                        synchronize(device)
                        for parameter, storage, gradient, value in zip(model.parameters(), stable, expected, initial):
                            self.assertIs(parameter.grad, storage)
                            torch.testing.assert_close(parameter.grad, gradient, rtol=0.001, atol=1e-8)
                            torch.testing.assert_close(parameter, value, rtol=0, atol=0)
                    self.assertEqual(backward.sink.calls, calls + 4)
                    if dtype == "fp16":
                        self.assertGreater(backward.scaler.get_scale(), before_scale)

    def test_accelerated_amp_record_carries_a_separate_f32_oracle(self):
        from contextlib import redirect_stdout
        import io
        import json
        from bench import _bench
        model = torch.nn.Linear(4, 2)
        with torch.no_grad():
            model.weight.fill_(0.125)
            model.bias.zero_()
        env = {"INFERENA_AUTOCAST": "bf16", "INFERENA_STRICT": "0",
               "INFERENA_TORCH_MODE": "eager", "INFERENA_GRAPH_REPLAY": "0", "INFERENA_BUDGET_ENFORCED": "1"}
        def measure(fn, *_, before=None):
            if before is not None:
                before()
            return fn(), [1.0] * 20, {"runs": 5, "seconds": 10.0}
        with patch.dict(os.environ, env), patch("bench.load_model", return_value=model), \
                patch("bench.prepare_inputs", return_value={"input_features": torch.ones(1, 4)}), \
                patch("bench._measure_call", side_effect=measure):
            output = io.StringIO()
            with redirect_stdout(output):
                _bench("test", {"type": "whisper"}, "cpu", None)
            record = json.loads(output.getvalue())
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["protocol"]["name"], "inferena-graph-replay-v18")
        self.assertEqual(record["precision"]["comparison_class"], "accelerated")
        self.assertEqual(record["correctness_reference"]["precision"], "strict-f32")
        self.assertIsNotNone(record["correctness_reference"]["outputs"]["grad_norm"])
        self.assertEqual(record["execution"]["precision_qualification"]["training"]["status"], "pass")
        self.assertFalse(record["execution"]["gradient_scaler"]["enabled"])
        import copy
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from amp_reference import check_record
        check_record(record, "test", "bf16", "eager", "cpu")
        forward_only = copy.deepcopy(record)
        forward_only["protocol"]["training_requested"] = False
        forward_only["protocol"]["warmup"].pop("training")
        forward_only["correctness_reference"]["training_requested"] = False
        forward_only["execution"]["precision_qualification"]["training"] = None
        forward_only["execution"]["graph_replay"]["phases"].pop("training")
        forward_only["timings"]["training_ms"] = None
        forward_only["timing_samples_ms"]["training"] = None
        self.assertEqual(check_record(forward_only, "test", "bf16", "eager", "cpu", training=False),
                         {"inference": "pass", "latency": "pass", "training": None})
        with self.assertRaisesRegex(ValueError, "training scope"):
            check_record(forward_only, "test", "bf16", "eager", "cpu")
        forward_only["timings"]["training_ms"] = 1.0
        with self.assertRaisesRegex(ValueError, "unrequested training"):
            check_record(forward_only, "test", "bf16", "eager", "cpu", training=False)
        for mutate in (lambda r: r["precision"].update(comparison_class="reduced-input-f32-accumulate"),
                       lambda r: r["execution"]["precision_qualification"]["training"].update(completed_updates=31),
                       lambda r: r["protocol"]["warmup"]["training"].update(seconds=2),
                       lambda r: r["timing_samples_ms"]["inference"].__setitem__(0, float("nan"))):
            changed = copy.deepcopy(record)
            mutate(changed)
            with self.assertRaises(ValueError):
                check_record(changed, "test", "bf16", "eager", "cpu")
        with patch.dict(os.environ, dict(env, INFERENA_GRAPH_REPLAY="1")), \
                self.assertRaisesRegex(ValueError, "no explicit whole-phase replay API"):
            _bench("test", {"type": "whisper"}, "cpu", None)

        partial = copy.deepcopy(record["execution"]["precision_qualification"])
        partial["status"] = "partial"
        from training_gate import assess_trajectories
        curve = partial["training"]
        curve["losses"] = [x * 1.1 for x in curve["losses"]]
        for loss, attempts in zip(curve["losses"], curve["scale_attempts"]):
            attempts[-1]["loss"] = loss
        assessment = assess_trajectories(curve, curve["reference_trajectory"], curve["workload"])
        curve.update(assessment["comparisons"]["training"], **assessment)
        with patch.dict(os.environ, env), patch("bench.load_model", return_value=model), \
                patch("bench.prepare_inputs", return_value={"input_features": torch.ones(1, 4)}), \
                patch("bench._measure_call", side_effect=measure), patch("bench.qualify_precision", return_value=partial):
            output = io.StringIO()
            with redirect_stdout(output):
                _bench("test", {"type": "whisper"}, "cpu", None)
            failed_training = json.loads(output.getvalue())
        self.assertEqual(check_record(failed_training, "test", "bf16", "eager", "cpu"),
                         {"inference": "pass", "latency": "pass", "training": "failed"})
        self.assertIsNone(failed_training["outputs"]["grad_norm"])

    def test_f32_oracle_disables_tf32_and_restores_requested_flags_on_failure(self):
        from amp import prepare_reference
        policy = AmpPolicy("cpu", "")
        model = torch.nn.Linear(4, 2)
        previous = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32)
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32 = True
            def forward(current, batch):
                self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                raise RuntimeError("oracle failure")
            with self.assertRaisesRegex(RuntimeError, "oracle failure"):
                prepare_reference(model, forward, lambda y, _: y.mean(), torch.ones(1, 4), policy)
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.set_float32_matmul_precision(previous[0])
            torch.backends.cudnn.allow_tf32 = previous[1]

    def test_accelerated_off_requires_the_training_gate_and_f32_oracle(self):
        from contextlib import redirect_stdout
        import io
        import json
        from bench import _bench
        from precision import check_precision_qualification
        model = torch.nn.Linear(4, 2)
        env = {"INFERENA_AUTOCAST": "off", "INFERENA_STRICT": "0", "INFERENA_TORCH_MODE": "eager",
               "INFERENA_GRAPH_REPLAY": "0"}
        def measure(fn, *_, before=None):
            if before is not None:
                before()
            return fn(), [1.0] * 20, {"runs": 5, "seconds": 10.0}
        with patch.dict(os.environ, env), patch("bench.load_model", return_value=model), \
                patch("bench.prepare_inputs", return_value={"input_features": torch.ones(1, 4)}), \
                patch("bench._measure_call", side_effect=measure):
            output = io.StringIO()
            with redirect_stdout(output):
                _bench("test", {"type": "whisper"}, "cpu", None)
            record = json.loads(output.getvalue())
        self.assertIsNone(record["precision"]["autocast"])
        self.assertEqual(check_precision_qualification(record)["training"], "pass")
        del record["execution"]["precision_qualification"]
        with self.assertRaises(KeyError):
            check_precision_qualification(record)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "qualification.json"
            with patch.dict(os.environ, {**env, "INFERENA_PRECISION_REPORT": str(receipt),
                                        "INFERENA_PREPARATION_REPORT": str(Path(directory) / "preparation.json")}), \
                    patch("bench.load_model", return_value=model), \
                    patch("bench.prepare_inputs", return_value={"input_features": torch.ones(1, 4)}), \
                    patch("bench.qualify_precision", side_effect=QualificationError("forward failure")), \
                    self.assertRaisesRegex(QualificationError, "forward failure"):
                _bench("test", {"type": "whisper"}, "cpu", None)
            failed = json.loads(receipt.read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertGreater(failed["reference_preparation_seconds"], 0)
            self.assertGreaterEqual(failed["elapsed_seconds"], 0)
            oracle = json.loads((Path(directory) / "torch-f32-oracle.json").read_text())
            self.assertEqual(oracle["model"], "test")
            self.assertEqual(oracle["correctness_reference"]["precision"], "strict-f32")
            self.assertTrue(oracle["correctness_reference"]["prepared_before_candidate_compilation"])
            self.assertIsNotNone(oracle["correctness_reference"]["outputs"]["grad_norm"])
            self.assertNotIn("timings", oracle)
            # An unusable F32 oracle must not turn the candidate's classified
            # numerical failure into a malformed-JSON collector failure.
            for invalid_field in ("loss", "full_output"):
                snapshot = json.loads(json.dumps(oracle["correctness_reference"]))
                reference = {"correctness_reference": snapshot, "output": torch.ones(1, 4)}
                if invalid_field == "loss":
                    snapshot["outputs"]["loss"] = float("nan")
                else:
                    reference["output"][0, 0] = float("nan")
                with patch.dict(os.environ, {**env, "INFERENA_PREPARATION_REPORT": str(Path(directory) / "preparation.json")}), \
                        patch("bench.load_model", return_value=model), \
                        patch("bench.prepare_inputs", return_value={"input_features": torch.ones(1, 4)}), \
                        patch("bench.prepare_reference", return_value=reference), \
                        patch("bench.qualify_precision", side_effect=QualificationError("forward failure")), \
                        self.assertRaisesRegex(QualificationError, "forward failure"):
                    _bench("test", {"type": "whisper"}, "cpu", None)
                unavailable = json.loads((Path(directory) / "torch-f32-oracle.json").read_text())
                self.assertIsNone(unavailable["correctness_reference"])
                self.assertIn("non-finite", unavailable["unavailable_reason"])

    def test_candidate_compilation_cannot_change_the_f32_oracle(self):
        from contextlib import redirect_stdout
        import io
        import json
        import math
        from bench import _bench
        model = torch.nn.Linear(4, 2)
        with torch.no_grad():
            model.weight.fill_(0.125)
            model.bias.zero_()

        def broken_compile(current, **_):
            # Model an incorrect compiler backward that also affects subsequent
            # eager uses of the same parameters. It must not redefine the oracle.
            for parameter in current.parameters():
                parameter.register_hook(lambda grad: grad * 2)
            return current

        def measure(fn, *_, before=None):
            if before is not None:
                before()
            return fn(), [1.0] * 20, {"runs": 5, "seconds": 10.0}

        env = {"INFERENA_AUTOCAST": "bf16", "INFERENA_STRICT": "0",
               "INFERENA_TORCH_MODE": "default", "INFERENA_GRAPH_REPLAY": "0"}
        with patch.dict(os.environ, env), patch("bench.load_model", return_value=model), \
                patch("bench.prepare_inputs", return_value={"input_features": torch.ones(1, 4)}), \
                patch("torch.compile", side_effect=broken_compile), patch("bench._measure_call", side_effect=measure):
            output = io.StringIO()
            with redirect_stdout(output):
                _bench("test", {"type": "whisper"}, "cpu", None)
            record = json.loads(output.getvalue())
        self.assertAlmostEqual(record["correctness_reference"]["outputs"]["grad_norm"], math.sqrt(2.5), places=6)
        self.assertEqual(record["execution"]["precision_qualification"]["training"]["status"], "failed")
        self.assertIsNone(record["timings"]["training_ms"])
        self.assertIsNotNone(record["timings"]["inference_ms"])

    def test_convergence_validates_dtype_within_accelerated(self):
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from convergence import validate_precision
        from bench import _configure_benchmark_precision

        for mode in ("strict", "bf16", "fp16"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"INFERENA_STRICT": "0"}):
                policy = AmpPolicy("cpu", "" if mode == "strict" else mode)
                precision = _configure_benchmark_precision("cpu", mode == "strict", policy)
                record = {"framework": "pytorch", "precision": precision["comparison_class"],
                          "precision_policy": precision, "gradient_scaler": {"enabled": mode == "fp16"}}
                validate_precision(record, f"pytorch-{mode}")
                record["precision_policy"]["autocast"] = "fp16" if mode != "fp16" else "bf16"
                with self.assertRaisesRegex(ValueError, "dtype/scaling"):
                    validate_precision(record, f"pytorch-{mode}")
        for mode, expected in (("strict", "strict-f32"), ("accelerated", "accelerated")):
            record = {"framework": "meganeura", "precision": expected,
                      "arithmetic": {"comparison_class": expected}}
            validate_precision(record, f"meganeura-{mode}")
            record["arithmetic"]["comparison_class"] = "wrong"
            with self.assertRaisesRegex(ValueError, "native arithmetic"):
                validate_precision(record, f"meganeura-{mode}")

    def test_real_data_gate_rejects_skips_stale_weights_and_missing_checkpoints(self):
        import copy
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from convergence import validate_run

        spec = {"protocol": "smollm2-text-v3", "model": "SmolLM2-135M", "steps": 256, "checkpoint_every": 64,
                "gate": {"parameter_must_change": "model.norm.weight"}}
        record = {"status": "complete", "protocol": spec["protocol"], "spec_sha256": "abc",
                  "model": "SmolLM2-135M", "completed_updates": 256, "skipped_updates": 0,
                  "losses": [1.0] * 256,
                  "checkpoints": [{"step": step, "probe_losses": [1 - step / 1024] * 4,
                                   "heldout_losses": [1 - step / 2048] * 4} for step in range(0, 257, 64)],
                  "parameter_check": {"name": "model.norm.weight", "changed": True,
                                      "initial": [1.0] * 576, "final": [0.9] * 576}}
        self.assertEqual(validate_run(record, spec, "abc")["status"], "pass")
        def fails(change):
            changed = copy.deepcopy(record)
            change(changed)
            with self.assertRaises(ValueError):
                validate_run(changed, spec, "abc")
        fails(lambda r: r.update(skipped_updates=1))
        fails(lambda r: r["parameter_check"].update(final=[1.0] * 576))
        fails(lambda r: r["checkpoints"].pop(1))
        fails(lambda r: r["checkpoints"][-1].update(probe_losses=[1.0] * 4))
        fails(lambda r: r["checkpoints"][-1].update(heldout_losses=[2.0] * 4))
        fails(lambda r: r["losses"].__setitem__(100, float("nan")))

        from convergence import compare_runs
        from training_gate import SMOLLM2_HIDDEN_SIZES
        for name, width in SMOLLM2_HIDDEN_SIZES.items():
            current = copy.deepcopy(record)
            current["model"] = name
            current["parameter_check"].update(initial=[1.] * width, final=[.9] * width)
            self.assertEqual(validate_run(current, dict(spec, model=name), "abc")["status"], "pass")
            self.assertEqual(compare_runs(current, current)["training"]["status"], "pass")
            other = copy.deepcopy(current)
            other["parameter_check"]["initial"][0] = 2.
            with self.assertRaisesRegex(ValueError, "same model, workload"):
                compare_runs(current, other)

    def test_xpu_embedding_probe_uses_real_text_before_preparing_oracle(self):
        from unittest.mock import Mock
        from amp import training_plan
        from bench import _bench

        inputs = {"input_ids": torch.arange(128).reshape(1, -1)}
        for name, strict, repeated in [("SmolLM2-135M", False, False),
                                      ("SmolLM2-360M", False, False),
                                      ("SmolLM2-1.7B", False, False),
                                      ("SmolLM2-360M", True, False),
                                      ("SmolLM2-360M", False, True)]:
            plan = training_plan(name, inputs, "cpu", repeated=repeated)
            expected = None if strict or repeated else plan["train"][0]["input_ids"].flatten().tolist()
            if expected is not None:
                self.assertLess(len(set(expected)), len(expected))
            model = Mock()
            env = {"INFERENA_STRICT": str(int(strict)), "INFERENA_AUTOCAST": "off",
                   "INFERENA_TORCH_MODE": "eager", "INFERENA_GRAPH_REPLAY": "0",
                   "INFERENA_TRAINING_GATE": "repeated-synthetic" if repeated else "auto",
                   "INFERENA_INFERENCE_ONLY": "0"}
            with self.subTest(model=name, strict=strict, repeated=repeated), patch.dict(os.environ, env), \
                    patch("bench.device_name", return_value="test XPU"), \
                    patch("bench.load_model", return_value=model), patch("bench.synchronize"), \
                    patch("bench._configure_benchmark_precision", return_value={}), \
                    patch("bench.prepare_inputs", return_value=inputs), \
                    patch("bench.training_plan", return_value=plan), \
                    patch("bench.prepare_reference") as oracle, \
                    patch("bench.qualify_embedding_backward", side_effect=RuntimeError("probe reached")) as probe:
                with self.assertRaisesRegex(RuntimeError, "probe reached"):
                    _bench(name, {"type": "causal_lm"}, "xpu", None)
                probe.assert_called_once_with(model, "xpu", expected)
                oracle.assert_not_called()

    def test_embedding_probe_includes_repeated_and_unused_rows(self):
        from bench import qualify_embedding_backward
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(513, 7)
            def get_input_embeddings(self):
                return self.embedding
        model = Model()
        report = qualify_embedding_backward(model, "cpu", [512, 0, 14, 14, 512])
        self.assertEqual(report["native_dense"]["matching_elements"], 513 * 7)
        self.assertEqual(report["selected"], "native-dense")
        self.assertIsNone(model.embedding.weight.grad)

    def test_overflow_repeats_the_same_batch_until_one_update_completes(self):
        with patch.dict(os.environ, {"INFERENA_STRICT": "0"}):
            policy = AmpPolicy("cpu", "fp16")
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        def loss():
            with policy.context():
                return model(torch.tensor([[2.0]])).float().square().mean()
        value, attempts = optimizer_update(optimizer, policy.scaler(), loss)
        self.assertEqual(value, 4.0)
        self.assertGreater(len(attempts), 1)
        self.assertEqual(sum(attempt["updated"] for attempt in attempts), 1)
        self.assertTrue(all(attempt["loss"] == 4.0 for attempt in attempts))
        torch.testing.assert_close(model.weight, torch.tensor([[0.2]]))

    def test_wrong_backward_fails_the_curve_but_preserves_forward_and_weights(self):
        with patch.dict(os.environ, {"INFERENA_STRICT": "0"}):
            policy = AmpPolicy("cpu", "bf16")
        model = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.125)
        initial = model.weight.detach().clone()
        class WrongBackward(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.original = original
            def forward(self, x):
                y = self.original(x)
                return y.detach() + 5 * (y - y.detach())
        report = qualify(model, WrongBackward(model), lambda m, x: m(x),
                         lambda y, _: y.float().square().mean(), torch.ones(1, 4), policy)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["training"]["status"], "failed")
        self.assertEqual(report["forward"]["max_abs_error"], 0)
        torch.testing.assert_close(model.weight, initial, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
