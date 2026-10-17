"""Portable numerical, compilation and replay regressions; no cohort tooling."""
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from execution import (QualificationError, capture_phase, check_gradient_set,
                       compare_tensors, graph_backend, profile_phase, synchronize)


class RunnerTest(unittest.TestCase):
    def test_uncaptured_qualification_and_partial_training_failure(self):
        from bench import _bench

        class Model(torch.nn.Module):
            def __init__(self, drift=False):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(8))
                self.calls = 0
                self.drift = drift

            def forward(self, x):
                if torch.is_grad_enabled():
                    self.calls += 1
                    if self.drift == "fault":
                        raise RuntimeError("device execution fault")
                return x * self.weight + (self.calls * 0.01 if self.drift else 0)

        x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        model = Model()
        def training():
            output = model(x)
            loss = output.square().mean()
            loss.backward()
            return output, loss
        _, report = capture_phase(training, model, device="cpu", capture=False)
        self.assertEqual(report["status"], "validated-uncaptured")
        self.assertEqual(report["validation"]["consecutive_replays"], 0)
        self.assertEqual(report["validation"]["gradient_tensors"], 1)
        model.drift = True
        with self.assertRaises(QualificationError) as error:
            capture_phase(training, model, device="cpu", capture=False)
        self.assertEqual(error.exception.details["stage"], "uncaptured repeat 1")
        self.assertEqual(error.exception.details["tensor"], "output 0")

        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "preparation.json"
            env = {"INFERENA_TORCH_MODE": "eager", "INFERENA_GRAPH_REPLAY": "0", "INFERENA_STRICT": "1",
                   "INFERENA_PREPARATION_REPORT": str(receipt), "INFERENA_REFERENCE_DIAGNOSTIC": "1"}
            def measure(fn, *_, **__):
                return fn(), [1.0], {"runs": 5, "seconds": 2.0}
            with patch.dict(os.environ, env), patch("bench.load_model", return_value=Model(True)) as load, \
                 patch("bench.prepare_inputs", return_value={"input_features": x}), \
                 patch("bench._measure_call", side_effect=measure):
                output = io.StringIO()
                with redirect_stdout(output):
                    _bench("test", {"type": "whisper"}, "cpu", None)
                result = json.loads(output.getvalue())
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["timing_samples_ms"],
                                 {"inference": [1.0], "latency": None, "training": None})
                self.assertIsNone(result["outputs"]["grad_norm"])
                self.assertEqual(result["outputs"]["logits_sample"], x.flatten().tolist())
                self.assertTrue(result["protocol"]["diagnostic"])
                self.assertEqual(json.loads(receipt.read_text())["qualification_failure"]["phase"], "training")
                load.return_value = Model("fault")
                with self.assertRaisesRegex(RuntimeError, "device execution fault"):
                    _bench("test", {"type": "whisper"}, "cpu", None)


    def test_diffusion_feed_forward_uses_exact_gelu(self):
        from bench import SpatialTransformerBlock

        torch.manual_seed(7)
        block = SpatialTransformerBlock(8, 4, 4, 1e-5)
        x = torch.randn(3, 8, requires_grad=True)
        context = torch.randn(2, 4)
        h = x + block.self_attn(block.norm1(x), block.norm1(x))
        h = h + block.cross_attn(block.norm2(h), context)
        z = block.ff.fc1(block.norm3(h))
        expected = h + block.ff.fc2(0.5 * z * (1 + torch.erf(z / (2 ** 0.5))))
        actual = block(x, context)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        actual_grad = torch.autograd.grad(actual.square().sum(), x)[0]
        expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)


    def test_smolvla_attention_uses_normalized_self_keys_and_external_cross_keys(self):
        from bench import ExpertLayer

        x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 7
        context = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) / 3
        for cross in (False, True):
            layer = ExpertLayer(8, 4, 16, 2, 1, 4, cross)
            normalized = layer.input_layernorm(x)
            residual = x + layer.self_attn(
                normalized, context if cross else normalized, causal=not cross)
            expected = residual + layer.mlp(layer.post_attention_layernorm(residual))
            actual = layer(x, context)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual_gradients = torch.autograd.grad(actual.sum(), tuple(layer.parameters()))
            expected_gradients = torch.autograd.grad(expected.sum(), tuple(layer.parameters()))
            for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


    def test_synthetic_parameters_match_native_layout_without_low_rank_weights(self):
        from bench import (_INIT_AMPLITUDE, _parameter_values, _name_seeded_init,
                           _transposed_init, _sd_parameter_name, _smolvla_parameter_name)

        name = "time_embed.0.weight"
        values = _parameter_values(name, 8)
        # Shared with Rust's synthetic_parameter_bits test, including u32 overflow.
        self.assertEqual(values.view(torch.uint32).tolist(), [
            0x3b10bfaf, 0xbb65c670, 0x3adfd0cc, 0xbb06e742,
            0x3b887a65, 0x3c9f5be6, 0xbcb7f94b, 0x3cafdd01,
        ])
        self.assertTrue(torch.equal(_parameter_values(name, 8, _INIT_AMPLITUDE * 0.5), values * 0.5))
        self.assertFalse(torch.equal(values, _parameter_values("time_embed.1.weight", 8)))
        self.assertEqual(_parameter_values("empty", 0).numel(), 0)
        for canonical in (
            _sd_parameter_name("encoder_attn.0.self_attn.q_proj.weight"),
            _smolvla_parameter_name("layers.0.self_attn.q_proj.weight"),
            "fc.weight", "layers.0.fc1.weight",
        ):
            native = _parameter_values(canonical, 32 * 64).reshape(32, 64)
            ordinary = torch.empty_like(native)
            transposed = torch.empty(64, 32)
            _name_seeded_init(ordinary, canonical)
            _transposed_init(transposed, canonical)
            self.assertTrue(torch.equal(ordinary, native))
            self.assertTrue(torch.equal(transposed.T, native))
            # A sinusoid reshaped as a matrix has rank <= 2 in exact arithmetic.
            energy = torch.linalg.svdvals(native.double()).square()
            self.assertLess(float(energy[:2].sum() / energy.sum()), 0.25)
            self.assertAlmostEqual(float(native.square().mean()), 0.02**2 / 2, delta=2e-5)


    def test_compilation_watchdog_preserves_failure_and_kills_workers(self):
        directory = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "compile.json"
            marker = Path(temporary) / "orphan.txt"
            env = dict(os.environ, INFERENA_COMPILE_SECONDS="0.3", INFERENA_PREPARATION_REPORT=str(report))
            for statement, code, status in (("pass", 0, "complete"),
                                            ("raise ValueError('test failure')", 1, "failed"),
                                            ("time.sleep(5)", 124, "timeout")):
                if report.exists():
                    report.unlink()
                child = f"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).touch()"
                source = ("import subprocess, sys, time; from budget import compilation_budget\n"
                          f"with compilation_budget():\n subprocess.Popen([sys.executable, '-c', {child!r}])\n {statement}\n"
                          if code == 124 else f"from budget import compilation_budget\nwith compilation_budget():\n {statement}\n")
                result = subprocess.run([sys.executable, str(directory / "budget.py"), "-c", source],
                                        cwd=directory, env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(json.loads(report.read_text())["status"], status)
            time.sleep(2)
            self.assertFalse(marker.exists(), "compiler descendant survived the deadline")


    def test_index_add_embedding_backward_accumulates_repeated_rows(self):
        from bench import _IndexAddEmbeddingBackward

        weight = torch.randn(5, 3, requires_grad=True)
        indices = torch.tensor([1, 1, 3])
        upstream = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        output = _IndexAddEmbeddingBackward.apply(indices, weight)
        output.backward(upstream)

        expected = torch.zeros_like(weight)
        expected.index_add_(0, indices, upstream)
        torch.testing.assert_close(output, weight[indices])
        torch.testing.assert_close(weight.grad, expected)


    def test_full_tensor_bounds_and_strided_accumulation(self):
        # Preserve f64 accumulation and logical element order for strided tensors.
        reference = torch.arange(20000, dtype=torch.float32).reshape(100, 200).T / 100
        actual = reference * (1 + 1e-6)
        metrics = compare_tensors(actual, reference)
        for key, values in (("rms_error", actual - reference), ("rms_reference", reference)):
            expected = float(torch.linalg.vector_norm(values, dtype=torch.float64)) / values.numel()**0.5
            self.assertAlmostEqual(metrics[key], expected, places=10)
        # Full-vector bounds admit cancellation noise, not corrupt/stale data.
        reference = torch.tensor([1.0, -1.0, 0.0])
        rounded = reference + torch.tensor([0.0, 0.0, 2e-6])
        self.assertEqual(compare_tensors(rounded, reference)["elementwise_mismatches"], 1)
        for gradient in (False, True):
            compare_tensors(rounded, reference, gradient=gradient)
            for invalid in (reference * 2, -reference, torch.zeros_like(reference),
                            torch.tensor([1.0, -1.0, 1e-3]), reference + float("nan"),
                            reference + float("inf"), reference[:2], reference.double()):
                with self.assertRaises(ValueError):
                    compare_tensors(invalid, reference, gradient=gradient)
        # The maximum catches sparse corruption; RMS catches diffuse drift
        # even when one large element would make a max-only gate permissive.
        reference = torch.zeros(10000)
        reference[0] = 1.0
        diffuse = reference + 1e-5
        for gradient in (False, True):
            with self.assertRaisesRegex(ValueError, "fixed bounds"):
                compare_tensors(diffuse, reference, gradient=gradient)
        compare_tensors(torch.zeros(3), torch.zeros(3), gradient=True)
        with self.assertRaises(ValueError):
            compare_tensors(torch.full((3,), 1e-3), torch.zeros(3), gradient=True)
        reference = torch.tensor([1.0, -1.0, 0.0])
        rounded = reference + torch.tensor([0.0, 0.0, 2e-3])
        check_gradient_set([
            compare_tensors(rounded, reference, gradient=True, reduced_precision=True)
        ], reduced_precision=True)
        with self.assertRaisesRegex(ValueError, "fixed bounds"):
            compare_tensors(rounded, reference, reduced_precision=True)
        with self.assertRaisesRegex(ValueError, "full-gradient"):
            check_gradient_set([
                compare_tensors(reference * 1.02, reference, gradient=True, reduced_precision=True)
            ], reduced_precision=True)

    def test_explicit_backend_is_probed_and_synchronized_without_fallback(self):
        from bench import MODEL_REGISTRY, attention_context, attention_policy, bench, detect_device
        initial_sdpa = torch.nn.attention._cur_sdpa_kernel_backends()
        for backend, device in (("cuda", "cuda:0"), ("rocm", "cuda:0"), ("xpu", "xpu:0"),
                                ("cpu", "cpu"), ("mps", "mps")):
            for model in MODEL_REGISTRY:
                default = "math" if backend == "xpu" else "auto"
                for override in (None, "auto", "math", "efficient"):
                    with patch.dict(os.environ), patch("torch.version.hip", "7.2" if backend == "rocm" else None):
                        os.environ.pop("INFERENA_SDPA", None)
                        if override is not None:
                            os.environ["INFERENA_SDPA"] = override
                        self.assertEqual(attention_policy(device, MODEL_REGISTRY[model]["type"]), override or default)
        for policy, expected in (("auto", initial_sdpa),
                                 ("math", [torch.nn.attention.SDPBackend.MATH]),
                                 ("efficient", [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION])):
            with patch.dict(os.environ, {"INFERENA_SDPA": policy}):
                with attention_context("cuda:0"):
                    self.assertEqual(torch.nn.attention._cur_sdpa_kernel_backends(), expected)
            self.assertEqual(torch.nn.attention._cur_sdpa_kernel_backends(), initial_sdpa)
        with patch.dict(os.environ, {"INFERENA_SDPA": "invalid"}):
            with self.assertRaisesRegex(ValueError, "INFERENA_SDPA"):
                attention_context("cuda:0")
        with patch.dict("os.environ", {"INFERENA_TORCH_BACKEND": "xpu"}), \
             patch("torch.xpu.is_available", return_value=True), \
             patch("bench._xpu_actually_works", return_value=False) as probe:
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                detect_device()
            probe.return_value = True
            self.assertEqual(detect_device(), "xpu:0")
        with patch("torch.xpu.synchronize") as xpu, patch("torch.cuda.synchronize") as cuda:
            synchronize("xpu:1")
            xpu.assert_called_once_with("xpu:1")
            cuda.assert_not_called()
        with patch("bench.detect_device", return_value="xpu:0"), \
             patch("bench._bench") as run, patch("bench.graph_backend") as api, \
             patch.dict(os.environ, {"TRITON_DEFAULT_BACKEND": "intel"}):
            run.side_effect = lambda *_, **__: self.assertEqual(
                torch.nn.attention._cur_sdpa_kernel_backends(), [torch.nn.attention.SDPBackend.MATH])
            bench("model", {})
            stream = api.return_value.Stream.return_value
            run.assert_called_once_with("model", {}, "xpu:0", stream)
            api.return_value.stream.assert_called_once_with(stream)
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        for model_type, hip in (("whisper", "7.2"), ("whisper", None), ("sd_unet", "7.2")):
            with patch("bench.detect_device", return_value="cuda:0"), \
                 patch("bench.select_compiler_backend"), patch("bench.graph_backend"), \
                 patch("torch.version.hip", hip), patch("bench._bench") as run:
                def fail(*args):
                    self.assertIs(torch.nn.functional.scaled_dot_product_attention, original_sdpa)
                    self.assertEqual(torch.nn.attention._cur_sdpa_kernel_backends(), initial_sdpa)
                    raise RuntimeError("test qualification failure")
                run.side_effect = fail
                with self.assertRaisesRegex(RuntimeError, "test qualification failure"):
                    bench("model", {"type": model_type})
            self.assertIs(torch.nn.functional.scaled_dot_product_attention, original_sdpa)
            self.assertEqual(torch.nn.attention._cur_sdpa_kernel_backends(), initial_sdpa)
        # MPS must attempt compilation, not report an eager result with zero
        # compile time. A mock checks routing, not real Metal backend support.
        from bench import _bench
        with patch.dict(os.environ, {"INFERENA_TORCH_MODE": "default", "INFERENA_GRAPH_REPLAY": "0", "INFERENA_STRICT": "1"}), \
             patch("bench.load_model"), patch("bench.synchronize"), \
             patch("bench.device_name", return_value="test MPS"), \
             patch("torch.compile", side_effect=RuntimeError("compiler failure")) as compile:
            with self.assertRaisesRegex(RuntimeError, "no eager timing substituted"):
                _bench("model", {"type": "resnet"}, "mps", None)
            compile.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available() or torch.xpu.is_available(), "CUDA/HIP/XPU required")
class ReplayTest(unittest.TestCase):
    def test_forward_backward_replay_observes_live_inputs_and_weights(self):
        torch.manual_seed(7)
        device = "cuda" if torch.cuda.is_available() else "xpu"
        from bench import select_compiler_backend
        select_compiler_backend(device)
        api = graph_backend(device)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.SiLU(),
            torch.nn.Flatten(), torch.nn.Linear(4 * 8 * 8, 3),
        ).to(device)
        inputs = torch.randn(2, 2, 8, 8, device=device)
        compiled = torch.compile(model, options={
            "max_autotune": False, "triton.cudagraphs": False,
        })

        def inference():
            with torch.no_grad():
                return compiled(inputs)

        def training():
            output = compiled(inputs)
            loss = output.square().mean()
            loss.backward()
            return output, loss

        stream = api.Stream(device=device)
        stream.wait_stream(api.current_stream(device))
        with api.stream(stream):
            # Keep this graph alive across capture, like compiled model caches.
            warmup = compiled(inputs)
            warmup.square().mean().backward()
        forward, _ = capture_phase(inference, stream=stream, device=device)
        backward, report = capture_phase(training, model, stream=stream, reduced_precision=True, device=device)
        self.assertEqual(report["validation"]["gradient_tensors"], 4)
        self.assertEqual(len(report["validation"]["uncaptured"]), 8)
        self.assertEqual(report["validation"]["uncaptured_calls"], 9)
        self.assertEqual(len(report["validation"]["replays"]), 2)
        captured_gradients = [p.grad for p in model.parameters()]
        with torch.no_grad():
            inputs.mul_(0.5)
            for parameter in model.parameters():
                parameter.add_(0.01)
        # Replay must compute new values, not return warmup/capture leftovers.
        torch.testing.assert_close(forward(), inference())
        reference = copy.deepcopy(model)
        expected_output = reference(inputs)
        expected_gradients = torch.autograd.grad(
            expected_output.square().mean(), tuple(reference.parameters())
        )
        for _ in range(3):
            output, loss = backward()
            synchronize(device)
            torch.testing.assert_close(output, expected_output)
            torch.testing.assert_close(loss, expected_output.square().mean())
            for parameter, storage, expected in zip(
                model.parameters(), captured_gradients, expected_gradients, strict=True
            ):
                self.assertIs(parameter.grad, storage)
                torch.testing.assert_close(parameter.grad, expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            # PTI may omit child kernels of Level Zero command-buffer replay.
            # Check ordinary XPU GPU profiling separately from replay correctness.
            with api.stream(stream):
                profile = profile_phase(backward if device == "cuda" else training, path, 2, device=device)
            self.assertEqual(len(profile["instrumented_wall_ms"]), 2)
            events = json.loads(path.read_text())["traceEvents"]
            self.assertTrue(any(event.get("cat") == "kernel" for event in events))
            if device == "cuda" and torch.version.cuda:
                self.assertTrue(any("cudaGraphLaunch" in event.get("name", "") for event in events))
