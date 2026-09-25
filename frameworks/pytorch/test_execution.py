"""Small CPU regressions for the runner contract; no checkpoints or GPU needed."""

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

import bench
from execution import QualificationError, capture_phase, compare_tensors


class ExecutionTests(unittest.TestCase):
    def test_qualification_checks_outputs_and_backward(self):
        model = torch.nn.Linear(4, 3)
        inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8

        def step():
            outputs = model(inputs)
            loss = outputs.square().mean()
            loss.backward()
            return outputs, loss

        for accelerated in (False, True):
            _, report = capture_phase(step, model, device="cpu", capture=False,
                                      reduced_precision=accelerated)
            self.assertEqual(report["status"], "validated-uncaptured")
            self.assertEqual(report["validation"]["gradient_tensors"], 2)
            self.assertGreater(report["validation_s"], 0)
            self.assertEqual(report["capture_s"], 0)
        reference = torch.ones(256)
        for broken in (reference + 0.1, torch.full_like(reference, float("nan")),
                       reference[:1], reference.double()):
            with self.subTest(shape=broken.shape, dtype=broken.dtype):
                with self.assertRaises(QualificationError):
                    compare_tensors(broken, reference)
        with self.assertRaisesRegex(ValueError, "no explicit whole-phase replay"):
            capture_phase(step, model, device="cpu", capture=True)
        calls = 0

        def unstable():
            nonlocal calls
            calls += 1
            return reference * calls

        with self.assertRaises(QualificationError) as error:
            capture_phase(unstable, device="cpu", capture=False)
        self.assertEqual(error.exception.details["stage"], "uncaptured repeat 1")

    def test_runner_keeps_partial_results_and_never_substitutes_eager(self):
        model = torch.nn.Linear(4, 256)
        inputs = torch.arange(4, dtype=torch.float32) / 4
        spec = {"type": "whisper"}

        def qualify(fn, training_model=None, **kwargs):
            if training_model is not None:
                raise QualificationError("injected training mismatch")
            return capture_phase(fn, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {
                "INFERENA_TORCH_MODE": "eager", "INFERENA_STRICT": "1",
                "INFERENA_TORCH_BACKEND": "cpu", "INFERENA_GRAPH_REPLAY": "0",
                "INFERENA_WARMUP_RUNS": "0", "INFERENA_MEASUREMENT_RUNS": "1",
            }, clear=True))
            stack.enter_context(patch.object(bench, "load_model", return_value=model))
            stack.enter_context(patch.object(bench, "prepare_inputs", return_value={}))
            stack.enter_context(patch.object(bench, "_benchmark_forward",
                                            side_effect=lambda *args: args[1](inputs)))
            stack.enter_context(patch.object(bench, "_benchmark_logits",
                                            side_effect=lambda kind, output: output))
            stack.enter_context(patch.object(bench, "_benchmark_loss",
                                            side_effect=lambda kind, output, inputs: output.square().mean()))
            stack.enter_context(patch.object(bench, "_benchmark_latency_call",
                                            return_value=lambda: model(inputs)))
            for partial in (False, True):
                output = io.StringIO()
                with patch.object(bench, "capture_phase", qualify if partial else capture_phase), redirect_stdout(output):
                    bench.bench("tiny-fixture", spec)
                result = json.loads(output.getvalue())
                self.assertEqual(result["status"], "partial" if partial else "ok")
                self.assertFalse(result["execution"]["compiled"])
                self.assertIsNotNone(result["timings"]["inference_ms"])
                self.assertEqual(result["timings"]["training_ms"] is None, partial)
                self.assertEqual(bool(result["outputs"]["gradient_norms"]), not partial)
                if partial:
                    self.assertEqual(result["execution"]["failure"]["phase"], "training")
                    self.assertIsNone(result["timings"]["latency_ms"])
            os.environ["INFERENA_TORCH_MODE"] = "default"
            with patch.object(torch, "compile", side_effect=RuntimeError("compiler failed")):
                with self.assertRaisesRegex(RuntimeError, "no eager timing substituted"):
                    bench.bench("tiny-fixture", spec)
            with patch.dict(os.environ, {"INFERENA_TORCH_BACKEND": "missing"}):
                with self.assertRaisesRegex(RuntimeError, "no fallback"):
                    bench.detect_device()

    def test_shared_fixture_and_self_attention_inputs(self):
        weight = torch.empty(2, 3)
        bench._transposed_init(weight, "test.weight")
        expected = torch.sin(torch.arange(6).float() * 0.01 + bench._name_seed("test.weight")) * 0.02
        torch.testing.assert_close(weight.T.flatten(), expected)
        layer = bench.ExpertLayer(4, 4, 8, 2, 1, 2, False)
        inputs = torch.arange(8).float().reshape(1, 2, 4) / 4
        seen = []
        handle = layer.self_attn.k_proj.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        try:
            layer(inputs, torch.zeros_like(inputs)).sum().backward()
        finally:
            handle.remove()
        torch.testing.assert_close(seen[0], layer.input_layernorm(inputs))
        self.assertTrue(all(p.grad is not None for p in layer.parameters()))
        with patch("os.path.isfile", return_value=False):
            with self.assertRaises(FileNotFoundError):
                bench.load_model("SmolLM2-135M", bench.MODEL_REGISTRY["SmolLM2-135M"], "cpu")

    def test_compile_cache_does_not_delete_user_files(self):
        with tempfile.TemporaryDirectory() as existing, patch.dict(os.environ, {}, clear=True):
            sentinel = Path(existing, "keep")
            sentinel.touch()
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = existing
            with bench.clear_compile_cache() as cache:
                self.assertNotEqual(cache, existing)
                self.assertTrue(sentinel.exists())
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(Path(cache, "triton")))


if __name__ == "__main__":
    unittest.main()
