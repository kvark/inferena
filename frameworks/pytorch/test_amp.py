"""Precision boundaries, scaling underflow, state restoration and curve failures."""

import os
import unittest
from unittest.mock import patch

import torch

from amp import AmpPolicy, BenchmarkBackward, compare_loss_curves, qualify, optimizer_update
from execution import QualificationError, compare_tensors


class AmpTest(unittest.TestCase):
    def test_classes_are_explicit(self):
        with patch.dict(os.environ, {"INFERENA_STRICT": "0"}):
            self.assertEqual(AmpPolicy("cpu", "bf16").metadata()["comparison_class"], "amp-bf16")
            with self.assertRaises(ValueError):
                AmpPolicy("cpu", "f16")
        with patch.dict(os.environ, {"INFERENA_STRICT": "1"}):
            with self.assertRaisesRegex(ValueError, "separate reference"):
                AmpPolicy("cpu", "fp16")

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

    def test_bfloat16_repeatability_checks_all_elements(self):
        reference = torch.ones(16, dtype=torch.bfloat16)
        self.assertEqual(compare_tensors(reference, reference)["elements"], 16)
        changed = reference.clone()
        changed[-1] += 1
        with self.assertRaises(QualificationError):
            compare_tensors(changed, reference)

    def test_benchmark_emits_a_separate_amp_record(self):
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
        self.assertEqual(record["protocol"]["name"], "inferena-amp-v3")
        self.assertEqual(record["precision"]["comparison_class"], "amp-bf16")
        self.assertEqual(record["execution"]["amp_qualification"]["training"]["status"], "pass")
        self.assertFalse(record["execution"]["gradient_scaler"]["enabled"])
        import copy
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from amp_reference import check_record
        check_record(record, "test", "bf16", "eager", "cpu")
        for mutate in (lambda r: r["precision"].update(comparison_class="reduced-input-f32-accumulate"),
                       lambda r: r["execution"]["amp_qualification"]["training"].update(completed_updates=31),
                       lambda r: r["protocol"]["warmup"]["training"].update(seconds=2),
                       lambda r: r["timing_samples_ms"]["inference"].__setitem__(0, float("nan"))):
            changed = copy.deepcopy(record)
            mutate(changed)
            with self.assertRaises(ValueError):
                check_record(changed, "test", "bf16", "eager", "cpu")
        with patch.dict(os.environ, dict(env, INFERENA_GRAPH_REPLAY="1")), \
                self.assertRaisesRegex(ValueError, "GradScaler bookkeeping is uncaptured"):
            _bench("test", {"type": "whisper"}, "cpu", None)

        partial = copy.deepcopy(record["execution"]["amp_qualification"])
        partial["status"] = "partial"
        partial["training"]["status"] = "failed"
        with patch.dict(os.environ, env), patch("bench.load_model", return_value=model), \
                patch("bench.prepare_inputs", return_value={"input_features": torch.ones(1, 4)}), \
                patch("bench._measure_call", side_effect=measure), patch("bench.qualify_amp", return_value=partial):
            output = io.StringIO()
            with redirect_stdout(output):
                _bench("test", {"type": "whisper"}, "cpu", None)
            failed_training = json.loads(output.getvalue())
        self.assertEqual(check_record(failed_training, "test", "bf16", "eager", "cpu"),
                         {"inference": "pass", "latency": "pass", "training": "failed"})
        self.assertIsNone(failed_training["outputs"]["grad_norm"])

    def test_real_data_gate_rejects_skips_stale_weights_and_missing_checkpoints(self):
        import copy
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from convergence import validate_run

        spec = {"protocol": "smollm2-text-v2", "steps": 256, "checkpoint_every": 64,
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
