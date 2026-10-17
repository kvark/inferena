"""Real-data gate evidence must cover updates, learning and independent probes."""

import copy
import unittest
from unittest.mock import patch

from quality import language_model_report
from training_gate import assess_trajectories, load_fixture, receipt_matches, workload_for


class TrainingGateTest(unittest.TestCase):
    def test_roundoff_in_diagnostics_cannot_change_a_gate_outcome(self):
        import math
        expected = dict(status="pass", relative_l2=0.003553088113636246, mismatched_steps=0)
        self.assertTrue(receipt_matches(dict(expected, relative_l2=0.00355308811363625), expected))
        for change in (dict(status="failed"), dict(mismatched_steps=1), dict(relative_l2=0.004),
                       dict(relative_l2=math.nan), dict(relative_l2=math.inf)):
            self.assertFalse(receipt_matches(dict(expected, **change), expected))

    def setUp(self):
        self.workload = workload_for("SmolLM2-135M")
        self.reference = {
            "optimizer": "SGD", "learning_rate": self.workload["learning_rate"],
            "completed_updates": 256, "skipped_steps": 0, "losses": [1.0] * 256,
            "scale_attempts": [[{"loss": 1.0, "scale": 1.0, "updated": True}] for _ in range(256)],
            "overflow_attempts": 0,
            "checkpoints": [{"step": step, "probe_losses": [1 - step / 1024] * 4,
                             "heldout_losses": [1 - step / 2048] * 4} for step in range(0, 257, 64)],
            "parameter_check": {"name": "model.norm.weight", "changed": True,
                                "initial": [1.0] * 576, "final": [0.9] * 576}}

    def test_frozen_fixture_is_portable_and_cannot_change_silently(self):
        fixture = load_fixture()
        self.assertEqual(len(fixture["train"]), 256)
        self.assertEqual(len({tuple(batch) for batch in fixture["train"]}), 185)
        self.assertTrue(all(len(batch) == 129 for key in ("train", "probe", "heldout") for batch in fixture[key]))
        self.assertNotIn("file", fixture["tokenizer"])
        with patch("pathlib.Path.read_bytes", return_value=b"{}"), self.assertRaisesRegex(ValueError, "frozen"):
            load_fixture()

    def test_training_agreement_cannot_mask_failed_probes_or_absent_learning(self):
        self.assertEqual(assess_trajectories(self.reference, self.reference, self.workload)["status"], "pass")
        candidate = copy.deepcopy(self.reference)
        for row in candidate["checkpoints"]:
            row["probe_losses"] = [x * 1.03 for x in row["probe_losses"]]
        report = assess_trajectories(candidate, self.reference, self.workload)
        self.assertEqual(report["comparisons"]["training"]["status"], "pass")
        self.assertEqual(report["comparisons"]["probe_losses"]["status"], "failed")
        self.assertEqual(report["status"], "failed")
        candidate = copy.deepcopy(self.reference)
        candidate["checkpoints"][-1]["probe_losses"] = [1.0] * 4
        self.assertEqual(assess_trajectories(candidate, candidate, self.workload)["status"], "failed")
        candidate = copy.deepcopy(self.reference)
        candidate["parameter_check"].update(changed=False, final=[1.0] * 576)
        self.assertEqual(assess_trajectories(candidate, candidate, self.workload)["status"], "failed")

    def test_l2_is_diagnostic_when_every_loss_meets_the_bound(self):
        candidate = copy.deepcopy(self.reference)
        for row in candidate["checkpoints"]:
            row["probe_losses"] = [x * 1.015 for x in row["probe_losses"]]
        report = assess_trajectories(candidate, self.reference, self.workload)
        self.assertGreater(report["comparisons"]["probe_losses"]["relative_l2"], 0.01)
        self.assertEqual(report["status"], "pass")
        candidate["checkpoints"][2]["probe_losses"][3] *= 1.02
        self.assertEqual(assess_trajectories(candidate, self.reference, self.workload)["status"], "failed")

    def test_incomplete_or_invalid_evidence_is_rejected(self):
        for mutate in (lambda r: r.update(skipped_steps=1), lambda r: r["checkpoints"].pop(1),
                       lambda r: r["scale_attempts"][99][-1].update(updated=False),
                       lambda r: r["losses"].__setitem__(1, float("nan")),
                       lambda r: r["checkpoints"][-1].update(heldout_losses=[float("inf")] * 4)):
            candidate = copy.deepcopy(self.reference)
            mutate(candidate)
            with self.assertRaises(ValueError):
                assess_trajectories(candidate, self.reference, self.workload)

    def test_collector_recomputes_probes_and_rejects_diagnostic_substitution(self):
        from precision import check_precision_qualification
        assessment = assess_trajectories(self.reference, self.reference, self.workload)
        curve = {**copy.deepcopy(self.reference), **assessment["comparisons"]["training"], **assessment,
                 "reference_trajectory": copy.deepcopy(self.reference), "reference_losses": [1.0] * 256,
                 "workload": self.workload}
        outputs = {"loss": 1.0, "output_shape": [1, 1, 2], "logits_sample": [1.0],
                   "grad_norm": 1.0, "gradient_norms": {"weight": 1.0}}
        record = {"model": "SmolLM2-135M", "status": "ok", "backend": "CUDA",
                  "protocol": {"training_requested": True, "training_gate": "auto", "diagnostic": False},
                  "precision": {"comparison_class": "accelerated", "autocast": None,
                                "torch_float32_matmul_precision": "high", "cuda_matmul_allow_tf32": True,
                                "cudnn_allow_tf32": True},
                  "outputs": outputs, "correctness_reference": {"precision": "strict-f32", "outputs": outputs,
                         "training_requested": True, "prepared_before_candidate_compilation": True},
                  "timings": {"training_ms": 1.0}, "timing_samples_ms": {"training": [1.0] * 20},
                  "execution": {"precision_qualification": {"status": "pass", "reference": "eager strict f32",
                         "forward": {"rtol": 0.02, "atol": 1e-6, "elements": 2, "output_shape": [1, 1, 2], "raw_logits_role": "diagnostic",
                                     "language_model": language_model_report([0.0], [1.0], [1.0], 2), "rms_error": 0.0,
                                     "rms_bound": 0.02, "max_abs_error": 0.0, "max_abs_bound": 0.02}, "training": curve}}}
        gate = record["execution"]["precision_qualification"]
        gate["latency_forward"] = copy.deepcopy(gate["forward"])
        self.assertEqual(check_precision_qualification(record)["training"], "pass")
        for row in curve["checkpoints"]:
            row["probe_losses"] = [x * 1.03 for x in row["probe_losses"]]
        with self.assertRaisesRegex(ValueError, "disagrees"):
            check_precision_qualification(record)
        assessment = assess_trajectories(curve, curve["reference_trajectory"], self.workload)
        curve.update(assessment)
        with self.assertRaisesRegex(ValueError, "failed accelerated training"):
            check_precision_qualification(record)
        record.update(status="partial")
        record["timings"]["training_ms"] = None
        record["timing_samples_ms"]["training"] = None
        record["execution"]["training_precision_failure"] = curve
        record["execution"]["precision_qualification"]["status"] = "partial"
        self.assertEqual(check_precision_qualification(record)["training"], "failed")
        record["protocol"].update(training_gate="repeated-synthetic", diagnostic=True)
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            check_precision_qualification(record)


if __name__ == "__main__":
    unittest.main()
