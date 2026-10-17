"""Quality receipts must cover every token and cannot promote failed evidence."""
import copy
import math
import unittest

from quality import check_language_model_report, language_model_report
from precision import check_forward_quality


class QualityTest(unittest.TestCase):
    def test_recomputes_all_bounds_and_rejects_sparse_corruption(self):
        report = language_model_report([0.0] * 512, [1.0] * 512, [1.0] * 512, 2)
        check_language_model_report(report)
        for key, value in (("tokens", 1), ("mean_kl_limit", 1.0), ("loss_bound", 1.0),
                           ("mean_kl", math.nan), ("policy", "other")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_language_model_report(dict(report, **{key: value}))
        kl = [0.0] * 512
        kl[17] = 0.37
        sparse = language_model_report(kl, [1.0] * 512, [1.0] * 512, 2)
        self.assertLess(sparse["mean_kl"], sparse["mean_kl_limit"])
        self.assertEqual(sparse["status"], "failed")
        with self.assertRaises(ValueError):
            check_language_model_report(dict(sparse, status="pass"))
        with self.assertRaises(ValueError):
            language_model_report([math.nan], [1.0], [1.0], 2)

    def test_tensor_failure_is_only_diagnostic_for_language_models(self):
        report = dict(elements=2, output_shape=[1, 1, 2], rtol=0.02, atol=1e-6, rms_error=10.0, rms_bound=0.02,
                      max_abs_error=10.0, max_abs_bound=0.02, raw_logits_role="diagnostic",
                      language_model=language_model_report([0.0], [1.0], [1.0], 2))
        check_forward_quality(report, language_model=True, shape=[1, 1, 2])
        with self.assertRaises(ValueError):
            check_forward_quality(report)
        with self.assertRaises(ValueError):
            check_forward_quality(report, language_model=True, shape=[1, 128, 2])
        invalid = copy.deepcopy(report)
        invalid["max_abs_error"] = math.inf
        with self.assertRaises(ValueError):
            check_forward_quality(invalid, language_model=True)

    def test_native_oracle_must_be_the_qualification_snapshot(self):
        from p3hpc import check_native_reference
        outputs = dict(loss=1.0, logits_hash="hash", logits_sample=[1.0],
                       grad_norm=None, gradient_norms={})
        record = dict(outputs=dict(output_shape=[1, 1, 2], logits_sample=[2.0]),
                      protocol=dict(training_requested=False),
                      optimizer=dict(sessions=[dict(mode="Inference", qualification=dict(
                          arithmetic_reference=dict(reference_outputs=outputs)))]),
                      correctness_reference=dict(precision="strict-f32", training_requested=False,
                          prepared_before_candidate_compilation=True,
                          outputs=dict(outputs, output_shape=[1, 1, 2])))
        check_native_reference(record)
        record["correctness_reference"]["outputs"]["logits_sample"] = [2.0]
        with self.assertRaises(ValueError):
            check_native_reference(record)


if __name__ == "__main__":
    unittest.main()
