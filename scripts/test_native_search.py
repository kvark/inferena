"""Termination receipts from the calibrated kernel race, without a GPU."""
import copy
import sys
import unittest

from p3hpc import check_native_kernel_search, TUNE_SCRATCH_BYTES


class KernelSearchContract(unittest.TestCase):
    def setUp(self):
        self.options = dict(strategy="Race", scope="All", max_classes=2 * sys.maxsize + 1,
                            min_improvement=0.05, max_scratch_bytes=TUNE_SCRATCH_BYTES,
                            max_time=dict(secs=27, nanos=188594235))
        # B570 SmolVLA reached the calibrated share cutoff after 7/8 classes,
        # well before its deadline. This is a valid, explicit early stop.
        self.receipt = dict(options=self.options, eligible_classes=8, visited_classes=7,
                            class_limit_reached=False, time_budget_exhausted=False,
                            share_cutoff=True, predicted_plan_ms=4.043221984517176,
                            machine_profile=dict(bandwidth=300962252389.2169,
                                                 matmul_flops=3205451246077.35))

    def check(self, receipt):
        check_native_kernel_search(receipt, self.options, 30)

    def test_reported_share_cutoff_is_accepted(self):
        self.check(self.receipt)

    def test_unexplained_incomplete_coverage_is_rejected(self):
        for flag in (False, None):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self.check(dict(self.receipt, share_cutoff=flag))
        self.check(dict(self.receipt, share_cutoff=False, time_budget_exhausted=True))
        self.check(dict(self.receipt, share_cutoff=False, visited_classes=8))

    def test_cutoff_requires_a_valid_prediction(self):
        for prediction in (None, 0, -1, float("nan"), float("inf")):
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                self.check(dict(self.receipt, predicted_plan_ms=prediction))
        for fields in (dict(machine_profile=None), dict(machine_profile={}),
                       dict(machine_profile=dict(bandwidth=float("nan"))), dict(share_cutoff="yes")):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.check(dict(self.receipt, **fields))

    def test_cutoff_does_not_hide_other_policy_violations(self):
        for fields in (dict(class_limit_reached=True), dict(visited_classes=9)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.check(dict(self.receipt, **fields))
        for options in (dict(strategy="Pairwise"), dict(max_classes=7),
                        dict(max_scratch_bytes=1), dict(max_time=dict(secs=31, nanos=0))):
            receipt = copy.deepcopy(self.receipt)
            receipt["options"].update(options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.check(receipt)
