"""A successful quality gate cannot excuse executing an undeclared recipe."""

import copy
import unittest

from precision import check_arithmetic_policy


class PrecisionPolicyTest(unittest.TestCase):
    def test_actual_recipe_must_match_every_declared_boundary(self):
        expected = dict(autocast_requested="auto", autocast="bf16",
                        autocast_selection="model-phase-policy-v3; no timing or numerical fallback",
                        model_guard="one-token-final-decoder-linears-f32-v1",
                        phase_autocast=dict(inference="bf16", training="bf16", latency="fp16"),
                        model_guard_phases=["latency"],
                        f32_linear_modules=["model.layers.29.mlp.down_proj"])
        record = dict(precision=copy.deepcopy(expected))
        check_arithmetic_policy(record, expected)
        for key, value in dict(autocast_requested="fp16", autocast="fp16",
                              autocast_selection="fallback", model_guard=None,
                              model_guard_phases=[], phase_autocast=dict(inference="bf16", training="bf16", latency="bf16"),
                              f32_linear_modules=[]).items():
            with self.subTest(key=key):
                changed = copy.deepcopy(record)
                changed["precision"][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    check_arithmetic_policy(changed, expected)
                del changed["precision"][key]
                with self.assertRaisesRegex(ValueError, key):
                    check_arithmetic_policy(changed, expected)


if __name__ == "__main__":
    unittest.main()
