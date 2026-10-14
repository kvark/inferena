"""A successful quality gate cannot excuse executing an undeclared recipe."""

import copy
import unittest

from precision import check_arithmetic_policy


class PrecisionPolicyTest(unittest.TestCase):
    def test_actual_recipe_must_match_every_declared_boundary(self):
        expected = dict(autocast_requested="auto", autocast="fp16",
                        autocast_selection="model-policy-v2; no timing or numerical fallback",
                        model_guard="final-decoder-linears-f32-v1",
                        f32_linear_modules=["model.layers.29.mlp.down_proj"])
        record = dict(precision=copy.deepcopy(expected))
        check_arithmetic_policy(record, expected)
        for key, value in dict(autocast_requested="fp16", autocast="bf16",
                              autocast_selection="fallback", model_guard=None,
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
