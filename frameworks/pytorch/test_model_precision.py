"""Fixed autocast boundaries retain the model's parameters and F32 oracle."""

from types import SimpleNamespace
import unittest

import torch

from model_precision import F32Linear, SMOLLM2_GUARD, SMOLLM2_LAYERS, apply_model_guard, guard_modules


def model_with_linears():
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_hidden_layers=SMOLLM2_LAYERS)
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for _ in range(SMOLLM2_LAYERS):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        for group, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                             (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                setattr(group, name, torch.nn.Linear(4, 4))
        model.model.layers.append(layer)
    return model


class ModelPrecisionTest(unittest.TestCase):
    def test_guard_preserves_parameters_state_and_f32_gradients(self):
        torch.manual_seed(17)
        model = model_with_linears()
        parameters = dict(model.named_parameters())
        state = {key: value.clone() for key, value in model.state_dict().items()}
        original = model.model.layers[-1].mlp.down_proj
        x = torch.randn(3, 4, requires_grad=True)
        reference = original(x)
        expected = torch.autograd.grad(reference.square().sum(), (x, original.weight, original.bias))
        paths = apply_model_guard(model, SMOLLM2_GUARD)
        self.assertEqual(paths, guard_modules(SMOLLM2_LAYERS))
        self.assertIsInstance(model.model.layers[0].mlp.down_proj, torch.nn.Linear)
        self.assertEqual(set(parameters), set(dict(model.named_parameters())))
        for name, parameter in model.named_parameters():
            self.assertIs(parameter, parameters[name])
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
        guarded = model.model.layers[-1].mlp.down_proj
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = guarded(x)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        gradients = torch.autograd.grad(actual.square().sum(), (x, guarded.weight, guarded.bias))
        for actual, reference in zip(gradients, expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        model.load_state_dict(state, strict=True)

    def test_incompatible_checkpoint_fails_before_any_replacement(self):
        model = model_with_linears()
        model.model.layers[-1].mlp.down_proj = torch.nn.Identity()
        with self.assertRaisesRegex(ValueError, "declared linear modules"):
            apply_model_guard(model, SMOLLM2_GUARD)
        self.assertFalse(any(isinstance(module, F32Linear) for module in model.modules()))
        self.assertEqual(apply_model_guard(model, None), [])
        with self.assertRaisesRegex(ValueError, "unknown model precision"):
            apply_model_guard(model, "unknown")
        model.config.num_hidden_layers -= 1
        with self.assertRaisesRegex(ValueError, "checkpoint architecture"):
            apply_model_guard(model, SMOLLM2_GUARD)


if __name__ == "__main__":
    unittest.main()
