"""The CPU oracle must not inject the timed benchmark's float32 loss rounding."""

import unittest

import torch
from torch.nn import functional as F

from cpu_reference import reference_loss


class ReferenceLossTest(unittest.TestCase):
    def test_cross_entropy_preserves_small_logit_differences_and_gradients(self):
        logits = torch.tensor([[[100.0, 100.000001, 99.999999]]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[1]])
        loss = reference_loss("causal_lm", logits, {"labels": labels})
        gradient, = torch.autograd.grad(loss, logits, retain_graph=True)
        expected = torch.softmax(logits.detach(), dim=-1)
        expected[0, 0, 1] -= 1
        torch.testing.assert_close(gradient, expected, rtol=1e-14, atol=1e-14)
        self.assertEqual(loss.dtype, torch.float64)
        rounded_loss = F.cross_entropy(logits.float().reshape(-1, 3), labels.reshape(-1))
        rounded_gradient, = torch.autograd.grad(rounded_loss, logits)
        self.assertGreater((rounded_gradient - gradient).abs().max().item(), 1e-7)

    def test_all_losses_keep_the_selected_reference_dtype(self):
        for dtype in (torch.float32, torch.float64):
            for kind in ("causal_lm", "resnet", "sd_unet", "smolvla", "whisper"):
                with self.subTest(dtype=dtype, kind=kind):
                    logits = torch.tensor([[0.5, -0.25, 1.25]], dtype=dtype, requires_grad=True)
                    inputs = {"labels": torch.tensor([2]), "noise_target": torch.zeros_like(logits)}
                    loss = reference_loss(kind, logits, inputs)
                    self.assertEqual(loss.dtype, dtype)
                    loss.backward()
                    self.assertEqual(logits.grad.dtype, dtype)
                    self.assertTrue(torch.isfinite(logits.grad).all())

    def test_unknown_workload_cannot_silently_use_language_model_loss(self):
        with self.assertRaisesRegex(ValueError, "unknown reference workload"):
            reference_loss("unknown", torch.ones(1), {})


if __name__ == "__main__":
    unittest.main()
