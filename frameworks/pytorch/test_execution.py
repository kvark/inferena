"""Campaign identity and broad CUDA replay checks; no retained artifacts."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from execution import capture_phase, profile_phase, synchronize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from p3hpc import TORCH_REVISION, TORCH_VERSION, check_torch_identity


class CampaignTest(unittest.TestCase):
    def test_common_source_with_platform_specific_builds(self):
        pin = next(line for line in (
            Path(__file__).resolve().parents[2] / "requirements-p3hpc.txt"
        ).read_text().splitlines() if line.startswith("torch=="))
        self.assertEqual(pin, f"torch=={TORCH_VERSION}")
        for suffix in ("", "+cu130", "+rocm7.2", "+xpu"):
            version = TORCH_VERSION + suffix
            check_torch_identity(version, TORCH_REVISION, version)
            for revision in (None, "unknown", "0" * 40):
                with self.assertRaises(ValueError):
                    check_torch_identity(version, revision, version)
        with self.assertRaises(ValueError):
            check_torch_identity(TORCH_VERSION, TORCH_REVISION, TORCH_VERSION + "+cu130")
        with self.assertRaises(ValueError):
            check_torch_identity("2.12.0", TORCH_REVISION, "2.12.0")

    def test_explicit_backend_is_probed_and_synchronized_without_fallback(self):
        from bench import detect_device
        with patch.dict("os.environ", {"INFERENA_TORCH_BACKEND": "xpu"}), \
             patch("torch.xpu.is_available", return_value=True), \
             patch("bench._xpu_actually_works", return_value=False) as probe:
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                detect_device()
            probe.return_value = True
            self.assertEqual(detect_device(), "xpu:0")
        with patch("torch.xpu.synchronize") as xpu, patch("torch.cuda.synchronize") as cuda:
            synchronize("xpu:0")
            xpu.assert_called_once()
            cuda.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda, "NVIDIA CUDA required")
class ReplayTest(unittest.TestCase):
    def test_forward_backward_replay_observes_live_inputs_and_weights(self):
        torch.manual_seed(7)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.SiLU(),
            torch.nn.Flatten(), torch.nn.Linear(4 * 8 * 8, 3),
        ).cuda()
        inputs = torch.randn(2, 2, 8, 8, device="cuda")
        compiled = torch.compile(model, options={
            "max_autotune": True, "triton.cudagraphs": False,
        })

        def inference():
            with torch.no_grad():
                return compiled(inputs)

        def training():
            output = compiled(inputs)
            loss = output.square().mean()
            loss.backward()
            return output, loss

        forward, _ = capture_phase(inference)
        backward, report = capture_phase(training, model)
        self.assertEqual(report["validation"]["gradient_tensors"], 4)
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
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected_output)
            torch.testing.assert_close(loss, expected_output.square().mean())
            for parameter, storage, expected in zip(
                model.parameters(), captured_gradients, expected_gradients, strict=True
            ):
                self.assertIs(parameter.grad, storage)
                torch.testing.assert_close(parameter.grad, expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            profile = profile_phase(backward, path, 2)
            self.assertEqual(len(profile["instrumented_wall_ms"]), 2)
            events = json.loads(path.read_text())["traceEvents"]
            self.assertTrue(any(event.get("cat") == "kernel" for event in events))
            self.assertTrue(any("cudaGraphLaunch" in event.get("name", "") for event in events))


if __name__ == "__main__":
    unittest.main()
