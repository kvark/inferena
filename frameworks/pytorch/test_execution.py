"""Campaign identity and broad CUDA replay checks; no retained artifacts."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from execution import capture_phase, check_gradient_set, compare_tensors, profile_phase, synchronize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from p3hpc import (MODELS, TORCH_REVISION, TORCH_VERSION, check_pair, check_torch_identity,
                  create_parser, gpu_matches, runner_bash, select_native_device)


class CampaignTest(unittest.TestCase):
    def test_common_source_with_platform_specific_builds(self):
        # Full-vector bounds admit cancellation noise, not corrupt/stale data.
        reference = torch.tensor([1.0, -1.0, 0.0])
        rounded = reference + torch.tensor([0.0, 0.0, 2e-6])
        compare_tensors(rounded, reference, gradient=True)
        with self.assertRaises(AssertionError):
            compare_tensors(rounded, reference)
        for invalid in (reference * 2, -reference, torch.zeros_like(reference),
                        torch.tensor([1.0, -1.0, 1e-3]), reference + float("nan"),
                        reference + float("inf"), reference[:2], reference.double()):
            with self.assertRaises(ValueError):
                compare_tensors(invalid, reference, gradient=True)
        # The maximum catches sparse corruption; RMS catches diffuse drift
        # even when one large element would make a max-only gate permissive.
        reference = torch.zeros(10000)
        reference[0] = 1.0
        diffuse = reference + 1e-5
        with self.assertRaisesRegex(ValueError, "fixed bounds"):
            compare_tensors(diffuse, reference, gradient=True)
        compare_tensors(torch.zeros(3), torch.zeros(3), gradient=True)
        with self.assertRaises(ValueError):
            compare_tensors(torch.full((3,), 1e-3), torch.zeros(3), gradient=True)
        reference = torch.tensor([1.0, -1.0, 0.0])
        rounded = reference + torch.tensor([0.0, 0.0, 2e-3])
        check_gradient_set([
            compare_tensors(rounded, reference, gradient=True, reduced_precision=True)
        ], reduced_precision=True)
        with self.assertRaisesRegex(ValueError, "full-gradient"):
            check_gradient_set([
                compare_tensors(reference * 1.02, reference, gradient=True, reduced_precision=True)
            ], reduced_precision=True)
        defaults = create_parser().parse_args([])
        self.assertTrue(defaults.collect)
        self.assertEqual(defaults.models, list(MODELS))
        self.assertEqual(defaults.precisions, ["strict", "accelerated"])
        self.assertEqual(defaults.replicates, 3)
        self.assertIsNone(defaults.backend)
        self.assertIsNone(defaults.gpu)
        self.assertIsNone(defaults.results_dir)
        self.assertFalse(create_parser().parse_args(["--qualify-only"]).collect)
        with self.assertRaisesRegex(ValueError, "pytorch failed: capture traceback"):
            check_pair([
                {"framework": "meganeura", "status": "ok"},
                {"framework": "pytorch", "status": "error", "error": "capture traceback"},
            ], defaults, "default", True, 1, "source")
        intel = {"name": "Intel Arc B570 Graphics", "device_id": 0xe20c,
                 "available": True, "software_emulated": False}
        nvidia = {**intel, "name": "NVIDIA GeForce RTX 5070", "device_id": 0x2f04}
        software = {**intel, "name": "llvmpipe", "software_emulated": True}
        self.assertIs(select_native_device([nvidia, intel, software], "Intel(R) Arc(TM) B570 Graphics"), intel)
        self.assertTrue(gpu_matches("RTX 5070", nvidia["name"]))
        self.assertFalse(gpu_matches("RTX 5080", nvidia["name"]))
        for devices, expected in (([nvidia, intel], None), ([nvidia, nvidia], "RTX 5070"),
                                  ([software], None), ([nvidia], "B570"),
                                  ([{**intel, "available": False}], "B570")):
            with self.assertRaises(ValueError):
                select_native_device(devices, expected)
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
        with tempfile.TemporaryDirectory(prefix="inferena Git with spaces ") as directory:
            shell = Path(directory) / "bin/bash.exe"
            shell.parent.mkdir()
            shell.touch()
            with patch.dict(os.environ), patch("p3hpc.sys.platform", "win32"), \
                 patch("p3hpc.shutil.which", return_value=str(Path(directory) / "cmd/git.exe")), \
                 patch("p3hpc.subprocess.check_output", return_value="MINGW64_NT\n") as uname:
                os.environ.pop("INFERENA_BASH", None)
                self.assertEqual(runner_bash(), str(shell))
                self.assertEqual(uname.call_args.args[0], [str(shell), "-c", "uname -s"])
                uname.return_value = "Linux\n"
                with self.assertRaisesRegex(RuntimeError, "not WSL"):
                    runner_bash()

    def test_explicit_backend_is_probed_and_synchronized_without_fallback(self):
        from bench import bench, detect_device
        with patch.dict("os.environ", {"INFERENA_TORCH_BACKEND": "xpu"}), \
             patch("torch.xpu.is_available", return_value=True), \
             patch("bench._xpu_actually_works", return_value=False) as probe:
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                detect_device()
            probe.return_value = True
            self.assertEqual(detect_device(), "xpu:0")
        with patch("torch.xpu.synchronize") as xpu, patch("torch.cuda.synchronize") as cuda:
            synchronize("xpu:1")
            xpu.assert_called_once_with("xpu:1")
            cuda.assert_not_called()
        with patch("bench.detect_device", return_value="xpu:0"), \
             patch("bench._bench") as run, patch("torch.cuda.stream") as stream:
            bench("model", {})
            run.assert_called_once_with("model", {}, "xpu:0", None)
            stream.assert_not_called()


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

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Keep this graph alive across capture, like compiled model caches.
            warmup = compiled(inputs)
            warmup.square().mean().backward()
        forward, _ = capture_phase(inference, stream=stream)
        backward, report = capture_phase(training, model, stream=stream, reduced_precision=True)
        self.assertEqual(report["validation"]["gradient_tensors"], 4)
        self.assertEqual(len(report["validation"]["uncaptured"]), 8)
        self.assertEqual(report["validation"]["uncaptured_calls"], 9)
        self.assertEqual(len(report["validation"]["replays"]), 2)
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
