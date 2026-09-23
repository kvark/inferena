"""Campaign contract, bounded compilation and backend replay; no retained artifacts."""

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import subprocess
import time
import unittest
from unittest.mock import patch

import torch

from execution import QualificationError, capture_phase, check_gradient_set, compare_tensors, graph_backend, profile_phase, synchronize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from p3hpc import (MODELS, PYTHON_VERSION, TORCH_REVISION, TORCH_VERSION, TUNE_SCRATCH_BYTES, SDPA_BACKENDS,
                  check_pair, check_torch_identity,
                  conditions, create_parser, gpu_matches, validate_replicated_gradients,
                  runner_bash, select_native_device, archive_results, reference_sdpa_policy,
                  assess_phases, eager_diagnostic)


class CampaignTest(unittest.TestCase):
    def test_uncaptured_qualification_and_partial_training_failure(self):
        from bench import _bench

        class Model(torch.nn.Module):
            def __init__(self, drift=False):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(8))
                self.calls = 0
                self.drift = drift

            def forward(self, x):
                if torch.is_grad_enabled():
                    self.calls += 1
                    if self.drift == "fault":
                        raise RuntimeError("device execution fault")
                return x * self.weight + (self.calls * 0.01 if self.drift else 0)

        x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        model = Model()
        def training():
            output = model(x)
            loss = output.square().mean()
            loss.backward()
            return output, loss
        _, report = capture_phase(training, model, device="cpu", capture=False)
        self.assertEqual(report["status"], "validated-uncaptured")
        self.assertEqual(report["validation"]["consecutive_replays"], 0)
        self.assertEqual(report["validation"]["gradient_tensors"], 1)
        model.drift = True
        with self.assertRaises(QualificationError) as error:
            capture_phase(training, model, device="cpu", capture=False)
        self.assertEqual(error.exception.details["stage"], "uncaptured repeat 1")
        self.assertEqual(error.exception.details["tensor"], "output 0")

        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "preparation.json"
            env = {"INFERENA_TORCH_MODE": "eager", "INFERENA_GRAPH_REPLAY": "0",
                   "INFERENA_PREPARATION_REPORT": str(receipt), "INFERENA_REFERENCE_DIAGNOSTIC": "1"}
            def measure(fn, *_, **__):
                return fn(), [1.0], {"runs": 5, "seconds": 2.0}
            with patch.dict(os.environ, env), patch("bench.load_model", return_value=Model(True)) as load, \
                 patch("bench.prepare_inputs", return_value={"input_features": x}), \
                 patch("bench._measure_call", side_effect=measure):
                output = io.StringIO()
                with redirect_stdout(output):
                    _bench("test", {"type": "whisper"}, "cpu", None)
                result = json.loads(output.getvalue())
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["timing_samples_ms"],
                                 {"inference": [1.0], "latency": None, "training": None})
                self.assertIsNone(result["outputs"]["grad_norm"])
                self.assertEqual(result["outputs"]["logits_sample"], x.flatten().tolist())
                self.assertTrue(result["protocol"]["diagnostic"])
                self.assertEqual(json.loads(receipt.read_text())["qualification_failure"]["phase"], "training")
                load.return_value = Model("fault")
                with self.assertRaisesRegex(RuntimeError, "device execution fault"):
                    _bench("test", {"type": "whisper"}, "cpu", None)

    def test_smolvla_attention_uses_normalized_self_keys_and_external_cross_keys(self):
        from bench import ExpertLayer

        x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 7
        context = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) / 3
        for cross in (False, True):
            layer = ExpertLayer(8, 4, 16, 2, 1, 4, cross)
            normalized = layer.input_layernorm(x)
            residual = x + layer.self_attn(
                normalized, context if cross else normalized, causal=not cross)
            expected = residual + layer.mlp(layer.post_attention_layernorm(residual))
            actual = layer(x, context)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual_gradients = torch.autograd.grad(actual.sum(), tuple(layer.parameters()))
            expected_gradients = torch.autograd.grad(expected.sum(), tuple(layer.parameters()))
            for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    def test_synthetic_parameters_match_native_layout_without_low_rank_weights(self):
        from bench import (_INIT_AMPLITUDE, _parameter_values, _name_seeded_init,
                           _transposed_init, _sd_parameter_name, _smolvla_parameter_name)

        name = "time_embed.0.weight"
        values = _parameter_values(name, 8)
        # Shared with Rust's synthetic_parameter_bits test, including u32 overflow.
        self.assertEqual(values.view(torch.uint32).tolist(), [
            0x3b10bfaf, 0xbb65c670, 0x3adfd0cc, 0xbb06e742,
            0x3b887a65, 0x3c9f5be6, 0xbcb7f94b, 0x3cafdd01,
        ])
        self.assertTrue(torch.equal(_parameter_values(name, 8, _INIT_AMPLITUDE * 0.5), values * 0.5))
        self.assertFalse(torch.equal(values, _parameter_values("time_embed.1.weight", 8)))
        self.assertEqual(_parameter_values("empty", 0).numel(), 0)
        for canonical in (
            _sd_parameter_name("encoder_attn.0.self_attn.q_proj.weight"),
            _smolvla_parameter_name("layers.0.self_attn.q_proj.weight"),
            "fc.weight", "layers.0.fc1.weight",
        ):
            native = _parameter_values(canonical, 32 * 64).reshape(32, 64)
            ordinary = torch.empty_like(native)
            transposed = torch.empty(64, 32)
            _name_seeded_init(ordinary, canonical)
            _transposed_init(transposed, canonical)
            self.assertTrue(torch.equal(ordinary, native))
            self.assertTrue(torch.equal(transposed.T, native))
            # A sinusoid reshaped as a matrix has rank <= 2 in exact arithmetic.
            energy = torch.linalg.svdvals(native.double()).square()
            self.assertLess(float(energy[:2].sum() / energy.sum()), 0.25)
            self.assertAlmostEqual(float(native.square().mean()), 0.02**2 / 2, delta=2e-5)

    def test_common_source_with_platform_specific_builds(self):
        # Full-vector bounds admit cancellation noise, not corrupt/stale data.
        reference = torch.tensor([1.0, -1.0, 0.0])
        rounded = reference + torch.tensor([0.0, 0.0, 2e-6])
        self.assertEqual(compare_tensors(rounded, reference)["elementwise_mismatches"], 1)
        for gradient in (False, True):
            compare_tensors(rounded, reference, gradient=gradient)
            for invalid in (reference * 2, -reference, torch.zeros_like(reference),
                            torch.tensor([1.0, -1.0, 1e-3]), reference + float("nan"),
                            reference + float("inf"), reference[:2], reference.double()):
                with self.assertRaises(ValueError):
                    compare_tensors(invalid, reference, gradient=gradient)
        # The maximum catches sparse corruption; RMS catches diffuse drift
        # even when one large element would make a max-only gate permissive.
        reference = torch.zeros(10000)
        reference[0] = 1.0
        diffuse = reference + 1e-5
        for gradient in (False, True):
            with self.assertRaisesRegex(ValueError, "fixed bounds"):
                compare_tensors(diffuse, reference, gradient=gradient)
        compare_tensors(torch.zeros(3), torch.zeros(3), gradient=True)
        with self.assertRaises(ValueError):
            compare_tensors(torch.full((3,), 1e-3), torch.zeros(3), gradient=True)
        reference = torch.tensor([1.0, -1.0, 0.0])
        rounded = reference + torch.tensor([0.0, 0.0, 2e-3])
        check_gradient_set([
            compare_tensors(rounded, reference, gradient=True, reduced_precision=True)
        ], reduced_precision=True)
        with self.assertRaisesRegex(ValueError, "fixed bounds"):
            compare_tensors(rounded, reference, reduced_precision=True)
        with self.assertRaisesRegex(ValueError, "full-gradient"):
            check_gradient_set([
                compare_tensors(reference * 1.02, reference, gradient=True, reduced_precision=True)
            ], reduced_precision=True)
        defaults = create_parser().parse_args([])
        self.assertTrue(defaults.collect)
        self.assertEqual(defaults.models, list(MODELS))
        self.assertEqual(defaults.precisions, ["strict", "accelerated"])
        self.assertEqual(defaults.replicates, 3)
        self.assertFalse(defaults.max_autotune)
        self.assertFalse(defaults.graph_ablation)
        self.assertEqual(defaults.compile_seconds, 120)
        self.assertEqual(defaults.tune_seconds, 60)
        self.assertIsNone(defaults.backend)
        self.assertIsNone(defaults.gpu)
        self.assertIsNone(defaults.results_dir)
        self.assertIsNone(defaults.sdpa)
        self.assertEqual(create_parser().parse_args(["--sdpa", "efficient"]).sdpa, "efficient")
        self.assertFalse(create_parser().parse_args(["--qualify-only"]).collect)
        self.assertFalse(create_parser().parse_args(["--no-max-autotune"]).max_autotune)
        for backend in ("cuda", "rocm", "xpu", "mps", "cpu"):
            replay = backend in ("cuda", "rocm", "xpu")
            self.assertEqual(conditions(backend), [("default", replay)])
            self.assertEqual(conditions(backend, True), [("default", replay), ("max-autotune", replay)])
        self.assertTrue(create_parser().parse_args(["--max-autotune"]).max_autotune)
        def pair(error):
            return {
                "pytorch": {
                    "precision": {"reduced_precision_allowed": True},
                },
                "meganeura": {
                    "validation": {
                        "forward_valid": True,
                        "parameter_gradient_relative_l2_error": error,
                        "total_gradient_relative_error": error,
                    },
                },
            }
        repeated = [pair(0.02), pair(0.052), pair(0.03)]
        groups = {("accelerated", "model", "default", True): repeated}
        report = validate_replicated_gradients(groups, 3)["groups"][0]
        self.assertEqual(report["status"], "pass")
        repeated[-1]["meganeura"]["validation"]["parameter_gradient_relative_l2_error"] = 0.11
        self.assertEqual(validate_replicated_gradients(groups, 3)["status"], "fail")
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
        import prepare_models
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "models/Test"
            directory.mkdir(parents=True)
            files = {"config.json": b"config", "model.safetensors": b"weights"}
            for name, contents in files.items():
                (directory / name).write_bytes(contents)
            pin = {"revision": "a" * 40,
                   "sha256": {name: hashlib.sha256(contents).hexdigest()
                               for name, contents in files.items()}}
            with patch.object(prepare_models, "ROOT", root), patch.object(prepare_models, "PINS", {"Test": pin}):
                prepare_models.prepare_model("Test")
                receipt = json.loads((directory / "source.json").read_text())
                self.assertEqual(receipt, {"repo": "HuggingFaceTB/Test", **pin})
                (directory / "model.safetensors").write_bytes(b"wrong")
                with self.assertRaisesRegex(ValueError, "does not match"):
                    prepare_models.prepare_model("Test")
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "run with spaces"
            campaign.mkdir()
            (campaign / "runner.log").write_text("runner output", encoding="utf-8")
            for status in ("incomplete", "complete"):
                (campaign / "campaign.json").write_text(json.dumps({"status": status}), encoding="utf-8")
                archive = archive_results(campaign)
                self.assertEqual(archive, campaign.parent / "latest.tgz")
                with tarfile.open(archive, "r:gz") as contents:
                    self.assertEqual(set(contents.getnames()), {
                        campaign.name, f"{campaign.name}/campaign.json", f"{campaign.name}/runner.log"})
                    self.assertEqual(json.load(contents.extractfile(f"{campaign.name}/campaign.json")), {"status": status})
                self.assertTrue((campaign / "runner.log").is_file())
            previous = archive.read_bytes()
            with patch("p3hpc.shutil.make_archive", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    archive_results(campaign)
            self.assertEqual(archive.read_bytes(), previous)

    def test_explicit_backend_is_probed_and_synchronized_without_fallback(self):
        from bench import MODEL_REGISTRY, attention_context, attention_policy, bench, detect_device
        initial_sdpa = torch.nn.attention._cur_sdpa_kernel_backends()
        args = create_parser().parse_args([])
        for backend, device in (("cuda", "cuda:0"), ("rocm", "cuda:0"), ("xpu", "xpu:0"),
                                ("cpu", "cpu"), ("mps", "mps")):
            args.backend = backend
            for model in MODELS:
                default = "math" if backend == "xpu" else "auto"
                for override in (None, "auto", "math", "efficient"):
                    args.sdpa = override
                    with patch.dict(os.environ), patch("torch.version.hip", "7.2" if backend == "rocm" else None):
                        os.environ.pop("INFERENA_SDPA", None)
                        if override is not None:
                            os.environ["INFERENA_SDPA"] = override
                        self.assertEqual(attention_policy(device, MODEL_REGISTRY[model]["type"]), override or default)
                        self.assertEqual(reference_sdpa_policy(args, model), override or default)
        for policy, expected in (("auto", initial_sdpa),
                                 ("math", [torch.nn.attention.SDPBackend.MATH]),
                                 ("efficient", [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION])):
            with patch.dict(os.environ, {"INFERENA_SDPA": policy}):
                with attention_context("cuda:0"):
                    self.assertEqual(torch.nn.attention._cur_sdpa_kernel_backends(), expected)
            self.assertEqual(torch.nn.attention._cur_sdpa_kernel_backends(), initial_sdpa)
        with patch.dict(os.environ, {"INFERENA_SDPA": "invalid"}):
            with self.assertRaisesRegex(ValueError, "INFERENA_SDPA"):
                attention_context("cuda:0")
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
             patch("bench._bench") as run, patch("bench.graph_backend") as api, \
             patch.dict(os.environ, {"TRITON_DEFAULT_BACKEND": "intel"}):
            run.side_effect = lambda *_, **__: self.assertEqual(
                torch.nn.attention._cur_sdpa_kernel_backends(), [torch.nn.attention.SDPBackend.MATH])
            bench("model", {})
            stream = api.return_value.Stream.return_value
            run.assert_called_once_with("model", {}, "xpu:0", stream)
            api.return_value.stream.assert_called_once_with(stream)
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        for model_type, hip in (("whisper", "7.2"), ("whisper", None), ("sd_unet", "7.2")):
            with patch("bench.detect_device", return_value="cuda:0"), \
                 patch("bench.select_compiler_backend"), patch("bench.graph_backend"), \
                 patch("torch.version.hip", hip), patch("bench._bench") as run:
                def fail(*args):
                    self.assertIs(torch.nn.functional.scaled_dot_product_attention, original_sdpa)
                    self.assertEqual(torch.nn.attention._cur_sdpa_kernel_backends(), initial_sdpa)
                    raise RuntimeError("test qualification failure")
                run.side_effect = fail
                with self.assertRaisesRegex(RuntimeError, "test qualification failure"):
                    bench("model", {"type": model_type})
            self.assertIs(torch.nn.functional.scaled_dot_product_attention, original_sdpa)
            self.assertEqual(torch.nn.attention._cur_sdpa_kernel_backends(), initial_sdpa)
        # MPS must attempt compilation, not report an eager result with zero
        # compile time. A mock checks routing, not real Metal backend support.
        from bench import _bench
        with patch.dict(os.environ, {"INFERENA_TORCH_MODE": "default", "INFERENA_GRAPH_REPLAY": "0"}), \
             patch("bench.load_model"), patch("bench.synchronize"), \
             patch("bench.device_name", return_value="test MPS"), \
             patch("torch.compile", side_effect=RuntimeError("compiler failure")) as compile:
            with self.assertRaisesRegex(RuntimeError, "no eager timing substituted"):
                _bench("model", {"type": "resnet"}, "mps", None)
            compile.assert_called_once()

    def test_executed_receipts_enforce_the_requested_protocol(self):
        args = create_parser().parse_args(["--backend", "cuda", "--gpu", "test GPU"])
        args.torch_version = TORCH_VERSION
        base = {
            "status": "ok", "benchmark_rev": "source", "gpu_name": args.gpu, "model": "Whisper-tiny",
            "validation": {"comparison_performed": True, "forward_valid": True,
                           "training_valid": True, "reference_framework": "pytorch"},
            "protocol": {"name": "inferena-paper-v3", "training_requested": True,
                         "synthetic_parameter_init": "name-index-uniform-v1",
                         "diagnostic": False, "warmup_runs": 5, "warmup_seconds": 2.0,
                         "warmup": {phase: {"runs": 5, "seconds": 2.1}
                                    for phase in ("inference", "latency", "training")}},
            "timing_samples_ms": {phase: [1.0] for phase in ("inference", "latency", "training")},
            "precision": {"comparison_class": "strict-f32", "reduced_precision_allowed": False,
                          "cooperative_matrix_policy": "NativeF32",
                          "native_f32_cooperative_matrix_permitted": True,
                          "f16_cooperative_matrix_permitted": False},
        }
        tuning = {"scope": "All", "max_classes": 2 * sys.maxsize + 1,
                  "max_time": {"secs": 60, "nanos": 0}, "max_scratch_bytes": TUNE_SCRATCH_BYTES}
        mg = {**copy.deepcopy(base), "framework": "meganeura", "optimizer": {
            "measured_construction": True, "mode": "egglog-outlined", "sessions": [{
                "mode": mode, "cooperative_matrix_policy": "NativeF32", "search": {
                    "options": {"max_time": {"secs": 60, "nanos": 0}, "max_graphs": 16,
                                "max_programs": 64, "warmup_runs": 2,
                                "warmup_time": {"secs": 0, "nanos": 250000000},
                                "max_plan_bytes": 3 * 1024**3, "tuning": tuning},
                    "selected": 0, "trials": [{"outcome": {"qualified": True}, "kernel_tuning": {
                        "options": tuning, "class_limit_reached": False,
                        "visited_classes": 1, "eligible_classes": 1, "time_budget_exhausted": False,
                    }}],
                },
                "memory_budget": {"device_budget_bytes": 4 * 1024**3, "device_usage_bytes": 0,
                                  "plan_fraction_of_available": 0.75},
                "qualification": {"policy": "fixed-full-tensor-v4", "rtol": 1e-4, "atol": 1e-6,
                                  "accelerated_gradient_rtol": 0.01, "qualified_calls": 2,
                                  "output_elements": 1, "gradient_elements": int(mode == "Training")},
            } for mode in ("Inference", "Training")],
        }}
        pt = {**copy.deepcopy(base), "framework": "pytorch", "backend": "CUDA",
              "protocol": {**copy.deepcopy(base["protocol"]), "name": "inferena-graph-replay-v7"},
              "torch_version": TORCH_VERSION,
              "environment": {"torch_git_version": TORCH_REVISION, "python_version": PYTHON_VERSION,
                              "triton_backend": "nvidia"},
              "execution": {
                  "stream_policy": "single dedicated preparation/run stream",
                  "requested_mode": "default", "compiled": True,
                  "sdpa_compile": "compiled",
                  "sdpa_policy": "auto", "sdpa_enabled_backends": sorted(SDPA_BACKENDS),
                  "compile_budget_seconds": args.compile_seconds, "compile_budget_enforced": True,
                  "compiler_options": {key: False for key in (
                      "max_autotune", "coordinate_descent_tuning", "max_autotune_gemm",
                      "max_autotune_pointwise", "triton.cudagraphs")},
                  "graph_replay": {"requested": True, "phases": {phase: {
                      "status": "captured-and-validated", "api": "torch.cuda.CUDAGraph",
                      "validation": {"policy": "fixed-full-tensor-v4", "uncaptured_calls": 3,
                                     "uncaptured_repeats": 2, "consecutive_replays": 2,
                                     "output_metric": "per-tensor RMS and maximum absolute error",
                                     "rtol": 1e-4, "atol": 1e-6,
                                     "accelerated_gradient_rtol": 0.01},
                  } for phase in ("inference", "latency", "training")}},
              }}
        check = lambda records: check_pair(records, args, "default", True, 1, "source", precision="strict")
        check([mg, pt])
        failure = {"kind": "numerical", "phase": "training", "stage": "uncaptured repeat 1",
                   "tensor": "output 0", "metrics": {"max_abs_error": 0.002}}
        partial = copy.deepcopy(pt)
        partial["status"] = "partial"
        partial["execution"]["failure"] = failure
        for phase, status in (("training", "failed"), ("latency", "not-attempted")):
            partial["execution"]["graph_replay"]["phases"][phase] = {"status": status}
            partial["timing_samples_ms"][phase] = None
        assess = lambda records, receipt={}: assess_phases(
            records, args, "default", True, 1, "source", receipt, precision="strict")[0]
        self.assertEqual({k: v["status"] for k, v in assess([mg, partial]).items()},
                         {"inference": "valid", "training": "failed", "latency": "not-attempted"})
        from generate_chart import load_summaries
        with tempfile.TemporaryDirectory() as temporary:
            summary = Path(temporary) / "Whisper-tiny_summary.json"
            chart_record = {**partial, "timings": {"inference_ms": 1.0}}
            summary.write_text(json.dumps([chart_record]))
            self.assertEqual(load_summaries(temporary)[0][1][0]["inference_ms"], 1.0)
            for invalid in (
                {**chart_record, "protocol": {"diagnostic": True}},
                {**chart_record, "execution": {}},
                {**chart_record, "timings": {"inference_ms": None}},
                {**chart_record, "validation": {"comparison_performed": True, "forward_valid": False}},
            ):
                summary.write_text(json.dumps([invalid]))
                self.assertEqual(load_summaries(temporary), [])
        mismatch = copy.deepcopy(mg)
        mismatch["validation"]["forward_valid"] = False
        self.assertEqual(assess([mismatch, pt])["inference"]["status"], "mismatch")
        error = {"framework": "pytorch", "status": "error", "error": "traceback"}
        with self.assertRaisesRegex(ValueError, "unclassified"):
            assess([mg, error])
        receipt = {"qualification_failure": {**failure, "phase": "inference"}}
        self.assertEqual(assess([mg, error], receipt)["inference"]["status"], "failed")
        self.assertEqual(assess([mg, pt])["training"]["status"], "valid")
        # A separate eager reference can validate native outputs, not rescue
        # the failed primary or enter its timing population.
        eager = copy.deepcopy(pt)
        eager["protocol"]["diagnostic"] = True
        eager["execution"].update(requested_mode="eager", compiled=False,
                                   sdpa_policy="math", sdpa_enabled_backends=["MATH"], sdpa_compile="eager")
        eager["execution"]["graph_replay"]["requested"] = False
        for report in eager["execution"]["graph_replay"]["phases"].values():
            report.update(status="validated-uncaptured", api=None)
            report["validation"]["consecutive_replays"] = 0
        self.assertEqual(assess_phases([mg, eager], args, "eager", False, 1, "source", {},
                                      precision="strict", oracle=True)[0]["training"]["status"], "valid")
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            check([mg, eager])
        groups = {("strict", "model", "default", True): []}
        self.assertEqual(validate_replicated_gradients(groups, 3)["groups"][0]["status"], "incomplete")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "Whisper-tiny_meganeura.json").write_text(json.dumps(mg))
            command = ["bash", "run.sh", "-f", "pytorch,meganeura", "--measurement-runs", "20",
                       "--results-dir", str(folder)]
            def run_diagnostic(command, **kwargs):
                target = Path(command[command.index("--results-dir") + 1])
                (target / "Whisper-tiny_pytorch.json").write_text(json.dumps(eager))
                self.assertEqual(kwargs["env"]["INFERENA_REFERENCE_DIAGNOSTIC"], "1")
                self.assertEqual(kwargs["env"]["INFERENA_GRAPH_REPLAY"], "0")
                self.assertEqual(kwargs["env"]["INFERENA_SDPA"], "math")
                self.assertEqual(command[command.index("-f") + 1], "pytorch")
                return subprocess.CompletedProcess(command, 0)
            with patch("p3hpc.subprocess.run", side_effect=run_diagnostic) as run, \
                 patch("p3hpc.subprocess.check_output", return_value=json.dumps([mg, eager])):
                report = eager_diagnostic(command, {}, folder, "Whisper-tiny", args, 1, "source", "strict")
                run.assert_called_once()
                self.assertFalse(report["timing_substituted"])
                self.assertEqual(report["phases"]["training"]["status"], "valid")
                self.assertEqual(json.loads((folder / "Whisper-tiny_meganeura.json").read_text()), mg)
                with self.assertRaises(FileExistsError):
                    eager_diagnostic(command, {}, folder, "Whisper-tiny", args, 1, "source", "strict")
        for engine, path, wrong in (
            (0, ("protocol", "warmup_seconds"), 0),
            (1, ("protocol", "warmup", "inference", "seconds"), 1.9),
            (1, ("protocol", "warmup", "latency", "runs"), 4),
            (0, ("protocol", "name"), "inferena-paper-v2"),
            (1, ("protocol", "name"), "inferena-graph-replay-v5"),
            (0, ("protocol", "synthetic_parameter_init"), None),
            (1, ("protocol", "synthetic_parameter_init"), "sinusoidal"),
            (0, ("optimizer", "measured_construction"), False),
            (0, ("precision", "cooperative_matrix_policy"), "Disabled"),
            (0, ("precision", "f16_cooperative_matrix_permitted"), True),
            (0, ("optimizer", "sessions"), []),
            (0, ("optimizer", "sessions", 0, "search", "options", "max_graphs"), 4),
            (0, ("optimizer", "sessions", 0, "search", "options", "warmup_time"), {"secs": 0, "nanos": 0}),
            (0, ("optimizer", "sessions", 0, "search", "options", "tuning", "max_classes"), 8),
            (0, ("optimizer", "sessions", 0, "search", "options", "tuning", "max_time"), {"secs": 2, "nanos": 0}),
            (0, ("optimizer", "sessions", 0, "search", "trials", 0, "kernel_tuning", "visited_classes"), 0),
            (0, ("optimizer", "sessions", 0, "search", "trials", 0, "outcome", "qualified"), False),
            (0, ("optimizer", "sessions", 1, "qualification", "gradient_elements"), 0),
            (1, ("execution", "compiled"), False),
            (1, ("execution", "sdpa_policy"), "math"),
            (1, ("execution", "sdpa_compile"), "eager"),
            (1, ("execution", "sdpa_enabled_backends"), ["MATH"]),
            (1, ("execution", "compile_budget_enforced"), False),
            (1, ("execution", "compiler_options", "max_autotune"), True),
            (1, ("execution", "graph_replay", "requested"), False),
            (1, ("execution", "graph_replay", "phases", "training", "api"), "torch.xpu.XPUGraph"),
            (1, ("execution", "graph_replay", "phases", "training", "validation", "rtol"), 0.01),
            (1, ("execution", "graph_replay", "phases", "training", "validation", "policy"), "fixed-full-gradient-v3"),
        ):
            records = copy.deepcopy([mg, pt])
            target = records[engine]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = wrong
            with self.subTest(path=path), self.assertRaises(ValueError):
                check(records)

        args.sdpa = "efficient"
        with self.assertRaisesRegex(ValueError, "attention policy"):
            check([mg, pt])
        alternate = copy.deepcopy(pt)
        alternate["execution"]["sdpa_policy"] = "efficient"
        with self.assertRaisesRegex(ValueError, "attention backends"):
            check([mg, alternate])
        alternate["execution"]["sdpa_enabled_backends"] = ["EFFICIENT_ATTENTION"]
        check([mg, alternate])
        args.sdpa = None
        with self.assertRaisesRegex(ValueError, "attention policy"):
            check([mg, alternate])

        args.backend = "rocm"
        rocm = copy.deepcopy(pt)
        rocm["backend"] = "ROCm 7.2"
        rocm["environment"]["triton_backend"] = "amd"
        check([mg, rocm])
        # A labelled diagnostic override may still request the failing math path.
        args.sdpa = "math"
        rocm["execution"]["sdpa_policy"] = "math"
        rocm["execution"]["sdpa_enabled_backends"] = ["MATH"]
        check([mg, rocm])
        args.sdpa = None
        rocm["model"] = mg["model"]
        rocm["execution"]["sdpa_policy"] = "auto"
        rocm["execution"]["sdpa_enabled_backends"] = sorted(SDPA_BACKENDS)
        check([mg, rocm])
        args.backend = "cuda"

        from bench import _measure_call
        for duration, expected_runs in ((0.125, 16), (1.0, 5)):
            clock = [0.0]
            def step():
                clock[0] += duration
            with patch("bench.time.perf_counter", side_effect=lambda: clock[0]), \
                 patch("bench.synchronize"):
                _, samples, warmup = _measure_call(step, 5, 3, "cpu", "inference")
            self.assertEqual(warmup, {"runs": expected_runs, "seconds": expected_runs * duration})
            self.assertEqual(samples, [duration * 1000] * 3)

    def test_compilation_watchdog_preserves_failure_and_kills_workers(self):
        directory = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "compile.json"
            marker = Path(temporary) / "orphan.txt"
            env = dict(os.environ, INFERENA_COMPILE_SECONDS="0.3", INFERENA_PREPARATION_REPORT=str(report))
            for statement, code, status in (("pass", 0, "complete"),
                                            ("raise ValueError('test failure')", 1, "failed"),
                                            ("time.sleep(5)", 124, "timeout")):
                if report.exists():
                    report.unlink()
                child = f"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).touch()"
                source = ("import subprocess, sys, time; from budget import compilation_budget\n"
                          f"with compilation_budget():\n subprocess.Popen([sys.executable, '-c', {child!r}])\n {statement}\n"
                          if code == 124 else f"from budget import compilation_budget\nwith compilation_budget():\n {statement}\n")
                result = subprocess.run([sys.executable, str(directory / "budget.py"), "-c", source],
                                        cwd=directory, env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(json.loads(report.read_text())["status"], status)
            time.sleep(2)
            self.assertFalse(marker.exists(), "compiler descendant survived the deadline")

    def test_index_add_embedding_backward_accumulates_repeated_rows(self):
        from bench import _IndexAddEmbeddingBackward

        weight = torch.randn(5, 3, requires_grad=True)
        indices = torch.tensor([1, 1, 3])
        upstream = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        output = _IndexAddEmbeddingBackward.apply(indices, weight)
        output.backward(upstream)

        expected = torch.zeros_like(weight)
        expected.index_add_(0, indices, upstream)
        torch.testing.assert_close(output, weight[indices])
        torch.testing.assert_close(weight.grad, expected)


@unittest.skipUnless(torch.cuda.is_available() or torch.xpu.is_available(), "CUDA/HIP/XPU required")
class ReplayTest(unittest.TestCase):
    def test_forward_backward_replay_observes_live_inputs_and_weights(self):
        torch.manual_seed(7)
        device = "cuda" if torch.cuda.is_available() else "xpu"
        from bench import select_compiler_backend
        select_compiler_backend(device)
        api = graph_backend(device)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.SiLU(),
            torch.nn.Flatten(), torch.nn.Linear(4 * 8 * 8, 3),
        ).to(device)
        inputs = torch.randn(2, 2, 8, 8, device=device)
        compiled = torch.compile(model, options={
            "max_autotune": False, "triton.cudagraphs": False,
        })

        def inference():
            with torch.no_grad():
                return compiled(inputs)

        def training():
            output = compiled(inputs)
            loss = output.square().mean()
            loss.backward()
            return output, loss

        stream = api.Stream(device=device)
        stream.wait_stream(api.current_stream(device))
        with api.stream(stream):
            # Keep this graph alive across capture, like compiled model caches.
            warmup = compiled(inputs)
            warmup.square().mean().backward()
        forward, _ = capture_phase(inference, stream=stream, device=device)
        backward, report = capture_phase(training, model, stream=stream, reduced_precision=True, device=device)
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
            synchronize(device)
            torch.testing.assert_close(output, expected_output)
            torch.testing.assert_close(loss, expected_output.square().mean())
            for parameter, storage, expected in zip(
                model.parameters(), captured_gradients, expected_gradients, strict=True
            ):
                self.assertIs(parameter.grad, storage)
                torch.testing.assert_close(parameter.grad, expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            # PTI may omit child kernels of Level Zero command-buffer replay.
            # Check ordinary XPU GPU profiling separately from replay correctness.
            with api.stream(stream):
                profile = profile_phase(backward if device == "cuda" else training, path, 2, device=device)
            self.assertEqual(len(profile["instrumented_wall_ms"]), 2)
            events = json.loads(path.read_text())["traceEvents"]
            self.assertTrue(any(event.get("cat") == "kernel" for event in events))
            if device == "cuda" and torch.version.cuda:
                self.assertTrue(any("cudaGraphLaunch" in event.get("name", "") for event in events))


if __name__ == "__main__":
    unittest.main()
