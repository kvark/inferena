"""Android collection and coverage contracts; no device needed."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from p3hpc import create_parser, check_native_search, TUNE_SCRATCH_BYTES
from p3hpc_android import (PROTOCOL, Device, battery_info, check_battery, check_native,
                           validate_oracle, write_json, sha256)
from p3hpc_report import campaign_rows


class AndroidContract(unittest.TestCase):
    def test_memory_fallback_is_bounded_and_android_only(self):
        import sys
        tuning = dict(scope="All", strategy="Race", max_classes=2 * sys.maxsize + 1, min_improvement=0.05,
                      max_time=dict(secs=30, nanos=0), max_scratch_bytes=TUNE_SCRATCH_BYTES)
        session = dict(mode="Training", cooperative_matrix_policy="NativeF32",
            search_input_policy="refresh before each step, outside timing",
            search=dict(options=dict(max_time=dict(secs=30, nanos=0),
            max_graphs=16, max_programs=64, max_submission_chunks=64, patience=None,
            min_improvement=0.01, warmup_runs=2, warmup_time=dict(secs=0, nanos=250000000),
            max_plan_bytes=9 * 1024**3 // 4, tuning=tuning), selected=0,
            trials=[dict(outcome=dict(qualified=True), qualifications=1, kernel_tuning=dict(
                options=tuning, class_limit_reached=False, visited_classes=1, eligible_classes=1,
                time_budget_exhausted=False, reused_classes=[], outcomes=[]))]),
            memory_budget=dict(device_budget_bytes=0, device_usage_bytes=0,
                               source="android-memavailable-capped-3gib", system_available_bytes=4 * 1024**3,
                               available_bytes=3 * 1024**3, plan_fraction_of_available=0.75),
            qualification=dict(policy="fixed-full-tensor-v4", reference="first checked program", rtol=1e-4,
                               arithmetic_reference=dict(policy="native-f32-full-tensor-v2", status="pass",
                                   reference="original graph; untuned native strict-f32", prepared_before_precision_conversion=True,
                                   output_policy="tensor-fidelity-v1", output_rtol=1e-4, gradient_rtol=1e-4, atol=1e-6, preparation_seconds=0.1,
                                   checked_calls=1, rejected_calls=0),
                               atol=1e-6, accelerated_gradient_rtol=0.01, qualified_calls=1,
                               output_elements=1, gradient_elements=1),
            kernel_memo=dict(policy="campaign-kernel-memo-v1", enabled=False, load="disabled", loaded_classes=0,
                             reused_class_visits=0, new_comparisons=0, key=None, saved_classes=None))
        check_native_search(session, 30, 64, android=True)
        # Meganeura omits empty reused-class arrays from its serialized receipt.
        del session["search"]["trials"][0]["kernel_tuning"]["reused_classes"]
        check_native_search(session, 30, 64, android=True)
        with self.assertRaises(ValueError):
            check_native_search(session, 30, 64)
        session["memory_budget"]["available_bytes"] = 4 * 1024**3
        with self.assertRaises(ValueError):
            check_native_search(session, 30, 64, android=True)

    def test_cli_retains_common_measurement_policy(self):
        args = create_parser().parse_args(["--backend", "android", "--adb-serial", "quest"])
        self.assertEqual(args.replicates, 3)
        self.assertEqual(args.precisions, ["strict", "accelerated"])
        self.assertEqual(args.tune_seconds, 30)
        self.assertTrue(args.kernel_memo)

    def test_native_phase_order_and_measurement_gate(self):
        args = create_parser().parse_args([])
        record = dict(framework="meganeura", framework_rev="revision", model="SmolVLA", gpu_name="Adreno",
            protocol=dict(name="inferena-paper-v6", diagnostic=False, synthetic_parameter_init="name-index-uniform-v1",
                          warmup_runs=5, warmup_seconds=2, measurement_runs=20, training_requested=True,
                          warmup={p: dict(runs=5, seconds=2.1) for p in ("inference", "latency", "training")}),
            timing_samples_ms={p: [1.] * 20 for p in ("inference", "latency", "training")},
            precision=dict(comparison_class="strict-f32"), optimizer=dict(measured_construction=True,
                mode="egglog-outlined", sessions=[dict(mode=mode, cooperative_matrix_policy="NativeF32", cooperative_dispatches=0)
                    for mode in ("Inference", "Training", "Inference")]))
        with patch("p3hpc_android.check_native_search") as search:
            check_native(record, args, "SmolVLA", "strict", 20, {"rev": "revision"}, {"name": "Adreno"})
            self.assertEqual(search.call_count, 3)
            record["precision"]["comparison_class"] = "accelerated"
            for session in record["optimizer"]["sessions"]:
                session["cooperative_matrix_policy"] = "Auto"
            check_native(record, args, "SmolVLA", "accelerated", 20, {"rev": "revision"}, {"name": "Adreno"})
            record["precision"]["comparison_class"] = "strict-f32"
            for session in record["optimizer"]["sessions"]:
                session["cooperative_matrix_policy"] = "NativeF32"
            record["optimizer"]["sessions"].pop()
            with self.assertRaises(ValueError):
                check_native(record, args, "SmolVLA", "strict", 20, {"rev": "revision"}, {"name": "Adreno"})
            record["model"] = "Whisper-tiny"
            check_native(record, args, "Whisper-tiny", "strict", 20, {"rev": "revision"}, {"name": "Adreno"})
            record["timing_samples_ms"]["training"] = [1.]
            with self.assertRaises(ValueError):
                check_native(record, args, "Whisper-tiny", "strict", 20, {"rev": "revision"}, {"name": "Adreno"})

    def test_charger_and_temperature(self):
        good = battery_info("  AC powered: true\n  Weak Charger: false\n  level: 70\n  temperature: 325")
        check_battery(good)
        check_battery(dict(good, weak_charger=True))
        for override in (dict(powered=False), dict(level=29), dict(temperature_c=45)):
            with self.subTest(override=override), self.assertRaises(RuntimeError):
                check_battery(dict(good, **override))

    def test_adb_timeout_kills_remote_workload(self):
        import subprocess
        device = Device("quest")
        with tempfile.TemporaryDirectory() as tmp, patch("p3hpc_android.subprocess.run", side_effect=subprocess.TimeoutExpired("adb", 1)), patch.object(device, "shell", side_effect=["1234", ""]) as shell, patch.object(device, "call", return_value="tombstone"):
            with self.assertRaises(subprocess.TimeoutExpired):
                device.run("runner", Path(tmp) / "log", "/tmp/pid", 1)
            self.assertEqual(shell.call_args.args, ("kill -9 1234",))
            self.assertEqual((Path(tmp) / "android-crash.log").read_text(), "tombstone\n")

    def test_native_crash_is_preserved_without_retry(self):
        device = Device("quest")
        with tempfile.TemporaryDirectory() as tmp, patch("p3hpc_android.subprocess.run", return_value=SimpleNamespace(returncode=139)) as run, patch.object(device, "call", return_value="SIGSEGV in driver"):
            folder = Path(tmp)
            self.assertEqual(device.run("runner", folder / "runner.log", "/tmp/pid", 1), 139)
            run.assert_called_once()
            self.assertIn("SIGSEGV", (folder / "android-crash.log").read_text())

    def test_oracle_requires_every_gradient_and_current_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cpu-float64-eager").mkdir()
            report = dict(reference_dtype="float64", attention="eager", policy=dict(atol=1e-6, rtol=1e-4), phases={}, passed=True)
            for phase in ("inference", "latency", "training"):
                capture = dict(phase=phase, precision="strict-f32", tuning=True, cooperative_dispatches=0,
                               gradients={"a": {}, "b": {}})
                write_json(root / f"{phase}.json", capture)
                reference = root / "cpu-float64-eager" / f"{phase}.npz"
                reference.write_bytes(b"reference")
                keys = ("loss", "gradient:a", "gradient:b") if phase == "training" else ("output:0",)
                report["phases"][phase] = dict(manifest_sha256=sha256(root / f"{phase}.json"),
                    reference_sha256=sha256(reference), compared_elements=10, passed=True,
                    checks={key: dict(passed=True, finite=True) for key in keys}, all_gradients=dict(passed=True))
            self.assertEqual(validate_oracle(root, report, "strict")["training"]["status"], "valid")
            missing = copy.deepcopy(report)
            del missing["phases"]["training"]["checks"]["gradient:b"]
            with self.assertRaises(ValueError):
                validate_oracle(root, missing, "strict")
            report["phases"]["training"]["checks"]["gradient:b"]["passed"] = False
            with self.assertRaises(ValueError):
                validate_oracle(root, report, "strict")
            report["phases"]["training"]["passed"] = report["passed"] = False
            self.assertEqual(validate_oracle(root, report, "strict")["training"]["status"], "mismatch")
            (root / "inference.json").write_text("{}")
            with self.assertRaises(ValueError):
                validate_oracle(root, report, "strict")

    def test_report_never_pairs_host_cpu_or_accepts_partial_replicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = dict(protocol=PROTOCOL, source="revision", status="complete", runs=[],
                            args=dict(replicates=3, collect=True, inference_only=False,
                                      models=["ResNet-50"], precisions=["strict"]))
            for replicate in (1, 2, 3):
                folder = root / str(replicate)
                folder.mkdir()
                write_json(folder / "ResNet-50_meganeura.json", dict(timing_samples_ms={p: [replicate * 10] * 20
                           for p in ("inference", "latency", "training")}))
                manifest["runs"].append(dict(model="ResNet-50", precision="strict", replicate=replicate,
                    path=str(replicate), phases={p: dict(status="valid") for p in ("inference", "latency", "training")}))
            write_json(root / "campaign.json", manifest)
            rows = campaign_rows(root)
            self.assertTrue(all(r["native_ms"] == 20 and r["speedup"] is None and r["reference_ms"] is None for r in rows))
            manifest["runs"][1]["phases"]["training"]["status"] = "mismatch"
            manifest["status"] = "complete-with-failures"
            write_json(root / "campaign.json", manifest)
            rows = campaign_rows(root)
            self.assertEqual(rows[-1]["coverage"], "2/3")
            self.assertIsNone(rows[-1]["native_ms"])
            self.assertEqual(rows[0]["native_ms"], 20)
            manifest["args"]["collect"] = False
            manifest["runs"] = manifest["runs"][:1]
            write_json(root / "campaign.json", manifest)
            self.assertTrue(all(r["native_ms"] is None for r in campaign_rows(root)))


if __name__ == "__main__":
    unittest.main()
