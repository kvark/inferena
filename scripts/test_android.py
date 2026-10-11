"""Android collection and coverage contracts; no device needed."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from p3hpc import create_parser
from p3hpc_android import (PROTOCOL, Device, battery_info, check_battery, check_native,
                           validate_oracle, write_json, sha256)
from p3hpc_report import campaign_rows


class AndroidContract(unittest.TestCase):
    def test_cli_retains_common_measurement_policy(self):
        args = create_parser().parse_args(["--backend", "android", "--adb-serial", "quest"])
        self.assertEqual(args.replicates, 3)
        self.assertEqual(args.precisions, ["strict", "accelerated"])
        self.assertEqual(args.tune_seconds, 30)
        self.assertTrue(args.kernel_memo)

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
        with tempfile.TemporaryDirectory() as tmp, patch("p3hpc_android.subprocess.run", side_effect=subprocess.TimeoutExpired("adb", 1)), patch.object(device, "shell", side_effect=["1234", ""]) as shell:
            with self.assertRaises(subprocess.TimeoutExpired):
                device.run("runner", Path(tmp) / "log", "/tmp/pid", 1)
            self.assertEqual(shell.call_args.args, ("kill -9 1234",))

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
