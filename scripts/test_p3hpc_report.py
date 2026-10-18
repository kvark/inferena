"""Keep measured timings, native qualification, and paired eligibility distinct."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from collection import write_json
from p3hpc_report import campaign_rows, cohort_rows


class PairedReport(unittest.TestCase):
    def campaign(self, root):
        manifest = dict(protocol="p3hpc-paired-campaign-v31", source="revision",
            status="complete-with-failures", runs=[],
            args=dict(replicates=3, collect=True, inference_only=False,
                      models=["Good", "Bad"], precisions=["accelerated"], gpu="GPU", backend="cuda"),
            reference_conditions=dict(selected=[dict(mode="default", graph_replay=True)]),
            replicated_gradient_validation=dict(status="fail", groups=[]))
        for model in ("Good", "Bad"):
            manifest["replicated_gradient_validation"]["groups"].append(dict(
                model=model, precision="accelerated", mode="default", graph_replay=True,
                status="pass" if model == "Good" else "fail"))
            for replicate in (1, 2, 3):
                folder = root / model / str(replicate)
                folder.mkdir(parents=True)
                for engine, value in (("meganeura", 1.), ("pytorch", 2.)):
                    write_json(folder / f"{model}_{engine}.json", dict(timing_samples_ms={
                        p: [value] * 20 for p in ("inference", "latency", "training")}))
                manifest["runs"].append(dict(model=model, precision="accelerated", replicate=replicate,
                    mode="default", graphs=True, path=str(folder.relative_to(root)),
                    phases={p: dict(status="valid") for p in ("inference", "latency", "training")}))
        write_json(root / "campaign.json", manifest)
        return manifest

    def test_failed_training_group_does_not_erase_other_models_or_forward_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.campaign(root)
            rows = {(r["model"], r["phase"]): r for r in campaign_rows(root)}
            self.assertEqual(rows["Good", "training"]["native_ms"], 1.)
            self.assertEqual(rows["Good", "training"]["speedup"], 2.)
            self.assertIsNone(rows["Bad", "training"]["native_ms"])
            self.assertIsNone(rows["Bad", "training"]["speedup"])
            self.assertEqual(rows["Bad", "inference"]["native_ms"], 1.)

    def test_windows_archive_paths_preserve_measurements_on_posix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.campaign(root)
            expected = campaign_rows(root)
            for run in manifest["runs"]:
                run["path"] = run["path"].replace("/", "\\")
            write_json(root / "campaign.json", manifest)
            self.assertEqual(campaign_rows(root), expected)

    def test_failed_reference_retains_native_measurement_without_claiming_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.campaign(root)
            for run in manifest["runs"]:
                run["phases"]["inference"] = dict(status="failed")
                write_json(root / run["path"] / f"{run['model']}_pytorch.json", dict(status="error"))
            write_json(root / "campaign.json", manifest)
            for row in campaign_rows(root):
                if row["phase"] == "inference":
                    self.assertEqual(row["native_measured_ms"], 1.)
                    self.assertIsNone(row["native_ms"])
                    self.assertIsNone(row["reference_ms"])
                    self.assertIsNone(row["speedup"])

    def test_preserved_oracle_qualifies_native_but_cannot_supply_reference_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.campaign(root)
            for run in manifest["runs"]:
                folder = root / run["path"]
                run["phases"] = {p: dict(status="failed") for p in ("inference", "latency", "training")}
                write_json(folder / f"{run['model']}_pytorch.json", dict(status="error"))
                write_json(folder / "torch-f32-oracle.json", dict(correctness_only=True))
                receipt = dict(policy="preserved-f32-oracle-v1", oracle="torch-f32-oracle.json",
                    timing_substituted=False,
                    native_sha256=hashlib.sha256((folder / f"{run['model']}_meganeura.json").read_bytes()).hexdigest(),
                    oracle_sha256=hashlib.sha256((folder / "torch-f32-oracle.json").read_bytes()).hexdigest(),
                    validation=dict(comparison_performed=True, forward_valid=True, training_valid=True,
                                    parameter_gradient_relative_l2_error=0.01, total_gradient_relative_error=0.01))
                write_json(folder / "native-oracle-comparison.json", receipt)
                run["native_oracle"] = dict(path="native-oracle-comparison.json", **receipt)
            write_json(root / "campaign.json", manifest)
            for row in campaign_rows(root):
                self.assertEqual(row["native_ms"], 1.)
                self.assertEqual(row["native_status"], "valid")
                self.assertIsNone(row["reference_ms"])
                self.assertIsNone(row["speedup"])
            # A stale receipt must not qualify a replaced measurement.
            path = root / manifest["runs"][0]["path"] / "Good_meganeura.json"
            record = json.loads(path.read_text())
            record["timing_samples_ms"]["inference"][0] = 5.
            write_json(path, record)
            with self.assertRaisesRegex(ValueError, "does not belong"):
                campaign_rows(root)

    def test_missing_or_duplicate_group_cannot_supply_training_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.campaign(root)
            good = manifest["replicated_gradient_validation"]["groups"][0]
            for groups in ([], [good, copy.deepcopy(good)]):
                manifest["replicated_gradient_validation"] = dict(status="pass", groups=groups)
                write_json(root / "campaign.json", manifest)
                self.assertTrue(all(r["native_ms"] is None for r in campaign_rows(root)
                                    if r["phase"] == "training"))

    def test_failed_phase_and_duplicate_replicates_cannot_supply_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.campaign(root)
            manifest["runs"][0]["phases"]["inference"]["status"] = "mismatch"
            manifest["runs"][1]["replicate"] = 1
            write_json(root / "campaign.json", manifest)
            self.assertTrue(all(r["native_ms"] is None for r in campaign_rows(root) if r["model"] == "Good"))

    def test_final_provenance_failure_withholds_previously_valid_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.campaign(root)
            manifest.update(status="incomplete", error="runner binary changed final_binary_check")
            write_json(root / "campaign.json", manifest)
            rows = campaign_rows(root)
            self.assertTrue(all(row["status"] == "incomplete-or-failed" for row in rows))
            self.assertTrue(all(row["native_ms"] is None and row["reference_ms"] is None for row in rows))

    def test_versions_are_readable_but_never_pooled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = []
            for version in range(18, 32):
                folder = root / str(version)
                manifest = self.campaign(folder)
                manifest["protocol"] = f"p3hpc-paired-campaign-v{version}"
                write_json(folder / "campaign.json", manifest)
                self.assertEqual(cohort_rows([folder])[0]["protocol"], manifest["protocol"])
                manifests.append(folder)
            with self.assertRaisesRegex(ValueError, "paired protocol"):
                cohort_rows(manifests)


if __name__ == "__main__":
    unittest.main()
