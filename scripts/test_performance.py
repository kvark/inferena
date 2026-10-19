"""Performance controls restore settings even when setup or the child fails."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from performance import run, settings


class PerformanceTest(unittest.TestCase):
    def intel(self, root):
        device = root / "0000:08:00.0"
        device.mkdir()
        (device / "vendor").write_text("0x8086")
        freq = device / "tile0/gt0/freq0"
        freq.mkdir(parents=True)
        (freq / "min_freq").write_text("1200")
        (freq / "max_freq").write_text("2750")
        return device, freq / "min_freq"

    def test_restore_on_success_and_child_failure(self):
        for code in (0, 7):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                device, minimum = self.intel(root)
                command = [sys.executable, "-c",
                           "import pathlib,sys; assert pathlib.Path(sys.argv[1]).read_text().strip() == '2750'; sys.exit(int(sys.argv[2]))",
                           str(minimum), str(code)]
                result = run(device, root / "output", command, apply=True, interval=.01)
                self.assertEqual(result, code)
                self.assertEqual(minimum.read_text().strip(), "1200")
                receipt = json.loads((root / "output/performance.json").read_text())
                self.assertEqual(receipt["restore_errors"], [])
                self.assertEqual(receipt["returncode"], code)

    def test_partial_setup_and_interruption_restore(self):
        for error in (OSError("launch failed"), KeyboardInterrupt()):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                device, minimum = self.intel(root)
                with patch("performance.subprocess.Popen", side_effect=error):
                    with self.assertRaises(type(error)):
                        run(device, root / "output", ["unused"], apply=True)
                self.assertEqual(minimum.read_text().strip(), "1200")
                receipt = json.loads((root / "output/performance.json").read_text())
                self.assertEqual(receipt["restore_errors"], [])

    def test_amd_manual_mode_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            device = Path(temporary)
            (device / "vendor").write_text("0x1002")
            (device / "power_dpm_force_performance_level").write_text("manual")
            with self.assertRaisesRegex(ValueError, "manual masks"):
                settings(device)

    def test_amd_restores_auto_after_child_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            device = Path(temporary)
            mode = device / "power_dpm_force_performance_level"
            (device / "vendor").write_text("0x1002")
            mode.write_text("auto")
            command = [sys.executable, "-c",
                       "import pathlib,sys; assert pathlib.Path(sys.argv[1]).read_text().strip() == 'high'; sys.exit(7)",
                       str(mode)]
            self.assertEqual(run(device, device / "output", command, apply=True, interval=.01), 7)
            self.assertEqual(mode.read_text().strip(), "auto")

    def test_nvidia_changes_only_recoverable_power_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            device = Path(temporary)
            (device / "vendor").write_text("0x10de")
            with patch("performance.query", return_value={"power.limit": "200", "power.max_limit": "250"}):
                changes = settings(device)
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0]["before"], "200.0")
            self.assertEqual(changes[0]["requested"], "250.0")
