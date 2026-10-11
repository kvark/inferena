"""CPU-only failure injection for campaign persistence and process cleanup."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from collection import run_recorded, write_json


class CollectionTest(unittest.TestCase):
    def test_failed_write_preserves_last_complete_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "campaign.json"
            write_json(path, {"status": "running"})
            for bad in (object(), {"value": float("nan")}):
                with self.assertRaises((TypeError, ValueError)):
                    write_json(path, bad)
                self.assertEqual(json.loads(path.read_text()), {"status": "running"})
            with patch("collection.os.replace", side_effect=OSError("rename failed")):
                with self.assertRaisesRegex(OSError, "rename failed"):
                    write_json(path, {"status": "complete"})
            self.assertEqual(json.loads(path.read_text()), {"status": "running"})
            self.assertEqual(list(path.parent.iterdir()), [path])
            write_json(path, {"status": "complete"})
            self.assertEqual(json.loads(path.read_text()), {"status": "complete"})

    def test_runner_preserves_success_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "watchdog.json"
            for code in (0, 7):
                result = run_recorded([sys.executable, "-c", f"raise SystemExit({code})"],
                                      report_path=path, seconds=5)
                self.assertEqual(result.returncode, code)
                report = json.loads(path.read_text())
                self.assertEqual(report["returncode"], code)
                self.assertEqual(report["status"], "complete" if code == 0 else "failed")
            with self.assertRaises(subprocess.TimeoutExpired):
                run_recorded([sys.executable, "-c", "import time; time.sleep(20)"],
                             report_path=path, seconds=0.2)
            report = json.loads(path.read_text())
            self.assertEqual(report["status"], "timeout")
            self.assertEqual(report["budget_seconds"], 0.2)
            self.assertGreaterEqual(report["elapsed_seconds"], 0.2)
            with self.assertRaises(OSError):
                run_recorded([str(Path(temporary) / "missing-executable")], report_path=path, seconds=1)
            self.assertEqual(json.loads(path.read_text())["status"], "failed")

    def test_watchdog_reaps_nested_compilation_session(self):
        budget = Path(__file__).resolve().parents[1] / "frameworks/pytorch/budget.py"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            marker = path / "orphan.txt"
            ready = path / "ready.txt"
            # This worker and its descendant run in budget.py's separate session.
            grandchild = f"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).touch()"
            worker = ("import subprocess, sys, time; from pathlib import Path; "
                      f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
                      f"Path({str(ready)!r}).touch(); time.sleep(20)")
            command = [sys.executable, str(budget), "-c", worker]
            with self.assertRaises(subprocess.TimeoutExpired):
                run_recorded(command, report_path=path / "watchdog.json", seconds=1,
                             env=dict(os.environ, INFERENA_PREPARATION_REPORT=str(path / "compile.json")))
            self.assertTrue(ready.exists(), "worker must start before testing its cleanup")
            time.sleep(2)
            self.assertFalse(marker.exists(), "compiler descendant survived the campaign timeout")

    @unittest.skipIf(sys.platform == "win32", "POSIX signal escalation")
    def test_watchdog_kills_worker_that_ignores_term(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            ready = path / "ready.txt"
            command = [sys.executable, "-c",
                       "import signal, time; from pathlib import Path; "
                       "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                       f"Path({str(ready)!r}).touch(); time.sleep(30)"]
            with self.assertRaises(subprocess.TimeoutExpired):
                run_recorded(command, report_path=path / "watchdog.json", seconds=1)
            self.assertTrue(ready.exists())
            report = json.loads((path / "watchdog.json").read_text())
            self.assertEqual(report["status"], "timeout")
            self.assertEqual(report["returncode"], -9)


if __name__ == "__main__":
    unittest.main()
