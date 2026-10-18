"""Durable campaign records and a wall-clock watchdog for a complete runner tree."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


@contextmanager
def keep_awake():
    """Hold macOS sleep assertions only for the campaign's lifetime."""
    if sys.platform != "darwin":
        yield {"enabled": False, "reason": "not macOS"}
        return
    # Keep the display awake as well: display sleep can change GPU behavior.
    # -w releases the assertions even if the collector is killed without cleanup.
    command = ["/usr/bin/caffeinate", "-dis", "-w", str(os.getpid())]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        print("Keeping macOS awake for the campaign (caffeinate -dis).", file=sys.stderr)
        yield {"enabled": True, "command": command,
               "scope": "campaign; display and idle sleep, system sleep on AC power"}
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def write_json(path, value):
    """Readers see either the previous record or the complete new record."""
    path = Path(path)
    staged = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False)
    temporary = Path(staged.name)
    try:
        with staged as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def terminate_tree(process):
    if sys.platform == "win32":
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True,
                           stdout=sys.stderr, stderr=sys.stderr)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        # The PyTorch compilation supervisor owns a separate session. Give its
        # SIGTERM handler time to reap that session before killing this group,
        # even if the outer shell has already exited.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def run_recorded(command, *, report_path, seconds, **kwargs):
    """Stop the campaign on expiration; preserve a receipt without retrying."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("watchdog seconds must be finite and positive")
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "budget_seconds": seconds, "scope": "runner-process-tree", "command": command}
    write_json(report_path, report)
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32"
               else {"start_new_session": True})
    started = time.monotonic()
    process = None
    try:
        process = subprocess.Popen(command, **kwargs, **options)
        report["pid"] = process.pid
        write_json(report_path, report)
        code = process.wait(timeout=seconds)
        report.update(status="complete" if code == 0 else "failed", returncode=code)
        return subprocess.CompletedProcess(command, code)
    except subprocess.TimeoutExpired:
        report["status"] = "timeout"
        raise
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=str(error))
        raise
    finally:
        try:
            if process is not None and (report["status"] in ("timeout", "interrupted")
                                        or process.poll() is None):
                terminate_tree(process)
        finally:
            report["elapsed_seconds"] = time.monotonic() - started
            if process is not None:
                report["returncode"] = process.poll()
            write_json(report_path, report)
