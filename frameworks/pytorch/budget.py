"""Bound compilation and its worker tree; never substitute an eager result."""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def save(path, report):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def terminate(process):
    if sys.platform == "win32":
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True,
                           stdout=sys.stderr, stderr=sys.stderr)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def interrupted(signum, _frame):
    # The campaign watchdog first sends SIGTERM. Exiting through finally reaps
    # our separate compiler session before the campaign kills its outer group.
    raise SystemExit(128 + signum)


@contextmanager
def compilation_budget():
    path = os.environ.get("INFERENA_PREPARATION_REPORT")
    report = {"phase": "compile", "status": "running", "started": time.monotonic(),
              "budget_seconds": float(os.environ.get("INFERENA_COMPILE_SECONDS", "120"))}
    if path:
        save(Path(path), report)
    try:
        yield
    except BaseException:
        report["status"] = "failed"
        raise
    else:
        report["status"] = "complete"
    finally:
        report["elapsed_seconds"] = time.monotonic() - report["started"]
        if path:
            save(Path(path), report)


def main():
    signal.signal(signal.SIGTERM, interrupted)
    seconds = float(os.environ.get("INFERENA_COMPILE_SECONDS", "120"))
    if not math.isfinite(seconds) or seconds <= 0:
        raise SystemExit("INFERENA_COMPILE_SECONDS must be finite and positive")
    with tempfile.TemporaryDirectory(prefix="inferena-budget-") as temporary:
        path = Path(os.environ.get("INFERENA_PREPARATION_REPORT", str(Path(temporary) / "compile.json")))
        env = dict(os.environ, INFERENA_PREPARATION_REPORT=str(path.resolve()),
                   INFERENA_BUDGET_ENFORCED="1")
        options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32"
                   else {"start_new_session": True})
        started = time.monotonic()
        process = subprocess.Popen([sys.executable, *sys.argv[1:]], env=env, **options)
        try:
            while process.poll() is None:
                report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                now = time.monotonic()
                compiling = report.get("status") == "running"
                elapsed = now - report.get("started", now) if compiling else now - started
                # Loading/qualification have a separate generous safety cap;
                # their CPU readback work is not called compilation or tuning.
                limit = seconds if compiling else max(600.0, 2 * seconds)
                if elapsed >= limit:
                    terminate(process)
                    report.update(status="timeout", phase="compile" if compiling else "process",
                                  budget_seconds=limit, elapsed_seconds=elapsed)
                    save(path, report)
                    print(f"[pytorch] {report['phase']} budget exhausted after {elapsed:.1f}s "
                          f"(limit {limit:g}s); terminating worker tree; no fallback", file=sys.stderr, flush=True)
                    return 124
                time.sleep(0.1)
            return process.returncode
        finally:
            terminate(process)


if __name__ == "__main__":
    sys.exit(main())
