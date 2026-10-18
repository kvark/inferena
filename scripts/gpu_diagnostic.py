"""One bounded GPU diagnostic with durable logs; never a publication measurement."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from collection import terminate_tree, write_json


def capture_lines(process, path, faults=None):
    with path.open("w") as output:
        next_sync = 0.0
        # Drain to EOF: a finished process can still have buffered diagnostics.
        for line in iter(process.stdout.readline, b""):
            text = line.decode("utf-8", errors="replace")
            output.write(json.dumps({"unix_s": time.time(), "line": text.rstrip()}) + "\n")
            if (faults is not None or "meganeura::submission" in text or " begin:" in text
                    or time.monotonic() >= next_sync):
                output.flush()
                os.fsync(output.fileno())
                next_sync = time.monotonic() + 0.2
            if faults is not None and re.search(r"job timed out|wedged|reset failed|reset.*failed", text, re.I):
                faults.append(text.strip())
        output.flush()
        os.fsync(output.fileno())


def observe(device, folder, stop):
    paths = {"actual_mhz": device / "tile0/gt0/freq0/act_freq",
             "requested_mhz": device / "tile0/gt0/freq0/cur_freq",
             "throttle": device / "tile0/gt0/freq0/throttle/status",
             "throttle_reasons": device / "tile0/gt0/freq0/throttle/reasons"}
    for hwmon in (device / "hwmon").glob("hwmon*"):
        paths.update({p.name: p for pattern in ("temp*_input", "energy*_input", "power*_cap")
                      for p in hwmon.glob(pattern)})
    captured = set()
    with (folder / "telemetry.jsonl").open("w") as output:
        while not stop.is_set():
            row = {"unix_s": time.time()}
            for key, path in paths.items():
                try:
                    row[key] = path.read_text().strip()
                except OSError as error:
                    row[key] = str(error)
            output.write(json.dumps(row) + "\n")
            output.flush()
            os.fsync(output.fileno())
            for dump in Path("/sys/class/devcoredump").glob("devcd*/data"):
                if str(dump) in captured:
                    continue
                captured.add(str(dump))
                # Do not clear the kernel's copy. Bound the copy's size and wait.
                try:
                    with (folder / f"{dump.parent.name}.dump").open("wb") as target:
                        result = subprocess.run(["head", "-c", str(16 << 20), str(dump)], stdout=target,
                                                stderr=subprocess.PIPE, timeout=5)
                        target.flush()
                        os.fsync(target.fileno())
                    write_json(folder / f"{dump.parent.name}.json", {
                        "source": str(dump), "returncode": result.returncode,
                        "stderr": result.stderr.decode(errors="replace"), "max_bytes": 16 << 20})
                except (OSError, subprocess.TimeoutExpired) as error:
                    write_json(folder / f"{dump.parent.name}.json", {"source": str(dump), "error": str(error)})
            stop.wait(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drm-card", required=True)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--memory-mib", type=int, default=6144)
    parser.add_argument("--trace", action="store_true", help="dump dispatch plans and every submission; changes timing")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.seconds <= 0 or args.memory_mib <= 0 or not re.fullmatch(r"card\d+", args.drm_card):
        parser.error("provide a command, positive limits, and a DRM card name")
    device = Path("/sys/class/drm") / args.drm_card / "device"
    identity = {name: (device / name).read_text().strip() for name in ("device", "vendor")}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_unix": time.time(), "command": command,
              "device": identity, "kernel": os.uname().release, "seconds": args.seconds,
              "binary_sha256": hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
              "policy": {key: os.environ.get(key) for key in (
                  "INFERENA_STRICT", "INFERENA_TUNE_SECONDS", "INFERENA_MAX_SUBMISSION_CHUNKS",
                  "INFERENA_WARMUP_SECONDS", "INFERENA_MEASUREMENT_RUNS", "FRAMEWORK_REV")},
              "diagnostic": True, "submission_tracing": args.trace,
              "durable_logging_affects_timing": True}
    write_json(args.output / "diagnostic.json", report)
    stop, faults, threads = threading.Event(), [], []
    process = journal = None
    try:
        journal = subprocess.Popen(["journalctl", "-k", "-f", "-n", "0", "--no-pager"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        threads.append(threading.Thread(target=capture_lines, args=(journal, args.output / "kernel.jsonl", faults), daemon=True))
        threads.append(threading.Thread(target=observe, args=(device, args.output, stop), daemon=True))
        for thread in threads:
            thread.start()
        env = dict(os.environ, MEGANEURA_DEVICE_ID=identity["device"],
                   RUST_LOG="warn,meganeura::train::search=info")
        if args.trace:
            env.update(MEGANEURA_DUMP_PLAN="1",
                       RUST_LOG="warn,meganeura::train::search=debug,meganeura::submission=debug")
        limited = [sys.executable, str(Path(__file__).with_name("limited.py")),
                   "--memory-mib", str(args.memory_mib), "--seconds", str(args.seconds), "--", *command]
        with (args.output / "record.json.tmp").open("w") as output:
            process = subprocess.Popen(limited, env=env, stdout=output, stderr=subprocess.PIPE, start_new_session=True)
            # Capture stderr through the same durable line writer as the kernel.
            class Stderr:
                stdout = process.stderr
            thread = threading.Thread(target=capture_lines, args=(Stderr(), args.output / "runner.jsonl"), daemon=True)
            threads.append(thread)
            thread.start()
            deadline = time.monotonic() + args.seconds
            while process.poll() is None and not faults and time.monotonic() < deadline:
                time.sleep(0.1)
            report["status"] = "gpu-fault" if faults else "timeout" if process.poll() is None else "complete" if process.returncode == 0 else "failed"
            if process.poll() is None:
                terminate_tree(process)
            report["returncode"] = process.returncode
            output.flush()
            os.fsync(output.fileno())
        if report["status"] == "complete":
            (args.output / "record.json.tmp").replace(args.output / "record.json")
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        for owned in (process, journal):
            if owned is not None and owned.poll() is None:
                try:
                    terminate_tree(owned)
                except (OSError, subprocess.TimeoutExpired) as error:
                    report.setdefault("cleanup_errors", []).append(str(error))
        stop.set()
        for thread in threads:
            thread.join(timeout=6)
        if faults:
            report["status"] = "gpu-fault"
        elif report["status"] == "running":
            report["status"] = "interrupted"
        report.update(faults=faults, finished_unix=time.time())
        write_json(args.output / "diagnostic.json", report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
