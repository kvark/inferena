#!/usr/bin/env python3
"""Run a profiler and all its children in a bounded Linux user cgroup."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid


def execute(limit, floor_mib, log_path, command):
    entry = next(line for line in Path("/proc/self/cgroup").read_text().splitlines()
                 if line.startswith("0::"))
    group = Path("/sys/fs/cgroup") / entry[3:].lstrip("/")
    for name, expected in (("memory.max", str(limit)), ("memory.swap.max", "0"),
                           ("memory.oom.group", "1")):
        if not (group / name).exists() or (group / name).read_text().strip() != expected:
            raise SystemExit(f"{name}: requested cgroup limit is not active; refusing launch")
    if not floor_mib:
        os.execvp(command[0], command)
    with Path(log_path).open("x") as log:
        log.write("elapsed_s,mem_available_mib,cgroup_memory_mib\n")
        start = time.monotonic()
        process = subprocess.Popen(command, start_new_session=True)
        try:
            while process.poll() is None:
                available = next(int(line.split()[1]) // 1024
                                 for line in Path("/proc/meminfo").read_text().splitlines()
                                 if line.startswith("MemAvailable:"))
                resident = int((group / "memory.current").read_text()) // 1024**2
                log.write(f"{time.monotonic() - start:.3f},{available},{resident}\n")
                log.flush()
                if available < floor_mib:
                    raise RuntimeError(f"host headroom {available} MiB below {floor_mib} MiB; stopping experiment")
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        raise SystemExit(process.returncode)


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-mib", type=positive, required=True)
    parser.add_argument("--seconds", type=positive, required=True)
    parser.add_argument("--minimum-available-mib", type=positive,
                        help="stop the experiment if global available RAM falls below this floor")
    parser.add_argument("--memory-log", type=Path, help="new CSV file, required with the global RAM floor")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- executable [arguments]")
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide a command after --")
    if bool(args.minimum_available_mib) != bool(args.memory_log):
        parser.error("provide both --minimum-available-mib and --memory-log")
    if sys.platform != "linux" or not shutil.which("systemd-run"):
        parser.error("requires Linux and a systemd user manager; no unbounded fallback")
    if not Path("/sys/fs/cgroup/cgroup.controllers").exists():
        parser.error("requires cgroup v2 memory controls")

    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemTotal:", "MemAvailable:"))}
    limit = args.memory_mib * 1024 * 1024
    reserve = max(1024**3, memory["MemTotal"] // 10)
    if memory["MemAvailable"] < limit + reserve:
        parser.error(f"insufficient host RAM: {memory['MemAvailable'] // 1024**2} MiB available; "
                     f"need {args.memory_mib} MiB plus {reserve // 1024**2} MiB reserve")

    # Scope mode inherits cwd/environment without copying secrets into arguments.
    # The limit covers children too, but not all GPU/driver-pinned allocations.
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    unit = f"inferena-profile-{uuid.uuid4().hex[:12]}"
    print(f"{unit}: {args.memory_mib} MiB, no swap, {args.seconds}s; "
          "stop with systemctl --user stop " + unit + ".scope", file=sys.stderr, flush=True)
    os.execvp("systemd-run", [
        "systemd-run", "--user", "--scope", "--collect", "--no-ask-password",
        "--expand-environment=no", f"--unit={unit}",
        f"--property=MemoryMax={limit}", "--property=MemorySwapMax=0",
        "--property=OOMPolicy=kill", f"--property=RuntimeMaxSec={args.seconds}",
        "--property=TimeoutStopSec=5s", "--", sys.executable,
        str(Path(__file__).resolve()), "--exec", str(limit),
        str(args.minimum_available_mib or 0), str(args.memory_log.resolve()) if args.memory_log else "-", *command,
    ])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--exec"]:
        execute(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5:])
    else:
        main()
