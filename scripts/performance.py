#!/usr/bin/env python3
"""Run a command with GPU telemetry and optional, reversible performance requests.

Example: python scripts/performance.py --pci 0000:08:00.0 --output /tmp/power \
    --apply -- python scripts/p3hpc.py ...

Settings require existing OS permissions; this tool never invokes sudo. NVIDIA
clock locks are deliberately untouched: nvidia-smi cannot recover an existing
lock reliably. A request cannot override thermal or firmware power limits.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import signal
import subprocess
import time

from collection import terminate_tree, write_json


def smi(pci, *arguments):
    return subprocess.check_output(["nvidia-smi", "-i", pci, *arguments],
                                   text=True, timeout=5).strip()


def query(pci, fields):
    values = next(csv.reader([smi(pci, "--query-gpu=" + ",".join(fields),
                                  "--format=csv,noheader,nounits")]))
    return dict(zip(fields, (v.strip() for v in values), strict=True))


def read(path):
    return Path(path).read_text().strip()


def settings(device):
    """Only settings whose previous state can be read and restored exactly."""
    vendor = read(device / "vendor")
    changes = []
    if vendor == "0x1002":
        path = device / "power_dpm_force_performance_level"
        before = read(path)
        if before not in ("auto", "low", "high", "profile_standard", "profile_peak",
                          "profile_min_sclk", "profile_min_mclk"):
            raise ValueError(f"cannot restore AMD mode {before!r}; manual masks are not recoverable")
        changes.append(dict(kind="sysfs", path=str(path), before=before, requested="high"))
    elif vendor == "0x8086":
        for path in sorted(device.glob("tile*/gt*/freq*/min_freq")):
            changes.append(dict(kind="sysfs", path=str(path), before=read(path),
                                requested=read(path.with_name("max_freq"))))
        if not changes:
            for path in sorted(device.glob("drm/card*/gt_min_freq_mhz")):
                changes.append(dict(kind="sysfs", path=str(path), before=read(path),
                                    requested=read(path.with_name("gt_max_freq_mhz"))))
        if not changes:
            raise ValueError("no supported Intel frequency controls found")
    elif vendor == "0x10de":
        values = query(device.name, ["power.limit", "power.max_limit"])
        before, maximum = float(values["power.limit"]), float(values["power.max_limit"])
        if not 0 < before <= maximum:
            raise ValueError("invalid NVIDIA power limits")
        changes.append(dict(kind="nvidia-power", pci=device.name,
                            before=str(before), requested=str(maximum)))
    else:
        raise ValueError(f"unsupported GPU vendor {vendor}")
    return changes


def set_value(change, value):
    if change["kind"] == "sysfs":
        Path(change["path"]).write_text(value + "\n")
        if read(change["path"]) != value:
            raise RuntimeError(f"setting readback differs: {change['path']}")
    else:
        smi(change["pci"], "--power-limit=" + value)
        actual = float(query(change["pci"], ["power.limit"])["power.limit"])
        if abs(actual - float(value)) > 0.01:
            raise RuntimeError("NVIDIA power limit readback differs")


def restore(changes):
    errors = []
    for change in reversed(changes):
        try:
            set_value(change, change["before"])
        except Exception as error:
            errors.append(str(error))
    return errors


def sample(device):
    result = dict(time_unix=time.time(), monotonic=time.monotonic(), sysfs={})
    patterns = ("hwmon/hwmon*/temp*_input", "hwmon/hwmon*/power*_average",
                "hwmon/hwmon*/power*_input", "hwmon/hwmon*/energy*_input",
                "tile*/gt*/freq*/*freq", "pp_dpm_sclk", "pp_dpm_mclk",
                "hwmon/hwmon*/freq*_input", "gpu_busy_percent", "drm/card*/gt_*freq_mhz")
    for pattern in patterns:
        for path in device.glob(pattern):
            try:
                result["sysfs"][str(path.relative_to(device))] = read(path)
            except OSError as error:
                result["sysfs"][str(path.relative_to(device))] = {"error": str(error)}
    if read(device / "vendor") == "0x10de":
        try:
            result["nvidia"] = query(device.name, ["temperature.gpu", "power.draw", "power.limit",
                "clocks.gr", "clocks.mem", "pstate", "utilization.gpu",
                "clocks_event_reasons.active"])
        except Exception as error:
            result["nvidia_error"] = str(error)
    return result


def run(device, output, command, apply=False, interval=1.0):
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(policy="scoped-performance-v1", pci=device.name, device=str(device),
                   apply=apply, command=command, started_unix=time.time(), status="preparing",
                   settings=[], limitation="requests only; thermal/firmware limits remain; NVIDIA clocks unchanged")
    attempted, process = [], None
    previous = {}

    def interrupted(number, _frame):
        raise KeyboardInterrupt(f"signal {number}")

    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, interrupted)
        if apply:
            receipt["settings"] = settings(device)
            # Persist recovery instructions before the first write. This file
            # also supports recovery after SIGKILL or a machine crash.
            write_json(output / "performance.json", receipt)
            for change in receipt["settings"]:
                if change["before"] == change["requested"]:
                    continue
                attempted.append(change)
                set_value(change, change["requested"])
        receipt["status"] = "running"
        write_json(output / "performance.json", receipt)
        process = subprocess.Popen(command, start_new_session=True)
        with (output / "telemetry.jsonl").open("w") as stream:
            while True:
                stream.write(json.dumps(sample(device), allow_nan=False) + "\n")
                stream.flush()
                try:
                    code = process.wait(timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    pass
        receipt.update(status="complete" if code == 0 else "failed", returncode=code)
        return code
    except BaseException as error:
        receipt.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=str(error))
        raise
    finally:
        # Prevent a second termination signal from interrupting restoration.
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        try:
            if process is not None:
                terminate_tree(process)
        finally:
            receipt["restore_errors"] = restore(attempted)
            receipt["finished_unix"] = time.time()
            write_json(output / "performance.json", receipt)
            for number, handler in previous.items():
                signal.signal(number, handler)
            if receipt["restore_errors"]:
                raise RuntimeError("performance restoration failed: " + "; ".join(receipt["restore_errors"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pci", help="exact Linux PCI address, e.g. 0000:08:00.0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--restore", type=Path, help="restore a trusted performance.json after interrupted cleanup")
    parser.add_argument("--apply", action="store_true", help="request supported performance settings and restore afterwards")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.restore:
        if args.pci or args.output or args.apply or args.command:
            parser.error("--restore is used alone")
        receipt = json.loads(args.restore.read_text())
        if receipt.get("policy") != "scoped-performance-v1":
            parser.error("unknown performance receipt")
        errors = restore(receipt["settings"])
        receipt.update(restore_errors=errors, recovered_unix=time.time())
        write_json(args.restore, receipt)
        if errors:
            raise RuntimeError("; ".join(errors))
        return
    if not args.pci or not args.output:
        parser.error("--pci and --output are required")
    if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", args.pci):
        parser.error("expected full PCI address")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    device = Path("/sys/bus/pci/devices") / args.pci.lower()
    if not device.is_dir():
        parser.error("PCI device does not exist")
    raise SystemExit(run(device, args.output, command, args.apply))


if __name__ == "__main__":
    main()
