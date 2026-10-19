"""Quest 3S collection: Android timings, full tensors, and a host CPU oracle."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import tomllib

from cpu_reference import LOSS_POLICY

from p3hpc import (ROOT, MODELS, PHASES, SMOLLM2_PINS, SYNTHETIC_PARAMETER_INIT,
                  archive_results, check_native_search, gpu_matches, input_hashes,
                  select_native_device)

PROTOCOL = "p3hpc-android-campaign-v15"
TARGET = "aarch64-linux-android"
WARMUP_SECONDS = 2


def measurement_environment(count):
    # The native runner defaults to a longer desktop warmup. Set the Android
    # policy explicitly so the launch and the receipt validator agree.
    return dict(INFERENA_WARMUP_RUNS="5", INFERENA_WARMUP_SECONDS=str(WARMUP_SECONDS),
                INFERENA_MEASUREMENT_RUNS=str(count))


def write_json(path, value):
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def battery_info(text):
    fields = {}
    for line in text.splitlines():
        key, separator, value = line.strip().partition(": ")
        if separator:
            fields[key] = value.strip()
    return dict(level=int(fields["level"]), temperature_c=int(fields["temperature"]) / 10,
                powered=any(fields.get(key) == "true" for key in
                            ("AC powered", "USB powered", "Wireless powered", "Dock powered")),
                weak_charger=fields.get("Weak Charger") == "true", raw=fields)


def check_battery(battery):
    if not battery["powered"] or battery["level"] < 30:
        raise RuntimeError("Quest needs external power and at least 30% battery; use wireless ADB on a strong charger")
    if not math.isfinite(battery["temperature_c"]) or battery["temperature_c"] >= 45:
        raise RuntimeError("Quest battery temperature is at least 45 C; preserve this run and let it cool")


class Device:
    def __init__(self, serial):
        self.prefix = ["adb", "-s", serial]

    def call(self, *args, timeout=60):
        return subprocess.check_output([*self.prefix, *map(str, args)],
                                       timeout=timeout, text=True, encoding="utf-8").strip()

    def shell(self, command, timeout=60):
        return self.call("shell", command, timeout=timeout)

    def health(self):
        return dict(battery=battery_info(self.shell("dumpsys battery")),
                    thermal=self.shell("dumpsys thermalservice"),
                    uptime=self.shell("cat /proc/uptime"))

    def verify_file(self, path, expected):
        actual = self.shell("sha256sum " + shlex.quote(path)).split()[0]
        if actual != expected:
            raise ValueError(f"device file differs from the pinned input: {path}")

    def run(self, command, log, pidfile, timeout):
        # ADB's host timeout alone does not stop the Android process. Keep the
        # exec'd shell PID so interruption also terminates the device workload.
        wrapped = f"echo $$ > {shlex.quote(pidfile)} && exec {command}"
        try:
            with log.open("wb") as stream:
                result = subprocess.run([*self.prefix, "shell", wrapped], stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=timeout)
            if result.returncode:
                self.capture_crash(log.parent)
            return result.returncode
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            pid = self.shell("cat " + shlex.quote(pidfile))
            if re.fullmatch(r"[1-9][0-9]*", pid):
                self.shell(f"kill -9 {pid}")
            self.capture_crash(log.parent)
            raise

    def capture_crash(self, folder):
        # Preserve the existing crash buffer without clearing unrelated device
        # diagnostics. A logging failure must not hide the native failure.
        try:
            crash = self.call("logcat", "-d", "-b", "crash", "-v", "threadtime")
        except Exception as error:
            crash = f"Could not retrieve Android crash buffer: {error}"
        (folder / "android-crash.log").write_text(crash + "\n", encoding="utf-8")


def android_environment(ndk):
    ndk = Path(ndk or os.environ.get("ANDROID_NDK_HOME", "")).resolve()
    properties = ndk / "source.properties"
    if not properties.is_file() or not re.search(r"Pkg.Revision\s*=\s*27\.", properties.read_text()):
        raise ValueError("use Android NDK r27: --android-ndk <sdk>/ndk/27.0.12077973")
    host = {"win32": "windows-x86_64", "linux": "linux-x86_64", "darwin": "darwin-x86_64"}[sys.platform]
    directory = ndk / "toolchains/llvm/prebuilt" / host / "bin"
    compiler = directory / ("aarch64-linux-android29-clang.cmd" if sys.platform == "win32"
                            else "aarch64-linux-android29-clang")
    archiver = directory / ("llvm-ar.exe" if sys.platform == "win32" else "llvm-ar")
    if not compiler.is_file() or not archiver.is_file():
        raise ValueError("NDK compiler/archiver missing for this host")
    env = dict(os.environ, CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=str(compiler),
               CC_aarch64_linux_android=str(compiler), AR_aarch64_linux_android=str(archiver))
    return env, properties.read_text()


def check_native(record, args, model, precision, count, dependency, native):
    if (record["framework"] != "meganeura" or record["framework_rev"] != dependency["rev"]
            or record["model"] != model or record["gpu_name"] != native["name"]):
        raise ValueError("native identity differs from the campaign")
    protocol = record["protocol"]
    if (protocol["name"] != "inferena-paper-v6" or protocol["diagnostic"]
            or protocol["synthetic_parameter_init"] != SYNTHETIC_PARAMETER_INIT
            or protocol["warmup_runs"] != 5 or protocol["warmup_seconds"] != WARMUP_SECONDS
            or protocol["measurement_runs"] != count or protocol["training_requested"] is not True):
        raise ValueError("native workload/measurement protocol differs")
    expected_class = "strict-f32" if precision == "strict" else "accelerated"
    if record["precision"]["comparison_class"] != expected_class:
        raise ValueError("arithmetic class differs")
    for phase in PHASES:
        warmup = protocol["warmup"][phase]
        samples = record["timing_samples_ms"][phase]
        if (warmup["runs"] < 5 or not math.isfinite(warmup["seconds"]) or warmup["seconds"] < WARMUP_SECONDS
                or len(samples) != count or any(not math.isfinite(x) or x <= 0 for x in samples)):
            raise ValueError(f"invalid {phase} warmup or samples")
    optimizer = record["optimizer"]
    sessions = optimizer["sessions"]
    # Whisper measures the same encoder session for inference and latency;
    # the other models construct a second input shape.
    expected_modes = ["Inference"] * (1 if model == "Whisper-tiny" else 2) + ["Training"]
    if (optimizer["measured_construction"] is not True or optimizer["mode"] != "egglog-outlined"
            or sorted(s["mode"] for s in sessions) != expected_modes):
        raise ValueError("missing calibrated sessions")
    inference_index = 0
    for session in sessions:
        if (session["cooperative_matrix_policy"] != ("NativeF32" if precision == "strict" else "Auto")
                or session["cooperative_dispatches"] != 0):
            raise ValueError("Quest f32/no-cooperative-matrix contract changed")
        shape = record["outputs"]["output_shape"] if model.startswith("SmolLM2-") else None
        check_native_search(session, args.tune_seconds, args.max_submission_chunks, args.kernel_memo, android=True,
                            language_model=model.startswith("SmolLM2-"),
                            output_shape=shape if inference_index == 0 or shape is None else [1, 1, shape[-1]])
        inference_index += session["mode"] == "Inference"


def validate_oracle(folder, report, precision):
    if (report["reference_dtype"] != "float64" or report["attention"] != "eager"
            or report.get("loss_dtype") != "float64" or report.get("loss_policy") != LOSS_POLICY
            or report["policy"]["atol"] != 1e-6 or report["policy"]["rtol"] != 1e-4
            or set(report["phases"]) != set(PHASES)):
        raise ValueError("CPU oracle contract changed")
    expected = "strict-f32" if precision == "strict" else "reduced-input-f32-accumulate"
    phases = {}
    for phase in PHASES:
        details = report["phases"][phase]
        capture = json.loads((folder / f"{phase}.json").read_text())
        if (details["manifest_sha256"] != sha256(folder / f"{phase}.json")
                or details["reference_sha256"] != sha256(folder / "cpu-float64-eager" / f"{phase}.npz")
                or capture["phase"] != phase or capture["precision"] != expected
                or capture["tuning"] is not True or capture["cooperative_dispatches"] != 0
                or details["compared_elements"] <= 0):
            raise ValueError("CPU report does not describe these full tensors")
        checks = details["checks"]
        expected_checks = ({"loss", *("gradient:" + key for key in capture["gradients"])}
                           if phase == "training" else {"output:0"})
        if set(checks) != expected_checks:
            raise ValueError("incomplete full tensor checks")
        passed = all(check["passed"] is True and check["finite"] is True for check in checks.values())
        if phase == "training":
            passed &= details["all_gradients"]["passed"] is True
        if details["passed"] is not passed:
            raise ValueError("oracle receipt contradicts tensor checks")
        phases[phase] = {"status": "valid" if passed else "mismatch",
                         "compared_elements": details["compared_elements"]}
    if report["passed"] is not all(p["status"] == "valid" for p in phases.values()):
        raise ValueError("oracle receipt contradicts phase checks")
    return phases


def collect(args, parser, packages, started_at, started):
    if not args.adb_serial:
        parser.error("--backend android requires --adb-serial (USB serial or IP:port)")
    if (args.inference_only or set(args.models) - set(MODELS) or args.max_autotune
            or args.graph_ablation or args.no_graphs or args.eager or args.sdpa):
        parser.error("Quest collection supports the five complete workloads and its fixed CPU f64/eager oracle")
    try:
        build_env, ndk_properties = android_environment(args.android_ndk)
        device = Device(args.adb_serial)
        identity = {key: device.shell("getprop " + key) for key in
                    ("ro.product.model", "ro.product.device", "ro.serialno", "ro.build.fingerprint",
                     "ro.build.version.sdk", "ro.product.cpu.abi")}
        if identity["ro.product.model"] != "Quest 3S" or identity["ro.product.cpu.abi"] != "arm64-v8a":
            raise ValueError("this collector requires an arm64 Quest 3S")
        initial_health = device.health()
        check_battery(initial_health["battery"])
        if initial_health["battery"]["weak_charger"]:
            print("Quest reports a weak charger; battery is monitored and collection stops below 30%.", flush=True)
        if not args.offline:
            from prepare_models import prepare_model
            for model in args.models:
                if model in SMOLLM2_PINS:
                    prepare_model(model)
        hashes = input_hashes(args.models)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dependency = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["dependencies"]["meganeura"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())
    destination = (args.results_dir or ROOT.parent / "inferena-results" /
                   f"{host}-quest3s-{stamp}-{revision[:8]}").resolve()
    if destination == ROOT or ROOT in destination.parents or destination.exists():
        parser.error("use a new results directory outside the checkout")
    destination.mkdir(parents=True)
    remote = f"/data/local/tmp/inferena-{stamp}-{revision[:8]}"
    executable = remote + "/inferena-meganeura"
    import torch
    manifest = dict(protocol=PROTOCOL, paired_protocol="p3hpc-paired-campaign-v33", source=revision,
                    started_at=started_at, platform="Quest 3S", backend="android-vulkan",
                    meganeura=dependency, python=sys.version, packages=packages,
                    torch=dict(version=torch.__version__, git_version=torch.version.git_version,
                               build_config=torch.__config__.show()),
                    args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    sha256=hashes, device=identity, initial_health=initial_health,
                    android_ndk=ndk_properties, target=TARGET, remote_directory=remote,
                    reference=dict(framework="pytorch", device="host CPU", dtype="float64", attention="eager",
                                   loss_dtype="float64", loss_policy=LOSS_POLICY,
                                   role="correctness-only", timing_competitor=False, rtol=1e-4, atol=1e-6),
                    arithmetic_note="No cooperative matrices on Quest; both classes execute f32 and use strict tensor bounds",
                    native_policy=dict(kernel_strategy="Race", negligible_plan_share_cutoff=0.001),
                    warmup=dict(minimum_runs=5, minimum_seconds=WARMUP_SECONDS),
                    measurement_runs=20 if args.collect else 1,
                    planned_runs=(args.replicates if args.collect else 1) * len(args.models) * len(args.precisions),
                    failure_policy=dict(retry_failed_primary=False, timing_substitution=False),
                    runs=[], status="in-progress")

    def save():
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_json(destination / "campaign.json", manifest)

    save()
    print(f"Quest campaign: {destination}", flush=True)
    original_stay_on = None
    try:
        subprocess.run(["git", "archive", "--format=tar", "--output=" + str(destination / "source.tar"), revision],
                       cwd=ROOT, check=True)
        build_start = time.monotonic()
        with (destination / "build.log").open("wb") as log:
            subprocess.run(["cargo", "build", "--locked", "--release", "--target", TARGET,
                            "-p", "inferena-meganeura"], cwd=ROOT, env=build_env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=args.android_timeout)
        binary = ROOT / "target" / TARGET / "release/inferena-meganeura"
        manifest["runner_build_seconds"] = time.monotonic() - build_start
        manifest["binary_sha256"] = sha256(binary)
        manifest["rustc"] = subprocess.check_output(["rustc", "-Vv"], text=True)
        shutil.copyfile(binary, destination / "inferena-meganeura")
        device.shell("mkdir -p " + shlex.quote(remote))
        device.call("push", binary, executable)
        device.verify_file(executable, manifest["binary_sha256"])
        device.shell("chmod 700 " + shlex.quote(executable))
        native = select_native_device(json.loads(device.shell(shlex.quote(executable) + " --list-devices")),
                                      args.gpu or "Adreno (TM) 740")
        manifest["native_device"] = native
        for model in args.models:
            if model in SMOLLM2_PINS:
                directory = remote + "/models/" + model
                device.shell("mkdir -p " + shlex.quote(directory))
                for filename, digest in SMOLLM2_PINS[model]["sha256"].items():
                    device.call("push", ROOT / "models" / model / filename, directory + "/" + filename, timeout=300)
                    device.verify_file(directory + "/" + filename, digest)
                shutil.copyfile(ROOT / "models" / model / "config.json", destination / f"{model}-config.json")
        original_stay_on = device.shell("settings get global stay_on_while_plugged_in")
        manifest["original_stay_on_while_plugged_in"] = original_stay_on
        save()
        device.shell("svc power stayon true")
        device.shell("am broadcast -a com.oculus.vrpowermanager.prox_close")
        count = 20 if args.collect else 1
        for replicate in range(1, (args.replicates if args.collect else 1) + 1):
            for precision in args.precisions:
                for model in args.models:
                    folder = destination / ("measurement" if args.collect else "qualification") / f"r{replicate}" / precision / model
                    folder.mkdir(parents=True)
                    remote_capture = remote + f"/r{replicate}/{precision}/{model}"
                    run = dict(path=folder.relative_to(destination).as_posix(), model=model,
                               precision=precision, replicate=replicate, status="running", phases={})
                    manifest["runs"].append(run)
                    save()
                    print(run["path"], flush=True)
                    run["health_before"] = device.health()
                    check_battery(run["health_before"]["battery"])
                    env = dict(MEGANEURA_DEVICE_ID=str(native["device_id"]), MEGANEURA_TUNE="1",
                               INFERENA_STRICT=str(int(precision == "strict")),
                               **measurement_environment(count),
                               INFERENA_TUNE_SECONDS=str(args.tune_seconds),
                               INFERENA_MAX_SUBMISSION_CHUNKS=str(args.max_submission_chunks),
                               INFERENA_TENSOR_EXPORT=remote_capture, FRAMEWORK_REV=dependency["rev"],
                               TMPDIR=remote)
                    if args.kernel_memo:
                        env["INFERENA_KERNEL_MEMO_DIR"] = remote + f"/kernel-memos/{precision}/{model}"
                    command = "env " + " ".join(shlex.quote(k + "=" + v) for k, v in env.items())
                    command += " " + shlex.quote(executable) + " " + shlex.quote(model)
                    run["command"] = command
                    run_start = time.monotonic()
                    returncode = device.run(command, folder / "runner.log", remote + "/pid", args.android_timeout)
                    run["device_seconds"] = time.monotonic() - run_start
                    run["returncode"] = returncode
                    run["health_after"] = device.health()
                    # Retain partial tensors even when later construction or
                    # receipt validation fails. They are not qualified timings.
                    if device.shell("if test -d " + shlex.quote(remote_capture) + "; then echo present; fi") == "present":
                        device.call("pull", remote_capture + "/.", folder, timeout=300)
                    check_battery(run["health_after"]["battery"])
                    if returncode:
                        # Keep a failed process visible and never re-run the condition.
                        run.update(status="failed", error="Android runner failed; inspect runner.log",
                                   phases={p: {"status": "failed"} for p in PHASES})
                        save()
                        continue
                    records = [json.loads(line) for line in (folder / "runner.log").read_text(encoding="utf-8").splitlines()
                               if line.startswith('{"')]
                    if len(records) != 1:
                        raise ValueError("missing/ambiguous native result")
                    record = records[0]
                    check_native(record, args, model, precision, count, dependency, native)
                    record.update(benchmark_rev=revision, platform="Quest 3S", status="ok")
                    write_json(folder / f"{model}_meganeura.json", record)
                    cpu_command = [sys.executable, str(ROOT / "scripts/compare_android.py"), str(folder),
                                   "--inferena", str(ROOT), "--dtype", "float64"]
                    if model in SMOLLM2_PINS:
                        cpu_command += ["--config", str(destination / f"{model}-config.json")]
                    run["cpu_command"] = cpu_command
                    save()
                    with (folder / "cpu.log").open("wb") as log:
                        cpu = subprocess.run(cpu_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=args.android_timeout,
                                             env=dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1",
                                                      TRANSFORMERS_OFFLINE="1"))
                    report_path = folder / "comparison-float64-eager.json"
                    if not report_path.exists():
                        raise RuntimeError(f"CPU oracle failed without a receipt: {folder / 'cpu.log'}")
                    report = json.loads(report_path.read_text())
                    run["phases"] = validate_oracle(folder, report, precision)
                    valid = all(p["status"] == "valid" for p in run["phases"].values())
                    if cpu.returncode != (0 if valid else 1):
                        raise RuntimeError("CPU process status contradicts the oracle receipt")
                    record["validation"] = dict(reference_framework="pytorch", reference_device="host CPU",
                                                policy="full-tensor-cpu-f64-v2", comparison_performed=True,
                                                forward_valid=all(run["phases"][p]["status"] == "valid" for p in PHASES[:2]),
                                                training_valid=run["phases"]["training"]["status"] == "valid",
                                                report="comparison-float64-eager.json")
                    write_json(folder / f"{model}_meganeura.json", record)
                    run["status"] = "valid" if valid else "mismatch"
                    print(f"  {run['status']}", flush=True)
                    save()
        if (input_hashes(args.models) != hashes
                or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip()
                or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != revision):
            raise ValueError("source or model inputs changed during collection")
        manifest["status"] = "complete" if all(r["status"] == "valid" for r in manifest["runs"]) else "complete-with-failures"
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status="incomplete", error=str(error))
        if manifest["runs"] and manifest["runs"][-1]["status"] == "running":
            manifest["runs"][-1]["status"] = "failed"
        print(f"Quest collection stopped: {error}", file=sys.stderr, flush=True)
    finally:
        if original_stay_on is not None:
            try:
                device.shell("am broadcast -a com.oculus.vrpowermanager.automation_disable")
                restore = ("settings delete global stay_on_while_plugged_in" if original_stay_on == "null"
                           else "settings put global stay_on_while_plugged_in " + shlex.quote(original_stay_on))
                device.shell(restore)
                manifest["device_settings_restored"] = True
            except Exception as error:
                manifest["device_settings_restored"] = False
                manifest["restore_error"] = str(error)
                manifest["status"] = "incomplete"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["phase_coverage"] = {p: dict(planned=manifest["planned_runs"],
            valid=sum(r.get("phases", {}).get(p, {}).get("status") == "valid" for r in manifest["runs"])) for p in PHASES}
        save()
        from p3hpc_report import campaign_rows, render
        (destination / "results.md").write_text(render(campaign_rows(destination)), encoding="utf-8")
        try:
            print("Archiving results and validation receipts...", flush=True)
            print(f"Archive: {archive_results(destination)}", flush=True)
        except OSError as error:
            print(f"Archive not updated: {error}; results remain in {destination}", file=sys.stderr)
    print(f"{manifest['status']}: {destination / 'campaign.json'}", flush=True)
    return 0 if manifest["status"] == "complete" else 1
