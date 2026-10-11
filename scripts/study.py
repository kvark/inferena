"""Reproducible, watchdog-protected supplementary studies outside the cohort."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

from collection import run_recorded, write_json

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def begin(destination, protocol, args):
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("use a new results directory outside the checkout")
    destination.mkdir(parents=True, exist_ok=False)
    files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                    cwd=ROOT).decode().split("\0")
    hashes = {}
    with zipfile.ZipFile(destination / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(files) - {""}):
            path = ROOT / name
            if path.is_file():
                archive.write(path, name)
                hashes[name] = sha256(path)
    return {"protocol": protocol, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
            "source": {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
                       "archive": "source.zip", "sha256": hashes},
            "host": {"platform": platform.platform(), "kernel": platform.release(), "python": sys.version},
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "runs": []}


def environment(backend):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("MEGANEURA_", "INFERENA_"))}
    env.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    env.update(PYTHON=sys.executable, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", INFERENA_REQUIRE_LOCAL_WEIGHTS="1", INFERENA_TORCH_BACKEND=backend,
               NVIDIA_TF32_OVERRIDE="0", TORCHINDUCTOR_COMPILE_THREADS="1")
    return env


def run_json(command, folder, env, seconds):
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "stdout.txt").open("w") as output, (folder / "runner.log").open("w") as log:
        result = run_recorded(command, report_path=folder / "watchdog.json", seconds=seconds,
                              cwd=ROOT, env=env, stdout=output, stderr=log)
    if result.returncode:
        raise RuntimeError(f"runner exited {result.returncode}; inspect {folder / 'runner.log'}")
    record = json.loads((folder / "stdout.txt").read_text())
    write_json(folder / "result.json", record)
    return record
