#!/usr/bin/env python3
"""256 matched SmolLM2 SGD updates on real text, with fixed loss-curve gates."""

import argparse
import json
import math
import platform
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys

from collection import write_json
from p3hpc import (check_torch_identity, cpu_frequency_policies, gpu_matches, input_hashes,
                  positive_seconds, select_native_device, SMOLLM2_REVISIONS, PYTHON_VERSION)
from study import ROOT, begin, environment, run_json, sha256

sys.path.insert(0, str(ROOT / "frameworks/pytorch"))
from amp import compare_loss_curves

CORPUS_URL = "https://www.gutenberg.org/ebooks/1514.txt.utf-8"
CORPUS_SHA256 = "e8626a628112990c8505dd1098276e6f75040cc268690d018ee6694c71888ea4"
MODEL = "SmolLM2-135M"
TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"


def prepare_spec(corpus, steps):
    if sha256(corpus) != CORPUS_SHA256:
        raise ValueError("corpus bytes differ from the frozen Project Gutenberg snapshot")
    text = corpus.read_text(encoding="utf-8")
    start = "*** START OF THE PROJECT GUTENBERG EBOOK A MIDSUMMER NIGHT'S DREAM ***"
    end = "*** END OF THE PROJECT GUTENBERG EBOOK A MIDSUMMER NIGHT'S DREAM ***"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("missing corpus boundary markers")
    text = text.split(start)[1].split(end)[0].strip()
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    tokenizer_path = Path(hf_hub_download(f"HuggingFaceTB/{MODEL}", "tokenizer.json",
                                        revision=SMOLLM2_REVISIONS[MODEL]))
    if sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("tokenizer differs from the frozen checkpoint revision")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokens = tokenizer.encode(text, add_special_tokens=False).ids
    sequence_length = 128
    width = sequence_length + 1
    boundary = int(len(tokens) * 0.8)
    def batches(part):
        return [part[index:index + width] for index in range(0, len(part) - width + 1, width)]
    train, heldout = batches(tokens[:boundary]), batches(tokens[boundary:])
    if len(train) < 8 or len(heldout) < 4:
        raise ValueError("insufficient real-data blocks")
    rng = random.Random(17)
    order = []
    while len(order) < steps:
        epoch = list(range(len(train)))
        rng.shuffle(epoch)
        order.extend(epoch)
    order = order[:steps]
    hashes = input_hashes([MODEL])
    # Both engines use the exact f32 learning-rate value.
    import struct
    learning_rate = struct.unpack("f", struct.pack("f", 0.001))[0]
    return {"protocol": "smollm2-text-v2", "model": MODEL, "steps": steps,
            "sequence_length": sequence_length, "learning_rate": learning_rate, "checkpoint_every": 64,
            "seed": 17, "optimizer": "SGD, no momentum, no weight decay, no clipping",
            "tokenizer": {"revision": SMOLLM2_REVISIONS[MODEL], "sha256": TOKENIZER_SHA256,
                          "file": str(tokenizer_path)},
            "corpus": {"url": CORPUS_URL, "sha256": CORPUS_SHA256, "tokens": len(tokens),
                       "split": "first 80% tokens train, last 20% held out; discard incomplete blocks",
                       "train_blocks": len(train), "heldout_blocks": len(heldout),
                       "special_tokens": False, "batch_order": order},
            "input_sha256": hashes, "train": [train[index] for index in order],
            "probe": train[:4], "heldout": heldout[:4],
            "gate": {"curve_policy": "loss-curve-v1", "per_step_rtol": 0.02, "atol": 1e-6,
                     "relative_l2_limit": 0.01, "skipped_updates": 0,
                     "max_scale_attempts_per_update": 8,
                     "fixed_training_probe_minimum_reduction": 0.05,
                     "heldout_maximum_relative_increase": 0.02,
                     "parameter_must_change": "model.norm.weight"}}


def validate_run(record, spec, spec_hash):
    if (record["status"] != "complete" or record["protocol"] != spec["protocol"]
            or record["spec_sha256"] != spec_hash or record["model"] != MODEL
            or record["completed_updates"] != spec["steps"] or record["skipped_updates"]
            or len(record["losses"]) != spec["steps"]):
        raise ValueError("incomplete, skipped or mismatched optimizer updates")
    if not all(math.isfinite(loss) and loss >= 0 for loss in record["losses"]):
        raise ValueError("non-finite training curve")
    expected_steps = list(range(0, spec["steps"] + 1, spec["checkpoint_every"]))
    if expected_steps[-1] != spec["steps"]:
        expected_steps.append(spec["steps"])
    checkpoints = record["checkpoints"]
    if [checkpoint["step"] for checkpoint in checkpoints] != expected_steps:
        raise ValueError("missing evaluation checkpoints")
    for checkpoint in checkpoints:
        for key in ("probe_losses", "heldout_losses"):
            if len(checkpoint[key]) != 4 or not all(math.isfinite(value) and value >= 0 for value in checkpoint[key]):
                raise ValueError("invalid evaluation checkpoint")
    parameter = record["parameter_check"]
    if (parameter["name"] != spec["gate"]["parameter_must_change"] or parameter["changed"] is not True
            or len(parameter["initial"]) != 576 or len(parameter["final"]) != 576
            or parameter["initial"] == parameter["final"]
            or not all(math.isfinite(value) for value in [*parameter["initial"], *parameter["final"]])):
        raise ValueError("no evidence of a finite parameter update")
    initial, final = checkpoints[0], checkpoints[-1]
    probe_ratio = statistics.mean(final["probe_losses"]) / statistics.mean(initial["probe_losses"])
    heldout_ratio = statistics.mean(final["heldout_losses"]) / statistics.mean(initial["heldout_losses"])
    if probe_ratio > 0.95 or heldout_ratio > 1.02:
        raise ValueError(f"learning gate failed: fixed training probe ratio {probe_ratio}, heldout ratio {heldout_ratio}")
    return {"status": "pass", "probe_loss_ratio": probe_ratio, "heldout_loss_ratio": heldout_ratio}


def compare_runs(actual, reference):
    reports = {"training": compare_loss_curves(actual["losses"], reference["losses"])}
    for key in ("probe_losses", "heldout_losses"):
        reports[key] = compare_loss_curves([loss for row in actual["checkpoints"] for loss in row[key]],
                                          [loss for row in reference["checkpoints"] for loss in row[key]])
    return reports


def validate_precision(record, name):
    framework, mode = name.split("-", 1)
    expected = "strict-f32" if mode == "strict" else "accelerated"
    if record["precision"] != expected or record["framework"] != framework:
        raise ValueError("requested engine/precision did not execute")
    if framework == "pytorch":
        dtype = None if mode == "strict" else mode
        if (record["precision_policy"].get("autocast") != dtype
                or record["gradient_scaler"]["enabled"] != (dtype == "fp16")):
            raise ValueError("requested autocast dtype/scaling policy did not execute")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cuda", "rocm", "xpu", "mps", "cpu"))
    parser.add_argument("--gpu", required=True, help="native GPU; PyTorch must match unless CPU is explicitly requested")
    parser.add_argument("--corpus", required=True, type=Path, help=f"frozen plain text from {CORPUS_URL}")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=256, help="use 256 for the declared study; smaller runs are smoke tests")
    parser.add_argument("--precisions", nargs="+", choices=("bf16", "fp16"), default=["bf16", "fp16"])
    parser.add_argument("--tune-seconds", type=positive_seconds, default=30)
    parser.add_argument("--run-seconds", type=positive_seconds, default=1200)
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.precisions)) != len(args.precisions):
        parser.error("positive step count and unique precisions required")
    import torch
    if platform.python_version() != PYTHON_VERSION:
        parser.error(f"use pinned Python {PYTHON_VERSION}")
    check_torch_identity(torch.__version__, torch.version.git_version, torch.__version__)
    spec = prepare_spec(args.corpus, args.steps)
    destination = args.results_dir.resolve()
    manifest = begin(destination, "smollm2-convergence-study-v2", args)
    manifest.update(cpu_frequency_policies=cpu_frequency_policies(), publication_ready=False,
                    scope="one checkpoint, one corpus, one seed; actual SGD updates, no performance claim")
    path = destination / "study.json"
    write_json(path, manifest)
    spec_path = destination / "spec.json"
    write_json(spec_path, spec)
    shutil.copyfile(args.corpus, destination / "corpus.txt")
    shutil.copyfile(spec["tokenizer"]["file"], destination / "tokenizer.json")
    spec_hash = sha256(spec_path)
    env = environment(args.backend)
    env.update(INFERENA_SDPA="math", INFERENA_GRAPH_REPLAY="0")
    records = {}
    try:
        with (destination / "build.log").open("w") as log:
            subprocess.run(["cargo", "build", "--release", "--locked", "-p", "inferena-meganeura", "-j1"],
                           cwd=ROOT, stdout=log, stderr=log, check=True)
        binary = ROOT / "target/release" / ("inferena-meganeura.exe" if sys.platform == "win32" else "inferena-meganeura")
        devices = json.loads(subprocess.check_output([str(binary), "--list-devices"], cwd=ROOT, env=env, text=True))
        native = select_native_device(devices, args.gpu)
        manifest["native_device"] = native
        manifest["binary_sha256"] = sha256(binary)
        # B570 is selected first when invoking this on the local two-GPU host.
        for name in ("meganeura-strict", "pytorch-strict", *[f"pytorch-{value}" for value in args.precisions]):
            folder = destination / name
            run = {"name": name, "status": "running"}
            manifest["runs"].append(run)
            write_json(path, manifest)
            current = dict(env, INFERENA_STRICT="1" if name.endswith("strict") else "0")
            if name.startswith("meganeura"):
                current.update(MEGANEURA_DEVICE_ID=str(native["device_id"]), MEGANEURA_TUNE="1",
                               INFERENA_TUNE_SECONDS=str(args.tune_seconds),
                               INFERENA_KERNEL_MEMO_DIR=str(destination / "cold-kernel-memo"))
                command = [str(binary), "--converge", str(spec_path), str(folder / "progress.json")]
            else:
                current["INFERENA_AUTOCAST"] = "" if name.endswith("strict") else name.split("-")[1]
                command = [sys.executable, str(ROOT / "frameworks/pytorch/train_curve.py"),
                           str(spec_path), str(folder / "progress.json")]
            print(name, flush=True)
            record = run_json(command, folder, current, args.run_seconds)
            records[name] = record
            validate_precision(record, name)
            if name.startswith("meganeura"):
                if record["environment"]["gpu_device_id"] != native["device_id"]:
                    raise ValueError("native device changed")
            elif (record["environment"]["backend"].split()[0].lower() != args.backend
                  or args.backend != "cpu" and not gpu_matches(args.gpu, record["environment"]["gpu_name"])):
                raise ValueError("reference backend/device changed")
            # Retain all complete trajectories even if a numerical/learning
            # gate fails; execution faults and watchdogs still abort immediately.
            try:
                run["learning"] = validate_run(record, spec, spec_hash)
                run["status"] = "pass"
            except ValueError as error:
                run.update(status="failed", error=str(error))
            write_json(path, manifest)
        manifest["comparisons"] = {}
        for name, record in records.items():
            if name != "pytorch-strict":
                try:
                    manifest["comparisons"][name] = {"status": "pass", **compare_runs(record, records["pytorch-strict"])}
                except ValueError as error:
                    manifest["comparisons"][name] = {"status": "failed", "error": str(error),
                                                      "details": getattr(error, "details", None)}
        if any(sha256(ROOT / file) != digest for file, digest in spec["input_sha256"].items()):
            raise ValueError("input changed during study")
        manifest["status"] = ("pass" if all(run["status"] == "pass" for run in manifest["runs"])
                              and all(row["status"] == "pass" for row in manifest["comparisons"].values()) else "failed")
    except BaseException as error:
        if manifest["runs"] and manifest["runs"][-1]["status"] == "running":
            manifest["runs"][-1].update(status="failed", error=str(error))
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(path, manifest)
    print(json.dumps({"status": manifest["status"], "study": str(path)}))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
