#!/usr/bin/env python3
"""SmolLM2 real-data SGD worker. Curves are evidence, not benchmark timings."""

import json
import os
from pathlib import Path
import sys
import time

import torch

from amp import AmpPolicy, optimizer_update
from bench import (MODEL_REGISTRY, load_model, detect_device, device_name, backend_name,
                   _configure_benchmark_precision, attention_context, qualify_embedding_backward)
from execution import QualificationError, synchronize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from collection import write_json
from study import sha256
from training_gate import SMOLLM2_HIDDEN_SIZES


def run(spec_path, output):
    spec = json.loads(spec_path.read_text())
    if (spec["protocol"] != "smollm2-text-v3" or spec["model"] not in SMOLLM2_HIDDEN_SIZES
            or len(spec["train"]) != spec["steps"]):
        raise ValueError("unexpected convergence workload")
    for path, expected in spec["input_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"changed input file: {path}")
    dev = detect_device()
    policy = AmpPolicy(dev)
    precision = _configure_benchmark_precision(dev, os.environ.get("INFERENA_STRICT") == "1", policy)
    torch.manual_seed(0)
    started = time.monotonic()
    model = load_model(spec["model"], MODEL_REGISTRY[spec["model"]], dev).to(dev).train()
    if any(isinstance(module, torch.nn.Dropout) and module.p for module in model.modules()):
        raise ValueError("matched native trajectory requires zero dropout")
    embedding_backward = (qualify_embedding_backward(model, dev, spec["train"][0][:-1])
                          if dev.startswith("xpu") else {"status": "not-required"})
    parameter_name = "model.norm.weight"
    parameter = dict(model.named_parameters())[parameter_name]
    initial = parameter.detach().cpu().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=spec["learning_rate"], foreach=False)
    scaler = policy.scaler()
    record = {
        "protocol": spec["protocol"], "status": "running", "framework": "pytorch", "model": spec["model"],
        "precision": precision["comparison_class"], "precision_policy": precision, "spec_sha256": sha256(spec_path),
        "environment": {"torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
                        "backend": backend_name(dev), "gpu_name": device_name(dev),
                        "torch_build_config": torch.__config__.show()},
        "execution": "eager, math SDPA, uncaptured, zero dropout",
        "embedding_backward": embedding_backward,
        "optimizer": {"name": "SGD", "learning_rate": spec["learning_rate"], "momentum": 0, "weight_decay": 0},
        "training_scope": "forward + f32 loss + backward + parameter update",
        "losses": [], "checkpoints": [], "completed_updates": 0, "skipped_updates": 0,
        "parameter_check": {"name": parameter_name, "initial": initial.tolist()},
        "gradient_scaler": {"enabled": scaler.is_enabled(), "attempts": [], "overflow_attempts": 0},
    }
    write_json(output, record)

    def loss(tokens):
        if len(tokens) != spec["sequence_length"] + 1:
            raise ValueError("wrong token batch length")
        batch = torch.tensor([tokens], dtype=torch.long, device=dev)
        with policy.context():
            logits = model(input_ids=batch[:, :-1], use_cache=False).logits
            return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                                      batch[:, 1:].reshape(-1))

    with attention_context(dev, "causal_lm"):
        for step in range(spec["steps"] + 1):
            if step % spec["checkpoint_every"] == 0 or step == spec["steps"]:
                with torch.no_grad():
                    checkpoint = {"step": step, "probe_losses": [float(loss(batch)) for batch in spec["probe"]],
                                  "heldout_losses": [float(loss(batch)) for batch in spec["heldout"]]}
                record["checkpoints"].append(checkpoint)
                write_json(output, record)
            if step == spec["steps"]:
                break
            try:
                value, attempts = optimizer_update(optimizer, scaler, lambda: loss(spec["train"][step]))
            except QualificationError as error:
                record.update(status="failed", failure={"step": step, **error.details})
                write_json(output, record)
                raise
            record["losses"].append(value)
            record["gradient_scaler"]["attempts"].append(attempts)
            record["gradient_scaler"]["overflow_attempts"] += len(attempts) - 1
            record["completed_updates"] = step + 1
            write_json(output, record)
            if (step + 1) % spec["checkpoint_every"] == 0:
                print(f"[pytorch] SGD step {step + 1}: loss={value:.6f}", file=sys.stderr, flush=True)
    synchronize(dev)
    final = parameter.detach().cpu()
    if not torch.isfinite(final).all() or torch.equal(initial, final):
        raise ValueError("SGD did not produce a finite, changed parameter")
    record["parameter_check"].update(final=final.tolist(), changed=True)
    record.update(status="complete", elapsed_seconds=time.monotonic() - started)
    write_json(output, record)
    print(json.dumps(record))


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
