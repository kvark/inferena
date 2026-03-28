#!/usr/bin/env python3
"""PyTorch benchmark runner for infermark.

Runs a fake training step (forward + backward) on a given model and prints
a JSON result to stdout matching the BenchResult schema.
"""

import hashlib
import json
import struct
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def sha256_f32_tensor(t: torch.Tensor) -> str:
    flat = t.detach().float().cpu().contiguous().flatten()
    raw = struct.pack(f"<{flat.numel()}f", *flat.tolist())
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def bench(model_name: str, hf_id: str):
    dev = device()

    # --- Load & compile ---
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32)
    model.to(dev)
    model.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000.0

    # --- Prepare dummy input ---
    # Fixed seed for reproducibility across frameworks.
    torch.manual_seed(42)
    seq_len = 128
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=dev)
    labels = torch.randint(0, vocab_size, (1, seq_len), device=dev)

    # --- Forward ---
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = model(input_ids=input_ids, labels=labels)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - t0) * 1000.0

    loss = outputs.loss
    logits = outputs.logits

    # --- Backward ---
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss.backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    backward_ms = (time.perf_counter() - t0) * 1000.0

    # --- Collect outputs ---
    logits_hash = sha256_f32_tensor(logits)
    logits_flat = logits.detach().float().cpu().flatten()
    logits_sample = logits_flat[:16].tolist()

    result = {
        "framework": "pytorch",
        "model": model_name,
        "device": dev,
        "gpu_name": gpu_name(),
        "timings": {
            "compile_ms": round(compile_ms, 3),
            "forward_ms": round(forward_ms, 3),
            "backward_ms": round(backward_ms, 3),
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": [round(v, 6) for v in logits_sample],
            "loss": round(loss.item(), 6),
        },
    }
    print(json.dumps(result))


# Map short model names to HuggingFace repo IDs.
MODEL_MAP = {
    "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M",
    "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B",
    "SmolVLM-256M": "HuggingFaceTB/SmolVLM-256M-Instruct",
}

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M"
    hf_id = MODEL_MAP.get(model_name)
    if hf_id is None:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_MAP.keys())}", file=sys.stderr)
        sys.exit(1)
    bench(model_name, hf_id)
