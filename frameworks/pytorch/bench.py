#!/usr/bin/env python3
"""PyTorch benchmark runner for infermark.

Runs a fake training step (forward + backward) on a given model and prints
a JSON result to stdout matching the BenchResult schema.

Features inspired by meganeura's bench/compare.sh (PR #30):
- torch.compile with fresh inductor cache for fair compile-time measurement
- torch.set_float32_matmul_precision("high") for TF32 on Ampere+
- Device name reporting (not just "cuda:0")
- torch version in output
- AMD ROCm HSA_OVERRIDE_GFX_VERSION hint
"""

import hashlib
import json
import os
import platform
import shutil
import struct
import sys
import time

import torch
from transformers import AutoModelForCausalLM


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name(dev: str) -> str:
    if dev.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    if dev == "mps":
        return f"Apple {platform.processor()}"
    return "cpu"


def sha256_f32_tensor(t: torch.Tensor) -> str:
    flat = t.detach().float().cpu().contiguous().flatten()
    raw = struct.pack(f"<{flat.numel()}f", *flat.tolist())
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def clear_compile_cache():
    """Clear torch inductor cache so we measure real compilation time."""
    torch._dynamo.reset()
    for d in [
        os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "torch", "inductor",
        ),
    ]:
        if d and os.path.isdir(d):
            print(f"  clearing compile cache: {d}", file=sys.stderr)
            shutil.rmtree(d, ignore_errors=True)


def bench(model_name: str, hf_id: str):
    dev = detect_device()
    dev_name = device_name(dev)
    torch.set_float32_matmul_precision("high")

    print(f"[pytorch] device: {dev_name} ({dev}), torch {torch.__version__}", file=sys.stderr)

    # --- Load model ---
    # Try: local models/ dir -> HF cache/download -> random-init fallback.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    local_model_dir = os.path.join(root_dir, "models", model_name)

    print(f"[pytorch] loading {hf_id}...", file=sys.stderr)
    t0 = time.perf_counter()
    model = None
    if os.path.isfile(os.path.join(local_model_dir, "config.json")):
        print(f"[pytorch] found local model at {local_model_dir}", file=sys.stderr)
        try:
            model = AutoModelForCausalLM.from_pretrained(local_model_dir, torch_dtype=torch.float32)
        except Exception as e:
            print(f"[pytorch] local load failed ({e})", file=sys.stderr)
    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32)
        except Exception as e:
            print(f"[pytorch] HF load failed ({e}), using random-init", file=sys.stderr)
            from transformers import LlamaConfig, LlamaForCausalLM
            fallback_configs = {
                "HuggingFaceTB/SmolLM2-135M": LlamaConfig(
                    vocab_size=49152, hidden_size=576, num_hidden_layers=30,
                    num_attention_heads=9, num_key_value_heads=3,
                    intermediate_size=1536, max_position_embeddings=2048,
                ),
                "HuggingFaceTB/SmolLM2-360M-Instruct": LlamaConfig(
                    vocab_size=49152, hidden_size=960, num_hidden_layers=32,
                    num_attention_heads=15, num_key_value_heads=5,
                    intermediate_size=2560, max_position_embeddings=2048,
                ),
                "HuggingFaceTB/SmolLM2-1.7B": LlamaConfig(
                    vocab_size=49152, hidden_size=2048, num_hidden_layers=24,
                    num_attention_heads=32, num_key_value_heads=32,
                    intermediate_size=8192, max_position_embeddings=2048,
                ),
            }
            config = fallback_configs.get(hf_id)
            if config is None:
                print(f"[pytorch] no fallback config for {hf_id}", file=sys.stderr)
                sys.exit(1)
            model = LlamaForCausalLM(config).to(torch.float32)
    model.to(dev)
    model.train()
    sync()
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[pytorch] loaded in {load_ms:.0f}ms", file=sys.stderr)

    # --- torch.compile (measure real compilation from clean state) ---
    print("[pytorch] compiling with torch.compile()...", file=sys.stderr)
    clear_compile_cache()
    compile_t0 = time.perf_counter()
    model = torch.compile(model)

    # Force compilation with a dummy forward pass.
    seq_len = 128
    vocab_size = model.config.vocab_size
    dummy_ids = torch.zeros(1, seq_len, dtype=torch.long, device=dev)
    dummy_labels = torch.zeros(1, seq_len, dtype=torch.long, device=dev)
    with torch.no_grad():
        model(input_ids=dummy_ids, labels=dummy_labels)
    sync()
    compile_s = time.perf_counter() - compile_t0
    print(f"[pytorch] compiled in {compile_s:.2f}s", file=sys.stderr)

    # --- Prepare deterministic input ---
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=dev)
    labels = torch.randint(0, vocab_size, (1, seq_len), device=dev)

    # --- Forward ---
    sync()
    t0 = time.perf_counter()
    outputs = model(input_ids=input_ids, labels=labels)
    sync()
    forward_ms = (time.perf_counter() - t0) * 1000.0

    loss = outputs.loss
    logits = outputs.logits

    # --- Backward ---
    sync()
    t0 = time.perf_counter()
    loss.backward()
    sync()
    backward_ms = (time.perf_counter() - t0) * 1000.0

    # --- Collect outputs ---
    logits_hash = sha256_f32_tensor(logits)
    logits_flat = logits.detach().float().cpu().flatten()
    logits_sample = logits_flat[:16].tolist()

    result = {
        "framework": "pytorch",
        "model": model_name,
        "device": dev_name,
        "gpu_name": dev_name,
        "torch_version": torch.__version__,
        "timings": {
            "compile_s": round(compile_s, 2),
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
