#!/usr/bin/env python3
"""PyTorch benchmark runner for infermark.

Runs a fake training step (forward + backward) on a given model and prints
a JSON result to stdout matching the BenchResult schema.

Features inspired by meganeura's bench/compare.sh (PR #30):
- torch.compile with fresh inductor cache for fair compile-time measurement
- torch.set_float32_matmul_precision("high") for TF32 on Ampere+
- Device name reporting (not just "cuda:0")
- torch version in output
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


# --- Model registry ---

MODEL_REGISTRY = {
    "SmolLM2-135M": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "type": "causal_lm",
    },
    "SmolLM2-360M": {
        "hf_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "type": "causal_lm",
    },
    "SmolLM2-1.7B": {
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B",
        "type": "causal_lm",
    },
    "SmolVLM-256M": {
        "hf_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "type": "vlm",
    },
}


def load_model(model_name: str, spec: dict, dev: str):
    """Load model, trying: local dir -> HF download -> random-init fallback."""
    from transformers import AutoModelForCausalLM

    hf_id = spec["hf_id"]
    model_type = spec["type"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    local_dir = os.path.join(root_dir, "models", model_name)

    model = None

    # Try local dir first.
    if os.path.isfile(os.path.join(local_dir, "config.json")):
        print(f"[pytorch] found local model at {local_dir}", file=sys.stderr)
        try:
            model = _load_pretrained(model_type, local_dir)
        except Exception as e:
            print(f"[pytorch] local load failed ({e})", file=sys.stderr)

    # Try HF download.
    if model is None:
        try:
            model = _load_pretrained(model_type, hf_id)
        except Exception as e:
            print(f"[pytorch] HF load failed ({e}), using random-init", file=sys.stderr)
            model = _random_init(model_type, model_name)

    return model


def _load_pretrained(model_type: str, path_or_id: str):
    if model_type == "vlm":
        from transformers import SmolVLMForConditionalGeneration
        return SmolVLMForConditionalGeneration.from_pretrained(path_or_id, torch_dtype=torch.float32)
    else:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(path_or_id, torch_dtype=torch.float32)


def _random_init(model_type: str, model_name: str):
    if model_type == "vlm":
        from transformers import SmolVLMConfig, SmolVLMForConditionalGeneration, LlamaConfig, SmolVLMVisionConfig
        vision = SmolVLMVisionConfig(
            hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
            image_size=384, patch_size=16,
        )
        text = LlamaConfig(
            vocab_size=49152, hidden_size=576, num_hidden_layers=30,
            num_attention_heads=9, num_key_value_heads=3,
            intermediate_size=1536, max_position_embeddings=2048,
        )
        config = SmolVLMConfig(vision_config=vision, text_config=text)
        return SmolVLMForConditionalGeneration(config).to(torch.float32)
    else:
        from transformers import LlamaConfig, LlamaForCausalLM
        configs = {
            "SmolLM2-135M": LlamaConfig(
                vocab_size=49152, hidden_size=576, num_hidden_layers=30,
                num_attention_heads=9, num_key_value_heads=3,
                intermediate_size=1536, max_position_embeddings=2048,
            ),
            "SmolLM2-360M": LlamaConfig(
                vocab_size=49152, hidden_size=960, num_hidden_layers=32,
                num_attention_heads=15, num_key_value_heads=5,
                intermediate_size=2560, max_position_embeddings=2048,
            ),
            "SmolLM2-1.7B": LlamaConfig(
                vocab_size=49152, hidden_size=2048, num_hidden_layers=24,
                num_attention_heads=32, num_key_value_heads=32,
                intermediate_size=8192, max_position_embeddings=2048,
            ),
        }
        config = configs.get(model_name)
        if config is None:
            print(f"[pytorch] no fallback config for {model_name}", file=sys.stderr)
            sys.exit(1)
        return LlamaForCausalLM(config).to(torch.float32)


def prepare_inputs(model_type: str, model, dev: str, seq_len: int = 128):
    """Build deterministic dummy inputs matching the model type."""
    vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else model.config.text_config.vocab_size
    input_ids = torch.arange(seq_len, device=dev, dtype=torch.long).unsqueeze(0)
    labels = (torch.arange(1, seq_len + 1, device=dev, dtype=torch.long) % vocab_size).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=dev)

    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    if model_type == "vlm":
        # One dummy 384x384 image.
        torch.manual_seed(42)
        kwargs["pixel_values"] = torch.randn(1, 1, 3, 384, 384, device=dev)
        kwargs["pixel_attention_mask"] = torch.ones(1, 1, 384, 384, dtype=torch.bool, device=dev)

    return kwargs


def bench(model_name: str, spec: dict):
    dev = detect_device()
    dev_name = device_name(dev)
    model_type = spec["type"]
    torch.set_float32_matmul_precision("high")

    print(f"[pytorch] device: {dev_name} ({dev}), torch {torch.__version__}", file=sys.stderr)

    # --- Load model ---
    print(f"[pytorch] loading {spec['hf_id']}...", file=sys.stderr)
    t0 = time.perf_counter()
    model = load_model(model_name, spec, dev)
    model.to(dev)
    model.train()
    sync()
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[pytorch] loaded in {load_ms:.0f}ms", file=sys.stderr)

    # --- torch.compile ---
    print("[pytorch] compiling with torch.compile()...", file=sys.stderr)
    clear_compile_cache()
    compile_t0 = time.perf_counter()
    model = torch.compile(model)

    # Force compilation with a dummy forward pass.
    dummy_kwargs = prepare_inputs(model_type, model, dev)
    with torch.no_grad():
        model(**dummy_kwargs)
    sync()
    compile_s = time.perf_counter() - compile_t0
    print(f"[pytorch] compiled in {compile_s:.2f}s", file=sys.stderr)

    # --- Prepare deterministic input ---
    fwd_kwargs = prepare_inputs(model_type, model, dev)

    # --- Forward ---
    sync()
    t0 = time.perf_counter()
    outputs = model(**fwd_kwargs)
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


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M"
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)
    bench(model_name, spec)
