#!/usr/bin/env python3
"""ONNX Runtime benchmark runner for inferena.

Exports the model to ONNX via PyTorch/optimum, then runs inference with
onnxruntime. Training (backward) is not supported.
"""

import hashlib
import json
import math
import os
import struct
import sys
import time

import numpy as np


def _name_seed(name: str) -> float:
    """Deterministic seed from parameter name — framework-independent init."""
    h = 0
    for c in name.encode('ascii'):
        h = ((h * 31) + c) & 0xFFFFFFFF
    return float(h % 10000)


MODEL_REGISTRY = {
    "SmolLM2-135M": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "type": "causal_lm",
    },
    "SmolVLA": {
        "hf_id": "lerobot/smolvla_base",
        "type": "smolvla",
    },
    "StableDiffusion": {
        "hf_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "type": "sd_unet",
    },
    "ResNet-50": {
        "hf_id": "torchvision/resnet50",
        "type": "resnet",
    },
    "Whisper-tiny": {
        "hf_id": "openai/whisper-tiny",
        "type": "whisper",
    },
}


def sha256_f32(data: np.ndarray) -> str:
    flat = data.astype(np.float32).flatten()
    raw = struct.pack(f"<{flat.size}f", *flat.tolist())
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _get_providers():
    """Return the best available ONNX Runtime execution providers."""
    import onnxruntime as ort
    available = ort.get_available_providers()
    # Prefer GPU providers, fall back to CPU.
    preferred = [
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "ROCMExecutionProvider",
        "MIGraphXExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    providers = [p for p in preferred if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    return providers


def _provider_label(sess):
    """Return (device, gpu_name, backend) from a live session's active provider."""
    active = sess.get_providers()
    first = active[0] if active else "CPUExecutionProvider"
    if "CUDA" in first or "Tensorrt" in first or "ROCM" in first:
        return ("gpu", "gpu", first)
    elif "CoreML" in first or "Dml" in first:
        return ("gpu", "gpu", first)
    return ("cpu", "cpu", first)


def cross_entropy_np(logits_2d, labels_1d):
    """Cross-entropy loss over (N, vocab) logits and (N,) labels."""
    total = 0.0
    for i in range(logits_2d.shape[0]):
        row = logits_2d[i].astype(np.float64)
        mx = row.max()
        lse = mx + math.log(np.exp(row - mx).sum())
        total += -(row[labels_1d[i]] - lse)
    return total / logits_2d.shape[0]


def export_resnet_onnx(onnx_path):
    """Export ResNet-50 to ONNX with name-seeded init matching PyTorch/meganeura."""
    import torch
    import torchvision.models as tv_models

    scale = 0.01  # small scale — identity BN, no explosion
    model = tv_models.resnet50(weights=None)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if '.bn' in name or '.downsample.1.' in name:
                if 'weight' in name:
                    p.fill_(1.0)
                else:
                    p.zero_()
            elif name == 'fc.weight':
                seed = _name_seed(name)
                out_f, in_f = p.shape
                w = torch.sin(torch.arange(in_f * out_f, dtype=torch.float32) * 0.01 + seed).view(in_f, out_f) * scale
                p.copy_(w.T)
            else:
                seed = _name_seed(name)
                n = p.numel()
                p.copy_(torch.sin(torch.arange(n, dtype=torch.float32) * 0.01 + seed).view_as(p) * scale)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, onnx_path, input_names=["images"],
                      output_names=["logits"], opset_version=17,
                      dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)


_WHISPER_TRANSPOSED = frozenset([
    'q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'out_proj.weight',
    'fc1.weight', 'fc2.weight',
])


def export_whisper_encoder_onnx(onnx_path):
    """Export Whisper-tiny encoder to ONNX with name-seeded init."""
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    config = WhisperConfig(
        d_model=384, encoder_layers=4, decoder_layers=4,
        encoder_attention_heads=6, decoder_attention_heads=6,
        encoder_ffn_dim=1536, decoder_ffn_dim=1536,
        vocab_size=51865, max_source_positions=1500,
        max_target_positions=448, num_mel_bins=80,
    )
    full_model = WhisperForConditionalGeneration(config)
    encoder = full_model.get_encoder()
    with torch.no_grad():
        for name, p in encoder.named_parameters():
            seed = _name_seed(name)
            if any(name.endswith(s) for s in _WHISPER_TRANSPOSED):
                out_f, in_f = p.shape
                w = torch.sin(torch.arange(in_f * out_f, dtype=torch.float32) * 0.01 + seed).view(in_f, out_f) * 0.1
                p.copy_(w.T)
            else:
                n = p.numel()
                p.copy_(torch.sin(torch.arange(n, dtype=torch.float32) * 0.01 + seed).view_as(p) * 0.1)
    encoder.eval()

    dummy_mel = torch.randn(1, 80, 3000)
    torch.onnx.export(encoder, dummy_mel, onnx_path, input_names=["mel"],
                      output_names=["hidden_states"], opset_version=17,
                      dynamo=False)


def _export_causal_lm_onnx(onnx_path, model_dir):
    """Export causal LM to ONNX via torch.onnx.export (no optimum dependency)."""
    import torch
    from transformers import AutoModelForCausalLM

    # Try local dir, then HF cache.
    src = model_dir
    if not os.path.isfile(os.path.join(src, "model.safetensors")):
        try:
            from huggingface_hub import snapshot_download
            HF_IDS = {"SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M"}
            hf_id = HF_IDS.get(os.path.basename(src))
            if hf_id:
                src = snapshot_download(hf_id, allow_patterns=["*.safetensors", "*.json"])
            else:
                raise FileNotFoundError(f"No weights at {src}")
        except Exception as e:
            print(f"[onnxruntime] cannot load model: {e}", file=sys.stderr)
            sys.exit(1)
    # Force eager attention implementation — SDPA/GQA paths have tracing
    # bugs in torch 2.11's dynamo ONNX exporter (SymbolicDim comparison) and
    # in transformers' legacy-exporter masking_utils (empty tuple shape).
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=torch.float32, attn_implementation="eager",
    )
    model.eval()

    # Wrap to return only logits (avoids DynamicCache tracing issues).
    class LogitsOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    wrapper = LogitsOnly(model)
    seq_len = 128
    dummy_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    # Try the dynamo exporter first; fall back to the legacy TorchScript path.
    # torch 2.11 + onnxscript 0.6 can fail on GQA's scaled_dot_product_attention
    # with a SymbolicDim comparison error — the legacy exporter still handles it.
    try:
        torch.onnx.export(
            wrapper, (dummy_ids, dummy_mask), onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            dynamo=True,
        )
    except Exception as e:
        msg = str(e).split("\n")[0][:200]
        print(f"[onnxruntime] dynamo export failed ({msg}); retrying with legacy exporter", file=sys.stderr)
        torch.onnx.export(
            wrapper, (dummy_ids, dummy_mask), onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            dynamo=False,
        )


def bench_causal_lm(model_name):
    import onnxruntime as ort

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    model_dir = os.path.join(root_dir, "models", model_name)
    onnx_path = os.path.join(model_dir, "causal_lm.onnx")

    seq_len = 128
    vocab_size = 49152

    # Always re-export so compile_s includes the full cold-start cost.
    print(f"[onnxruntime] exporting {model_name} to ONNX...", file=sys.stderr)
    if os.path.isfile(onnx_path):
        os.remove(onnx_path)

    providers = _get_providers()
    print(f"[onnxruntime] providers: {providers}", file=sys.stderr)
    t0 = time.perf_counter()
    _export_causal_lm_onnx(onnx_path, model_dir)
    sess = ort.InferenceSession(onnx_path, providers=providers)
    compile_s = time.perf_counter() - t0
    print(f"[onnxruntime] active: {sess.get_providers()}", file=sys.stderr)

    input_ids = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    labels = np.array([[(i + 1) % vocab_size for i in range(seq_len)]], dtype=np.int64)

    sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    t0 = time.perf_counter()
    outputs = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    inference_ms = (time.perf_counter() - t0) * 1000.0

    logits = outputs[0]  # [1, seq_len, vocab]
    loss = cross_entropy_np(logits[0], labels[0])

    # Latency (single-token).
    lat_ids = np.zeros((1, 1), dtype=np.int64)
    lat_mask = np.ones((1, 1), dtype=np.int64)
    sess.run(None, {"input_ids": lat_ids, "attention_mask": lat_mask})
    t0 = time.perf_counter()
    sess.run(None, {"input_ids": lat_ids, "attention_mask": lat_mask})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    emit(model_name, compile_s, inference_ms, latency_ms, logits, loss, sess)


def bench_resnet(model_name):
    import onnxruntime as ort

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    onnx_path = os.path.join(root_dir, "models", model_name, "resnet50.onnx")

    # Always re-export so compile_s includes the full cold-start cost.
    print("[onnxruntime] exporting ResNet-50 to ONNX...", file=sys.stderr)
    if os.path.isfile(onnx_path):
        os.remove(onnx_path)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    t0 = time.perf_counter()
    export_resnet_onnx(onnx_path)
    sess = ort.InferenceSession(onnx_path, providers=_get_providers())
    compile_s = time.perf_counter() - t0

    batch = 4
    images = np.sin(np.arange(batch * 3 * 224 * 224, dtype=np.float32) * 0.001).reshape(batch, 3, 224, 224)
    labels = np.arange(batch, dtype=np.int64) % 1000

    sess.run(None, {"images": images})
    t0 = time.perf_counter()
    outputs = sess.run(None, {"images": images})
    inference_ms = (time.perf_counter() - t0) * 1000.0
    logits = outputs[0]

    loss = cross_entropy_np(logits, labels)

    # Single-image latency.
    lat_img = np.zeros((1, 3, 224, 224), dtype=np.float32)
    sess.run(None, {"images": lat_img})
    t0 = time.perf_counter()
    sess.run(None, {"images": lat_img})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    emit(model_name, compile_s, inference_ms, latency_ms, logits, loss, sess)


def bench_whisper(model_name):
    import onnxruntime as ort

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    onnx_path = os.path.join(root_dir, "models", model_name, "whisper_encoder.onnx")

    # Always re-export so compile_s includes the full cold-start cost.
    print("[onnxruntime] exporting Whisper-tiny encoder to ONNX...", file=sys.stderr)
    if os.path.isfile(onnx_path):
        os.remove(onnx_path)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    t0 = time.perf_counter()
    export_whisper_encoder_onnx(onnx_path)
    sess = ort.InferenceSession(onnx_path, providers=_get_providers())
    compile_s = time.perf_counter() - t0

    mel = np.sin(np.arange(80 * 3000, dtype=np.float32) * 0.001).reshape(1, 80, 3000)

    sess.run(None, {"mel": mel})
    t0 = time.perf_counter()
    outputs = sess.run(None, {"mel": mel})
    inference_ms = (time.perf_counter() - t0) * 1000.0

    hidden_states = outputs[0]  # [1, seq_len, d_model]
    loss = float(np.mean(hidden_states ** 2))  # MSE vs zero

    emit(model_name, compile_s, inference_ms, 0.0, hidden_states, loss, sess)


def _import_pytorch_model(cls_name):
    """Import a model class from the PyTorch bench (used for ONNX export only)."""
    import importlib.util
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pytorch_bench = os.path.join(script_dir, "..", "pytorch", "bench.py")
    spec = importlib.util.spec_from_file_location("pytorch_bench", pytorch_bench)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def _name_seeded_init(model, scale=0.1):
    """Deterministic init matching PyTorch bench's _deterministic_init."""
    import torch
    with torch.no_grad():
        for name, p in model.named_parameters():
            seed = _name_seed(name)
            n = p.numel()
            p.copy_(torch.sin(torch.arange(n, dtype=torch.float32) * 0.01 + seed).view_as(p) * scale)
    model.eval()
    return model


def export_sd_unet_onnx(onnx_path):
    """Export simplified SD U-Net to ONNX."""
    import torch
    SDUNet = _import_pytorch_model("SDUNet")
    model = SDUNet(in_channels=4, base_channels=64, num_levels=3, num_groups=16, eps=1e-5)
    model = _name_seeded_init(model)
    dummy = torch.randn(2, 4, 32, 32)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(model, dummy, onnx_path, input_names=["noisy_latent"],
                      output_names=["noise_pred"], opset_version=17, dynamo=False)


def bench_sd(model_name):
    import onnxruntime as ort

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    onnx_path = os.path.join(root_dir, "models", model_name, "sd_unet.onnx")

    print("[onnxruntime] exporting SD U-Net to ONNX...", file=sys.stderr)
    if os.path.isfile(onnx_path):
        os.remove(onnx_path)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    providers = _get_providers()
    t0 = time.perf_counter()
    export_sd_unet_onnx(onnx_path)
    sess = ort.InferenceSession(onnx_path, providers=providers)
    compile_s = time.perf_counter() - t0

    # Match PyTorch bench inputs: batch=2, 4 channels, 32×32.
    batch, in_c, res = 2, 4, 32
    in_size = batch * in_c * res * res
    noisy = np.sin(np.arange(in_size, dtype=np.float32) * 0.01).reshape(batch, in_c, res, res)

    sess.run(None, {"noisy_latent": noisy})
    t0 = time.perf_counter()
    outputs = sess.run(None, {"noisy_latent": noisy})
    inference_ms = (time.perf_counter() - t0) * 1000.0
    pred = outputs[0]

    # MSE loss against zeros (noise prediction target).
    loss = float(np.mean(pred ** 2))

    emit(model_name, compile_s, inference_ms, 0.0, pred, loss, sess)


def export_smolvla_onnx(onnx_path):
    """Export SmolVLA action expert to ONNX."""
    import torch
    ActionExpert = _import_pytorch_model("ActionExpert")
    model = ActionExpert()
    model = _name_seeded_init(model)
    noisy_actions = torch.sin(torch.arange(50 * 32, dtype=torch.float32) * 0.01).view(1, 50, 32)
    timestep = torch.sin(torch.arange(720 * 2, dtype=torch.float32) * 0.005).view(1, 1, 1440)
    vlm_kv = torch.cos(torch.arange(16 * 320, dtype=torch.float32) * 0.01).view(1, 16, 320)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model, (noisy_actions, timestep, vlm_kv), onnx_path,
        input_names=["noisy_actions", "timestep", "vlm_kv"],
        output_names=["predicted_actions"], opset_version=17, dynamo=False,
    )


def bench_smolvla(model_name):
    import onnxruntime as ort

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    onnx_path = os.path.join(root_dir, "models", model_name, "smolvla.onnx")

    print("[onnxruntime] exporting SmolVLA to ONNX...", file=sys.stderr)
    if os.path.isfile(onnx_path):
        os.remove(onnx_path)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    providers = _get_providers()
    t0 = time.perf_counter()
    export_smolvla_onnx(onnx_path)
    sess = ort.InferenceSession(onnx_path, providers=providers)
    compile_s = time.perf_counter() - t0

    # Match PyTorch bench inputs.
    noisy_actions = np.sin(np.arange(50 * 32, dtype=np.float32) * 0.01).reshape(1, 50, 32)
    timestep = np.sin(np.arange(720 * 2, dtype=np.float32) * 0.005).reshape(1, 1, 1440)
    vlm_kv = np.cos(np.arange(16 * 320, dtype=np.float32) * 0.01).reshape(1, 16, 320)

    feeds = {"noisy_actions": noisy_actions, "timestep": timestep, "vlm_kv": vlm_kv}
    sess.run(None, feeds)
    t0 = time.perf_counter()
    outputs = sess.run(None, feeds)
    inference_ms = (time.perf_counter() - t0) * 1000.0
    pred = outputs[0]

    # MSE loss against zeros.
    loss = float(np.mean(pred ** 2))

    emit(model_name, compile_s, inference_ms, 0.0, pred, loss, sess)


def emit(model_name, compile_s, inference_ms, latency_ms, logits, loss, sess=None):
    import onnxruntime as ort

    logits_np = np.asarray(logits, dtype=np.float32)
    logits_hash = sha256_f32(logits_np)
    logits_flat = logits_np.flatten()
    logits_sample = [round(float(v), 6) for v in logits_flat[:16]]

    if sess is not None:
        device, gpu_name, backend = _provider_label(sess)
    else:
        device, gpu_name, backend = "cpu", "cpu", "CPUExecutionProvider"

    result = {
        "framework": "onnxruntime",
        "model": model_name,
        "device": device,
        "gpu_name": gpu_name,
        "onnxruntime_version": ort.__version__,
        "backend": backend,
        "timings": {
            "compile_s": round(compile_s, 2),
            "inference_ms": round(inference_ms, 3),
            "latency_ms": round(latency_ms, 3),
            "training_ms": 0.0,
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": logits_sample,
            "loss": round(float(loss), 6),
        },
    }
    print(json.dumps(result))


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M"
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    if os.environ.get("INFERENA_DRY_RUN") == "1":
        print(f"[onnxruntime] dry-run OK: {model_name} ({spec['type']})", file=sys.stderr)
        sys.exit(0)

    model_type = spec["type"]
    if model_type == "causal_lm":
        bench_causal_lm(model_name)
    elif model_type == "resnet":
        bench_resnet(model_name)
    elif model_type == "whisper":
        bench_whisper(model_name)
    elif model_type == "smolvla":
        bench_smolvla(model_name)
    elif model_type == "sd_unet":
        bench_sd(model_name)
    else:
        print(f"[onnxruntime] unsupported: {model_type}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
