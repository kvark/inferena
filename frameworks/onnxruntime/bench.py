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


MODEL_REGISTRY = {
    "SmolLM2-135M": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "type": "causal_lm",
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
    """Export ResNet-50 to ONNX."""
    import torch
    import torchvision.models as tv_models

    model = tv_models.resnet50(weights=None)
    # Match PyTorch bench.py's deterministic init.
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            n = p.numel()
            p.copy_(torch.sin(torch.arange(n, dtype=torch.float32) * 0.01 + i).view_as(p) * 0.1)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, onnx_path, input_names=["images"],
                      output_names=["logits"], opset_version=17,
                      dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)


def export_whisper_onnx(onnx_dir):
    """Export Whisper-tiny to ONNX via optimum."""
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    import tempfile

    config = WhisperConfig(
        d_model=384, encoder_layers=4, decoder_layers=4,
        encoder_attention_heads=6, decoder_attention_heads=6,
        encoder_ffn_dim=1536, decoder_ffn_dim=1536,
        vocab_size=51865, max_source_positions=1500,
        max_target_positions=448, num_mel_bins=80,
    )
    model = WhisperForConditionalGeneration(config)
    # Match PyTorch bench.py's deterministic init.
    import torch
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            n = p.numel()
            p.copy_(torch.sin(torch.arange(n, dtype=torch.float32) * 0.01 + i).view_as(p) * 0.1)

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(tmp, export=True)
        ort_model.save_pretrained(onnx_dir)


def bench_causal_lm(model_name):
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForCausalLM
    import torch

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    model_dir = os.path.join(root_dir, "models", model_name)
    onnx_dir = os.path.join(model_dir, "onnx")
    onnx_path = os.path.join(onnx_dir, "model.onnx")

    seq_len = 128
    vocab_size = 49152

    if not os.path.isfile(onnx_path):
        print(f"[onnxruntime] exporting to ONNX via optimum...", file=sys.stderr)
        os.makedirs(onnx_dir, exist_ok=True)
        ort_model = ORTModelForCausalLM.from_pretrained(model_dir, export=True)
        ort_model.save_pretrained(onnx_dir)
    else:
        print(f"[onnxruntime] loading from {onnx_dir}...", file=sys.stderr)

    t0 = time.perf_counter()
    ort_model = ORTModelForCausalLM.from_pretrained(onnx_dir)
    compile_s = time.perf_counter() - t0

    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    labels = np.array([[(i + 1) % vocab_size for i in range(seq_len)]], dtype=np.int64)

    ort_model(input_ids=input_ids, attention_mask=attention_mask)
    t0 = time.perf_counter()
    outputs = ort_model(input_ids=input_ids, attention_mask=attention_mask)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    logits = outputs.logits.detach().numpy()
    loss = cross_entropy_np(logits[0], labels[0])

    # Latency (single-token).
    lat_input = torch.zeros(1, 1, dtype=torch.long)
    lat_mask = torch.ones(1, 1, dtype=torch.long)
    ort_model(input_ids=lat_input, attention_mask=lat_mask)
    t0 = time.perf_counter()
    ort_model(input_ids=lat_input, attention_mask=lat_mask)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    emit(model_name, compile_s, inference_ms, latency_ms, logits, loss)


def bench_resnet(model_name):
    import onnxruntime as ort

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    onnx_path = os.path.join(root_dir, "models", model_name, "resnet50.onnx")

    if not os.path.isfile(onnx_path):
        print("[onnxruntime] exporting ResNet-50 to ONNX...", file=sys.stderr)
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        export_resnet_onnx(onnx_path)

    t0 = time.perf_counter()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
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

    emit(model_name, compile_s, inference_ms, latency_ms, logits, loss)


def bench_whisper(model_name):
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    import torch

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    onnx_dir = os.path.join(root_dir, "models", model_name, "onnx")

    if not os.path.isdir(onnx_dir) or not any(f.endswith(".onnx") for f in os.listdir(onnx_dir)):
        print("[onnxruntime] exporting Whisper-tiny to ONNX...", file=sys.stderr)
        os.makedirs(onnx_dir, exist_ok=True)
        export_whisper_onnx(onnx_dir)

    t0 = time.perf_counter()
    ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(onnx_dir)
    compile_s = time.perf_counter() - t0

    mel = torch.sin(torch.arange(80 * 3000, dtype=torch.float32) * 0.001).reshape(1, 80, 3000)
    dec_ids = torch.tensor([[50258, 50259, 50359, 50363]], dtype=torch.long)

    ort_model(input_features=mel, decoder_input_ids=dec_ids)
    t0 = time.perf_counter()
    outputs = ort_model(input_features=mel, decoder_input_ids=dec_ids)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    logits = outputs.logits.detach().numpy()
    loss = cross_entropy_np(
        logits.reshape(-1, logits.shape[-1]),
        dec_ids[:, :logits.shape[1]].numpy().reshape(-1),
    )

    emit(model_name, compile_s, inference_ms, 0.0, logits, loss)


def emit(model_name, compile_s, inference_ms, latency_ms, logits, loss):
    import onnxruntime as ort

    logits_np = np.asarray(logits, dtype=np.float32)
    logits_hash = sha256_f32(logits_np)
    logits_flat = logits_np.flatten()
    logits_sample = [round(float(v), 6) for v in logits_flat[:16]]

    result = {
        "framework": "onnxruntime",
        "model": model_name,
        "device": "cpu",
        "gpu_name": "cpu",
        "onnxruntime_version": ort.__version__,
        "backend": "CPUExecutionProvider",
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

    model_type = spec["type"]
    if model_type == "causal_lm":
        bench_causal_lm(model_name)
    elif model_type == "resnet":
        bench_resnet(model_name)
    elif model_type == "whisper":
        bench_whisper(model_name)
    else:
        print(f"[onnxruntime] unsupported: {model_type}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
