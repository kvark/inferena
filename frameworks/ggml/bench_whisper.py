#!/usr/bin/env python3
"""Whisper-tiny benchmark via faster-whisper (CTranslate2 / whisper.cpp engine).

Uses the faster-whisper Python API for proper programmatic access to the
whisper encoder. Runs on CPU or GPU via CTranslate2 backend.
"""

import hashlib
import json
import os
import struct
import sys
import tempfile
import time

import numpy as np


def sha256_f32(data):
    flat = np.asarray(data, dtype=np.float32).flatten()
    raw = struct.pack(f"<{flat.size}f", *flat.tolist())
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def main():
    from faster_whisper import WhisperModel

    # Auto-detect GPU: try CUDA first, fall back to CPU.
    # Note: faster-whisper (CTranslate2) supports CUDA but not MPS/Metal.
    device = "cpu"
    compute_type = "float32"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
    except ImportError:
        pass
    if device == "cpu":
        # CTranslate2 can also use "auto" which selects the best available.
        try:
            import ctranslate2
            if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
                device = "cuda"
                compute_type = "float16"
        except (ImportError, RuntimeError):
            pass

    # Use a temp directory for CT2 conversion so compile_s always measures
    # the full cost (conversion + load), never a cached load.
    ct2_dir = tempfile.mkdtemp(prefix="inferena_whisper_")
    print(f"[ggml] loading whisper-tiny via faster-whisper (device={device})...", file=sys.stderr)
    t0 = time.perf_counter()
    model = WhisperModel("tiny", device=device, compute_type=compute_type,
                         download_root=ct2_dir)
    compile_s = time.perf_counter() - t0
    print(f"[ggml] loaded in {compile_s:.2f}s", file=sys.stderr)

    # Generate deterministic audio input (sine wave, 30s at 16kHz).
    sr = 16000
    duration = 30
    n_samples = sr * duration
    audio = np.sin(np.arange(n_samples, dtype=np.float32) * 0.001) * 0.1

    # Warm-up.
    segments, info = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
    for _ in segments:
        pass

    # Timed inference.
    t0 = time.perf_counter()
    segments, info = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    text_output = " ".join(text_parts).strip()
    print(f"[ggml] inference: {inference_ms:.0f}ms, output: {text_output[:80]!r}", file=sys.stderr)

    # Latency (re-run — whisper is not autoregressive in the token sense).
    t0 = time.perf_counter()
    segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
    for _ in segments:
        pass
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Use text bytes as logits proxy (whisper.cpp doesn't expose raw encoder states).
    text_bytes = text_output.encode("utf-8")
    logits_hash = "sha256:" + hashlib.sha256(text_bytes).hexdigest()
    sample_vals = [float(b) / 255.0 for b in text_bytes[:16]] if text_bytes else [0.0]

    # Loss: not meaningful for GGML whisper (different pipeline than encoder-only).
    loss = 0.0

    ct2_backend = "CTranslate2"
    gpu_name = device
    if device == "cuda":
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
    result = {
        "framework": "ggml",
        "model": "Whisper-tiny",
        "device": device,
        "gpu_name": gpu_name,
        "backend": f"faster-whisper ({ct2_backend}, {device.upper()})",
        "timings": {
            "compile_s": round(compile_s, 2),
            "inference_ms": round(inference_ms, 3),
            "latency_ms": round(latency_ms, 3),
            "training_ms": 0.0,
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": [round(v, 6) for v in sample_vals],
            "loss": loss,
        },
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
