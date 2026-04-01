#!/usr/bin/env python3
"""llama.cpp benchmark wrapper for inferena.

Uses llama-cpp-python bindings for proper inference with logit output.
Backward pass is not supported (inference-only framework).
"""

import hashlib
import json
import math
import os
import struct
import sys
import time


def main():
    if len(sys.argv) < 4:
        print("Usage: bench.py <model_name> <gguf_path> <llama_cpp_dir>", file=sys.stderr)
        sys.exit(1)

    model_name = sys.argv[1]
    gguf_path = sys.argv[2]
    llama_cpp_dir = sys.argv[3]

    try:
        from llama_cpp import Llama
    except ImportError:
        print("[llama-cpp] llama-cpp-python not installed, trying pip install...", file=sys.stderr)
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python", "--quiet"],
            stdout=sys.stderr, stderr=sys.stderr,
        )
        from llama_cpp import Llama

    seq_len = 128

    # --- Load model ---
    print(f"[llama-cpp] loading {gguf_path}...", file=sys.stderr)
    t0 = time.perf_counter()
    llm = Llama(
        model_path=gguf_path,
        n_ctx=seq_len + 1,
        logits_all=True,
        verbose=False,
    )
    compile_s = time.perf_counter() - t0
    print(f"[llama-cpp] loaded in {compile_s:.2f}s", file=sys.stderr)

    # --- Deterministic input (same as other frameworks) ---
    vocab_size = llm.n_vocab()
    input_ids = list(range(seq_len))
    labels = [(i + 1) % vocab_size for i in range(seq_len)]

    # --- Forward (inference) ---
    print(f"[llama-cpp] running forward pass (seq_len={seq_len})...", file=sys.stderr)
    llm.reset()
    t0 = time.perf_counter()
    llm.eval(input_ids)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # --- Collect logits ---
    # llama-cpp-python stores logits for all tokens after eval.
    import ctypes
    import numpy as np

    n_vocab = llm.n_vocab()
    # Get logits for all positions.
    all_logits = []
    for i in range(seq_len):
        row_ptr = llm._ctx.get_logits_ith(i)
        row = np.ctypeslib.as_array(
            ctypes.cast(row_ptr, ctypes.POINTER(ctypes.c_float)),
            shape=(n_vocab,),
        ).copy()
        all_logits.append(row)
    logits_np = np.stack(all_logits)  # (seq_len, vocab_size)

    # --- Compute loss (cross-entropy, matching PyTorch) ---
    # Loss = mean of -log(softmax(logits)[label]) over all positions.
    total_loss = 0.0
    for i in range(seq_len):
        row = logits_np[i].astype(np.float64)
        max_val = row.max()
        log_sum_exp = max_val + math.log(np.exp(row - max_val).sum())
        total_loss += -(row[labels[i]] - log_sum_exp)
    loss = total_loss / seq_len

    # --- Logits hash & sample ---
    logits_flat = logits_np.astype(np.float32).flatten()
    raw = struct.pack(f"<{logits_flat.size}f", *logits_flat.tolist())
    logits_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
    logits_sample = [round(float(v), 6) for v in logits_flat[:16]]

    # --- Latency (single-token forward) ---
    llm.reset()
    llm.eval([0])  # warm up with 1 token
    llm.reset()
    t0 = time.perf_counter()
    llm.eval([0])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # --- GPU info ---
    gpu_name = "cpu"

    # Extract llama.cpp git rev if available.
    rev_file = os.path.join(llama_cpp_dir, ".git", "refs", "heads", "master")
    framework_rev = ""
    if os.path.isfile(rev_file):
        framework_rev = open(rev_file).read().strip()[:7]

    result = {
        "framework": "llama-cpp",
        "framework_rev": framework_rev,
        "model": model_name,
        "device": gpu_name,
        "gpu_name": gpu_name,
        "backend": "CPU",
        "timings": {
            "compile_s": round(compile_s, 2),
            "inference_ms": round(inference_ms, 3),
            "latency_ms": round(latency_ms, 3),
            "train_ms": 0.0,
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": logits_sample,
            "loss": round(float(loss), 6),
        },
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
