#!/usr/bin/env python3
"""llama.cpp benchmark wrapper for inferena.

Runs llama-bench for forward-only (prompt processing) performance and
outputs JSON matching the inferena BenchResult schema.
Backward pass is not supported (inference-only framework).
"""

import hashlib
import json
import os
import struct
import subprocess
import sys
import time


def main():
    if len(sys.argv) < 4:
        print("Usage: bench.py <model_name> <gguf_path> <llama_cpp_dir>", file=sys.stderr)
        sys.exit(1)

    model_name = sys.argv[1]
    gguf_path = sys.argv[2]
    llama_cpp_dir = sys.argv[3]

    llama_bench = os.path.join(llama_cpp_dir, "build", "bin", "llama-bench")
    if not os.path.isfile(llama_bench):
        print(f"[llama-cpp] llama-bench not found at {llama_bench}", file=sys.stderr)
        sys.exit(1)

    seq_len = 128

    # Run llama-bench: prompt processing (pp) measures forward pass speed.
    # -p <seq_len>: prompt length
    # -n 0: don't generate tokens (forward only)
    # -r 1: 1 repetition (we measure a single pass)
    print(f"[llama-cpp] running llama-bench on {gguf_path}...", file=sys.stderr)

    t0 = time.perf_counter()
    proc = subprocess.run(
        [llama_bench,
         "-m", gguf_path,
         "-p", str(seq_len),
         "-n", "0",
         "-r", "1",
         "-o", "json"],
        capture_output=True, text=True,
    )
    wall_time = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"[llama-cpp] llama-bench failed: {proc.stderr[:500]}", file=sys.stderr)
        sys.exit(1)

    # Parse llama-bench JSON output.
    try:
        bench_results = json.loads(proc.stdout)
        if isinstance(bench_results, list) and len(bench_results) > 0:
            entry = bench_results[0]
        else:
            entry = bench_results
    except json.JSONDecodeError:
        print(f"[llama-cpp] failed to parse JSON: {proc.stdout[:500]}", file=sys.stderr)
        sys.exit(1)

    # Extract timing. llama-bench reports t/s (tokens per second for prompt).
    # Forward time = seq_len / (tokens_per_second) * 1000 ms.
    pp_ts = entry.get("avg_ts", entry.get("t/s", 0))
    if pp_ts > 0:
        forward_ms = seq_len / pp_ts * 1000.0
    else:
        forward_ms = wall_time * 1000.0

    gpu_name = entry.get("gpu_info", "cpu")

    result = {
        "framework": "llama-cpp",
        "model": model_name,
        "device": gpu_name,
        "gpu_name": gpu_name,
        "backend": entry.get("type_k", "CPU"),
        "timings": {
            "compile_s": 0.0,
            "forward_ms": round(forward_ms, 3),
            "backward_ms": 0.0,
        },
        "outputs": {
            "logits_hash": "",
            "logits_sample": [],
            "loss": 0.0,
        },
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
