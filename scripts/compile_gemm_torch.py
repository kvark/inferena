#!/usr/bin/env python3
"""f32 GEMM with runtime dimensions, matching native tile/K/warp counts."""

from dataclasses import asdict
import hashlib
import json
import os
import platform
import sys
import time

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver

from p3hpc import PYTHON_VERSION, check_torch_identity


@triton.jit(do_not_specialize=["M", "N", "K"], do_not_specialize_on_alignment=["M", "N", "K"])
def gemm(A, B, C, M, N, K, TILE: tl.constexpr):
    rows = tl.program_id(1) * TILE + tl.arange(0, TILE)
    cols = tl.program_id(0) * TILE + tl.arange(0, TILE)
    inner = tl.arange(0, 32)
    acc = tl.full((TILE, TILE), 0, tl.float32)
    for block in range(tl.cdiv(K, 32)):
        reduction = block * 32 + inner
        a = tl.load(A + rows[:, None] * K + reduction[None, :],
                    (rows[:, None] < M) & (reduction[None, :] < K), 0)
        b = tl.load(B + reduction[:, None] * N + cols[None, :],
                    (reduction[:, None] < K) & (cols[None, :] < N), 0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(C + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


def inputs(elements, operand, scale):
    bits = (torch.arange(elements, dtype=torch.int64) + (0x9E3779B9 * (operand + 1) & 0xFFFFFFFF)) & 0xFFFFFFFF
    bits ^= bits >> 16
    bits = (bits * 0x85EBCA6B) & 0xFFFFFFFF
    bits ^= bits >> 13
    return ((bits >> 8).float() / 16777216.0 - 0.5) * scale


def main():
    m, n, k, tile = map(int, sys.argv[1:])
    assert min(m, n, k) > 0 and tile in (32, 64)
    assert (m*k + k*n + m*n) * 4 <= 64 * 1024**2
    assert platform.python_version() == PYTHON_VERSION
    check_torch_identity(torch.__version__, torch.version.git_version, torch.__version__)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    context_start = time.perf_counter_ns()
    torch.cuda.init()
    driver.active.get_current_target()
    context_ns = time.perf_counter_ns() - context_start
    a, b, c = (torch.empty(shape, device="cuda", dtype=torch.float32) for shape in ((m, k), (k, n), (m, n)))
    torch.cuda.synchronize()
    records, loads, starts = [], [], {}

    def compiled(*, src, metadata, metadata_group, times, cache_hit):
        records.append({"name": src.name, "metadata": metadata, "cache_hit": cache_hit,
                        "times_us": asdict(times), "total_us": times.total})

    def load_start(module, function, name, metadata_group, key):
        starts[key] = time.perf_counter_ns()

    def load_end(module, function, name, metadata_group, key):
        loads.append({"name": name, "duration_ns": time.perf_counter_ns() - starts.pop(key)})

    knobs.compilation.listener = compiled
    knobs.runtime.kernel_load_start_hook.add(load_start)
    knobs.runtime.kernel_load_end_hook.add(load_end)
    warmup = None
    if os.environ.get("INFERENA_GEMM_WARM_COMPILER") == "1":
        warmup_start = time.perf_counter_ns()
        warmup_tile = 96 - tile
        warmup_kernel = gemm.warmup(a, b, c, m, n, k, TILE=warmup_tile,
                                   grid=(triton.cdiv(n, warmup_tile), triton.cdiv(m, warmup_tile)),
                                   num_warps=8, num_stages=1)
        warmup_kernel._init_handles()
        assert not starts and len(records) == 1
        warmup = {"prepare_ns": time.perf_counter_ns() - warmup_start,
                  "compilations": records[:], "loads": loads[:]}
        records.clear(); loads.clear()
    grid = (triton.cdiv(n, tile), triton.cdiv(m, tile))
    prepare = time.perf_counter_ns()
    kernel = gemm.warmup(a, b, c, m, n, k, TILE=tile, grid=grid, num_warps=8, num_stages=1)
    compiled_ns = time.perf_counter_ns() - prepare
    kernel._init_handles()
    prepare_ns = time.perf_counter_ns() - prepare
    validation = []
    for scale in (1.0, 1.0e-12):
        cpu_a = inputs(m*k, 0, scale).reshape(m, k)
        cpu_b = inputs(k*n, 1, 1.0).reshape(k, n)
        a.copy_(cpu_a); b.copy_(cpu_b); c.fill_(float("nan"))
        gemm[grid](a, b, c, m, n, k, TILE=tile, num_warps=8, num_stages=1)
        result = c.cpu().double()
        reference = cpu_a.double() @ cpu_b.double()
        error = (reference - result).abs()
        bound = float(torch.tensor(scale, dtype=torch.float32)) * 1.0e-5 + reference.abs() * 2.0e-4
        failures = (~torch.isfinite(result) | (error > bound)).sum().item()
        output_hash = hashlib.sha256(result.float().numpy().astype("<f4", copy=False).tobytes()).hexdigest()
        validation.append({"scale": scale, "failures": failures, "max_abs_error": error.max().item(),
                           "elements": m*n, "output_hash": "sha256:" + output_hash})
    assert not starts and len(records) == 1, "unexpected new specialization or pending load"
    print(json.dumps({"engine": "triton", "shape": [m, n, k], "tile": tile,
                      "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                      "torch_source": torch.version.git_version, "triton": triton.__version__,
                      "context_ns": context_ns, "warmup": warmup,
                      "prepare_ns": prepare_ns, "compile_ns": compiled_ns,
                      "launcher_and_load_ns": prepare_ns - compiled_ns,
                      "dimensions": "runtime, no value/alignment specialization",
                      "compilations": records, "loads": loads, "validation": validation}, default=str))
    if any(row["failures"] for row in validation):
        raise SystemExit("full-output f64 qualification failed")


if __name__ == "__main__":
    main()
