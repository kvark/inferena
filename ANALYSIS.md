# Native gap analysis — September 9–10

The evidence separates inefficient resident kernels from weight-representation
and memory-placement costs. It does **not** establish a removable barrier-cost percentage.
September 9's post-OOM work was offline. September 10 resumed bounded captures
after the author's cache reclaim restored host-memory headroom. Diagnostic
and unprofiled performance runs remain separate from the collection tag.

## Qualified Systems evidence

Source `experiment/p3hpc-scaling-2026-09-08` (`c7d9e79`), Meganeura `43b606ff`,
RTX 5070 / 595.71.05, Nsight Systems 2026.4.1. Both saved pairs use strict f32,
**default compilation plus explicit whole-phase CUDA Graphs**, not max-autotune.
They cannot be subtracted from the replicated max-autotune table in EXPERIMENT.
All measured CUDA kernels have graph-node IDs; the paired runner's numerical,
source and capture gates passed. These are three diagnostic calls per phase.

Run `python scripts/nsys_report.py <capture>/pytorch.sqlite` and repeat for
`meganeura.sqlite`. The report excludes preparation/warmup, counts actual samples
and checks GPU events fall inside them. Mean summed kernel durations below are
not wall-time components when kernels overlap.

| PyTorch workload / kernel | Launches/call | GPU ms/call | Registers/thread |
|---|---:|---:|---:|
| 1.7B prefill: CUTLASS SIMT SGEMM 256×128, K tile 8 | 73 | 19.252 | 212 |
| 1.7B prefill: CUTLASS SIMT SGEMM 128×64, K tile 8 | 96 | 7.435 | 130 |
| 1.7B token: cuBLAS `gemv2T_kernel_val` | 169 | 11.126 | 64 |
| ResNet F+L+B: `convolve_common_engine_float_NHWC`, 128 variant | 15 | 2.821 | 76 |
| ResNet F+L+B: cuDNN `dgrad_engine`, 512 variant | 44 | 2.473 | 95 |
| ResNet F+L+B: `wgrad_alg0_engine_NHWC`, 128 variant | 28 | 1.399 | 81 |

These are named kernel variants, not totals for every convolution or gradient.
The two SGEMMs contribute about 96% of the prefill kernel-duration sum; GEMV
contributes about 97% for a token. The strict reference is using scalar-f32
matrix kernels here, not simply winning by substituting reduced-precision
tensor-core arithmetic. Register counts alone do not establish occupancy or spills.

For ResNet training, Meganeura's host `step` averages 4.275 ms, its wait
41.880 ms, and grouped GPU work 41.622 ms. PyTorch's host sample averages
14.867 ms. For 1.7B token, native host `step` is 2.496 ms versus 1951.810 ms
of grouped GPU work. CPU encoding alone cannot explain either gap. Wait is
elapsed synchronization, not busy CPU time; no subtraction here isolates barriers.

## Resident-kernel target: convolution weight gradients

The earlier qualified dispatch sidecar at `experiment/p3hpc-methodology-2026-09-08`
uses the same Meganeura revision but a different, per-pass instrumentation
contract. Its ResNet training profile has two samples, 512 dispatches and 273
barrier groups; instrumentation expands wall time by about 1.17×. Use it to
rank candidates, not to subtract against the Systems table above.

| Native pipeline | Dispatches | Sum of pass medians | Share |
|---|---:|---:|---:|
| `Conv2dGradWeightGemmSmall:scalar` | 12 | 15.593 ms | 32.2% |
| `Conv2dGradInputGemm:scalar` | 47 | 11.101 ms | 22.9% |
| `Conv2dGradWeightGemm:scalar` | 41 | 9.559 ms | 19.7% |

The slowest dispatch, #508 / origin 826, is the stem weight gradient:
`[64, 147]` output, reduction length `4×112×112 = 50176`. Its 32×32 output
tiles launch only `[5,2,1]` workgroups. The shared shader advances K by 16,
so each group executes 3136 iterations with two workgroup barriers each.
Ten groups cannot occupy all 48 SMs simultaneously. This is a concrete
parallelism limit, **not** a measured fraction of time lost to barriers.
Its pass interval is 4.475 ms, about 9.2% of the profile sum.

The qualified Systems trace's final cuDNN `wgrad_alg0_engine_NHWC` launch
(graph 5, node 21474837191) uses grid `[5,2,100]`, block `[8,8,1]`: **1000 blocks**,
81 registers/thread and 2304 shared bytes/block. Its mean kernel interval is
0.109 ms across three calls. Matching it to the native stem is an inference
from ordering and geometry, not retained operator/shape correlation. Replay
traces do not contain those CPU operator ranges. The 100× block-count difference
is evidence of different launch geometry, not a 100× speedup or a paired timing
ratio; the native pass profile uses different instrumentation. Any cuDNN
conversion/initialization/reduction kernels must also be charged to its operation.
Use `scripts/nsys_report.py <capture>/pytorch.sqlite --launches --top 1000`
to inspect individual graph nodes instead of aggregating a shared kernel name.

The general candidate is split reduction plus a final sum, using the existing
bounded sequence search and plan-before-allocation lowering. This is **not a
new discovery or a ready promotion**: the September 6 experiments already found
unsplit-control and split-partial accuracy failures on long, tiny-gradient
cases, and the compensated attempt failed 10/240 rows. Preserve those gates.
First resolve accumulation accuracy, then measure a few legal split counts,
charge all partial storage and the final pass, and confirm whole-step gains.
No ResNet/card-name rule, giant sweep or relaxed validation is warranted.
[Existing experiment conclusions](https://github.com/kvark/meganeura/blob/43b606ff45b99d23e7e57b5f120f5fc347039082/docs/experiments.md).

The source-only [CPU arithmetic diagnostic](https://github.com/kvark/meganeura/blob/1224323/bench/dw_rounding.rs)
narrows that accuracy question. On the old long uneven-K, tiny structured-input
case, f32 FMA dot products inside 16-term tiles already cause relative L2 error
`3.491e-3` even with an f64 outer sum, versus the unchanged `2e-4` gate. Rounding
f64-evaluated split partitions once passes all four structured shapes. Thus
outer compensation alone need not fix inaccurate inner subtotals; inspect
inner accumulation before blaming f32 partial storage or relaxing the gate.
This is a CPU arithmetic model of a subset, not the full GPU qualification or
proof of the historical GPU failure's instruction-level cause. Disabling FMA
is not a generally justified fix. No new shader or performance result is promoted.

## Weight representation and placement first for 1.7B

Header-only accounting of the pinned checkpoints explains almost all the native
token plan: SwiGLU packing retains both original gate/up matrices **and** their
horizontal-concatenation copy. The strict cohort expands weights to f32; this
table counts tensor shapes × 4 bytes, not the checkpoint's BF16 file size.

| SmolLM2 | Original f32 weights (MiB) | Extra packed copy (MiB) | Recorded token plan (MiB) |
|---|---:|---:|---:|
| 135M | 513.134 | 202.500 | 715.849 |
| 360M | 1380.238 | 600.000 | 1980.472 |
| 1.7B | 6528.383 | 3072.000 | 9600.703 |

Reproduce without loading tensors: sum `product(shape) * 4` over the safetensors
header, then separately over `.mlp.gate_proj.` and `.mlp.up_proj.` tensors.
Their sum is within 0.321 MiB of `memory.phases.latency.allocated_bytes` in each
saved `smollm-strict/measurement/r3/strict/<model>/max-autotune-graph1` native row.
For 1.7B, `24 layers × 2 × 2048 × 8192 × 4 bytes = 3 GiB` of extra storage.
The tied embedding/head is counted once. This is static byte accounting,
**not** a measured unpacked speedup or proof that packing alone causes fallback.

The [optimizer](https://github.com/kvark/meganeura/blob/43b606ff45b99d23e7e57b5f120f5fc347039082/src/optimize.rs)
creates the derived parameter, but `sweep_dead_nodes` deliberately keeps original
parameters for named access. Compilation allocates all of them and the memory
planner pins them. `set_parameter` writes both the original and its packed slice;
named reads and updates still require the original contract. Simply deleting
unconsumed source buffers is therefore not a safe dead-code optimization.
Treat packed/unpacked storage as legal representation choices: include persistent
bytes, preparation and actual placement in an eventual measured comparison.
Eliminating duplicate backing must preserve named reads/updates and external
buffer contracts, not just the forward output.

Systems confirms roughly 4.9 GiB of native bindings on the non-device-local
host-visible/coherent/cached heap despite a roughly 9.4 GiB plan. This is not
an all-VRAM scaling point. `Shared` permits fallback; the budget preflight
does not guarantee placement. The September 10 experiment below controls
placement with explicit bounded staging, retaining named parameter access,
host-visible inputs and external-buffer contracts. Unified-memory backends
still need qualification; do not impose a discrete-VRAM interpretation on them.

Only after placement is controlled should resident matmul/GEMV tile, coalescing,
occupancy and barrier comparisons be interpreted as kernel scaling. Keep any
fallback result separately labelled; do not silently exclude it or reduce f32 weights.

### Placement and allocation ablation — September 10

The source-only `experiment/parameter-allocation-2026-09-10` tags in Inferena,
Meganeura and Blade distinguish memory preference from allocator strategy.
Inferena `b9ef2d5` pins Meganeura `2e71371` and Blade `ae46130`. Parameters can
request `Device` (buddy) or `DeviceTransient` (free list), with at most 16 MiB
of upload staging. Original/derived parameter storage and all kernels remain
unchanged. Optional streaming reads one checkpoint tensor at a time, retaining
the existing conversion/transposition code. The resident/streaming 135M and
360M controls have identical recorded output fields. Existing broad smoke,
device-local and cross-session parameter-lifetime tests pass; no new test file.

The four-arm pilot separates two effects:

| Strict f32, five warmups / twenty samples | Shared buddy | Shared free list | Device buddy | Device free list |
|---|---:|---:|---:|---:|
| 360M process device bytes, GiB | 3.137 | 2.176 | 3.145 | 2.239 |
| 1.7B host-heap plan bindings, GiB, prefill | 4.930 | 5.195 | 0 | 0 |
| 1.7B prefill, ms | 955.291 | 968.235 | 54.629 | 54.783 |
| 1.7B stateless token, ms | 1937.933 | 1979.101 | 16.387 | 16.433 |

These are **single-process diagnostic pilots, not replicated performance
estimates**. CPU setup/allocation tracing is on; the small-model pilot's trace
was written on NFS and must not be treated as publication timing. All recorded
outputs, including full prefill hashes, match exactly across each model's arms.
For 360M, the free list largely removes buddy rounding without a runtime gain.
For 1.7B, both allocators already have negligible rounding: changing placement
puts the same 10,108,674,560-byte prefill plan entirely on the device heap.
Thus rounding and host fallback are different problems. `Device` remains a
backend preference; actual binding properties, not its name, establish residency.
Run `scripts/allocation_report.py <completed-study>` for per-plan heap accounting.
Sequential prefill/token allocations must not be added as simultaneous residency.

An untraced confirmation completed **one** control/candidate pair:
951.234→54.887 ms prefill and 1941.822→16.282 ms token, with the same full hash.
It was stopped during the next candidate as a precaution, not a failed numerical
gate. Its manifest remains `incomplete`; do not report six replicates. The
Shared controls emitted NVIDIA `dmaAllocMapping_GM107` / `NV_ERR_NO_MEMORY`
mapping failures. A [first-hand driver report](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1270)
describes the same messages preceding an unrecoverable lock in a different
workload on this driver/architecture. That does not identify our exact driver
bug, but deliberately repeating mapping exhaustion is not warranted. No Xid,
watchdog, host OOM, GPU reset or reboot occurred in these bounded runs.
The card reports a 16 GiB BAR1 aperture, not a legacy 256 MiB aperture; free
global BAR1 bytes alone do not establish which internal mapping can succeed.

### Fresh resident-only Systems pairs

Inferena `111579d` pins the same Meganeura with Blade `7b6d97a`'s allocation
spans. Both 135M and 1.7B pairs pass the unchanged cross-engine and full CUDA
Graph replay gates, with strict f32/default compilation. Native uses device
parameters and streamed checkpoints. PyTorch uses the ordinary Transformers
single-device loader with `device_map={"": device}` and f32, avoiding a whole
CPU-resident f32 model. This experiment-only option needs Accelerate 1.15.0;
135M's 272 parameters and both buffers match the CPU-then-device loader bitwise.
There is no quantization or CPU offload. The collection environment is unchanged.

| Three-call diagnostic means, ms | 135M prefill | 135M token | 1.7B prefill | 1.7B token |
|---|---:|---:|---:|---:|
| Meganeura host sample | 14.295 | 2.525 | 55.532 | 22.199 |
| Meganeura host encoding (`step`) | 3.253 | 0.533 | 2.557 | 2.317 |
| Meganeura grouped GPU interval | 10.769 | 1.877 | 52.672 | 19.448 |
| PyTorch host sample | 6.228 | 1.922 | 28.395 | 11.679 |
| PyTorch summed CUDA kernel intervals | 5.951 | 1.622 | 27.908 | 11.488 |

All measured CUDA kernels have graph-node IDs. In 1.7B, the two SIMT SGEMM
families total 26.818 ms per prefill; cuBLAS GEMV totals 11.148 ms per token.
The remaining resident gap is mainly GPU execution, not host encoding or the
earlier host-memory fallback. Profiling inflates some intervals; retain untraced
confirmation for performance claims. These rows still do not isolate barriers.

Setup profiling also rejects a premature row-copy explanation: 135M parameter
preparation takes 4.55 / 4.40 seconds for its two sequential plans, of which
`vkAllocateMemory` plus `vkFreeMemory` take 3.74 / 3.74 seconds. Summed GPU copy
intervals are only about 41 ms per plan. There are 272 upload allocations per
plan. Reusing bounded staging is the next controlled experiment; API allocation
durations are measured directly, not inferred by subtracting GPU time.

The 1.7B pair finished in 109 seconds inside a 3 GiB, swap-disabled cgroup;
global `MemAvailable` stayed above 4989 MiB. Driver-pinned pages can escape
cgroup accounting, so `scripts/limited.py` can additionally sample global RAM
and stop its own process group below a requested floor. Each Systems launch
logged one `refcntRequestReference_IMPL` status `0x56`
([`NV_ERR_NOT_SUPPORTED`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/595.71.05/src/common/sdk/nvidia/inc/nvstatuscodes.h)),
but neither resident capture logged the earlier mapping-allocation failures.
The measured traces and full runner output are complete; do not claim a
warning-free driver. Local results live under `nsys-resident-*-20260910`,
`parameter-allocation-1.7b-20260910-pilot` and the explicitly incomplete
`parameter-placement-1.7b-20260910-confirm` in `/x/Code/inferena-results/`.

### Bounded upload reuse and CPU transposition

The `experiment/upload-reuse-2026-09-10` tags retain the allocation ablation
and qualified Systems follow-up. One lazily grown upload buffer, capped at
16 MiB, survives between writes and is destroyed with its session. Every copy
still completes before its staging bytes can be reused. Existing smoke tests
pass (83 active), as do device-local and cross-policy shared-parameter lifetime
checks; no new test file was added. In the 135M trace, upload allocations drop
272→1 per plan: preparation is 588 / 562 ms, tensor preparation 468 / 444 ms,
and uploads 96 / 95 ms. GPU copy intervals remain about 39 ms. The new source
separates those CPU ranges directly; run `scripts/nsys_report.py <native.sqlite>
--setup`. Absent spans in an older trace are reported as unrecorded, not zero.

The 1.7B reuse trace likewise makes one upload allocation per plan. Of 16.54 /
16.81 s preparation, tensor reading/conversion/transposition takes 15.22 /
15.48 s, while upload takes 1.01 s each. This motivated a separate cache-blocked
CPU transpose, without changing arithmetic, decoded bytes or device layout.
At Meganeura `experiment/checkpoint-transpose-2026-09-10`, compile
`rustc --edition=2024 -O bench/transpose.rs -o /tmp/transpose-study`, then run
the binary with a new output CSV path inside a bounded scope. It compares
unchanged row-major traversal with 16/32/64 tiles; all 2,259,756,320 copied
elements match bitwise, including non-finite patterns and edge shapes. This is
one process with rotated trials, not four independent process replicates.
Tile 16 is the conservative full-model candidate, not a universal CPU optimum.

Inferena `experiment/parameter-preparation-2026-09-10` (`9464cc3`) pins
Meganeura `854b5b6` and confirms three **resident-only** arms across six fresh
processes per model/arm, using all six execution orders. Strict f32, streamed
weights, five warmups/twenty samples, no profiler, ordinary warm driver caches:

| Median complete runner process, seconds | Fresh upload allocations | Reused upload buffer | Reuse + CPU tile 16 |
|---|---:|---:|---:|
| 135M | 10.537 | 2.939 | 2.625 |
| 360M | 14.164 | 5.718 | 4.521 |
| 1.7B | 43.163 | 36.179 | 28.167 |

These include both sequential plans, loading, warmup, measurement and cleanup;
they are **not** `compile_s` or inference speedups. Paired process savings for
reuse are 7.601 / 8.395 / 6.990 s, versus twice MAD 0.151 / 0.226 / 0.172 s.
Tiling adds 0.314 / 1.221 / 7.990 s savings versus 0.039 / 0.079 / 0.260 s noise.
All 54 processes complete and all recorded output fields repeat exactly.
Steady-state prefill remains about 12.7 / 21.7 / 54.5 ms, and no step gain
clears the 5%-plus-noise guard. The existing comparison tolerances are unchanged.
The 15-minute study stayed above 8868 MiB globally available inside a 3 GiB,
swap-disabled cgroup, with no new kernel log entries. Process records are under
`/x/Code/inferena-results/parameter-preparation-20260910-confirm`.

Reproduce with `scripts/tune_study.py --stream-weights --baseline
device-params-buddy --variants device-params-buddy device-params-reuse
device-params-tiled --models SmolLM2-135M SmolLM2-360M SmolLM2-1.7B
--replicates 6 --output <new-dir>`. Use the same memory wrapper and global
floor as the resident Systems recipe. These environment-driven prototypes
remain source-only experiments, not a new public configuration contract or a
change to the collection tag.

## Packing, layout and warmup controls — September 10

The source-only `experiment/packing-layout-2026-09-10` tag pins Meganeura
`3024539` and Blade `7b6d97a`. All arms stream weights into device-buddy
allocations, reuse bounded upload staging and use CPU transpose tile 16.
The six-arm, five-model pilot passes the ordinary forward/gradient gates.
Interleaving scalar-matmul output columns is slower (about 8–18% for the
SmolLM2 prefills); it is not a promoted bank-layout fix. It separately passes
21 CPU shader checks, the existing rectangular tile qualification, 83 broad
smoke checks and 599,492 full-f64 GEMM output checks, including tiny inputs.

Disabling the greedy SwiGLU packed-weight copy leaves original parameters
intact. Prefill plan requests fall from 742→540 MiB (135M), 2009→1409 MiB
(360M) and 9640→6567 MiB (1.7B). The 1.7B process-driver peak falls from
9704→6632 MiB. This is a graph representation choice, not allocator rounding.
The runtime can still horizontally group the unpacked prefill projections.

Six three-arm fresh-process repetitions (untuned / default tuning / unpacked,
all six orders; 5 warmups, 20 samples) preserve all recorded output fields and
the full stateless-token hash. Default tuning again improves 135M prefill
12.689→11.596 ms, but changes no 360M/1.7B tile. Unpacking changes 1.7B token
latency 16.258→14.520 ms; it does not establish a small-model latency gain.
Increasing tuning scratch to 256 MiB in the pilot admits another 1.7B class,
which retains its baseline. Packed-up and vocabulary classes still exceed
that budget; GEMV remains outside the current tuner. This is not exhaustive
evidence against larger-model tuning.

The source-only `experiment/packing-warmup-2026-09-10` tag (`ab60d02`) repeats
packed/unpacked with **100 warmups and 100 samples**, six AB/BA process pairs
per model. All full prefill and token hashes still match exactly. The 1.7B
token gain survives: 16.316→14.558 ms, 1.121×, median paired gain 1.772 ms
versus twice its MAD 0.033 ms. Prefill stays about 54.8 ms. The other models
do not clear the whole-step guard. No default or collection revision changes.

This control also exposes a timing transient: ordinary 135M token latency is
about 2.56 ms in the short protocol versus 3.58 ms after sustained execution;
prefill changes little. Some short 360M-unpacked windows drift from about
3.9 to 5.5 ms; long-warmup quarters stay near 5.47 ms. Do not call the fastest
short-window result steady-state throughput. Clock/CPU/GPU correlation is a
separate diagnostic; these observations alone do not identify its cause.
All 36 longer runs finish with at least 9066 MiB globally available and no new
kernel warnings. The earlier 54-run confirmation also retains every sample.

Reproduce with `scripts/tune_study.py --resident --stream-weights --models
SmolLM2-135M SmolLM2-360M SmolLM2-1.7B --variants untuned unpacked
--replicates 6 --warmup-runs 100 --measurement-runs 100 --output <new-dir>`,
under the same 3 GiB scope and global floor. Add `default` and use the earlier
5/20 counts to reproduce the three-arm confirmation. `--profile` is a separate
serialized-pass diagnostic, not another set of publication timings.

### Host-side latency transient

The source-only `experiment/host-latency-2026-09-10` tag retains the diagnosis.
A qualified Systems pair with 100 warmups/100 samples puts native 135M token
GPU time at 1.877 ms, versus 1.873 ms in the earlier short capture. Host
`step` grows from about 0.59 to 1.75 ms. The GPU stays at 2895 MHz SM /
14001 MHz memory throughout the measured token window, with no clock-event
flags. This growth is on the host, not a growing GPU barrier cost.

Separate unprofiled-by-Nsight runs add opt-in Linux thread-CPU clocks and
read-only `scaling_cur_freq` snapshots (`--host-trace`). Three resident runs
show command recording changing from about 0.59 ms at 2.7–3.2 GHz to
1.44–1.48 ms at 800 MHz. Thread CPU time matches elapsed recording time;
descheduling does not account for the increase. Three Shared-allocation
controls reproduce the downclock. Three fixed-core runs and three per-task
utilization-hint runs do not prevent it. These diagnostics are not paper
timings or a paired comparison of scheduling policies.

The host uses active `intel_pstate`, `powersave`, `balance_performance`,
800–4400 MHz, with HWP dynamic boost disabled. Frequency snapshots are not
instruction-level measurements; see the kernel's [frequency interface](https://www.kernel.org/doc/html/latest/admin-guide/pm/cpufreq.html)
and [Intel policy documentation](https://www.kernel.org/doc/html/latest/admin-guide/pm/intel_pstate.html).
No governor, EPP, driver setting or global clock limit was changed. Affinity
and utilization hints affect only the diagnostic child and disappear on exit.
Nsight CPU sampling is unavailable under the current perf paranoid level 4.
The actual thread clock avoids guessing busy CPU time from GPU subtraction.

Reproduce with the preceding resident command, `--variants untuned --replicates 3
--host-trace`; compare explicit `--cpu 2` and `--cpu 2 --cpu-util-min 1024`
controls on an allowed core. Omit both `--resident` and `--stream-weights` for
the small Shared control; do not extend that control to 1.7B. The diagnostic
records all warmup/measurement samples and CPU IDs. Even 100 warmups cannot
guarantee a fixed CPU performance state: the later GEMV pilot contains a
mid-window acceleration. Collection must disclose power policy and distinguish
short-window latency from sustained execution, not cherry-pick the faster state.

### Token Graphics source correlation

The `experiment/native-token-graphics-2026-09-10` tag adds an explicit NGFX SDK
phase trigger. A strict resident 135M capture starts after 100 token warmups,
retains three submissions in a 7.17 ms trace, and preserves the complete
prefill and token hashes. The labelled GPU intervals are 2.12–2.14 ms; these
instrumented durations are not ordinary benchmark samples. No hardware-event
overflow is reported. Actual PMA is 400 MB; profiler-tree peak is 1379 MiB,
with at least 8349 MiB globally available. The SDK also passes an initial
capture without shader debug information; use the labelled follow-up for
source attribution.

PC sample shares are 47.52% residual-add GEMV, 40.98% RMSNorm-fused GEMV and
11.09% transposed GEMV. Their workgroups use 32/256/32 threads, 18/22/72
registers per thread and 512/5136/128 shared bytes respectively. Aggregate SM
throughput is 6.53% of peak and DRAM throughput 33.97%. Long-scoreboard and
workgroup-barrier occupancy are 17.58% and 3.89% of peak warp slots; active
compute warps occupy 29.22%. These are **not fractions of removable wall time**,
and workgroup barriers are not Vulkan resource barriers. The profile suggests
latency-hiding/memory-access candidates, not a measured API ceiling.

The preceding `experiment/gemv-width-2026-09-10` pilot varies only plain and
RMSNorm-fused f32 GEMV (32/64/128/256 threads), keeping residual-add and BT at
32. Its 24 processes cover three SmolLM2 sizes, packed and unpacked, 100/100
counts. All prefill records match exactly; full token/prefill-prefix L2 is at
most 1.64e-5 despite changed reduction order. Existing shader/parity/smoke
checks and 217,792 full-f64 output checks pass. No broad width win emerges;
the unusually fast 135M/64 run drifts within its window. Residual-add GEMV,
the largest PC-sample family, needs its own measured width experiment.

## Residual-add GEMV width — September 10

The source-only Inferena/Meganeura `experiment/gemv-add-width-2026-09-10` tags
pin Inferena `054db61`, Meganeura `2c8b5df` and Blade `7b6d97a`. They reuse one
width generator for the residual-add f32 kernel as well as plain/RMSNorm GEMV.
`MEGANEURA_GEMV_ADD_THREADS=32|64|128|256` changes workgroup/reduction width;
other kernels, four outputs per workgroup and the grouped schedule stay fixed.
The prototype keeps 32 as default and fixes the environment per process. This
is an implementation ablation, not a production pipeline-key/selection API.

A 24-process pilot covers all four widths and packed/unpacked weights on all
three SmolLM2 sizes. A separate 54-process confirmation uses six replicates,
all six three-arm execution orders, 100 warmups and 100 retained samples,
strict f32, resident parameters, streamed loading, reused upload staging and
CPU transpose tile 16. No tracing or system power-policy changes. All processes
complete; minimum globally available RAM is 8711 MiB under the memory guard.

| Token latency, median ms | Original 32 | 128 | 128 + unpacked |
|---|---:|---:|---:|
| 135M | 3.586 | 2.984 | 2.912 |
| 360M | 5.368 | 4.7135 | 4.828 |
| 1.7B | 16.3135 | 15.5195 | 13.7375 |

Width alone clears the 5% plus paired-noise guard on 135M/360M, not 1.7B.
Its paired median gains / twice-MAD are 0.601/0.007, 0.6505/0.015 and
0.7945/0.031 ms. The combined arm clears the guard on all three; 1.7B gives
2.575/0.028 ms. Prefill has no material gain. Keep the raw quarter-window
variation: several 135M candidates drift, and one 1.7B control starts faster.
No run or sample is excluded, and the guard is not a confidence interval.

Every prefill output record is exact across arms. Token hashes change with
reduction order but repeat exactly within each variant; packed/unpacked agree
at equal width. Full token/prefill-prefix relative L2 is at most 1.6373e-5.
All four widths pass the existing 21 shader checks and 14 GEMV parity cases;
217,792 full-f64 output checks cover ordinary/tiny inputs. At 128, unpacked and
resident, the existing broad smoke suite passes 83 tests with two already
ignored. No tolerance relaxation or new regression fixture.

Reproduce with `scripts/tune_study.py --models SmolLM2-135M SmolLM2-360M
SmolLM2-1.7B --variants untuned gemv-add128 unpacked-gemv-add128 --replicates 6
--stream-weights --resident --warmup-runs 100 --measurement-runs 100
--output <new-dir>` under the documented memory guard. The pilot uses one
replicate and the additional `gemv-add64`, `gemv-add256`, `unpacked`,
`unpacked-gemv-add64` and `unpacked-gemv-add256` arms. Existing production
tuning excludes GEMV; bounded per-class selection remains the integration step.

### GPU localization of the accepted candidate

Qualified Systems pairs at the same source use 100/100 counts, default Torch
compilation and validated explicit CUDA Graphs. The 135M arm retains packing;
1.7B uses unpacked weights. Both use 128 residual-add threads. Native grouped
GPU intervals are 1.389 ms/token on 135M, versus 1.877 in the preceding 32-thread
sustained capture, and 12.163 ms/token on 1.7B. Host step spans are 1.071 and
2.545 ms, with total host samples 2.652 and 15.042 ms. Those are instrumented
elapsed intervals, not the unprofiled estimates above or additive CPU/GPU cost.
Native prefill GPU intervals remain 10.695/53.063 ms; the candidate affects
token GEMV, not prefill matmul.

Torch's 135M token kernel sum is 1.636 ms and its first-kernel-start to
last-kernel-end span is 1.673 ms, on one stream with no measured memcpy/memset.
Its host sample is 1.922 ms. This localizes an important remaining native host
cost; it does not establish an unprofiled end-to-end win. The report script now
distinguishes observed GPU span from summed event durations and refuses a
sample with no GPU events. Neither subtraction yields removable barrier cost.

On 1.7B, Torch's token kernel sum/span is 11.499/11.506 ms and host sample
11.723 ms. Its prefill kernel span is 28.904 ms. The improved native token path
is close in observed GPU time but still pays host command recording; prefill
retains a much larger GPU gap. Both statements concern strict f32, not the
accelerated-precision engines. A reusable-command path needs a real lifetime
and invalidation contract: Blade currently records one-time-submit Vulkan
command buffers, which must not simply be resubmitted.

The short 135M Graphics candidate follows the SDK-triggered recipe with
`MEGANEURA_GEMV_ADD_THREADS=128`. Complete prefill/token records match its
ordinary control; there is no reported hardware-event overflow. Three token
GPU intervals are 1.64–1.66 ms, versus 2.12–2.14 in the labelled 32-thread
capture. The residual-add pipeline still uses 18 registers/thread; shared
memory rises 512→2048 bytes and reported maximum resident warps 24→48.
Its PC-sample share falls 47.52→25.78%; RMSNorm-fused GEMV now has 57.90% and
BT 15.74%. The latter two implementations are unchanged.

Whole-trace peak-normalized active-compute warp occupancy rises 29.22→38.03%,
DRAM throughput 33.97→42.71%, and SM throughput 6.53→8.46%. Workgroup-barrier
occupancy also rises 3.89→6.29% while the GPU work gets faster. These are warp
state/throughput observations, **not wall-time percentages or Vulkan barrier
overpayment**. They support a latency-hiding explanation without uniquely
identifying a hardware bottleneck. Keep the verified early pipeline CSV fields;
the later ambiguous headers remain unsuitable for attribution.

Local records: `gemv-add-width-20260910-{pilot,confirm}`,
`nsys-gemv-add128-{135m,1.7b}-20260910` and `ngfx-token-add128-20260910` under
`/x/Code/inferena-results`. These are separate from collection/paper data.

## Qualified Graphics source correlation — September 10

The short resident-model captures now have complete runner output, matching
ordinary-control hashes/numerics, shader/pipeline source correlation and **no
hardware-event overflow**. Keep `--time-every-action` off and the production
grouped schedule. The source-only Inferena tag
`experiment/native-graphics-2026-09-10` pins Meganeura `8ec9f2a` and Blade
`cccf47a`; these add CPU compilation spans to the collection runtime, not new
kernels. `INFERENA_SHARED_CAPTURE_GPU=1` keeps one explicitly owned context
across phases for diagnostic capture only. Default execution is unchanged.

The 135M prefill trace covers three steps in 41.11 ms and contains one complete
GPU-projected measurement range. In that range, peak-normalized SM throughput
is 19.40%, active-warp occupancy 11.75%, barrier-stalled warp occupancy 0.82%,
and incoming PCIe traffic 0.031%. None is a wall-time barrier-cost fraction.
MatMul, MatMul+Add, horizontal MatMul and MatMulBT account for about 86% of
whole-trace shader PC samples; flash attention accounts for about 11%.
The 64-tile scalar matmul uses 91 registers/thread and 16,768 shared bytes.
Source correlation places many samples on shared-B loads in the inner loop.
This motivates layout/occupancy experiments, not an automatic barrier diagnosis.

The labelled ResNet training capture exits successfully, covers three steps
in 127.54 ms and contains a complete `meganeura/training/sample` GPU range of
42.09 ms. Whole-trace PC samples group as follows:

| Scalar convolution family, both tiles | Shader PC sample share |
|---|---:|
| Input gradient | 33.08% |
| Weight gradient | 38.52% |
| Forward | 21.69% |

These are **sample shares, not shares of wall time**. Whole-trace SM throughput
is 32.37% of peak. The sampled instruction mix attributes 22.28% of samples to
shared stores, 13.00% to FP32 FMA and 11.34% to integer FMA; these are not
dynamic instruction counts. This is evidence to investigate staging and
index calculation; counters alone do not prove which transformation will help.
The immutable-parameter ablation below tests that hypothesis with unchanged
arithmetic, buffers and schedule.

For this tool version, the pipeline CSV's early sample/register/shared-memory
columns agree with the UI, but some later headers do not align with their data.
Do not consume ambiguous stall columns. Use the source export and verified UI
metrics. [GPU Trace](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-ui.html),
[shader profiler semantics](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html).

The source-only Inferena/Blade `experiment/native-ir-2026-09-10` tags probe
`VK_KHR_pipeline_executable_properties` with the required internal-representation
capture flag. This driver advertises the feature but returns zero representations
for the tested f32 GEMM; both ordinary/tiny full-f64 output checks pass. It does
not supply native assembly through that query. The public Graphics source view
offers WGSL/SPIR-V correlation here, not SASS; avoid pretending the sampled mix
is disassembly or executed-instruction counts.

The shared-context Systems control also matches the original per-session
context's complete outputs and gradient norms. Its training host sample is
46.585 ms: host `step` 4.308 ms, `wait` 41.922 ms, grouped GPU work 41.539 ms.
Those nested/overlapping intervals must not be added or subtracted to invent
a CPU-utilization or barrier-overpayment metric.

## Immutable convolution parameters — September 10

The source-only `experiment/conv-specialization-2026-09-10` tags in Inferena
and Meganeura replace the scalar convolution's immutable u32 parameter uniform
with a WGSL constant struct. This permits ordinary driver constant folding;
there is no model/card-name rule. Exact parameter values and tile geometry key
each pipeline. A later `experiment/conv-native-division-2026-09-10` tag also
uses native u32 division with those constant divisors, retaining exact indexing.
These are whole-model implementation ablations, **not automatic selections**.
They do not combine with the current tile tuner or change the collection tag.

Run `scripts/tune_study.py --models ResNet-50 Whisper-tiny --variants untuned
fixed-params fixed-native-div --replicates 6 --output <new-dir>`. Five warmups
and twenty samples per fresh process, strict f32, no tracing, normal per-session
contexts. The six-process three-arm cohort rotates execution order. Preparation
includes both the ordinary fallback pipelines and additional exact variants.

| ResNet F+loss+backward | Untuned | Constant parameters | Constants + native division |
|---|---:|---:|---:|
| Median step, ms | 44.153 | 34.444 | 33.329 |
| Median preparation, s, warm driver cache | 0.716 | 0.8235 | 0.826 |

All recorded logits hashes, loss and gradient norms repeat exactly in both
models across all 36 processes. Both variants separately pass the existing
full-f64 forward/dX/dW regression oracles, including ordinary/tiny inputs,
padding, stride and tile edges. ResNet's roughly 1.28× / 1.325× speedups clear
the 5% plus paired-noise guard. Whisper improves by only about 1%, below it.
The incremental native-division gain is about 1.1 ms, less than the same 5%
guard relative to constant parameters; it is not another large independent win.

The earlier constant-parameter-only six-pair confirmation agrees. Repeating it
with `--fresh-driver-cache` gives ResNet 44.151→34.497 ms, but adds a median
3.067 seconds of preparation: about **319 training steps to amortize**. Warm
cache extra preparation was about 113 ms, or 12 steps. Do not use frontend-only
compilation time to erase that first-use cost. The unchanged 135M control has
no qualifying convolution and no guarded gain.

At Inferena `cbb1621`, a separate six-pair fresh-driver-cache confirmation of
the native-division arm gives 44.091→33.304 ms, with 2.757 s median extra
preparation: about **256 training steps to amortize**. All recorded outputs
remain exact. This charges the native driver rather than extrapolating from
the constants-only arm.

A qualified constant-parameter Graphics capture follows the same short-window
recipe as the baseline: three steps in 98.42 ms, one complete training GPU
range of 32.21 ms versus 42.09 ms in the earlier baseline capture. Runner output
matches its ordinary control and the baseline; no hardware-event overflow.
This independently places the improvement on the GPU, but these diagnostic
intervals are not the unprofiled performance estimate. The sampled integer-FMA
share changes from 11.34% to 7.95%; **that is not an executed-instruction count**.
Stalls on a consumer instruction can reflect earlier loads or synchronization.
Shared-store samples alone do not prove that shared stores caused the delay.

The follow-up `experiment/conv-k-stage-2026-09-10` tags double K staging from
16 to 32, preserving the accumulator order while roughly halving loop/barrier
rounds and doubling shared storage. Both integer-division forms pass the same
full-f64 convolution oracles. Six three-arm process replicates (all six order
permutations) confirm that the native-division K=32 arm **regresses ResNet
training**: 33.340→36.834 ms, versus 44.133 ms untuned. Recorded output fields
remain exact across all 36 processes. Minimal-forward latency goes the other
way, 4.281→4.053 ms; Whisper changes negligibly. No global K=32 promotion is
justified, and fewer barrier rounds alone do not predict a faster model.
Reproduce with `--variants untuned fixed-native-div fixed-native-div-k32` and
the same models/replicate count. These are warm-driver-cache results.

## Earlier Graphics pilot and incident disposition

Graphics 2026.3.1.0 counter access now works. The 135M run at Inferena `f51431a`
completed both phases, with full prefill output hash and loss matching its
ordinary control (256 sampled values differ at most 3.6e-15 after JSON parsing).
Two complete GPU-projected prefill sample ranges have about 18.76% peak SM
throughput, 11.82% peak active-warp occupancy and 0.82% peak barrier-stalled
warp occupancy. These last two share a denominator; **0.82% is not wall-time
barrier overhead**. Incoming PCIe traffic is low (~0.036% peak); total PCIe
traffic also includes profiler writes. The run reported hardware-event buffer
overflow, so its per-dispatch timeline is incomplete. Counter observations
remain pilot diagnostics, not a qualified per-kernel breakdown or speed result.
[GPU Trace](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-ui.html),
[warp-stall semantics](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html).

The subsequent 1.7B trace had overflow and no complete runner JSON. A later
detached retry exhausted host RAM and swap at 06:11 UTC and was OOM-killed.
Neither is qualified evidence. No reboot occurred after the 05:51 boot.
Use [bounded capture instructions](EXPERIMENT.md#nvidia-paper-analysis-captures);
do not rerun that detached recipe or grow buffers blindly.

The source/name support is now in the collection runtime: default-off
`MEGANEURA_GPU_CAPTURE` enables shader debug information/command labels
independently of pass timestamps and dispatch grouping. The qualified short
captures above supersede the pilot's missing correlation. A causal removable
barrier-cost estimate still requires a legal schedule A/B with unchanged kernels
and full validation; warp-barrier stalls are not Vulkan resource-barrier cost.

Local evidence stays outside Git: `/mnt/data/inferena-native-analysis.nC5jZj/`
(`resnet-nsys-final`, `smollm-1.7b-nsys`),
`/mnt/data/inferena-profile-handoff.bzLgOQ/profiles/`, and
`/mnt/data/inferena-ngfx-markers.bx2oTf/`. Failed Graphics attempts are retained
separately at `inferena-ngfx-large.WKyJMP` and `inferena-ngfx-attached.dvS9uD`.
Git retains the source/recipe and this conclusion, not trace binaries or raw arrays.
September 10 records live outside Git under `/x/Code/inferena-results`, in
`ngfx-short-20260910.m4RoCh`, `ngfx-resnet-training-labelled-20260910.AwJMn3`
and `nsys-native-shared-20260910.Qi2pXf`. Compiler and tuning study scripts/tags
likewise retain procedures, not binaries/caches/raw arrays.
