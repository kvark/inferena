# Native gap analysis — September 9

The evidence separates inefficient resident kernels from weight-representation
and memory-placement costs. It does **not** establish a removable barrier-cost percentage.
No new speed samples were collected during the post-OOM offline analysis.

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
does not guarantee placement. A separate general experiment should allocate
persistent parameters device-locally with explicit upload/readback staging,
then inspect actual binding heaps and rerun the unchanged numerical gates.
Keep host-visible inputs/external buffers' contracts intact, bound staging,
and account for training/checkpoint access. On unified memory, use the actual
backend's memory semantics rather than imposing a discrete-VRAM rule.

Only after placement is controlled should resident matmul/GEMV tile, coalescing,
occupancy and barrier comparisons be interpreted as kernel scaling. Keep any
fallback result separately labelled; do not silently exclude it or reduce f32 weights.

## Graphics pilot and incident disposition

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

Next capture: a short, bounded resident-model window without event loss, shader
source/pipeline correlation, and matching PyTorch compiler mode. Then inspect
the expensive kernel's instruction/load/stall mix. A current barrier-overpayment
estimate still requires a legal schedule A/B with unchanged kernels and full
validation; warp-barrier stalls are not Vulkan resource-barrier cost.

[Meganeura PR #165](https://github.com/kvark/meganeura/pull/165) prepares this:
existing structured pipeline keys become native-tool names, and default-off
`MEGANEURA_GPU_CAPTURE` enables Blade's shader debug information/command labels
independently of pass timestamps and dispatch grouping. All CI jobs passed;
hardware source/name correlation is still unqualified, so the PR remains draft.
It is **not** in Inferena's `43b606ff` collection pin. Qualify a separately recorded
diagnostic revision before using the flag; existing measurement tags stay unchanged.

Local evidence stays outside Git: `/mnt/data/inferena-native-analysis.nC5jZj/`
(`resnet-nsys-final`, `smollm-1.7b-nsys`),
`/mnt/data/inferena-profile-handoff.bzLgOQ/profiles/`, and
`/mnt/data/inferena-ngfx-markers.bx2oTf/`. Failed Graphics attempts are retained
separately at `inferena-ngfx-large.WKyJMP` and `inferena-ngfx-attached.dvS9uD`.
Git retains the source/recipe and this conclusion, not trace binaries or raw arrays.
