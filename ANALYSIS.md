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

## Pre-cohort checklist review (October 11, 2026)

Durable records and a process-tree watchdog are implemented. The follow-up
search-order study below isolates candidate coverage from a warmup sensitivity.
Before collection, integrate the reviewed library change and freeze the source,
budgets and cache policy.
PyTorch CPU remains the correctness oracle on the graphics-only platforms;
its times are excluded from GPU comparisons. No ggml/ONNX reference is needed.
Qualification-only coverage must remain labelled separately from replicated
native timing results. B570 should be the first device qualified and collected
once the source is frozen.

The recovered work is present locally, including uncommitted changes. These
paths identify experiments, not branches certified for the cohort:

| Item | Recovered work and disposition |
| --- | --- |
| 1. Unsupported PyTorch GPUs | Keep the CPU correctness oracle and native portability rows. B570 SmolLM2-360M now passes a compiled, uncaptured diagnostic in both classes; the failed captured run is retained. Replicated collection remains to be done. |
| 2. B570 hang | The harness now checkpoints each completed engine with atomic replacement. The collector writes atomic manifests and bounds each pair with `--run-seconds` (1,200 s by default), including the nested PyTorch compiler session. A timeout stops the campaign. |
| 3. Frozen settings | The current pin declares 30 s/session, 64 programs, 16 graphs and no patience stop. V17 already creates an empty campaign-local kernel memo and records reuse; report its cold and warm preparation separately. Host/kernel identity is now in the manifest before a run; native driver/Mesa identity is already recorded. Do not import a different branch's early-stop assumptions. |
| 4. Mixed precision reference | A prototype and one-process results already exist in `/x/Code/inferena-halfprec` and `../inferena-results/autocast-20261010`. It wraps forward in autocast but has **no gradient scaling**. The new scaling diagnostic below removes most of Whisper's gradient discrepancy; implement scaling before choosing a new gate. A PyTorch AMP column can be added without waiting for native f16 activations. |
| 5. Convergence | High-value follow-up: pin real data, batch order, initialization, optimizer, learning rate, update count and seeds; compare loss trajectories and held-out quality. Existing timings cover forward/loss/backward only, with no optimizer update. A single short run supports a bounded convergence claim, not equivalence for all five workloads. |
| 6. Cross-device decisions | This is not a cache-copy experiment: `KernelMemo::bind` rejects another device/driver and the memo stores private kernel decisions, not a selected whole program. Add an explicit capability-checked export/import of decisions, reconstruct and qualify on the target, and report invalid/unavailable choices as well as slowdown against target-local search. Never disable ordinary cache identity checks. |
| 7. Energy | Useful secondary study after latency is stable. Prefer energy-counter differences over a long repeated batch, divided by completed steps. Record sensor scope, units, sampling interval, idle baseline and missing sensors; GPU-board and package energy are different quantities. Include preparation energy separately if claiming search pays for itself. |
| 8. Footprint/cold start | Worth adding as a small separate study: runtime binary plus required shared libraries versus the minimal runnable PyTorch environment, with model weights reported separately. Measure process launch to first completed step in a deployed runner, excluding Cargo/package setup. State filesystem, framework and driver cache conditions separately. `compile_s` is not this metric. |
| 9. Ablations | The saved `search-vs-greedy-20261010` study covers greedy, kernel tuning, graph-count and time budgets; it does not isolate all four mechanisms. Use explicit fusion/layout/chunking/cooperative controls with the same budget and correctness gates. Strict-f32 runs already exclude reduced-input cooperative tiles, so toggling those there would be a no-op. |
| 10. B570 kernels | GEMV changes and sweeps survive in `/x/Code/meganeura-next` and `../inferena-results/b570-gemv-20261011`. Convolution register-column interleaving survives in `/x/Code/meganeura-conv`; three saved greedy ResNet runs improve B570 training median from 53.35 to 52.43 ms but worsen 5070 from 28.56 to 30.12 ms. This is not the proposed cooperative convolution implementation. ANV's recorded 8×16×16 tile remains incompatible with the square-tile codegen. |
| 11. f16 activations | `/x/Code/meganeura-f16` contains completed f16 **weight** work and `docs/reduced-precision-storage.md`; stage 2 activations and stage 3 training remain unimplemented. Treat weight storage and AMP as distinct labelled precision policies. Do not make stage 2 a prerequisite for measuring the reference. |
| 12. Whisper search gap | The recovered prototype remains in `/x/Code/meganeura-next`; the isolated change and coverage tests are in `/x/Code/meganeura-search-order`. The 30 s study below separates candidate starvation from warmup-sensitive B570 timings. Integrate and qualify the reviewed library revision before the cohort. |

The old strict 5070 Whisper `r1` study is concrete evidence of candidate
starvation: `graph1` selects the 128-thread, 16-key flash layout at trial 17,
giving 5.707 ms inference, while `search60` keeps the ordinary candidate at
6.341 ms. Both reach the 64-program cap at about 29 s, so increasing the time
budget alone cannot fix that inference result. The layout exists in the search
space; its ordering prevents it being visited within the program bound.

In the subsequent 60 s coordinate-order pilot, 5070 Whisper inference changes
from 5.690/5.788 ms to 5.417/5.175 ms across two processes. B570 diffusion
training changes from 10.958/10.918 ms to 11.470/12.275 ms. These are incomplete
native-only pilot results on other source revisions, not qualification of the
current 30 s campaign policy. Preserve the negative results when refining the
search and then qualify the frozen implementation across the full model set.

Two new diagnostics are retained in `../inferena-results/precohort-20261011`,
with scripts, source snapshot, logs and watchdog receipts. B570 SmolLM2-360M
passes all uncaptured phase checks in strict and accelerated modes; training
medians are 54.501 and 54.558 ms. The separate comparison against the archived
strict native result passes output and gradient gates in both cases. These
single-process diagnostics preserve, rather than replace, the failed replay run.

The 5070 Whisper experiment uses the current deterministic fixture, eager
execution and math SDPA to isolate scaling. With the same fp16 forward and
f32 loss, enabling `torch.amp.GradScaler` (scale 65,536, gradients unscaled
before comparison) reduces parameter-norm-vector relative L2 error from
20.2585% to 0.001887%, and full-gradient relative L2 error from 36.6109% to
0.027458%. No optimizer step is taken. This supports gradient underflow as
the cause of the large unscaled discrepancy; it does not establish compiled
AMP replay correctness or convergence. See `whisper_scaling.py` and
`whisper-scaling.json` in the diagnostic directory.

For the precision follow-up, PyTorch documents gradient scaling as protection
against fp16 gradient underflow; bf16 does not have the same exponent-range
limitation. See the [official AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
For energy, NVML exposes a cumulative energy query where supported, while its
power query can report a one-second average. Millisecond-step power snapshots
therefore need a longer measurement window; see the
[NVML device-query reference](https://docs.nvidia.com/deploy/archive/R550/nvml-api/group__nvmlDeviceQueries.html).

## Search order and warmup diagnostic (October 11, 2026)

The isolated library change is commit `975fafa` in
`/x/Code/meganeura-search-order`, branch `fix/physical-search-coverage`,
based on the current Inferena pin `2c42b44`.
It carries only the recovered search changes, with coverage tests and boundary
fixes; the GEMV and convolution prototypes remain separate. After the first
three graph baselines, it alternates graph forms using the incumbent's physical
choice with physical choices on the incumbent's graph. A winner carries its
choice to the other axis. Remaining interactions follow diagonally if the
budget permits. The exact previously missed Whisper layout has a regression
test that reserves half the 64-program allowance for submission probes.

The old B570 diffusion slowdown could be reproduced with the isolated change
and a 30 s budget, but its sign depended on the run. With two-second warmup,
control/candidate training was 8.803/10.004 ms; reversing process order gave
12.315/10.005 ms. Both controls selected graph 2 and both candidates graph 13.
Ten additional one-second windows after selection showed the candidate drifting
from 10.093 to roughly 11.3 ms. Late clock samples differed between the two
programs, around 2,050 MHz for the control and 2,250 MHz for the candidate,
with no reported GPU throttling. These observations establish sensitivity to
operating state; they do not identify a single driver or hardware cause.

Protocol v18 therefore uses a fixed ten-second, five-call minimum warmup for
both engines, followed by all twenty samples. This changes held-out warmup,
not the 30 s search budget, 250 ms comparison warmup, candidate cap or
numerical gates. A fixed longer warmup reduces the observed drift; it does
not promise constant clocks. Old receipts are rejected by the collector.

With ten-second warmup, both process orders retain graph 2 for the B570
diffusion control and graph 13 for the candidate. Training is
11.622/11.688 ms for the control and 11.439/11.483 ms for the candidate.
The earlier training regression is absent in these matched runs. Inference
and latency still show run-to-run variation, so this is not evidence of
eliminated clock variation or a precise speedup claim.

On the 5070, the same 30 s policy reaches the missing 128-thread, 16-key,
non-interleaved flash layout at trial 49 in both process orders. Inference
search finishes its 64 trials in 28.51/26.01 s. With ten-second warmup,
Whisper inference changes from 5.719/5.790 ms to 5.333/5.335 ms; latency
changes from 5.657/5.797 ms to 5.321/5.406 ms. Training remains close:
24.369/24.651 ms versus 24.671/24.445 ms. The paired output samples and
parameter-gradient norm vectors are identical for Whisper; diffusion's
gradient-norm-vector relative L2 difference is below 0.000001.

Raw attempts, source patches, binary hashes, paired-search samples, hardware
telemetry, watchdog receipts and the analysis script are retained outside Git
in `../inferena-results/search-order-30s-20261011`. Each process starts with
an empty learned-kernel memo; driver shader caches are not cleared. These are
strict-f32 native diagnostics with unchanged full-tensor candidate checks,
not a replicated PyTorch comparison or a replacement cohort. The new library
revision still needs integration and full cohort qualification.

Validation passes 16 library search tests, nine campaign tests and eleven
harness tests, plus formatting, Clippy and library documentation checks.
Debug GPU tests emit the existing SPIR-V validation-layer message
`VUID-StandaloneSpirv-None-10684`; it also reproduces on a freshly rebuilt
unchanged `2c42b44`, and the test assertions pass on both revisions.
