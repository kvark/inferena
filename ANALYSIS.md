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
based on the then-current Inferena pin `2c42b44`.
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
not a replicated PyTorch comparison or a replacement cohort. The library
revision is now published in [PR #245](https://github.com/kvark/meganeura/pull/245)
and integrated through pin `5b37124`, whose additional commit only fixes a
test's assumed wall-clock trial count. Full cohort qualification remains.

Validation passes 16 library search tests, nine campaign tests and eleven
harness tests, plus formatting, Clippy and library documentation checks.
Debug GPU tests emit the existing SPIR-V validation-layer message
`VUID-StandaloneSpirv-None-10684`; it also reproduces on a freshly rebuilt
unchanged `2c42b44`, and the test assertions pass on both revisions.

## AMP and real-data training (October 11, 2026)

Inferena `273ac31` adds separate AMP references and a real-data training
study. Later commits align compiler concurrency/attention with the cohort and
retain independently qualified timing phases. The library pin is `5b37124`;
[PR #245](https://github.com/kvark/meganeura/pull/245) passes all CI jobs,
including Metal, Vulkan/lavapipe, Windows, MSRV and quality checks. It remains
a published PR branch, not a claim that the full cohort has been qualified.

The real-data run uses SmolLM2-135M, 128-token next-token batches from
Project Gutenberg's *A Midsummer Night's Dream*, 256 completed SGD updates,
learning rate 0.001, and identical seeded token order across engines. Every
run begins from the pinned checkpoint. Four fixed training sequences and
four held-out sequences are evaluated at steps 0/64/128/192/256. The protocol
and predeclared gates are in [EXPERIMENT.md](EXPERIMENT.md#amp-reference-and-real-data-convergence).
These are single-seed correctness studies with no throughput claim.

The following relative L2 errors are **percent**, against PyTorch strict f32.
All eight final trajectories completed 256 updates, changed the checked
parameter and passed the learning gate.

| GPU | Engine / arithmetic | Training curve | Fixed training probe | Held out | Curve gate |
| --- | --- | ---: | ---: | ---: | --- |
| B570 | Meganeura f32 | 0.000763% | 0.000809% | 0.000648% | pass |
| B570 | PyTorch bf16 | 0.5525% | 1.0603% | 0.4175% | fail |
| B570 | PyTorch fp16 + scaling | 0.2270% | 0.4134% | 0.1541% | pass |
| RTX 5070 | Meganeura f32 | 0.000861% | 0.000926% | 0.000703% | pass |
| RTX 5070 | PyTorch bf16 | 0.5621% | 0.9764% | 0.3378% | pass |
| RTX 5070 | PyTorch fp16 + scaling | 0.2271% | 0.4121% | 0.1554% | pass |

Native f32 lowers fixed training-probe loss by about 6.70% and held-out loss
by 3.60% on both GPUs. bf16 also learns, but the B570 fixed-probe curve fails
the declared 1% L2 bound; the 5070 result is close to that boundary too.
Neither justifies widening the gate after seeing these runs. Each fp16 run
records one initial overflow retry on the same batch, then completes all 256
updates. The original attempt-counting diagnostic completed only 255 updates
and remains recorded as failed.

The first real-text XPU run failed in native embedding backward, with
`Trying to create tensor with negative dimension -1`. A separate probe of
the first real batch also found non-finite gradients. The existing dense
`index_add` autograd workaround matches every one of the 28,311,552 expected
gradient elements, including repeated and unused rows. Real-text workers now
qualify that input pattern and record the selected primitive. With it, the
XPU f32 trajectory matches native f32 closely. No CPU timing replaces a GPU
reference.

Compiled Whisper on the RTX 5070 passes the full-output, 32-update precision
and ordinary repeatability gates in both AMP modes. These are one-process
diagnostics, with backend-auto SDPA, explicit uncaptured execution, normal
compiler concurrency, 120 s compilation deadline and ten-second warmups:

| Reference | Compile | Inference | Latency | Forward/loss/backward |
| --- | ---: | ---: | ---: | ---: |
| PyTorch AMP bf16 | 8.796 s | 1.313 ms | 1.317 ms | 4.023 ms |
| PyTorch AMP fp16 + scaling | 8.152 s | 1.370 ms | 1.441 ms | 4.724 ms |

The respective 32-update loss-curve L2 errors are 0.00207% and 0.00122%.
fp16 timings include gradient unscaling, finite checks and scale maintenance;
there is no timed optimizer update. These columns are not paired speedup
claims against earlier native measurements and do not qualify AMP graph replay.

B570 SmolLM2 AMP still needs work before the cohort. With normal compiler
concurrency, bf16 compilation completed in 108.6 s, but its fixed-input
32-update training curve differed by 13.08% L2. A later phase-preserving
diagnostic compiled in 97.2 s and failed the forward maximum-error bound
(0.7799 versus 0.7272 allowed; RMS passed). fp16 compilation reached the
120 s deadline. The separately labelled eager fp16 diagnostic qualifies
inference/latency (33.697/30.835 ms), but fails the fixed-input training curve
at 5.43% L2, so it emits no training timing. Passing the real-data trajectory
does not override a failed benchmark-specific gate. All failures remain
visible, with no tolerance changes or timing substitution.

Two earlier compilation attempts forced one compiler worker through the new
study launcher. That differed from the cohort's default twelve workers here;
those timeouts are retained as serial-compilation diagnostics and excluded
from the compiler comparison. The launcher override is removed and effective
compiler concurrency is now reported.

Artifacts are under `../inferena-results/amp-convergence-20261011`:
`b570-v2` and `rtx5070-v2` contain the committed real-data runs, raw losses,
checkpoints, source archives, corpus/tokenizer/spec hashes and PNG/SVG plots;
`convergence-summary.json` tabulates all curve metrics. AMP directories retain
every compilation/qualification receipt, raw timing samples, source archive
and watchdog result; qualified phases have `columns.csv`. Library and Inferena
checks pass, including ten AMP tests, nine campaign tests, eleven harness
tests and four native runner tests (one separate GPU cache test is ignored).

Next, resolve the B570 AMP failures and review the separate GEMV/convolution
prototypes before freezing a full B570-first qualification/cohort. The new
AMP references quantify the motivation for native f16 activations; stage 2
remains unimplemented. Cross-GPU transfer, energy and footprint studies remain
separate roadmap work.

## B570 AMP diagnosis and kernel review (October 11, 2026)

The SmolLM2 synthetic-input AMP failures are reproducible on **both GPUs**,
including eager execution. Untimed diagnostics at `0c78dff` use the same
checkpoint, 128-token benchmark input, math SDPA and 32 completed SGD updates
as the declared gate. Each trajectory restores the checkpoint and RNG;
f32 is repeated before testing bf16 and fp16. The following errors compare
each trajectory to its device's initial f32 trajectory, in percent:

| GPU | Repeated f32 | AMP bf16 | AMP fp16 |
| --- | ---: | ---: | ---: |
| B570 | 0.02114% | 13.0641% | 5.4744% |
| RTX 5070 | 0.00000% | 13.0150% | 5.4259% |

Both f32 repeats pass the unchanged gate; all four AMP trajectories fail.
Each fp16 trajectory retries two overflowed attempts and completes all 32
updates. The eager bf16 forward maximum-error gate also fails on both GPUs:
1.0780 on B570 and 0.8898 on 5070, against a 0.7272 bound. fp16 forward passes
on both. The corresponding AMP curves agree across GPUs much more closely
than they agree with f32 (0.3796% L2 for bf16, 0.1516% for fp16).
This points to sensitivity of the repeated synthetic workload to AMP
arithmetic, rather than an XPU-only or compiler-only defect. It does not
establish a particular operator as the cause. The real-data study remains
separate evidence; no thresholds or failed benchmark cells are changed.

A single, explicitly extended-deadline B570 fp16 compilation diagnostic
finishes in 96.36 s, qualifies forward execution and again fails its training
curve (5.4780% L2). It retains inference/latency at 8.872/8.089 ms and leaves
training empty. The configured deadline was 300 s, so this is a diagnostic,
not a replacement for the earlier 120 s timeout or a qualified cohort replicate.
It establishes that this configuration can finish, not that a retry fixes the
earlier failure. The cohort compilation limit stays 120 s.

The kernel prototypes are excluded from the frozen candidate:

- **GEMV:** the current library already searches subgroup reductions, row
  groups and forward column groups. The additional prototype reads
  `MEGANEURA_GEMV_UNROLL` inside code generation, outside the typed, serialized
  tuning configuration. The saved `sweep.sh` never sets it, so its results
  evaluate the existing shape choices, not unrolling. Its example times
  chains without reading back or validating their outputs. An analytic lower
  bound using its positive initializer also exceeds f32 range in the 40-pair
  chains for 576×576, 576×1536 and 576×3072; these need a bounded fixture.
  Promotion requires
  a represented tuning choice, numerical checks and matched full-model
  measurements; these sweeps cannot establish an unrolling improvement.
- **Convolution:** the process-global interleaving prototype changes scalar
  register tiling, not cooperative matrix execution. The three saved greedy
  ResNet training runs have medians 53.349 → 52.434 ms on B570 (1.7% lower)
  and 28.559 → 30.122 ms on 5070 (5.5% higher), under the older protocol.
  This does not justify a universal default. A future implementation should
  make it a measured, persisted per-kernel choice.
- **Xe2 cooperative matrices:** a fresh capability query of the pinned native
  runner reports f16-input/f32-accumulate shape `[8, 16, 16]` on B570 and no
  native-f32 shape. ANV exposes the capability; the existing square-tile
  implementation cannot use that shape. Rectangular code generation is
  separate library work, and would not accelerate the strict-f32 column.

Raw diagnostics, all losses, initial full logits, per-parameter first-gradient
comparisons, source archives, watchdogs, capability logs and prototype patches
are retained in `../inferena-results/b570-amp-diagnosis-20261011`. Its first
diagnostic attempt stopped after the f32 repeats because the diagnostic
launcher incorrectly combined the strict environment flag with explicit AMP;
the corrected `xpu-eager-v2` and `cuda-eager` runs above are complete.

The first frozen B570 qualification at `0c78dff` completed SmolLM2-135M in
both engines, then exposed a collector schema bug: Rust omits an empty
`TuneReport.reused_classes`, but the Python memo validator indexed it as
mandatory. The validator now interprets that omitted list as empty while
still rejecting inconsistent aggregate reuse counts. Contract tests cover
omitted/explicit empty lists, nonempty hits and missing claimed hits. Read-only
reassessment of the retained first pair passes all three phases; the original
campaign remains marked incomplete. A new qualification campaign is required
for the corrected source, with its own empty memo directory.

The corrected campaign at `0a718b6` then completes all five models in both
strict and accelerated classes on B570: **10/10 valid conditions and 30/30
valid phase comparisons**, with captured PyTorch inference, latency and
training. SmolVLA completes in both classes without a hang. PyTorch compilation
takes 18.8–85.0 seconds, within the unchanged 120-second limit. The full
qualification takes 1,942.6 seconds. These are one-sample qualification
records, not replicated throughput estimates.

Every model/precision starts with a memo miss and zero loaded classes in its
first native session. Later sessions in that same process reuse qualified
decisions, as their receipts explicitly record. A separate campaign directory
is used for replicated collection, so qualification does not seed its cache.
Receipts are in `../inferena-results/b570-qualification-memo-fix-20261011`;
the first, schema-rejected attempt and its read-only reassessment remain in
`../inferena-results/b570-qualification-0c78dff-20261011`.

## Frozen local cohort (October 11, 2026)

The B570 primary cohort at Inferena
`0a718b6fb9d6c990410dc274618978ad4988a36d` and Meganeura `5b37124` completes
all **30/30 paired processes and 90/90 phase comparisons**. All ten
replicated-gradient groups pass. The five models run in strict and accelerated
classes, with three independent processes per condition and twenty held-out
samples per phase. Every PyTorch phase uses qualified graph replay. There are
no hangs, watchdog failures or substituted diagnostic timings. Collection takes
5,758.6 seconds (96.0 minutes); compilation takes 18.9–85.5 seconds per reference
process, within the unchanged 120-second limit.

Meganeura is slower in every B570 cell. The following ratios are **PyTorch time
divided by Meganeura time**, so values above one would favor Meganeura. Each
engine's time is the median of its three process medians, with twenty samples
in each process. These are execution timings; preparation is recorded separately.

| Arithmetic | Model | Inference | Latency | Training |
| --- | --- | ---: | ---: | ---: |
| Strict | SmolLM2-135M | 0.438× | 0.574× | 0.310× |
| Strict | SmolVLA | 0.502× | 0.611× | 0.541× |
| Strict | StableDiffusion | 0.255× | 0.266× | 0.311× |
| Strict | ResNet-50 | 0.413× | 0.325× | 0.306× |
| Strict | Whisper-tiny | 0.639× | 0.636× | 0.333× |
| Accelerated | SmolLM2-135M | 0.443× | 0.574× | 0.305× |
| Accelerated | SmolVLA | 0.520× | 0.571× | 0.534× |
| Accelerated | StableDiffusion | 0.249× | 0.268× | 0.307× |
| Accelerated | ResNet-50 | 0.411× | 0.373× | 0.304× |
| Accelerated | Whisper-tiny | 0.629× | 0.628× | 0.335× |

The driver remains Mesa 26.0.3-1ubuntu1 on Linux 7.0.0-38-generic throughout
the cohort. Native execution uses scalar kernels on this Xe2 stack; the
rectangular cooperative-matrix limitation described above still applies.
The results establish a complete, numerically qualified baseline, and leave
the B570 kernel gap as substantial library work.

The first native session of every model/precision in replicate one misses the
new campaign memo (ten misses). Its later sessions already reuse decisions
(eighteen hits). All twenty first sessions in replicates two and three hit the
memo; all fifty-six sessions in those replicates record hits. Thus replicate
one is not an all-cold phase set. Every PyTorch process owns a fresh Inductor
and Triton cache. Native preparation, search costs and memo counts remain
available per session; the execution table does not hide them inside a
purported cold-start comparison.

Raw receipts are in `../inferena-results/b570-cohort-0a718b6-20261011`.
`../inferena-results/local-cohort-20261011` retains a named B570 archive,
derived JSON/CSV with process medians and ranges, the analysis script, and the
portable source bundle `inferena-cohort-0a718b6.bundle`. The bundle contains
the complete history ending at the frozen revision and passes a fresh clone
check. Its SHA-256 is
`05c626eb2e95f327590a653b1621d6974dbcb8a11d3e309bc56a744a76b7abe8`.
It excludes environments, model checkpoints and untracked workspace files.

RTX 5070 first passes all ten qualification conditions and thirty phases on
the same frozen source, then completes **30/30 paired measurement processes,
90/90 phase comparisons and all ten replicated-gradient groups**. Qualification
takes 1,513.4 seconds; collection takes 4,469.7 seconds (74.5 minutes). Reference
compilation in collection takes 7.2–47.4 seconds. All reference phases use
qualified graph replay; every watchdog receipt records a successful exit.
The native driver is NVIDIA 595.91.07 on the same Linux kernel.

Using the same ratio and median definitions as the B570 table:

| Arithmetic | Model | Inference | Latency | Training |
| --- | --- | ---: | ---: | ---: |
| Strict | SmolLM2-135M | 0.946× | 1.164× | 0.589× |
| Strict | SmolVLA | 0.929× | 0.872× | 0.744× |
| Strict | StableDiffusion | 1.162× | 1.146× | 0.801× |
| Strict | ResNet-50 | 1.342× | 1.437× | 0.616× |
| Strict | Whisper-tiny | 0.643× | 0.655× | 0.513× |
| Accelerated | SmolLM2-135M | 0.364× | 1.158× | 0.371× |
| Accelerated | SmolVLA | 0.518× | 0.928× | 0.529× |
| Accelerated | StableDiffusion | 0.740× | 0.740× | 0.494× |
| Accelerated | ResNet-50 | 0.479× | 0.509× | 0.293× |
| Accelerated | Whisper-tiny | 0.602× | 0.607× | 0.437× |

Meganeura wins six of the thirty RTX 5070 timing cells: strict ResNet and
diffusion inference/latency, plus SmolLM2's stateless one-token latency in
both classes. PyTorch remains faster in every training cell on both local
GPUs. The primary accelerated class is separate from the autocast study;
these results do not establish native f16 activation storage or mixed-precision
training. On RTX 5070, 21 of 42 accelerated native sessions select programs
with cooperative dispatches, compared with none of the 42 strict sessions.

The NVIDIA campaign has the same recorded memo pattern as B570: ten first-session
misses and eighteen subsequent hits in replicate one, followed by twenty-eight
hits in each later replicate. Preparation remains separate from the execution
ratios. The combined figure shows the range of paired process-median ratios;
those whiskers are observed ranges across three processes, not confidence
intervals.

The RTX qualification and measurement receipts are in
`../inferena-results/rtx5070-qualification-0a718b6-20261011` and
`../inferena-results/rtx5070-cohort-0a718b6-20261011`. Named archives, both
GPU summaries, analysis/plot scripts, PNG/SVG/PDF figures and
`RUN-COHORT.txt` are in `../inferena-results/local-cohort-20261011`.
The source bundle and instructions provide the frozen handoff for H100/M3
collection and 780M/9600X/RPL-U CPU-oracle qualification on the user's machines.
Those devices still need their own evidence. No CPU timing is substituted for
a GPU reference.
