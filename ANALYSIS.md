# Inferena experiment analysis

Results retain their experiment's revision, protocol and limitations. The earlier
post-OOM offline analysis separates inefficient resident kernels from
weight-representation and memory-placement costs; it does not establish a
removable barrier-cost percentage or contain new speed measurements.

## Final-program qualification (October 19, 2026)

Meganeura `7cf2935` ([#267](https://github.com/kvark/meganeura/pull/267))
separates final qualification from search checkpoints. Inferena requests one
execution per checkpoint and two fresh final executions, with state/input
restoration and final-check time reserved inside the soft search deadline.
Paired v33 / Android v15 receipts require the final count and duration explicitly.

A serialized SmolLM2-135M accelerated comparison against the preserved `5cb84bd`
runner used 30-second searches, ten-second warmups, 20 held-out samples and no
persisted decisions: two B570 pairs (second order reversed) and one RTX 5070
pair. Both B570 variants used watchdog logging. All six processes pass saved
qualified PyTorch forward/training comparisons; all nine new-policy sessions
pass receipt validation and record two final executions. No candidate failed
qualification. Totals below include final checks; B570 values are medians across
its two processes per variant.

| GPU / phase | Qualification seconds, old → new | Candidate trials, old → new | Step-time change |
| --- | ---: | ---: | ---: |
| B570 prefill | 8.42 → 5.76 | 37–38 → 43 | −1.75% |
| B570 latency | 0.305 → 0.168 | 14 → 14 | +3.06% |
| B570 training | 19.83 → 16.27 | 5 → 9 | −7.83% |
| RTX 5070 prefill | 6.75 → 3.96 | 64 → 56 | −1.62% |
| RTX 5070 latency | 0.216 → 0.118 | 15 → 15 | +1.26% |
| RTX 5070 training | 19.13 → 12.76 | 14 → 19 | +0.38% |

Lower qualification overhead does not guarantee more trials or a faster winner:
search ordering and kernel budgets follow measured incumbents. B570 latency
selected four submission chunks instead of eight; its paired changes were
+5.46% and +0.72%, retained rather than dismissed. This small study supports
lower qualification cost and improved training coverage, not a universal
zero-regression claim or cohort replacement. Original records, comparisons,
binary hashes and scripts are retained in `inferena-results/final-qualification-20261019`.
The earlier measurements below retain their original revisions and policies.

## Search ordering and independent native reporting (October 18 follow-up)

Meganeura `5cb84bd` preserves each training seed's backward-layout rank and
prioritizes other forward forms with the incumbent's rank. The initial cover,
alternation with physical choices, numerical gates and whole-program timing
gate remain intact. This changes exploration order rather than increasing the
thirty-second search allowance. There are no model names, GPU identities or
preferred graph indices in the policy.

All 42 final-pinned RTX processes pass, including numerical comparison with
preserved qualified PyTorch references. The core comparison covers five models,
both profiles and two alternating-order pairs; one additional accelerated
SmolLM2 pair brings the target to three repetitions. Accelerated 135M training
falls from 24.363 to 22.341 ms (8.3%). All three new runs select the faster
family with one cooperative dispatch, and training search takes 28.6–28.8
seconds. Strict training improves 9.3% across two pairs. The forty-process core
matrix has zero correctness rejections and takes 62.8 minutes; summed native
process time increases 1.35% (1871 to 1897 seconds). Its largest phase increase
is 4.1%, below the existing 5% material-regression criterion, with the selection
differences described below. These are diagnostic comparisons, not a new
publication cohort or a claim of zero losses.

A fresh thirty-second export selects the faster training family with eight
submission chunks and independently passes CPU all-F64 checking of its outputs,
loss and all 134,515,008 gradient elements. A separate thirty-second real-text
search selects that family with four chunks and completes 256 matched SGD
updates against the preserved PyTorch F32 reference. Training, probe and
held-out curves have zero mismatched steps; maximum training-loss difference
is 0.001884. The fixed training-probe loss falls 6.66%, and held-out loss falls
3.60%. This covers one checkpoint, corpus and seed.

Accelerated collection now saves its already-computed PyTorch F32 snapshot
before candidate compilation. A separate numerical comparison binds that
snapshot to the native record by file hashes; it supplies no reference timing.
The report distinguishes native qualification from paired eligibility and
retains complete native measurements with an explicit unqualified label when
independent evidence is missing. Training still requires its replicated gate.

Reprocessing copies of the MI300 core and large-model archives retains all
42 native model/profile/phase cells. Thirty-two qualify independently, while
31 have qualified PyTorch pairs. The additional qualified cell is strict
ResNet latency: its already-qualified inference oracle remains usable after
the reference's later failure. The other ten cells lack sufficient passing
evidence and remain unqualified. No original archive was modified and no
server workload was rerun. Fourteen usable archived comparisons reproduce
the existing F32 metrics exactly through the new oracle comparison command.

The B570 follow-up uses serialized, memory-limited processes with a watchdog,
durable candidate/kernel logs and read-only clock/temperature sampling. Detailed
submission tracing is opt-in because it materially affects search coverage.
One fully traced old-control Whisper process completes in 95 seconds without
a fault, but checks only ten training candidates; this is diagnostic evidence,
not a performance comparison or proof that the earlier hang is fixed.

The subsequent light-logging B570 matrix completes all 36 processes: Whisper,
ResNet and SmolLM2-135M, both profiles, three paired repetitions with alternating
order. Control is cohort Meganeura `378d8e2` with logging-only instrumentation;
the fixed build uses `5cb84bd`. Every process passes output/loss/gradient
comparison with a preserved, qualified PyTorch reference. The largest increase
across eighteen phase comparisons is 1.52%, for accelerated SmolLM2 prefill
(12.114 to 12.298 ms). Accelerated ResNet training increases 1.37%; all other
increases are smaller. No GPU fault or watchdog timeout occurs. Collection takes
60 minutes; summed process times are 1803 seconds for control and 1798 for fixed.
Package and VRAM telemetry both peak at 74 C. These observations do not identify
or fix the earlier queue/recovery failure.

ResNet latency medians decrease 6.1% strict and 13.1% accelerated, but the
fixed build's accelerated repetitions span 5.171–6.338 ms. Read-only telemetry
also records substantial clock variation, so these diagnostic comparisons do
not establish that the code caused those gains. The largest measured increase
is below the existing 5% material-regression criterion; this is not a claim of
zero losses or publication-quality timing under the diagnostic monitor.

The RTX strict ResNet training comparison exposes a small loss that must not
be hidden: both control runs choose eight convolution weight-gradient splits,
while both new runs retain four, measuring 3.21% slower across the two pairs.
The candidate prefix is identical through this decision. Both builds test the
eight-split candidate, but the new runs' gain does not clear the unchanged
one-percent improvement plus measured-noise requirement. One new run improves
26.191 to 25.767 ms during search, but its 0.204 ms noise margin keeps the
incumbent; the corresponding control's 0.051 ms margin permits replacement.
The final held-out comparison uses four submission chunks, so its gain need
not equal the earlier one-chunk probe. This identifies the different search
decision, not a demonstrated arithmetic slowdown or a reason to relax the gate.
The raw decisions are retained in `rtx-resnet-strict-decision.json`.

The largest RTX increases are strict Whisper latency (4.1%) and prefill
(3.9%). Both new runs retain two submission chunks while both controls
select four, with the same selected graph and flash-attention description.
Both builds test four chunks, including with the final flash variant; the
new runs' search comparisons retain two under the existing timing gate.
Inference form order is unchanged by the backward-rank policy. These records
identify an associated selection difference, but do not isolate every private
kernel choice or prove the cause of the changed timing observations.
`rtx-whisper-strict-decision.json` retains the complete search receipts.

Evidence is retained in `../inferena-results/cohort-followups-20261018/`,
including source/binary hashes, the excluded overlapping-GPU-test pilot pair,
MI300 report copies and the guarded comparison manifests. Start with
`rtx-summary.json`, `b570-summary.json`, `rtx-target-search.json`,
`fast-path-validation/manifest.json` and the two archived-reference comparison
files. Final CPU tests pass (49 PyTorch tests, three GPU-only cases skipped,
and 42 script tests); diagnostic success, timeout and launch-error paths pass.
The library's 21 search tests, 16 harness tests and Clippy passed earlier in
this follow-up. Both PRs remain unmerged.

## Search accuracy versus repeatability (October 18, 2026)

The v30 native checker compared every candidate with the first accelerated
program using its repeatability bound. On RTX 5070, all 55 rejected
SmolLM2-135M prefill candidates had already passed the independent original-F32
language-model gate. Every one had smaller mean KL error than the initial
program. A separate diagnostic recorded 186 same-program repeat comparisons
with zero maximum error. These were cross-program numerical differences,
not demonstrated unstable kernels.

A second reproduction retains the old gate and captures complete tensors for
all 55 rejections. Every rejected output independently passes PyTorch CPU
float64 evaluation, and all 55 have smaller mean KL than the initial program:
the largest rejected mean KL is `2.788e-6`, versus `3.595e-6` for the anchor.
The largest rejected worst-token KL is `4.458e-5`. There are 17 distinct full
output tensors across 71 qualification callbacks. Diagnostic writes use a
sixty-second search allowance to retain the complete 64-trial frontier;
these captures are correctness evidence, not additional performance samples.
The final independent oracle explicitly uses F64 RMSNorm, rotary frequencies
and attention softmax: Hugging Face retains F32 casts in those operations even
after `model.double()`. A Torch dispatch audit verifies F64 floating results
through the corrected evaluation. Earlier diagnostics using the ordinary
Hugging Face double model remain archived; the correction changes none of the
55 pass/fail decisions. Benchmark references and numerical gates are unchanged.

All 55 captured rejected plans were then replayed with their original submission
counts on the original token sequence, real text and random tokens. The replay
asserts complete execution-plan equality after construction and starts scratch
memory with NaNs. All 165 cases pass the native and independent all-F64 CPU
gates; all 21,120 additional executions reproduce their complete output exactly.
All 55 original-input output hashes match the historical captures. Maximum
mean KL across plans is `2.788e-6` for the original input, `1.749e-6` for real
text and `2.029e-6` for random tokens. The initial replay setup stopped at its
plan-equality guard because construction reset a workgroup tile; supplying the
captured tile settings restores exact equality. Both failed setup attempts and
the corrected replay's source/binary provenance are retained.

Inferena `03e6c9c` gives each complete execution plan its own repeatability
anchor and requests two fresh executions at each qualification checkpoint.
Every distinct plan must still satisfy the original F32 accuracy contract.
Exact full-tensor repeats reuse that anchor's quality evidence. No numerical
threshold, profile, search budget, or whole-program timing gate changes.
Meganeura `9dafafa` supplies repeated qualification with state/input restoration;
its API defaults to one execution for existing callers. Paired protocol v31 and
Android v13 identify the new acceptance policy. Earlier measurements remain
valid for their original protocol and must not be pooled with v31.

The RTX 5070 comparison uses three paired fresh processes per condition,
alternating variant order, no kernel memo, 30-second searches, ten-second
warmups and 20 held-out samples. The control uses cohort Meganeura `378d8e2`
with its archived diagnostic instrumentation disabled. All 60 core-model
processes pass. SmolLM2 results
are medians of three process medians; the 360M study measures forward phases:

| Model / accelerated phase | Control (ms) | Fixed (ms) | Change |
| --- | ---: | ---: | ---: |
| 135M prefill | 7.289 | 5.498 | -24.6% |
| 135M one-token latency | 1.497 | 1.418 | -5.3% |
| 135M training | 24.361 | 24.347 | -0.1% |
| 360M prefill | 15.806 | 12.147 | -23.2% |
| 360M one-token latency | 3.149 | 3.144 | -0.2% |

The largest increase across the 30 core model/profile/phase comparisons is
2.85%, for strict SmolVLA latency; this is not a claim of zero regressions.
Rejection counts fall from 183 in the control to zero in the fixed core matrix. The faster
135M prefill program retains one cooperative dispatch for the output projection
and uses scalar split-K layer products, versus 31 cooperative dispatches in the
control. Its mean KL against native F32 falls from `3.597e-6` to `2.616e-6`. Total native process
time rises 3.2%, from 2719 to 2805 seconds: the corrected prefill search uses
nearly 30 seconds instead of quickly rejecting candidates in about ten seconds.

A separate post-reboot RTX 5070 budget sweep completes 21 fresh processes,
three per budget, with no kernel memo and zero correctness rejections. Median
135M prefill / one-token times are 9.870 / 2.657 ms at one second, 7.398 / 1.692
at three seconds, 5.848 / 1.416 at five seconds, 5.395 / 1.416 at ten seconds,
5.547 / 1.431 at thirty seconds and 5.509 / 1.420 at sixty seconds. Ten seconds
recovers the prefill improvement in this workload; longer searches do not
produce a further measured gain. Small nonmonotonic differences remain. This
diagnostic does not change the cohort's thirty-second budget or establish a
budget sufficient for other models or GPUs.

Training has a different budget tradeoff. Nine additional post-reboot processes
compare 30-, 60- and 120-second searches, three per budget with reversed order
on the second repetition. Training medians are respectively 24.200, 22.321 and
22.030 ms: 60 seconds improves 7.8% over 30 seconds in this diagnostic, and 120
seconds improves 9.0%. The shorter searches check 14–15 training candidates,
60 seconds checks 34, and the 120-second allowance reaches the 64-program cap
after about 104 seconds. None is rejected for correctness. The later winner
keeps one cooperative dispatch instead of 31. This exposes additional useful
training search coverage; it does not change the cohort budget or establish
that a larger allowance is worthwhile for every workload.
The later training family passes an independent CPU F64 check of the full
outputs, loss and all gradients. A standalone 120-second real-text search
selects that family and passes 256 matched updates, probe losses and held-out
losses, with no mismatched steps; maximum training-loss difference is 0.00188.
A 60-second standalone curve also passes, but reaches only 26 candidates and
selects the earlier family, so it is not counted as learning evidence for the
faster family. Search coverage depends on the construction context as well as
the configured allowance.
Read-only RTX telemetry reaches 86 C during the later sustained training runs.
The final driver query reports zero software/hardware thermal-slowdown counters,
and the resumed kernel log contains no NVIDIA Xid or GPU wedge. Approximate
final-training-window clock medians span 2842–2857 MHz across the nine budget
runs; the two-second telemetry does not resolve individual timing samples.
These later RTX readings are separate from the earlier B570 failure below.

The matrix's fixed binary uses `8bf5efd`, whose production code is identical to
`9dafafa`; the only subsequent library changes share an existing GPU context in
two tests. The archived Inferena production patch exactly matches `03e6c9c`.
A clean build of the actual git pin then passes the full collector qualification:
ten model/profile conditions and 30 phase checks. Twelve cold/warm memo processes
also pass; cached programs still receive repeated qualification and whole-program
acceptance. This qualification-only collection is not a new publication cohort.

The final pinned 135M program passes independent PyTorch CPU float64 checks of
both forward phases, loss, and all 134,515,008 gradient elements under the
unchanged aggregate max/RMS contract. Both 360M forward exports pass the CPU
float64 distribution/loss gate too. Eight matched 256-update real-text
trajectories pass: both native versions/profiles, PyTorch F32/BF16/F16, and an
explicit native F16-permitted recipe. The fixed default accelerated curve's
maximum per-step relative loss error is 0.297%. This covers one checkpoint,
one corpus and one seed, not general training convergence.

Forty additional shape/input cases pass both native qualification and the
independent all-F64 CPU oracle. They cover 135M sequence lengths 1, 7, 31, 63,
127, 128, 129 and 257 with synthetic, real-text and random-token inputs;
selected cases use NaN-initialized scratch. Four ragged lengths and two poisoned
cases also cover 360M. All 1280 additional executions reproduce their selected
program's complete output exactly. These diagnostics test correctness and
repeatability, not cohort performance.

BF16 support is distinct from automatic selection. H100 exposes BF16 cooperative
matrices, but the archived default programs select no BF16 dispatches. MI300's
recorded RADV driver reports no BF16 shader/cooperative support; this is Vulkan
driver exposure, not a statement about MI300 hardware or ROCm. Explicit local
BF16 operands with F32 matrix outputs do qualify for forward execution. Three
alternating repetitions show:

| Model / phase | F16-permitted search (ms) | BF16 operands, F32 output (ms) |
| --- | ---: | ---: |
| 135M prefill | 5.501 | 5.744 |
| 135M one-token latency | 1.417 | 5.440 |
| 360M prefill | 12.112 | 11.843 |
| 360M one-token latency | 3.128 | 10.143 |

The F16-permitted search retains F32 scalar kernels. Blanket BF16 conversion
uses 211 or 225 cooperative dispatches, including in single-token execution.
Its small 360M prefill gain does not justify switching the whole model. Rounding
matrix outputs to BF16 additionally fails the 135M one-token mean-KL gate
(`0.010557 > ln(1.01)`). All three BF16 training recipes fail the existing initial
gradient-fidelity gate, so no learning trajectory was run for them. These are
contract rejections, not proof of faulty BF16 kernels. The diagnostic recipes
do not add profiles or change the collection default.

The offline MI300 audit covers the core and large-model archives: all 42 native
processes completed, but only 98 of 126 paired phases qualify. Missing pairs
come from PyTorch repeatability failures in core models and the BF16 1.7B
prefill worst-token KL gate (`0.103 > ln(1.1)`); later phases of that run were
not attempted. Recorded campaign durations total 142 minutes 20 seconds. Native
execution completion must not be confused with a fully qualified paired cohort.

The archived H100 135M prefill searches reject 55 candidates per repetition at
the old cross-program anchor; MI300 rejects 56. This establishes that the
acceptance defect also affects their search coverage. It does not establish
which alternatives would win whole-program timing on either server.

**The initial B570 performance comparison was interrupted.** Ten of 36 planned native
processes completed before the machine hung during the old control binary's
Whisper strict training construction/search. At 12:02:03 UTC the Intel `xe`
driver reported a failed GuC engine reset; at 12:02:04 it declared the device
wedged. Immediately before the timeout, sampled temperatures were 60 C package
and 64 C VRAM, with no throttling; the retained trace reaches 72 C package and
reports only `pl2` power-limit throttling. The journal contains no matching
thermal alarm, OOM, NVIDIA Xid, MCE or PCIe AER event during this investigation.
The timed-out job is logged as "not started." This points to GPU queue/recovery
failure rather than demonstrated overheating, but does not identify the
initiating shader, driver, firmware or hardware cause. B570 workloads remained
stopped for the rest of this initial investigation; the follow-up above records
the subsequent guarded runs.

The first accelerated B570 ResNet latency pair increased 14.5% (5.517 to
6.316 ms). Only one convolution class selected a different column layout,
affecting three dispatches. Six frozen-plan controls changing only those
dispatches produce process-median summaries of 6.342331 and 6.342615 ms, with
matching outputs on both the original zero image and a nonzero image. The same
program runs around 5.25 ms at 2750 MHz and 6.34 ms at 2100 MHz in read-only
clock traces; clock-matched layout differences are small. These controls do
not reconstruct the original pair's unrecorded clocks or turn the incomplete
B570 matrix into a passing no-regression result.

A post-reboot audit verifies all 60 completed core result documents, their
twenty samples per phase, the medians recomputed from those samples, serial
process execution and both retained binary hashes. All 84 fixed sessions have
repeated qualification; their 3292 recorded trials contain zero correctness
rejections. An unfinished analysis file and the sensor log's damaged trailing
row are retained separately from valid measurements. The hang interrupted the
six-hour investigation; downtime is excluded from its active-work accounting.

Evidence is in `../inferena-results/search-correctness-20261018/`: start with
`evidence-index.json`, `rejected-plan-replays-v2-5070/audit.json`,
`matched-v4-5070/comparison.json`,
`large-final-5070/comparison.json`, `bf16-forward-comparison.json`,
`bf16-training-checks.json`, and `qualified-final-5070/campaign.json`.
Binary hashes, source patches, individual samples, failures and CPU comparisons
are retained. The changes are in draft [Meganeura #265](https://github.com/kvark/meganeura/pull/265)
and [Inferena #77](https://github.com/kvark/inferena/pull/77); neither was merged.
No new H100 or MI300 execution was performed, so server recovery is unmeasured.

## Quest SmolLM2 CPU-oracle loss precision (October 17, 2026)

The v9 Quest qualification at Inferena `663e419` reported SmolLM2-135M training
mismatches in both arithmetic classes. Prefill, one-token latency, training loss,
and aggregate gradients passed. Four of 272 individual parameter gradients
failed only the maximum-error bound: layer 19 K projection, layer 19 input
RMSNorm, layer 27 input RMSNorm, and layer 29 post-attention RMSNorm. All RMS
bounds passed. Earlier `bce7adf` and `3a0129a` campaigns reproduce these errors
exactly; the failures were not transient device state or precision selection.

The CPU comparator called `bench._benchmark_loss`, which deliberately converts
logits to float32 for the timed AMP benchmark. Consequently the nominal float64
oracle injected float32 cross-entropy rounding into its backward pass. A small
scalar-loss difference concealed larger errors in some individual gradients.

`scripts/diagnose_android_lm.py` reproduces the original oracle and changes one
precision boundary at a time, using the identical saved native tensors:

| CPU-reference change | Failed parameter gradients |
| --- | ---: |
| Original float64 model with float32 loss | 4 |
| Float64 RMSNorm only | 4 |
| Float64 attention softmax only | 4 |
| Float64 rotary embeddings only | 4 |
| Float64 cross-entropy only | 0 |
| All four operations in float64 | 0 |

Promoting only the loss reduces aggregate gradient RMS error from `7.4556e-8`
to `2.3077e-8`, across 134,515,008 elements. Every individual gradient passes
the original `atol=1e-6, rtol=1e-4` maximum and RMS bounds. Diagnostic receipts
are in `../inferena-results/smollm2-root-cause-20261017/oracle-*.json`.

The correction gives both CPU tensor comparators a reference loss that preserves
their selected dtype. It does not change timed benchmark arithmetic. Android v10
receipts explicitly record and validate the loss dtype and policy; v9 failures
remain preserved. Fresh device qualification is required before cohort collection.

## PR #257 on current main (October 15, 2026)

PR [#257](https://github.com/kvark/meganeura/pull/257) at `056562b` is stacked
on `9e05731`; it does not apply cleanly to current main `f648a45`. A local
investigation branch, `investigate/pr257-on-main` in
`../meganeura-pr257-main`, ports its two commits while retaining main's Blade/Naga
pins, source-based shader API, rectangular BF16 support and balanced split-K
partitioning. The tested runtime is `b41fe20`. This is an adapted port, not
qualification of the stale PR head as-is.

The port also recognizes typed products in BF16 capability checks, updates
uniform packing and structural test assertions, and uses plan cache format 31:
main and the original PR both use 30 for different representations. Numerical
assertions are unchanged. The old BF16 generator API forwards to the new shared
generator. The PR reintroduces a separate typed shader entry; this is not a
recommendation to reverse main's shared-entry architecture.

Focused correctness checks pass on B570 and RTX 5070: 32 full BF16 matrix/partial
cases per GPU (859,216 exact values), 12 long-gradient cases, 18 master-gradient
cases, 18 reduced-gradient cases and 32 connected optimizer updates. Typed
partition/tuning probes cover three BF16 layouts on B570 and six F16/BF16 layouts
on RTX. Stored-F16-weight and skinny-matmul regressions also pass. Neither GPU
exposes native F32 cooperative matrices; those device tests return without
exercising that path. CPU checks pass 43 codegen tests, two BF16 capability/selection
contract tests and 16 cache tests.

Performance probes use the same probe source and dependency lock on each side, with
three fresh processes per variant, alternating order, disabled search, 20 held-out
samples of 32 steps, and complete F64 output checks on exactly representable
inputs. The first study uses two-second warmups, three shapes, both ordinary and
explicit BF16 storage, and ordinary/transposed-B products on both GPUs. All B570
median changes stay within 5% (explicit BF16: -0.11% to +2.35%). Its ordinary
products select unchanged scalar kernels. Small RTX timings have broad,
overlapping process ranges, including apparent BF16 regressions; every sample
and both signs of change remain in the retained analysis.

A second study uses five-second warmups and larger RTX products that all select
the affected ordinary F16-input cooperative path. Medians of three process
medians, in milliseconds:

| Product | Main | Adapted PR | Change |
| --- | ---: | ---: | ---: |
| 512 x 512 x 512, A B | 0.063781 | 0.060800 | -4.67% |
| 512 x 512 x 512, A B-transpose | 0.062257 | 0.096954 | +55.73% |
| 1024 x 1024 x 1024, A B | 0.199926 | 0.181779 | -9.08% |
| 1024 x 1024 x 1024, A B-transpose | 0.214321 | 0.417886 | +94.98% |

Every candidate process is slower than every main process for both transposed-B
cases. At 512 cubed the process ranges are 0.058886–0.077381 ms on main versus
0.096850–0.097675 ms with the PR; at 1024 cubed, 0.198296–0.215656 ms versus
0.383539–0.418353 ms. Ordinary-product ranges overlap, so their median reductions
are not a general speedup claim. All 18 probe processes and 168 matrix cases
pass, covering 26,370,048 exact F64-checked output values. These are kernel probes,
not full-model cohort timings or evidence of native-F32 correctness.

The PR routes ordinary native16 cooperative products into its checked generator.
Unlike the previous transposition-aware vec4 staging, its transposed-B loads
vary the output column across neighboring lanes and consequently read with a
K-sized stride. This is a plausible explanation for the reproducible transpose
regression; no hardware-counter attribution or isolated load-only repair was run.
Clock traces are retained and observational; clocks and power limits were not
changed. The typed-storage/tuning work should be separated from an unconditional
replacement of the existing transpose kernels, or the load strategy repaired
and requalified before merging.

Raw records, binary/source hashes, all process medians, build/test logs, retained
integration failures and the exact adapted patch are in
`../inferena-results/pr257-on-main-20261015/`. Start with `review.json`,
`analysis.json`, `probe-manifest.json` and `large/probe-manifest.json`. The
production Inferena dependency pin and frozen v27 collection are unchanged.

## V27 full local validation (October 15, 2026)

The repaired accelerated profile passes the complete local matrix: **60/60
publication cells, 180/180 phase checks and 20/20 repeated-gradient groups**.
Both B570 and RTX 5070 run five models, strict and accelerated, with three fresh
processes per condition. B570 takes 101.3 minutes and RTX 5070 takes 77.8 minutes.
The frozen measurement source is Inferena `f587313`, Meganeura `b344d49`, paired
protocol v27 and PyTorch runner v16. Later report-only commits do not change that
measurement identity; use the frozen revision when extending this collection.

Independent audits recompute the phase and repeated-gradient checks, accelerated
loss/KL/learning arithmetic, and publication eligibility. They also verify all
124 executable guards, archived binaries and source, input hashes, actual
compiler benchmarkers, and the 20 empty initial model/profile kernel memos.
Subsequent repeats reuse their declared native memo; PyTorch compiler caches are
fresh per process, and driver cache state remains uncontrolled. There are 168
constructed native sessions: Whisper uses its qualified forward session for
both inference and latency. All 180 timed phase checks are still required.

Evidence is in `../inferena-results/accelerated-v27-20261015-r2/`, especially
`audit.json`, `quality-evidence.json`, `completion-evidence.json`, `cohort.md` and
`summary-for-report.json`. The original records and separate v22/v26 comparisons
are retained. No current phase median exceeds the 5% review threshold against
the frozen v22 baseline. The previously excluded v22 SmolLM2 latency cells have
no old timing and cannot establish speedups.

The XPU profiler repair carries into the primary collection. B570 PyTorch
SmolVLA inference/latency/training medians are **2.517/2.223/7.142 ms**, versus
**2.584/2.459/7.304 ms** in v26: changes of **-2.58%/-9.58%/-2.22%**. Each
phase's three process medians span less than 0.2%. These primary deltas are
separate from the earlier alternating benchmarker controls below.

The native search repair retains the Whisper improvement over merged `99e3de2`:

| GPU | Phase | v22 (ms) | v27 (ms) | Change |
| --- | --- | ---: | ---: | ---: |
| B570 | Inference | 13.791 | 12.367 | -10.3% |
| B570 | Latency | 13.798 | 12.301 | -10.8% |
| B570 | Training | 61.823 | 61.417 | -0.7% |
| RTX 5070 | Inference | 4.835 | 4.498 | -7.0% |
| RTX 5070 | Latency | 4.868 | 4.523 | -7.1% |
| RTX 5070 | Training | 23.840 | 23.023 | -3.4% |

The separate v26 comparison retains two B570 native flags: accelerated diffusion
inference **4.823 to 5.793 ms (+20.12%)**, and strict ResNet latency **5.460 to
6.143 ms (+12.50%)**. Every comparable RTX phase stays below the review threshold against
both baselines. The native executable is byte-identical between v26 and v27
(using the v26 build-provenance reconstruction documented below);
selected programs and operating clocks can still differ. These flags remain
visible even though they do not appear against v22. Single-choice, qualified
program-replay controls now complete the timing review below.

### Native timing controls

Both campaigns run the same native executable. Selected tuning receipts expose
one changed convolution class in each flagged phase: diffusion changes its
16-wide convolution's K stage from 16 to 32; ResNet changes a 32-wide, K16
convolution from contiguous to interleaved columns. The latter class appears in
five dispatches. The controls capture qualified programs, lock their complete
execution plans, and change only those kernel fields. Actual pipeline keys
confirm exactly one changed diffusion dispatch or five changed ResNet
dispatches, with every other phase and dispatch identical between variants.

Each variant runs in three fresh processes, alternating order. All **36 replay
phase checks and six capture phase checks pass** against original F32. Every
process also passes the retained strict PyTorch output/loss/gradient-norm
comparison. Every replay preserves its complete supplied execution plan.
The retained instrumented binary uses Meganeura `b344d49`; its native runner
source matches `f587313` apart from trace
and replay instrumentation. These diagnostics use 40 seconds of traced warmup
and 20 held-out samples, so they remain separate from the primary 10-second
warmup measurements. No clocks, power limits or primary samples are changed.

| Controlled phase | v26 kernel choice (ms) | v27 kernel choice (ms) | Change |
| --- | ---: | ---: | ---: |
| Diffusion accelerated inference | 5.632 | 5.114 | -9.19% |
| ResNet strict latency | 6.174 | 6.365 | +3.10% |

These are medians of all three process medians. Diffusion process medians span
5.000–5.640 ms with the earlier choice and 4.960–5.644 ms with the current one.
At observed clocks repeated in at least two processes per variant, diffusion's
warmup-trace differences are -0.92% at 1350 MHz, -1.14% at 1500 MHz and +0.26%
at 2750 MHz. ResNet's are +1.27% at 2100 MHz and -0.85% at 2750 MHz.

The same fixed diffusion program has stable one-second bins around 3.32 ms at
2750 MHz and 5.78 ms at 1300 MHz. ResNet likewise moves from about 5.26 ms at
2750 MHz to 6.37 ms at 2100 MHz. This exposes operating-state variation large
enough to affect the cohort comparisons. Neither single-choice intervention
reproduces its original slowdown. Clock matching uses warmup iterations,
requires equal nonzero readings bracketing each complete bin and no throttle
flag or sensor gap above 0.5 seconds, and remains observational. It supplies no
frequency-normalized replacement timings or exact decomposition of the
historical flags. The unchanged-phase controls stay within 1.74% in their
process medians.

The first diffusion capture selected a different attention layout for inference.
Its qualified latency plan matches the intended layout and has identical
inputs, parameters, buffers and original-F32 reference outputs; that plan is
the common inference replay background. The initial identity-check failure is
retained. A diagnostic reader also initially assumed that strict records carried
the accelerated-only reference field; fixing the reader rechecks the completed
ResNet capture. Neither GPU capture is repeated or selected by its timing.
Evidence, original failures, source hashes, complete plans, per-step traces and
the final independent audit are in
`../inferena-results/slowdown-quality-20261014/v27-b570-native-choice-controls/`,
especially `analysis.json` and `manifest.json`.

The quality policy is unchanged. RTX search rejects 186 candidates under its
existing numerical checks; B570 rejects 13 for scratch limits. All 186 RTX
failures are the tighter search-anchor raw-output check, so the count must not
be described as 186 failures of the broader language-model KL/loss contract.
Every selected program passes its original-F32 gate. No native session selects
BF16, so this collection establishes BF16 permission and qualified PyTorch AMP,
not a native BF16 performance benefit. The 256-update real-text learning check
applies to PyTorch SmolLM2; native training retains full-gradient qualification
and the separate cross-engine convergence study. Other models' repeated
synthetic updates are numerical checks, not real-data convergence evidence.

The local profile is qualified and ready for collection under the frozen
protocol, with the timing variability and both v26 flags retained. H100, M3 and
the integrated GPUs still need their own runs; these results do not certify
those devices or prove equivalence on untested data. Qualification and reference
construction remain outside step timing, with their costs reported separately.
There are still only two profiles and 30 pairs per GPU.

## XPU kernel autotuning repair (October 15, 2026)

The fixed-configuration intervention below establishes that Triton launch
choices can cause an 11.19% timing difference for identical source and graph.
The repair uses PyTorch's built-in `TorchProfilerBenchmarker` on XPU, selected
through its public environment setting before importing the compiler. It does
not seed a fast cache or remove candidate optimizations. Every process retains
default shape padding, fresh compiler caches and the same compilation deadline,
arithmetic, replay, numerical gates and final step timer.

After one successful feasibility probe, a separate alternating comparison runs
three fresh processes per method from the same archived v26 source `a220bf3`:

| SmolVLA phase | Default benchmarker (ms) | Profiler benchmarker (ms) | Change |
| --- | ---: | ---: | ---: |
| Inference | 2.768 | 2.520 | -8.98% |
| Latency | 2.468 | 2.225 | -9.84% |
| Training | 7.684 | 7.141 | -7.06% |

All 18 phase checks pass. Profiler inference spans 2.516–2.521 ms, latency
2.224–2.228 ms and training 7.139–7.144 ms. Its compilation spans
49.647–49.793 seconds, within the unchanged 120-second limit, versus
51.672–53.713 seconds for the default benchmarker. All three profiler inference
modules have identical generated source, while the default modules differ.
The feasibility probe is retained separately and excluded from these medians.
Evidence is in `../inferena-results/slowdown-quality-20261014/`, under
`v26-b570-smolvla-profiler-feasibility` and `v26-b570-smolvla-benchmarker-controls`.
The latter contains raw quality receipts, clock traces, generated code, selected
configurations and the independently recomputed analysis.

The v27 collector and PyTorch runner v16 declare the XPU policy for both strict
and accelerated. The wrapper selects it before the runner imports torch; each
compiled result reports the actual instantiated benchmarker and both controlling
settings. The collector verifies those receipts and rejects inherited method
overrides. Eager diagnostics remain outside the compilation policy. CUDA and
other backends retain the pinned default method. No quality threshold changes.

Both runner suites pass 39 tests, including GPU replay checks, and all 28
collector tests pass. New regression coverage checks that the wrapper overrides
conflicting inherited settings before launching the runner, and that the
collector rejects missing or wrong benchmarker receipts, including an XPU run
that still used the default method. Historical campaign versions remain readable
and cannot be pooled with v27. The full five-model validation above now passes
on both GPUs; its separate native timing flags have qualified replay controls.

Production-wrapper SmolLM2 preflights also pass all three phases and the full
256-update real-text gate on both GPUs. XPU uses `TorchProfilerBenchmarker` and
compiles in 89.598 seconds; CUDA retains `InductorBenchmarker` and compiles in
52.772 seconds. Their inference/latency/training timings are
4.344/1.776/12.587 ms and 2.944/1.639/8.838 ms, respectively. These remain
single-process diagnostics in `xpu-benchmarker-integration`.

The final standalone harness check catches a separate dependency declaration
bug before collection: building the harness alone loses
`serde_json/float_roundtrip`, changing a retained F64 probe-loss ratio by one ULP.
Its existing exact-receipt test fails in that build. The combined harness/native
build used for earlier collections already enables that feature through the
native package, masking the missing declaration. The harness now inherits the
workspace's explicit JSON feature, and CI tests the two crates separately.
This preserves the original failed test and prepared source evidence; no v27
cohort had started at that point.

## V26 full local collection (October 15, 2026)

The frozen collection uses Inferena `a220bf3`, Meganeura `b344d49`, paired
protocol v26 and PyTorch runner v15. Both GPUs complete all five models,
strict and accelerated, with three independent processes per condition:
**60/60 publication cells, 180/180 phase checks and 20/20 repeated-gradient
groups pass**. B570 takes 102.3 minutes and RTX 5070 takes 77.8 minutes.
SmolLM2 uses declared F32 one-token latency and BF16 prefill/training. Native
Auto selects no BF16 sessions, so these results do not establish a native BF16
performance benefit. Numerical thresholds and search budgets are unchanged.

The complete records, raw quality audit and comparison to frozen v22 are in
`../inferena-results/accelerated-v26-20261015/`, especially `cohort.md`,
`audit.json` and `quality-evidence.json`. The v22 SmolLM2 BF16 latency exclusions
remain exclusions; a missing old cell cannot establish a speedup.

Native Whisper accelerated inference/latency improve 11.0%/10.9% on B570 and
5.6%/6.1% on RTX. B570 training is unchanged within 0.2%; RTX training improves
3.5%. The selected search receipts confirm that formerly skipped layouts now
receive timing comparisons. Every RTX phase in both profiles stays within the
predeclared 5% regression limit. B570 has two remaining flags:

| Engine and accelerated phase | v22 (ms) | v26 (ms) | Change |
| --- | ---: | ---: | ---: |
| Meganeura diffusion training | 11.499 | 12.149 | +5.66% |
| PyTorch SmolVLA latency | 2.318 | 2.459 | +6.09% |

These are medians of three process medians. They remain in the primary report;
supplemental controls cannot replace them. The native diffusion processes select
graph 2, graph 2 and graph 13, with held-out training times 12.149, 13.147 and
11.476 ms. The second search reaches and qualifies graph 13, but its paired gain
of 0.268 ms falls just below the unchanged 0.273 ms noise-aware acceptance bound.
Its incumbent drifts from 10.270 ms when selected to 13.217 ms in the later
comparison. Coverage alone therefore does not explain this flag.

Three alternating old/new controls per engine are retained in
`../inferena-results/slowdown-quality-20261014/v26-b570-diffusion-controls/` and
`v26-b570-smolvla-pytorch-controls/`. Native controls retain full plans and
40-second traces; PyTorch controls use each archived source, fresh compiler
caches and the ordinary ten-second warmup. Both record clocks without changing
power or frequency settings. Read-only clock matching alone cannot establish
causality, and the older PyTorch controls lack v26's independent latency F32 gate.

All six native controls pass the original-F32 and preserved cross-engine checks:

| Diffusion control phase | Merged `99e3de2` (ms) | Fixed `b344d49` (ms) | Change |
| --- | ---: | ---: | ---: |
| Inference | 5.264 | 5.105 | -3.03% |
| Latency | 4.855 | 5.650 | +16.39% |
| Training | 11.443 | 11.653 | +1.84% |

The training flag does not repeat at the 5% level. Old and fixed searches both
select graphs 2 and 13, with differing private convolution kernel decisions;
the graph label alone does not establish an identical program. The supplemental
latency gap is retained too: the longer warmup reaches different clock states.
Stable frequency bins present in at least two processes of each variant give
old/new latency medians of 5.632/5.762 ms at 1,300 MHz and 4.860/4.981 ms at
1,550 MHz (+2.31%/+2.48%). These warmup comparisons support clock variation as a
contributor, but cannot substitute for the held-out observations or identify its
driver/firmware cause. Training has no frequency with two processes of each
variant, so its individual clock matches remain diagnostic. The original
cohort's training flag remains visible; this study does not reproduce a
consistent material regression from the search change.

All six PyTorch controls pass their archived qualification policies:

| SmolVLA control phase | v22 (ms) | v26 (ms) | Change |
| --- | ---: | ---: | ---: |
| Inference | 2.625 | 2.818 | +7.37% |
| Latency | 2.314 | 2.282 | -1.40% |
| Training | 7.287 | 7.467 | +2.48% |

The original latency flag does not reproduce. All six generated latency modules,
their launch methods and all 26 embedded Triton definitions have identical
Python ASTs, including their numerical constants, decorators and compiler/launch
settings. Only comments and source positions are removed, including those in
embedded Triton source. `latency-code-identity.json` retains source hashes and
the comparison method. This proves generated source identity, not identical
driver machine code or runtime autotuner choices: the diagnostic archive did
not retain `.zebin` or `.best_config` files. Latency varies within each source
at the same observed 2,750 MHz, so clocks alone do not explain that variation.

The **supplemental inference flag requires separate controls**. It persists at the same observed
2,750 MHz. Unlike latency, the generated inference plans differ across fresh
processes, including within the same source revision: they contain 249, 265 or
281 Triton calls alongside the same 115 `mm`, 32 `bmm` and one `addmm` calls.
V26's second and third processes generate identical inference modules and run
at 2.818 and 2.827 ms. V22 also has a slower 2.857 ms process, but its median is
2.625 ms. This identifies compiler plan variability; it does not yet establish
why the plans differ or rule out a source-dependent bias. Subsequent controls
below identify two compiler mechanisms and establish the effect of kernel
selection. The full primary cohort still has only the two original flags.
Neither successful quality gates nor the disappearance of those flags in
controls establishes universal absence of performance regressions.

### SmolVLA: compiler choices cause repeatable timing differences

Fixing `PYTHONHASHSEED=0` does not stabilize the inference graph. Three further
processes per source all qualify, but inference medians are 2.563/2.852 ms
(+11.27%); latency is 2.459/2.235 ms (-9.14%). These unfavorable inference
results are retained in `v26-b570-smolvla-pytorch-hashseed0` under the same
diagnostic root. Hash seed is not a production repair.

A separate old/new compiler-IR pair identifies a difference before fusion:
Inductor's shape-padding pass sometimes expands 50-row matmuls to 56 rows, then
slices their outputs. The captured pair differs by 16 such padding operations
across the transformer layers. The pass benchmarks each shape and reuses its
decision; model inputs and parameters have identical graph argument signatures.
`v26-b570-smolvla-compiler-ir/padding-ir-finding.json` binds the IR evidence.
This explains varying graph construction, but does not by itself establish its
full-model cost.

Six same-source v26 controls compare default padding with the public
`shape_padding=False` option. All 18 phase checks pass. Disabling padding changes
the inference median from 2.587 to 2.599 ms (+0.46%), latency from 2.242 to
2.243 ms (+0.03%) and training from 7.664 to 7.254 ms (-5.36%). It does not
consistently improve inference. The three padding-off inference modules have
identical complete Python ASTs, including embedded Triton definitions, yet take
2.822, 2.507 and 2.599 ms. The retained `.best_config` files reveal differing
Triton block sizes and warp counts despite identical generated source.

A controlled intervention then freezes those choices. Both variants use the
same v26 source, padding disabled, BF16, XPU graph replay, 120-second compile
budget, ten-second warmup, twenty samples and every v26 numerical gate. Each
fresh compiler cache is seeded **only with the 19 inference kernel choice
records** from the first or second padding-off process. No generated source,
machine code or tensors are seeded, and PyTorch's cache validation stays enabled.
Thirteen choices differ between these two sets. The six processes alternate
order and produce the following inference times:

| Seeded choices | Process 1 (ms) | Process 2 (ms) | Process 3 (ms) | Median (ms) |
| --- | ---: | ---: | ---: | ---: |
| Padding-off process 1 | 2.822 | 2.835 | 2.820 | 2.822 |
| Padding-off process 2 | 2.505 | 2.506 | 2.508 | 2.506 |

All 18 phase checks pass. All six inference modules and their 28 kernel source
ASTs match; each process retains its expected 19 selected configurations and
matching compiled artifacts. All measured blocks have the same observed
2,750 MHz. This establishes a repeatable **11.19% reduction from changing the
selected kernel configurations**, with less than 0.6% spread within either
set. It is stronger evidence than a graph-count or clock correlation.
`v26-b570-smolvla-fixed-configs/causal-finding.json` records the checks and binds
the raw measurements and artifact analysis. Other phases still tune normally;
their timings are retained, without attributing their variation to this
inference intervention.

These controls establish a cause of substantial within-source variation. They
do not measure how much of either earlier old/new inference gap came from each
compiler choice, or exclude a source-dependent bias in autotuning. The installed
PyTorch uses `InductorBenchmarker` for those choices, with its experimental
event-based benchmarker enabled and profiler-based benchmarking disabled.
Default-reference timing stability remains the open investigation. Production
still uses its declared fresh compiler caches and default padding/tuning policy;
it does not adopt a hand-selected fast cache, discard a slower qualified run,
or replace any primary timing. Numerical qualification on both GPUs is complete;
blanket performance clearance is not claimed.

### Executable provenance

The enclosing two-GPU supervisor reports a final native binary hash failure,
although both device campaigns completed. Exact cached Cargo reproduction
resolves this as a provenance bug: a combined harness/native build selects
SHA-256 `51db385e736a7abcdf786fb627e7c1a2bbf3e93353216aceef29f536a01be41f`,
whereas the native wrapper's single-package build selects
`eee1fd3d672355a796db7076dbc122f5c0bd6e68c9d8144d34d601161c4ce60a`.
The feature union adds `bitflags/std` only to the combined build. Both cached
artifacts predate the collection; all 60 native wrapper invocations use the
single-package recipe and show no compilation. The harness hash is unchanged.
Per-process executable hashes were not originally captured; native identity is
reconstructed from the wrapper logs, preserved Cargo fingerprints, clean frozen
source and exact artifact reproduction. `binary-provenance-resolution.json`
records this limitation and binds the audit evidence. The original failed
supervisor states and initial source receipt remain unchanged.

The collector repair prepares both build recipes before device probing,
archives the wrapper-selected executables and checks their hashes around every
pair and at campaign completion. A provenance failure keeps the campaign
incomplete and withholds its publication timings. All 28 collector tests pass,
including injected feature-union replacement and missing-binary failures and
publication exclusion after a final provenance failure. Running the repaired
build/archive helper against the frozen source selects byte-identical measured
executables; `collector-provenance-check/result.json` records the proof. This
repair changes collection metadata and build preparation, not timed binaries,
quality thresholds or benchmark settings.

## Simpler SmolLM2 latency policy (October 15, 2026)

The v25 attempt at `1787ddc` exposed avoidable cost in the FP16/F32 projection
wrapper. Its first B570 SmolLM2 strict one-token result is 1.802 ms, versus
2.857 ms for the accelerated wrapper, with the same source and hardware.
The prior RTX strict median is 1.644 ms, versus 1.937 ms in the qualified wrapper
preflight. Accelerated permits reduced precision; it need not require it when
F32 is faster and passes the same quality contract.

V26 therefore declares F32 for SmolLM2-135M one-token latency, with TF32 disabled,
and retains capability-selected BF16 for prefill and training on these GPUs.
It removes all projection wrappers and architecture-specific module paths.
Explicit dtype requests remain unchanged. The recipe is frozen before execution;
there is no retry after failed qualification and no quality threshold change.
The paired protocol is v26 and the shared PyTorch runner protocol is v15.

The v25 collection was stopped after seven completed B570 pairs and an interrupted
eighth pair; RTX had not started. It is retained in
`../inferena-results/accelerated-v25-20261015/` as incomplete diagnostic evidence.
Its first accelerated pass qualifies all 15 phases, and native Whisper inference
and latency improve about 10% versus v22. B570 SmolVLA latency in both engines and
native diffusion training each show roughly 10% slower first-repeat timings.
These need replicated confirmation and, if persistent, matched controls before
claiming regression clearance. Their selected graph descriptions alone do not
establish identical plans or explain their performance.

Both Python runner suites pass 38 tests, the collector suite passes 26 tests,
and the Rust harness/native suites, scoped formatting and locked release build
pass after removing the obsolete wrapper tests. The regression tests retain
independent latency autocast checks and reject undeclared phase arithmetic.

Production preflights pass inference, latency, graph replay and all 256 matched
real-text updates on both GPUs, with the same 120-second compile limit, fresh
compiler cache, ten-second phase warmup and twenty samples:

| GPU | Prefill (ms) | One-token latency (ms) | Training (ms) | Compilation (s) |
| --- | ---: | ---: | ---: | ---: |
| B570 | 4.453 | 1.813 | 12.639 | 90.64 |
| RTX 5070 | 2.934 | 1.640 | 8.835 | 53.30 |

One-token mean KL is `5.44e-11` / `1.26e-11`, against the unchanged `ln(1.01)`
limit. B570 reference preparation and precision qualification take 18.66 and
9.35 seconds, separate from compilation and step timing. Evidence and archived
source are in `../inferena-results/slowdown-quality-20261014/qualify-f32-latency-{xpu,cuda}`.
These are single-process preflights; the new full collection must establish
replicated qualification and performance before readiness is claimed.

## Accelerated latency coverage audit (October 15, 2026)

The v24 collection at `d501a36` was stopped after eight completed B570 pairs.
The first accelerated pass passed the existing checks for all five models, but
an independent receipt audit found that those checks did not cover every latency
shape. PyTorch compared the full ResNet batch and SmolVLA action sequence against
F32; their batch-one and one-action latency workloads only had same-arithmetic
replay checks. Native sessions already checked their separate latency programs.

A CPU reproducer makes only the smaller-shape output incorrect. Before the fix,
both ResNet and SmolVLA runner records incorrectly report success. The regression
test now excludes latency timings while retaining qualified inference and
training. Compilation, F32 qualification and timing share the latency input
definition. The collector requires a separate receipt and checks its output
shape, preventing an inference receipt from qualifying a different workload.
All numerical thresholds and arithmetic recipes remain unchanged.

The stopped collection and reproducer are retained in
`../inferena-results/accelerated-v24-20261014-r2/`; its partial observations are
diagnostics, not cohort results. This coverage correction requires fresh
collection under paired protocol v25 / PyTorch runner v14. Native search and
its `b344d49` pin remain unchanged. For non-language models it adds two forward
checks per accelerated process, outside step timing, without another column or
a larger search budget. The failing-before/passing-after CPU regression covers
both affected shapes. All 40 Python runner tests pass on each GPU; all 26
collector tests, the Rust harness/native suites, scoped formatting and the locked
release build pass.

Production preflights for ResNet and SmolVLA pass inference, latency, training and
replay checks on both B570 and RTX 5070. Each uses automatic BF16, a fresh compiler
cache, the 120-second compilation limit, ten-second phase warmup and 20 samples.
The new latency comparisons cover `[1, 1000]` and `[1, 1, 32]` respectively; the
largest maximum/RMS error uses 31% of its unchanged bound. Compilation takes
19.6–54.8 seconds. These four standalone preflights are in
`../inferena-results/slowdown-quality-20261014/qualify-latency-coverage-{xpu,cuda}`;
they establish coverage and numerical behavior, not replicated cohort performance.

## Search slowdown and accelerated recipe (October 14, 2026)

The v22 failures are retained unchanged. This investigation keeps all v22
numerical thresholds and the two-profile matrix. Evidence is in
`../inferena-results/slowdown-quality-20261014/`; prototype measurements below
are diagnostic preflights, not replacement cohort records.

### Whisper: dispatch equality was skipping different shaders

Meganeura's search skipped whole-program timing when dispatch lists and
submission counts matched. Attention knobs can produce different shaders with
identical dispatch geometry. The B570 v22 strict Whisper run qualified trial 46
(EPT16, 256 threads, eight keys, interleaved, fused), then retained the incumbent
without collecting a single timing sample. That layout had won the historical
cohort. RTX trial 50 similarly skipped the fused layout before timing and
selecting its slower unfused counterpart. The retained candidate receipts are
summarized in `whisper-skipped-candidates.json`.

Commit `acb799a` compares complete execution plans before skipping a trial. A GPU
regression test fails before the fix (zero timing pairs instead of four) and
passes after it; an exact duplicate still skips timing. This alone exposed a
budget problem: timing the formerly skipped shaders exhausted the cold 30-second
search before reaching B570's fast layout. Commit `b344d49` prioritizes attention
layouts with the incumbent's winning fusion setting, while retaining every
combination. It does not add a vendor-specific layout preference or enlarge the
search budget. All 20 search tests pass on B570 and RTX 5070, and all-target,
all-feature Clippy passes.
The changes are published for review in
[Meganeura PR #260](https://github.com/kvark/meganeura/pull/260).
The dispatch-only shortcut was introduced by `07445e8` after the frozen October
11 pin; its removal has a failing-before/passing-after GPU regression test.
PR #260's GitHub CI also passes.

A cold B570 control and fixed run use fresh driver caches, no native memo,
30-second search, 40-second traced warmup and 20 held-out samples. Both pass the
preserved PyTorch F32 output/loss/gradient comparison:

| Phase | Merged `99e3de2` (ms) | Fixed `b344d49` (ms) |
| --- | ---: | ---: |
| Inference | 13.636 | 12.373 |
| Latency | 13.675 | 12.282 |
| Training | 61.696 | 61.673 |

The fixed search reaches the previously skipped B570 layout. These are single
process controls, with a longer diagnostic warmup; the full collection is still
required to establish replicated performance and correctness.

The corresponding RTX 5070 control also chooses the previously skipped fused
layout. Inference improves from 5.643 to 5.482 ms, latency from 5.680 to 5.455 ms,
and training from 24.689 to 23.861 ms. Both processes pass the same preserved F32
cross-engine comparison. These are also single-process controls, retained in
`rtx-whisper-{current,ordered}-plan-trace` and `whisper-controls.json`.

### Diffusion: changing operating clocks confound the historical comparison

Rebuilding the historical Inferena `0a718b6` / Meganeura `5b37124` binary reproduces
slow current-day B570 timings. Historical and current programs have the same
300-dispatch inference kernel inventory. Within one fixed historical session,
inference drifts from about 3.93 ms near 2,025 MHz to 5.78 ms at 1,300 MHz.
At frequency-matched stable one-second bins, old/current medians are 5.785/5.786 ms
at 1,300 MHz and 5.497/5.498 ms at 1,400 MHz; training is 11.640/11.593 ms at
2,250 MHz. No thermal or power throttle flag is observed. Actual and requested
GPU clocks track each other, but this does not identify the underlying driver
or firmware policy. No clock, power or governor settings were changed.

This is evidence against attributing the historical timing gap to a kernel
regression. It does not reconstruct the original cohort's operating conditions.
Raw traces and sensors are in `b570-historical-current-trace`.

### SmolLM2: a declared precision boundary, with unchanged quality bounds

Plain BF16 fails the one-token distribution gate in eager as well as compiled
execution. Keeping only the output head in F32 does not resolve it. F32 compiled
execution passes; protecting MLP projections localizes the sensitivity more
strongly than protecting attention projections. Ordinary FP16 improves the
result but still fails one of 41 compiled input probes on each GPU. Changing
Inductor's precision-cast emulation does not fix that failure.

The initial prototype uses FP16 with the seven linear projections in the
final decoder block in F32. It preserves the same master parameters and state
names, and passes all 41 compiled probes on both GPUs (the original nine plus
32 seeded held-out token/target pairs). This boundary covers only the linear
projections, not every operation in the block. Explicit `fp16`, `bf16` and `off`
requests remain unmodified diagnostics. The manifest declares each model's
recipe before execution; a failed gate cannot select another recipe.

The full prototype PyTorch runner also passes prefill, one-token latency,
execution/replay checks, 256 matched real-text SGD updates, and all fixed training
probe and held-out checks on both GPUs. No threshold was widened:

| GPU | One-token mean KL | KL limit | Training curve relative L2 (diagnostic) | Updates |
| --- | ---: | ---: | ---: | ---: |
| B570 | 0.0001531 | 0.0099503 | 0.002153 | 256/256 |
| RTX 5070 | 0.0001439 | 0.0099503 | 0.002143 | 256/256 |

Each run uses explicit whole-phase replay, a fresh compiler cache, the 120-second
compile budget, ten-second phase warmup and twenty measurement samples.
Compilation takes 109.3 seconds on B570 and 57.1 seconds on RTX. These prototypes
inject the boundary before model preparation and label it separately in their
records; they do not claim to be v23 collection. Production wiring uses PyTorch
runner v12 and paired protocol v23, and requires its own full frozen-source run.
The quality contract remains bounded numerical fidelity on these inputs, not
universal generation quality or convergence evidence for every model.

The recipe has a measured cost that must remain visible: prototype FP16 training
is 16.72 ms on B570 and 10.12 ms on RTX, versus v22 BF16 process-median aggregates
of 12.61 and 8.86 ms. FP16's required unscale/finite-check/scale-update work is
inside the timed step. These unmatched prototype numbers do not isolate all of
the cost, but they rule out claiming a blanket no-regression result. Prefill is
similar (4.63/2.90 ms versus 4.67/2.96 ms). The full v23 run must report the new
recipe's training cost separately from unchanged-workload native regressions.

That cost is avoidable because the failure is specific to one-token latency.
The v23 collection at `7cbb866` was interrupted during its first pair and retained
as superseded in `../inferena-results/accelerated-v23-20261014/`; it supplies no
completed cohort. V24 instead declares FP16/F32 only for the one-token workload,
while prefill and training retain their already-qualified capability-selected
BF16 policy on these GPUs. A static sequence-length branch in the seven wrapper
projections leaves longer inputs and their backward arithmetic unchanged. Unit
tests compare those BF16 outputs and gradients exactly, check the independent
latency autocast context, and reject mismatched phase-policy receipts. The
production runner is v13. This narrower recipe requires fresh full qualification
and collection; neither earlier failures nor interrupted v23 attempts are reused.

Both production preflights pass all phases, replay checks and 256 training
updates with the narrower recipe:

| GPU | Prefill (ms) | One-token latency (ms) | Training (ms) | Compilation (s) |
| --- | ---: | ---: | ---: | ---: |
| B570 | 4.267 | 2.760 | 12.905 | 95.98 |
| RTX 5070 | 2.960 | 1.937 | 8.861 | 55.28 |

One-token mean KL remains 0.0001531 / 0.0001439. Training is within 2.4% / 0.1%
of the prior BF16 cohort medians (12.607 / 8.857 ms), avoiding the full-model FP16
prototype's 16.722 / 10.119 ms cost. These remain single-process preflights;
all phase metadata and raw gates are retained in `qualify-phase-{xpu,cuda}`.
The Python runner suite passes 39 tests on each GPU, and all 26 collector tests
pass. The native pin and Meganeura binary are unchanged from the tested search fix.

The first v24 paired attempt at `bd69a08` caught a consumer-version bug: the
Rust harness still expected PyTorch v11 and rejected the v13 result. The attempt
was stopped and retained in `../inferena-results/accelerated-v24-20261014/`, with
no promoted timings. The runner, collector and Rust harness now read one shared
`frameworks/pytorch/PROTOCOL` file. A subprocess regression test reproduces the
rejection before the fix, accepts the current protocol after it, and continues
to reject stale records. Fresh collection requires the corrected common revision;
this integration fix does not change arithmetic or quality thresholds. Both real
GPU preflight receipts also fail through the preserved old harness and pass
through the rebuilt one (`protocol-receipt-replay/verification.json`).

## V22 full local collection (October 14, 2026)

The agreed acceptance changes are implemented in `p3hpc-paired-campaign-v22`:
pretrained language-model logits use full distribution and labelled-loss checks,
and training retains the per-loss 2% bound without the additional 1% curve-L2
veto. Strict checks, native full-gradient checks and execution repeatability are
unchanged. There are still exactly two profiles, `strict` and `accelerated`.

Both full collections use **Inferena `1ed4df7`**, merged Meganeura `99e3de2`,
Blade `96c1dc7`, Naga `a97559b6`, and PyTorch 2.13.0 vendor wheels from the same
PyTorch commit. Each GPU runs five models × two profiles × three fresh processes:
SmolLM2-135M, SmolVLA, StableDiffusion, ResNet-50 and Whisper-tiny. B570 runs first;
GPU jobs are serial. The source stays clean and frozen across both collections.

| GPU | Strict phases | Accelerated phases | Replicated-gradient groups | Wall time |
| --- | ---: | ---: | ---: | ---: |
| B570 | 45/45 | 42/45 | 10/10 pass | 106.9 min |
| RTX 5070 | 45/45 | 42/45 | 10/10 pass | 80.0 min |

All 60 paired runs finish within their watchdogs. All inference and training
phases qualify. The six failed observations are the three accelerated SmolLM2
one-token latency repeats on each GPU. Their PyTorch timings remain absent;
eager diagnostic timings never replace them. The paired report contains **58/60
qualified model/profile/phase cells**, requiring all three replicates for each
cell. Both campaigns correctly finish `complete-with-failures`.

Search remains 30 seconds per native session, reference compilation 120 seconds,
warmup at least ten seconds and five calls per phase, and measurement twenty
samples. Each campaign starts with empty kernel memos; each records ten misses
and 74 later hits. Driver shader-cache contents are separately labelled
uncontrolled. Native F32-reference preparation is included in preparation cost:
115.8 seconds total on B570 and 38.8 seconds on RTX. Wall times above include the
failed-setting eager diagnostics. Mesa is `26.0.3-1ubuntu1`, NVIDIA's driver is
`595.91.07`, and the host kernel is `7.0.0-38-generic`.

### Remaining one-token precision exclusion

PyTorch's compiled BF16 prefill now passes the distribution/loss contract, and
all six 256-update real-text training/probe/held-out qualifications pass. The
latency failure is repeatable:

| GPU | Prefill mean KL | One-token KL, all three repeats | One-token limit |
| --- | ---: | ---: | ---: |
| B570 | 0.001547–0.001608 | 0.012834 | ln(1.01) = 0.009950 |
| RTX 5070 | 0.001806 | 0.013273 | ln(1.01) = 0.009950 |

The compiled one-token labelled-loss check passes; mean KL fails. With one token,
the mean budget is also a per-token budget. The first prefill token also exceeds
that mean limit individually, while the entire sequence passes. Eager BF16 fails
one-token qualification too, including its labelled loss, so graph replay alone
does not explain the exclusion.

An untimed follow-up checks the benchmark token plus the first token/target pair
from each of the eight frozen probe/held-out blocks, without a KV cache. Compiled
BF16 passes only **2/9 cases on B570 and 3/9 on RTX**. Pooling those nine cases
still fails the unchanged mean-KL limit: 0.021741 and 0.028905. Keeping the final
projection in F32 does not fix it. This is broader than a single unlucky
benchmark token; hardware BF16 support is not an accuracy guarantee.

FP16 passes the benchmark token on both GPUs and has much smaller pooled KL
(0.000424 / 0.000569). It passes 8/9 individual compiled cases; one labelled-loss
bound still fails, despite passing KL. These are forward-only, uncaptured
diagnostics, not replacement cohort results or a new training qualification.
The earlier FP16 256-update study remains separate evidence under its recorded
revision. No dtype or threshold was changed in response to these results.

### Collection and regression evidence

The implementation also fixes three issues exposed by preflight collection:
latency qualification must not discard valid prefill/training; a qualified
untuned initial program may legitimately win calibrated search; and JSON must
preserve floating-point evidence across the Rust harness. Derived receipt
metrics tolerate serialization-scale roundoff, while acceptance is recomputed
from raw curves with unchanged bounds. Failed phases cannot carry timings.
Earlier preflights and stopped attempts remain archived and are excluded from
the full collection.

The historical native comparison flags B570 diffusion inference/latency about
28–35% slower than the October 11 cohort, Whisper about 11–12% slower on B570 and
5–7% on RTX, and isolated B570 latency differences in ResNet and SmolVLA. These
comparisons span library and protocol changes. B570 ResNet's accelerated process
medians also overlap substantially: 5.232–6.171 ms before and 5.101–6.368 ms now.
The historical flags alone cannot identify a regression caused by this patch.

A matched supplementary study compares pre-change Inferena `1288461` with
`1ed4df7`, both using the same merged Meganeura revision. Each variant has three
fresh processes, alternating order, strict arithmetic, the same 30-second search,
ten-second warmup and twenty samples. Kernel memos start empty per variant/model;
driver cache directories start empty per process. Every process also passes the
preserved same-input PyTorch strict output/loss/gradient checks. The predeclared
material-slowdown limit is 5% for each phase's median of process medians.

| GPU | Model | Inference change | Latency change | Training change |
| --- | --- | ---: | ---: | ---: |
| B570 | StableDiffusion | +2.19% | +0.27% | +0.25% |
| B570 | Whisper-tiny | −1.38% | −0.89% | −0.48% |
| RTX 5070 | Whisper-tiny | +0.06% | +0.06% | +1.11% |

All nine comparisons pass that limit, and all 18 supplementary processes pass
correctness. B570 diffusion's pre-change latency itself ranges from 4.749 to
5.629 ms; the current range is 5.576–5.789 ms. Clock traces record substantial
changes during both variants' B570 runs. The large historical diffusion/Whisper
gaps therefore do not reproduce when changing only the acceptance implementation.
This does not establish that every intervening library/protocol change is
performance-neutral. Raw controls, ranges and B570 clock samples are retained in
`matched-native-b570` and `matched-native-rtx5070`.

Rust and Python unit suites, Clippy, and compiled AMP/live-input replay checks
on both GPUs pass. Full results, raw failure receipts, source archives, scripts,
and the 60-cell table are in `../inferena-results/accelerated-v22-20261014/`;
the final campaign directories are `b570-full-r3` and `rtx5070-full`.

**The default accelerated policy is not ready for a fully populated cohort.**
Strict is locally qualified, and failed accelerated phases are correctly
excluded. Before a publication freeze, explicitly choose and qualify the
SmolLM2 precision policy rather than treating capability-based `auto` as an
accuracy guarantee. Retain the two profiles and all failed evidence.

## Acceptance threshold audit (October 14, 2026)

Meganeura #259 is merged as `99e3de2`; its complete tree is identical to the
tested `cfab3d1` branch. Inferena now pins the merged commit. `cargo check
--locked -p inferena-meganeura` passes with that pin.

**The accelerated acceptance criteria need revision.** Preserving a frozen
experiment's original thresholds is necessary for honest reporting; it does
not justify carrying an unsuitable criterion into a new experiment. The earlier
claim that the B570 BF16 probe failure ruled out reconsidering the 1% bound was
too strong. These are our engineering tolerances, not established limits on
acceptable mixed-precision learning.

Fresh SmolLM2-135M forward probes compare eager and default-compiled BF16/FP16
against eager strict F32 on each GPU. They use the unchanged checkpoint,
synthetic benchmark sequence, four frozen training probes and four held-out
sequences: nine inputs per arithmetic/execution combination. This is an untimed
audit with serial GPU jobs and process-tree watchdogs. It produces no benchmark
performance samples. Math SDPA is used on B570 and automatic SDPA on RTX.

| BF16 evidence, eager and compiled combined | B570 | RTX 5070 |
| --- | ---: | ---: |
| Passing current forward gate | 3 / 18 | 3 / 18 |
| Largest relative logit L2 error | 2.618% | 2.814% |
| Largest logit error / reference peak | 10.732% | 12.053% |
| Largest absolute relative cross-entropy change | 0.2435% | 0.3780% |
| Largest mean KL divergence, nats/token | 0.001547 | 0.001806 |
| Largest single-token KL divergence, nats | 0.07387 | 0.06441 |

FP16 passes the current forward gate in all 36 cases. BF16's ordinary eager
execution also fails many cases, so this is not solely a compiler/replay issue.
Top-1 agreement is not perfect: the lowest batch agreement is 92.97% on B570
and 95.31% on RTX. Small distribution changes can change the winning token when
probabilities are close; these probes do not establish identical generation or
general task accuracy.

The present raw-logit test also rejects adding the same constant to every
logit, although softmax probabilities and cross-entropy are unchanged. In the
retained control, adding one gives 13.05% relative tensor error and KL below
2e-14. This exposes a mismatch between raw tensor fidelity and the practical
language-model behavior that the accelerated profile is intended to preserve.
Changing the maximum-logit allowance to 5% or 10% would still reject ordinary
BF16 on these inputs without resolving that mismatch.

Fresh, separately compiled BF16 trajectories complete all 256 varying real-text
SGD updates. They are recorded directly even if the forward gate would have
stopped the benchmark; they cannot supply benchmark timings. Both satisfy the
existing learning checks and, in these runs, even the original 1% curve limit:

| Fresh BF16 training evidence | B570 | RTX 5070 |
| --- | ---: | ---: |
| Training-curve relative L2 | 0.5511% | 0.5562% |
| Fixed-probe relative L2 | 0.9472% | 0.9882% |
| Held-out relative L2 | 0.3683% | 0.3257% |
| Fixed-probe loss reduction | 5.617% | 5.399% |
| Held-out loss reduction | 3.249% | 3.233% |

The prior B570 integration run's probe error was 1.0018%, despite all individual
losses satisfying the 2% bound and its learning checks passing. These standalone
trajectory probes are not exact repetitions of the integration/capture workflow,
so the difference must not be attributed solely to run-to-run noise. Together
they show that the earlier rejection does not establish a failure to learn.

The proposed next contract, evaluated separately from the current collector,
is:

- For pretrained language-model logits, compare the complete predicted
  distributions to F32: mean `KL(reference || candidate) <= log(1.01)` and
  maximum token KL `<= log(1.10)`. These correspond to 1% average and 10%
  per-token cross-perplexity budgets under the reference distribution. Require
  the labelled cross-entropy to remain within the existing 2% bound as well.
  Retain raw-logit errors as diagnostics. These are explicit proposed budgets,
  not a claim of a universal AMP error bound or a guarantee of generation quality.
- For training, keep every observed loss within the existing 2% bound, all
  completed finite updates, the fixed-probe learning requirement and the
  held-out checks. Report curve L2 without its additional 1% veto. A curve can
  satisfy every per-loss bound and learn successfully while failing that veto.
- Keep strict arithmetic, same-policy execution/replay repeatability and
  independent implementation checks. The new inference criterion alone says
  nothing about backward correctness. Do not loosen the native full-gradient
  guard merely to admit BF16; a native training path also needs the matched
  learning study before promotion.
- Apply the same task-quality criterion to both engines before freezing the
  next cohort protocol. Non-probability outputs and synthetic, untrained models
  need their own suitable checks; do not apply a language-model softmax gate
  indiscriminately to them. Keep the same two benchmark profiles.

All 36 BF16 forward cases pass the proposed probability/loss criterion. It
accepts the harmless constant shift and rejects zeroed, sign-reversed and
vocabulary-permuted outputs. A separate sparse-error control passes mean KL
and labelled loss but fails the per-token KL bound, exercising that safeguard.
Both fresh training curves and the older B570 BF16 curve meet the proposed
training criterion. Injected missing updates, non-finite losses, absent parameter
updates and held-out loss drift still fail. This supports revising the gate; it does not make
hardware support an unconditional correctness certificate. PyTorch's own
[mixed-precision guidance](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
also notes that some models require selective precision management.

The production v21 thresholds remain unchanged in this audit. Old failures
remain failures under their recorded protocol; none are relabelled as accepted
benchmark results. Evidence and runnable probes are retained in
`../inferena-results/acceptance-audit-20261014`.

## Rectangular BF16 and cohort qualification (October 14, 2026)

[Meganeura #259](https://github.com/kvark/meganeura/pull/259), commit `cfab3d1`,
allows the B570's advertised BF16/F32 M×N×K shape `[8,16,16]`. The existing
32×32 output block uses eight rectangular native tiles. Square devices retain
four 16×16 tiles. Dense transpose/addend products and implicit-GEMM convolutions
share this loop; checked tails, F32 partials and whole 32-element split-K
partitions are preserved. This enables explicit BF16 graphs; default F32/FP16
selection is unchanged.

Both B570 and RTX 5070 pass the 15-test BF16 regression filter, including the
ignored full-F64 convolution oracle. The unsupported-device test correctly skips
on these supported GPUs. Coverage includes every dense transpose/addend layout,
unaligned dimensions, split partials, F32/BF16 results, master and reduced
gradients, and repeated updates. The library CPU suite passes 439 tests with
seven ignored; formatting, Clippy and PR CI pass.

**The old native qualification could miss a precision error in its first
program.** It used that program as its full-tensor anchor. The paired harness's
independent PyTorch check compared sampled outputs and gradient norms, which
cannot detect every element error or reversed derivative. The earlier Whisper
FP16 search therefore produced raw timings despite failing the separate full
F32 comparison. Explicit dtype diagnostics were already excluded from the
cohort, so those timings were not promoted to publication results.

Inferena now builds the original graph in untuned strict F32 before applying
precision conversions or searching. Every candidate passes both that full-tensor
arithmetic check and the existing within-search check. Accelerated forward
maximum/RMS bounds use `atol=1e-6, rtol=0.02`; aggregate full-gradient bounds use
`atol=1e-6, rtol=0.01`. Strict uses `rtol=1e-4`. Initial programs and memo hits
also pass these checks. A bad challenger is discarded; a bad initial or final
program aborts construction. Graph1 diagnostics enforce the check too. Failed
native construction emits no result JSON; the collector cannot accept its
timings. Rejection logs and watchdog receipts remain available.

This is a native arithmetic control, **not an independent implementation**.
The PyTorch cross-engine check remains; a shared native F32 compiler error can
still require a full independent tensor oracle to diagnose. Neither fixed-step
gradients nor the primitive update tests replace real-data convergence.

Actual negative checks now reject:

- B570 SmolLM2-135M BF16 inputs: one-token output maximum error 0.6009 exceeds
  the fixed 0.5553 bound. No benchmark result is emitted.
- B570 diffusion BF16 inputs: aggregate gradient RMS error 3.721e-5 exceeds
  3.038e-5. No benchmark result is emitted.
- RTX Whisper unrestricted FP16 with the ordinary 30-second search: the initial
  training program's gradient maximum error 1.963e-5 exceeds 7.696e-6. Training
  warmup/timing never begins. The previous first-candidate anchor accepted this
  arithmetic, so this is an exercised rejection, not just a schema test.

RTX Whisper Auto and BF16 inputs both pass the searched path. Training is
23.531 ms and 28.739 ms respectively, with ten-second warmups and twenty retained
samples. The earlier Auto measurement was 23.565 ms. B570 BF16 now executes and
passes Whisper, SmolVLA and ResNet, but is slower in each screen. SmolLM2 and
diffusion fail the numerical gate. No BF16 policy is promoted to the cohort.

All 60 default-path phase comparisons (five models × two profiles × two GPUs,
three phases) are bit-identical to the prior merged-pin build, including full
outputs, losses and canonical gradients, with matching parameters and inputs.
The final separate-context build repeats all 60 comparisons with bit-identical
results (`final-validation.json`); that pass uses one sample and brief warmup
only for numerical validation, with CPU test work allowed concurrently. Its
timings are not used for performance claims. These are graph1 regression checks,
not a full searched cohort. Alternating
old/new binaries with ten-second warmups remove the B570 training difference:
SmolVLA medians are 26.228 ms before and 26.258 ms after (+0.11%). Its inference
median differs by 2.7%; latency by 0.5%. B570 diffusion training is 13.625 versus
13.626 ms in a matched followup.

RTX SmolVLA graph1 one-token latency retains a difference: 1.881 versus 2.125 ms
across the three alternating pairs; one old process also takes 2.137 ms. The
128-dispatch plans are identical. Isolating the F32 reference context does not
remove this difference. The **actual 30-second searched protocol** does not
reproduce a slowdown in the matched followup: inference is 2.390 versus 2.408 ms,
latency 1.131 versus 1.105 ms, and training 7.388 versus 7.300 ms. These are one
fresh searched process per build, not replicated cohort estimates. The graph1
difference remains disclosed; there is no blanket claim that every timing is
unchanged. Both reference construction and the added tensor readbacks are
outside the step timer, but preparation can affect later execution.

The final implementation constructs the F32 reference in a separate GPU context
on the same device, then drops it before candidate construction. This removes
64 MiB of reference-related retained device allocation observed with the shared
context in the RTX SmolVLA probe; it is not presented as a latency fix. The
`guarded` and `isolated` binaries/source archives distinguish the two stages.

**Collection protocol v21 still has two profiles and 30 pairs per GPU.** It now
requires the F32-reference receipt and refreshes inputs before every search step,
outside timing, just as the benchmark does. The previous search kept them resident.
F32 reference preparation is included in `compile_s`, separately reported, and
precedes the unchanged 30-second search allowance. In the B570 screen it costs
about 0.7 seconds for Whisper and 9.8 seconds for SmolLM2-135M across all requested
sessions. Summing the initial screen's reference costs projects about two minutes
on B570 and 36 seconds on RTX across 30 pairs, plus context lifecycle overhead.
This is a projection, not a measured complete cohort duration. These checks add
preparation work, not benchmark columns.

Before starting a new publication cohort:

1. Prequalify and freeze the PyTorch arithmetic policy. Capability-only `auto`
   still selects BF16 where supported, with no numerical fallback. The existing
   SmolLM2 BF16 failures are real exclusions; FP16 passed the prior 256-update
   real-text/probe/held-out test on both local GPUs. Do not rerun failed BF16
   and silently substitute FP16 timings. A different policy needs an explicit
   receipt and fresh qualification under the pinned source.
2. Keep memo-cold and memo-warm preparation separate, or use
   `--no-kernel-memo`. Driver shader caches are independent. The collector now
   records their environment controls and labels their initial contents
   uncontrolled. A cold-start claim needs verified empty driver caches; a fresh
   kernel memo alone is insufficient.
3. Keep v21 separate from frozen v18/v20 evidence, run B570 first with the
   watchdog, and preserve failed trials. Native BF16 remains diagnostic until a
   useful qualifying path exists; promoting its training also needs the matched
   learning check. The Android collector's stale protocol/profile labels were
   corrected, but no Android hardware run was performed here.

The report reader now accepts v19/v20/v21 and Android v2, and refuses to pool
different paired protocol versions even if source identifiers match. Training
acceptance uses the matching model/profile/condition's replicate group; one
failed group's median no longer hides other groups' passing training results.
Missing/duplicate groups or failed phases still produce no timing or speedup.
Inferena's 20 Rust unit tests, 33 PyTorch execution tests and 20 script tests pass,
as do Clippy and formatting for the native runner and harness. The ignored GPU
memo regression passes separately on both GPUs: saved decisions are reused,
while each fresh session rechecks its new weights and inputs against F32.

Evidence: `../inferena-results/rectangular-protocol-20261014`, including library
logs, retained binaries, serial watchdog receipts, full default-path tensors,
`regression.json`, positive/negative precision checks, searched Whisper runs and
alternating timing followups. The older October 14 artifact remains unchanged.

## BF16 permission and regression checks (October 14, 2026)

Accelerated permits BF16 alongside FP16 and F32, with the same two collection
profiles and existing numerical bounds. That permission does not require an
engine to use BF16. The updated harness accepts declared mixed F32/BF16
storage and no longer requires an accelerated implementation to permit FP16.
Strict still rejects reduced precision. Meganeura is pinned to merged
`7574ad1`, Blade to `96c1dc7`, and Naga to `a97559b6`. Its default `Auto`
selector still uses the existing F32/FP16 paths. Explicit BF16 diagnostics are
excluded from the paired collector; they add no default collection time.

**The default paths preserve their numerical results.** Forty fresh processes
compare the previous Meganeura `eb8c90f` / Blade `7909fc1` pins with the merged
pins: five models, strict and accelerated, on B570 and RTX 5070. All 60 paired
phase comparisons have bit-identical full outputs, losses and canonical
gradients, with identical source parameters and inputs. This is graph1 coverage
(`MEGANEURA_TUNE=0`), not a repeat of the full searched cohort. Each phase has
at least five warmups and one second of warmup, then twenty retained samples.
The same batched tensor-export change is applied to both builds, outside timing.
Serial jobs use a 300-second process-tree watchdog; the B570 runs first.

Most short-screen timing differences are within 2%. RTX strict diffusion's
latency has one 38% outlier. Three alternating pairs with ten-second warmups
remove it: medians are 2.959 ms before and 2.963 ms after (+0.14%); training is
8.986 versus 9.010 ms (+0.27%). The original short-screen receipt is retained.

A preparation-time difference (roughly 0.5 versus 1.25 seconds) is caused by
driver cache state. CPU profiles locate the extra work in NVIDIA's shader
compiler, and both versions of the new-pin runner emit the same 106 WGSL
sources. With separate empty driver caches, preparation is 1.005 versus
0.950 seconds; the next process using each cache takes 0.492 versus 0.475
seconds. Disabling the cache yields 1.299 versus 1.250 seconds. The
previous-pin control takes 1.018 seconds with its own empty cache and 0.483
seconds on the next process, also showing no preparation regression here. The
[NVIDIA cache controls](https://download.nvidia.com/XFree86/Linux-x86_64/550.144.03/README/openglenvvariables.html)
and their values are retained in `cache-control.json` receipts. These short
preparation measurements diagnose a confound, not replicated cold-start claims.

**BF16 fixes a real FP16 range failure, but the current graph1 kernels are not
faster.** On RTX Whisper, unrestricted `AllowF16` loses entire gradient tensors:
aggregate full-gradient relative L2 error is 11.851%, and 41 of 66 individual
gradient tensors fail the diagnostic 1% maximum/RMS gate. Explicit BF16 reduces
aggregate error to 0.105%, with all individual tensors passing. Keeping matrix
results in F32 reduces it to 0.101%. These are fixed-parameter gradient checks,
not new training-convergence evidence.

The normal **30-second search budget** with fresh kernel-decision caches and
ten-second warmups confirms this distinction on Whisper. Auto trains in
23.565 ms with 0.0068% gradient L2 error. Unrestricted FP16 takes 24.730 ms
and still fails at 11.851%. BF16 inputs take 28.736 ms and pass at 0.101%.
All three full-forward checks pass. BF16 therefore rescues numerical range
here, but there is no correct-and-faster replacement to promote. These are
single fresh searched processes, not a replicated cohort speedup estimate.

The closest operand-only BF16 control retains F32 matrix results, accumulation,
interfaces, parameters and master gradients. This separates operand precision
from the convenience transform's extra BF16 result rounding. Both controls
include conversion of F32 parameters during every timed step. Diagnostic gates
use the existing `1e-6` absolute allowance, 2% full-forward maximum/RMS bound and
1% aggregate full-gradient maximum/RMS bound; bounds are not fitted to results.

| RTX graph1 workload | Auto training | BF16-input training | BF16-input gradient L2 | Gradient gate |
| --- | ---: | ---: | ---: | --- |
| SmolLM2-135M | 34.532 ms | 36.628 ms | 1.838% | Fail |
| Whisper-tiny | 24.331 ms | 29.664 ms | 0.101% | Pass |
| SmolVLA | 12.852 ms | 14.329 ms | 0.551% | Pass |
| Diffusion U-Net | 8.800 ms | 11.351 ms | 1.227% | Fail |
| ResNet-50 | 28.732 ms | 31.129 ms | 0.385% | Pass |

These single-process short-warmup medians screen candidates; failed cells are
raw diagnostic measurements and cannot become collection speedups. The
forward-only BF16 transform and the transform with BF16 dense derivatives are
also retained in the evidence. Neither produces a faster qualifying replacement
in this screen. SmolLM2's ordinary BF16 transform fails forward maximum error
(2.492% of the reference maximum, against 2%); FP16 passes. Retaining F32
matrix results improves BF16 prefill to 1.767%, but one-token forward still
fails at 2.284%, and training fails. Its one-token timing rises from 3.064 ms
under Auto to 7.594 ms with BF16 inputs. BF16's exponent range is useful for
Whisper's small derivatives; its coarser mantissa is not an accuracy improvement
for every workload.

**B570 needs another tile shape.** Mesa `26.0.3-1ubuntu1` exposes native BF16
storage and an 8×16×16 BF16/F32 cooperative tile. Meganeura's current BF16
kernels require 16×16×16 and reject the explicit experiment before compilation.
The default B570 path remains unchanged. RTX 5070 (`595.91.07`) exposes the
required tile. Both runs use Linux `7.0.0-38-generic`.

Evidence lives in `../inferena-results/bf16-accelerated-20261014`: source
archives and binary hashes, retained binaries, shared content-addressed full
tensors, per-process environment/watchdog receipts, `regression.json`,
`candidates.json`, and `input-candidates.json`. The interrupted original
per-parameter exporter attempt is preserved separately. Batched readback fixes
that diagnostic bottleneck without changing the timed model execution.

The next useful library work is BF16 kernel selection/tuning and support for
the B570's rectangular tile. Broad BF16 graph conversion is not justified by
these results. Existing PyTorch dtype selection and all learning-quality gates
remain unchanged; the earlier SmolLM2 real-text findings below still apply.
Validation passes: 19 Rust tests and 49 Python tests, targeted rustfmt/clippy,
and the results-Markdown checker. The optional saved-memo GPU test remains
ignored; fresh-cache search is exercised by the Whisper runs above.

## Accelerated real-text qualification (October 12, 2026)

Development v20 implements a common PyTorch training gate for `off`/TF32,
FP16 and BF16 within the existing accelerated profile. SmolLM2-135M uses the
same frozen 256-update real-text workload, fixed training probes and held-out
blocks as the preceding investigation. The portable token fixture is bundled
and hash checked. Other models retain their explicitly labelled 32-update
synthetic numerical check. Native qualification and the separate cross-engine
learning study are unchanged. See the [protocol](EXPERIMENT.md#development-v20-two-profiles).

Fresh local integration runs retain the existing forward bounds, per-loss 2%
bound and 1% curve relative-L2 limit. Each row below is one process with the
requested execution mode, fresh compiler cache and the normal phase warmups;
these are qualification diagnostics, not replicated performance comparisons.

| GPU / arithmetic | Execution | Training curve L2 | Fixed-probe L2 | Held-out L2 | Result |
| --- | --- | ---: | ---: | ---: | --- |
| B570 / off | Eager, uncaptured | <0.0001% | <0.0001% | <0.0001% | All gates pass |
| B570 / FP16 | Compiled, replay | 0.2272% | 0.4081% | 0.1561% | All gates pass |
| B570 / BF16 | Compiled, replay | 0.5557% | **1.0018%** | 0.3875% | Probe fails; inference/latency retained |
| RTX 5070 / TF32 | Compiled, replay | 0.2276% | 0.4084% | 0.1565% | All gates pass |
| RTX 5070 / FP16 | Compiled, replay | 0.2255% | 0.4107% | 0.1521% | All gates pass |
| RTX 5070 / BF16 | Compiled, replay requested | — | — | — | Forward gate fails; no timing |

All complete real-text trajectories also pass the fixed learning requirements:
at least 5% reduction in mean training-probe loss, at most 2% held-out loss
increase, finite changed parameters and no skipped optimizer updates. B570
BF16 still fails its separate probe-agreement bound; its proximity to 1% does
not justify changing that bound. RTX BF16's maximum forward error is
**1.030688** against **0.727245**, although its RMS error passes. BF16 remains
unqualified for these respective cells; capability-based `auto` does not
silently switch to FP16.

The optional repeated-synthetic diagnostic remains available through
`scripts/amp_reference.py --repeated-batch`. A new RTX eager TF32 run still
fails it at **5.4673%** training-curve relative L2 while retaining qualified
inference and latency. The paired collector rejects this diagnostic as a
substitute for SmolLM2's real-text gate. It recomputes training, probe and
held-out decisions from raw trajectories rather than trusting status labels.
The f32 oracle is now prepared before compilation for every accelerated setting,
with TF32 temporarily disabled and all arithmetic flags restored afterward.

Integration exposed one additional scaler issue. The first B570 FP16 compiled
run passed the real-text curves, then overflowed on the restored synthetic
benchmark input. Transferring the real-text scaler state was inappropriate for
that input's larger gradients. The timing scaler now settles on its own input
before the repeatability baseline, without updating parameters. The successful
B570 replay run records an overflowing attempt at scale 32768 and finite
gradients at 16384. Full-gradient repeatability and consecutive replay gates
then pass on both GPUs. The original failed integration attempt is retained.

The compiled SmolLM2 reference-plus-learning qualification takes **28.9 s**
on B570 FP16 and **15.2–15.7 s** on RTX TF32/FP16, separate from compilation,
replay preparation, warmup and step timing. The primary matrix is still
**30 pairs per GPU**; no extra mandatory dtype columns were added. These runs
do not qualify the remaining workloads or remote devices for a new cohort.

Evidence is in `../inferena-results/accelerated-realdata-20261012/`: raw curves,
phase receipts, watchdog records, `summary.json`/`.csv`, reproduction scripts
and eight verified source archives. `study-notes.json` identifies the superseded
scaler attempt. `SHA256SUMS` verifies 68 artifacts. CPU regression tests cover
state/flag restoration, scaler preparation, independent pre-compilation oracles,
fixture integrity, missing updates, probe failures and diagnostic exclusion;
the harness tests and clippy also pass. The frozen v18 evidence is unchanged.

## Accelerated gap follow-up (October 12, 2026)

The follow-up separates batch diversity from trajectory length and tests
whether correcting half-precision rounding can satisfy the existing gates.
It uses source `97a27f1`, the pinned SmolLM2-135M checkpoint and both local GPUs.
These are supplementary diagnostics, outside the frozen v18 collection.

**Changing batches explains the earlier real-data success.** A controlled
comparison uses the same public-domain text corpus, SGD rate, token shapes and
initial checkpoint, with math SDPA on both devices. One schedule repeats the
first training block; the other consumes the frozen sequence of 256 blocks.
Both use the existing 2% per-loss and 1% curve-L2 bounds. The 32-step comparison
is a prefix of each 256-update trajectory.

| GPU | Arithmetic | Repeated batch, 32 / 256 updates | Changing batches, 32 / 256 updates |
| --- | --- | ---: | ---: |
| B570 | fp16 | 1.664% / 6.161% | 0.078% / 0.228% |
| B570 | bf16 | 3.465% / 16.594% | 0.199% / 0.547% |
| RTX 5070 | fp16 | 1.663% / 6.155% | 0.075% / 0.227% |
| RTX 5070 | bf16 | 3.444% / 15.316% | 0.196% / 0.562% |

Values are curve L2 errors against the corresponding f32 schedule. All changing-
batch training curves pass both bounds at both lengths; all repeated-batch
curves fail. More updates alone do not resolve the discrepancy. Fixed probe and
held-out evaluations at updates 0, 64, 128, 192 and 256 also pass for fp16 on both
GPUs. BF16's B570 probe still fails at 1.037% L2; its held-out curve passes at
0.369%. Both RTX bf16 evaluation curves pass (probe 0.976%, held-out 0.338%).
Thus the changing-batch training result is not a blanket bf16 qualification.
This limited corpus and short run do not establish general convergence quality.

**Explicit rounding corrections close the eager arithmetic gap.** Each linear
weight is split into its half-precision value and the remainder of its f32
master value. FP16 adds one matrix product using the remainder, with f32 output
addition. BF16 additionally splits the activation and adds its corresponding
product; all three BF16 products return f32 tensors. The small remainder-times-
remainder term is omitted. This decomposition follows the approach described
by [Henry, Tang and Heinecke](https://arxiv.org/abs/1904.06376), with the exact
experimental forward and backward definitions retained in `policies.py`.

| GPU | Arithmetic | Ordinary AMP curve L2 | Corrected curve L2 | Corrected forward / curve |
| --- | --- | ---: | ---: | --- |
| B570 | fp16 | 5.450% | 0.154% | pass / pass |
| B570 | bf16 | 13.111% | 0.219% | pass / pass |
| RTX 5070 | fp16 | 5.412% | 0.186% | pass / pass |
| RTX 5070 | bf16 | 12.834% | 0.123% | pass / pass |

These use the original repeated synthetic input, 32 updates, and unchanged
full-forward and training-curve bounds (B570 math SDPA; RTX auto SDPA). Merely
returning f32 tensors from ordinary half-input linears leaves the training
failure intact. Adding extra products constitutes a different arithmetic
algorithm with additional work; these corrections are not ordinary autocast.
Autograd uses straight-through derivatives for the high components and detaches
the remainders. The f32-output matrix product requires an explicit backward in
this pinned PyTorch build;
the prototype uses half-precision backward products. Compiled two-layer tests
check every gradient, graph replay, stable gradient-buffer addresses, unchanged
parameters, and fp16 scale changes on both devices. Full-model compilation and
capture must qualify independently of these eager results.

**Default compilation erases the correction.** A minimal split reproducer
returns nonzero eager weight remainders but all-zero compiled remainders for
both fp16 and bf16 on both GPUs. The pinned Inductor source documents removal
of intermediate downcast/upcast pairs by default. That transformation is
incompatible with a decomposition that explicitly needs the rounded value.
On RTX, `emulate_precision_casts=True` restores both splits exactly. On B570,
the same setting restores the bf16 split but still returns zero fp16
remainders; the option alone is not a portable fix. A second B570 reproducer
returns the remainder in both f32 and fp16 and scales inputs by 1, 100 and
10,000. Every compiled remainder is zero, including normal-sized f32 values
whose eager maximum is 0.466. This rules out flushing only small fp16 output
values. The retained generated Triton contains the explicit casts, narrowing
the remaining problem to execution/compilation downstream of that code; the
exact downstream pass is not yet identified.
`repro_xpu_fp16_roundtrip.py` is a standalone three-value reproducer requiring
only the tested PyTorch XPU wheel: `w - w.half().float()` for f32 `w=1000.25`
returns 0.25 eagerly and zero when compiled, even with cast preservation requested.

This explains why the default compiled RTX fp16 prototype passes forward but
fails training at 5.408% curve L2, despite its eager result. The default bf16
prototype instead fails forward (maximum error 0.734 versus a 0.727 bound),
so no training curve or timings qualify. Both B570 prototypes exceed the
unchanged 120-second cold-compile limit. Fresh strict controls compile in
84.5 seconds on B570 and 46.8 seconds on RTX and qualify all three captured
phases. Small relative-error tests on unchanged weights were insufficient to
detect the removed correction; the targeted split test exposes it directly.

Enabling cast preservation restores full-model qualification on RTX for both
corrections. FP16's compiled curve error is 0.179%, with an 87.7-second compile;
bf16's is 0.145%, with a 119.5-second compile, very close to the 120-second limit.
These are explicit prototype settings, not a dtype or execution fallback.

| RTX arithmetic | Inference | One-token latency | Forward/loss/backward |
| --- | ---: | ---: | ---: |
| Strict f32 | 6.064 ms | 1.637 ms | 15.606 ms |
| Corrected fp16, casts preserved | 4.163 ms | 2.687 ms | 12.950 ms |
| Corrected bf16, casts preserved | 6.327 ms | 3.554 ms | 19.262 ms |

Each row is one fresh process with an empty compiler cache, qualified explicit
graph replay, ten-second phase warmups and twenty samples. The fp16 training
measurement includes scaler housekeeping; timed training uses fixed parameters.
These diagnostic medians establish costs to investigate, not replicated cohort
speedups. The bf16 correction costs
more than strict f32 in every phase; fp16 improves inference/training here but
increases one-token latency. Together with B570's compilation failures, this
does not support adopting the prototypes as a common collection policy.

**The older accelerated setting fails the same trajectory criterion on RTX.**
With autocast off, backend `high` precision, TF32 enabled, and the NVIDIA
environment override unset, the 32-update synthetic curve differs from eager
strict f32 by 5.467% in eager execution and 5.432% after compilation. Both
forward checks pass. B570's eager `high` control passes at 0.029%. The current
runner only applies this trajectory gate when AMP is active, so selecting
`--autocast off` bypasses a criterion that the RTX TF32 path would also fail.
This does not retroactively invalidate the frozen v18 measurements: those
measure forward/backward at fixed parameters and did not claim this SGD-curve
qualification. It does mean that the next accelerated training contract should
apply its learning-quality criterion consistently across arithmetic options.

The next policy step should use changing real training batches and independent
probes to qualify practical learning, while retaining the repeated-batch curve
as an arithmetic stress test. BF16's remaining B570 probe failure shows why a
passing changing-batch training curve alone is insufficient. Such a contract
must be frozen and requalified before collection; it still needs only the
existing strict and accelerated profiles.

The diagnostic scripts, source archives, raw trajectories, exact token spec,
watchdogs, tables and `gap-diagnosis.png`/`.svg` are retained under
`../inferena-results/accelerated-gap-20261012`; `artifact-index.json` records
source/checkpoint hashes and retained failed attempts. The correction prototypes
remain outside the benchmark runtime. No gate, dtype selection or collection
axis has changed.

## Accelerated failure diagnosis (October 12, 2026)

The SmolLM2 failures have two distinct numerical causes. The new diagnostics
use source `9998622`, the same checkpoint, inputs, SGD rate and 32-update gate,
with eager execution on B570 (math SDPA) and RTX 5070 (auto SDPA). They extend
the earlier eager/compiler controls; no benchmark settings or gates change.

**Training: rounding the updated weights explains almost all the divergence.**
An ablation casts only linear weights to bf16/fp16 and back to f32 on every
forward, while keeping activations, matrix arithmetic, gradients and SGD in
f32. A straight-through derivative isolates forward weight quantization from
gradient rounding. The original checkpoint contains 272 BF16 tensors; rounding
its initial linear weights to bf16 changes neither logits nor the first
gradient materially. Subsequent f32 master-weight updates are nevertheless
rounded away when constructing the next forward's weights.

| GPU | Dtype | Full AMP curve L2 error | Rounded weights, f32 computation |
| --- | --- | ---: | ---: |
| B570 | fp16 | 5.447% | 5.458% |
| B570 | bf16 | 13.029% | 12.910% |
| RTX 5070 | fp16 | 5.412% | 5.417% |
| RTX 5070 | bf16 | 12.834% | 12.838% |

After the first f32 SGD update on B570, 108,375,146 linear-weight elements
change. Of those changes, 98.95% are invisible after an fp16 cast and 99.86%
after a bf16 cast. They remain stored in the f32 master weights and can become
visible after later updates accumulate. The same updated f32 weights give
loss 5.110976, versus 5.176702 with fp16-rounded weights and 5.234647 with
bf16-rounded weights (initial loss 6.009892). The weight-only ablation needs
neither autocast nor a scaler. This isolates the dominant cause from gradient
underflow, skipped steps, graph replay and compiler behavior.

**BF16 forward: activation arithmetic is the limiting factor.** Initial
bf16-rounded weights alone produce exactly the f32 logits, but full AMP has
maximum logit error 1.078 on B570 and 0.913 on RTX 5070 against a 0.727 bound.
Both RMS checks pass. The layer trace exposes large MLP activations and
residual cancellation, especially around layers 11 and 28. Keeping all MLPs
in f32 lowers maximum error to 0.642/0.533 and passes both GPUs. Keeping only
the output head in f32 does not resolve the B570 failure. A single-layer
exemption is insufficient across devices: layer 28 in f32 passes B570 but
still fails RTX 5070. These are diagnostic ablations, not adopted policies;
MLP-only protection also leaves the B570 training curves outside their bounds.
The f32 regions use PyTorch's documented
[autocast-disabled subregions](https://docs.pytorch.org/docs/2.14/amp.html#torch.autocast).

Changing synthetic input to text does not by itself solve the short-curve
failure. Three blocks were selected before execution from the frozen corpus
(first train, probe and held-out source blocks), each used as a repeated
training batch for this diagnostic. All six fp16 curves fail at 1.67–2.93%
L2; all six bf16 curves fail at 3.38–5.74%. All fp16 full-forward gates pass,
while all bf16 maximum-error gates fail. Initial cross-entropy changes by
less than 0.17% in every case. These runs are not held-out generalization
evaluations. The prior 256-update study on changing real-text batches remains
separate: fp16 passed on both GPUs; bf16 missed the B570 fixed-probe bound.

The evidence supports revisiting the accelerated training criterion around
real-data learning quality before another cohort, with the short repeated-batch
curve retained as an arithmetic diagnostic. It does not justify silently
widening the current limits, switching dtype after a failure or promoting a
device-specific f32 exemption. Strict/accelerated remain the only two profiles,
and every existing failed cell retains its status.

`../inferena-results/accelerated-failures-20261012` retains the scripts, source
archives, 36 complete diagnostic trajectories, per-layer/parameter comparisons,
update-visibility counts, raw losses, watchdogs, summary JSON/CSV and
`weight-rounding.png`/`.svg`. A first text-probe attempt is retained separately:
a repeated native XPU embedding probe raised an invalid-SymInt exception.
The completed rerun reused the already selected dense index-add fallback and
verified all gradient elements for each text batch. No runtime code changes
or new performance measurements are included in this investigation.

## Accelerated includes AMP (October 12, 2026)

Development protocol v19 keeps exactly two paired profiles: strict and
accelerated (practical). Accelerated now permits AMP/bf16/fp16; dtype is a
recorded setting, not another collection axis. PyTorch selects native bf16
when available and otherwise fp16 on GPU backends, with an explicit
`--autocast bf16`, `--autocast fp16` or `--autocast off` override.
Strict and the graphics-only CPU oracle
remain f32. Meganeura retains its existing f32 storage and eligible reduced-input
kernels; this change does not implement native f16 activations. The five-model,
three-replicate collection still has 30 pairs per GPU.

The AMP run prepares its strict-f32 output/loss/gradient oracle and 32-update
reference trajectory before candidate compilation, restores weights and RNG,
then qualifies the requested AMP candidate. Cross-engine gradient checks use
that oracle, not AMP gradients. fp16 forward/loss/scaled backward can be
captured; unscaling, finite checks and scale updates run after every replay
inside timing. Neither engine performs an optimizer update in timed training.
All prior numerical thresholds are retained. Failed training gates preserve
qualified inference and latency, with no dtype or execution-mode substitution.

Ten one-process integration cases exercise both local GPUs, both AMP dtypes,
and strict controls. These are qualification diagnostics with one held-out
sample per phase, not a replicated performance cohort. Selected dtype checks
reuse an explicitly identified native result rather than repeat native search.

| GPU | Model | PyTorch arithmetic | Inference / latency | Training |
| --- | --- | --- | --- | --- |
| B570 | Whisper-tiny | strict f32, bf16, fp16 | pass | pass |
| RTX 5070 | Whisper-tiny | strict f32, bf16, fp16 | pass | pass |
| B570 | SmolLM2-135M | fp16 | pass | curve fails, 5.432% L2 |
| B570 | SmolLM2-135M | bf16 | pass | curve fails, 12.892% L2 |
| RTX 5070 | SmolLM2-135M | fp16 | pass | curve fails, 5.406% L2 |
| RTX 5070 | SmolLM2-135M | bf16 | forward gate fails | not attempted |

RTX SmolLM2 bf16 has maximum forward error 1.031 against a 0.727 bound;
RMS passes. Training curve L2 must remain at most 1%. These failed cells
remain failures; no tolerance is widened and no faster dtype is selected
after observing timings. Availability of a dtype does not qualify every model.

Whisper's added f32-oracle and AMP-curve work totals about 2.5 seconds on B570
and 0.8–0.9 seconds on RTX 5070 in these diagnostics. SmolLM2's completed gates
take about 7.1–7.3 and 3.1 seconds respectively. This is preparation overhead,
not another benchmark process, and is separate from compilation and warmup.

The pre-compilation oracle ordering matters: a B570 pilot that prepared eager
f32 after AMP compilation produced gradient norm 41.789 instead of 0.003827
and a non-finite reference trajectory after 16 updates. Preparing f32 first
restores the independent reference and both compiled AMP Whisper curves pass.
The underlying PyTorch cause is not established. A regression test makes a
mock compiler corrupt backward and verifies that it cannot redefine the f32
oracle or turn the bad AMP trajectory into a passing result.

Python contract/replay tests pass in both pinned GPU environments, including
live loss-scale changes, stable captured gradients, unchanged parameters,
state restoration, and partial failures. Harness/native tests and clippy pass.
Raw records, source snapshots, watchdogs, classifications and test logs are in
`../inferena-results/accelerated-amp-20261012`; `summary.json`/`summary.csv`
identify the final ten cases. Superseded pilots and the initial launcher
argument error are retained separately. The frozen v18 cohort and its handoff
bundle are unchanged. The remaining models/backends still need v19 qualification.

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

## B570 kernel candidates on merged upstream (October 11, 2026)

Inferena development commit `60d4b43` pins upstream Meganeura
`430f14b153afe1d92839b50b4c9f897933e335ba`. Its complete tree matches the
cohort's `5b37124` exactly; the frozen source and records above retain their
original identities. Subsequent kernel experiments use a separate Meganeura
worktree and branch, `experiment/b570-kernel-tuning-2026-10-11`, at
`91fb22d5423d96fac4feaed9f600efa092a9c148`.

The GEMV experiment replaces the overflowing positive chain with independent,
signed products. A bank of distinct weights totals at least 64 MiB; every
output is checked against an f64 dot product before and after timing. Each
variant warms for ten seconds, followed by twenty rotated/reversed timing
rounds, each timing ten completed steps. Construction, uploads and readbacks
are excluded. These are effective weight bytes per wall-clock second for a
reused bank, not hardware DRAM counters or whole-model latency.

For the 576×3072 projection, the best tested existing geometry and the best
load-batching prototype give:

| GPU | Weight layout | Existing GB/s | Existing time / best batched time |
| --- | --- | ---: | ---: |
| B570 | forward | 181.3 | 0.932× |
| B570 | transposed B | 179.5 | 1.006× |
| RTX 5070 | forward | 361.1 | 1.005× |
| RTX 5070 | transposed B | 361.3 | 1.001× |

These are one process per device/layout, with three geometries and K-load
batching factors 1, 2 and 4. None of the added choices beats the best existing
geometry by the 5% kernel-selection threshold in this fixture. Batching helps
the B570's default transposed kernel, but existing row grouping is faster
still. The prototype is retained as evidence and omitted from the candidate
library. `examples/bench_gemv_layout.rs` keeps the bounded, checked fixture
for the existing layouts; a separate B570 576×1536 run verifies its explicit
device selection and reaches 168.9 GB/s with grouped columns.

The initial diagnostic launcher called `init_gpu_context()`, which ignores
`MEGANEURA_DEVICE_ID`; all four initial records identify the RTX 5070. Two
filenames incorrectly said B570. Those attempts remain in
`initial-default-adapter/` and are excluded from B570 evidence. The corrected
B570 reruns restrict the Intel ICD and check the recorded adapter name; the
committed example uses `GpuOptions::from_env()` and records the requested ID.

The convolution candidate makes column interleaving a typed, serialized
per-kernel choice for forward, input gradients and weight gradients. Pipeline
keys include the choice; plan-cache format 24 and kernel-memo revision 2
invalidate older decisions. Ordinary builds keep the previous column layout.
The candidate adds four legal alternatives per convolution class, with the
existing numerical gates and improvement thresholds. It does not enable
rectangular Xe2 cooperative matrices or f16 activations.

Validation passes clippy across all targets/features, 96 targeted unit tests,
and three GPU suites on each device: explicit convolution kernels against the
f64 oracle, tuning across padding/stride/tile edges with preserved training
state, and indexing at reciprocal boundaries. The rejected GEMV prototype
also passes its five regression tests on each GPU, including f16/RMSNorm and
loop tails.

Raw records, source patches, dependency locks, retained binaries, validation
logs and analysis scripts are in
`../inferena-results/b570-kernel-tuning-20261011/`. This study is separate from
the frozen paired cohort.

### Whole-model acceptance

The strict-f32 ResNet-50 comparison completes twelve processes: three each for
upstream and candidate on B570 and RTX 5070, with order reversed in replicate
two. Runner source is Inferena `60d4b43`; library revisions are recorded above.
All builds finish before timing. Each process starts independent 30-second
searches with cross-process kernel memos disabled, then uses the same minimum
five-call/ten-second warmup and twenty samples per phase. This cold-search
study has a different memo policy from the primary cohort; its timings must
not replace that cohort's warm-reuse cells.

All twelve processes pass the harness's unchanged output, loss and parameter
gradient gates against their corresponding preserved PyTorch strict-f32
reference records. All four groups pass the replicated-gradient gate. This
reuses numerical references, not their timing as a new paired measurement.
Every watchdog exits successfully. Native construction also qualifies complete
outputs and gradients against its first checked program.

Times below are medians of three process medians, in milliseconds. Ratios are
upstream time divided by candidate time; greater than one favors the candidate.

| GPU | Phase | Upstream | Candidate | Ratio |
| --- | --- | ---: | ---: | ---: |
| B570 | Inference | 12.812 | 12.710 | 1.008× |
| B570 | Small-batch latency | 5.333 | 6.365 | 0.838× |
| B570 | Training | 53.397 | 53.421 | 1.000× |
| RTX 5070 | Inference | 4.948 | 4.947 | 1.000× |
| RTX 5070 | Small-batch latency | 2.547 | 2.510 | 1.015× |
| RTX 5070 | Training | 27.253 | 27.792 | 0.981× |

B570 small-batch latency regresses 19.4%; its observed process ranges are
5.231–6.078 ms upstream and 6.252–6.385 ms with the candidate. NVIDIA training
regresses 2.0%, with ranges 27.235–27.362 and 27.790–27.825 ms. These are
observed ranges, not confidence intervals. Median preparation increases from
65.492 to 71.020 seconds on B570, and 59.237 to 68.123 seconds on NVIDIA.

**Keep `91fb22d` experimental.** Inferena retained merged upstream `430f14b`
at this checkpoint; the later development update is described below.
Faster isolated convolution choices do not establish a whole-model gain. The
first B570 candidate process records interleaved winners in forward and
weight-gradient classes, but still returns graph/program zero for all three
sessions. In `src/train/search/measure.rs`, the first program installs private
kernel winners before becoming the incumbent; it has no paired timing against
its pre-tuning form. Later program comparisons therefore do not by themselves
guarantee that this initial kernel tuning helped. This is a concrete acceptance
gap to investigate before expanding the kernel search further; it does not
establish the complete cause of the observed regression.

A separate **single-process** B570 ordinary-build control (`MEGANEURA_TUNE=0`)
passes the same preserved PyTorch gates and measures 12.825 ms inference,
5.224 ms small-batch latency and 53.368 ms training, with 1.763 seconds of
preparation. It retains the default graph optimizer but disables measured
construction. This is not an identical-plan, kernel-only ablation, and it is
not a replicated cohort. It provides a useful performance baseline for the
next acceptance change: retain an ordinary/pre-tuning incumbent and require a
whole-program improvement before replacing it with private kernel decisions.
Only then revisit the additional convolution choices and rectangular Xe2
cooperative kernels. Native f16 activation storage remains separate work.

`model-summary.json`/`.csv` retain all process timings and ranges;
`model-summary.json` also retains kernel-comparison decisions, qualification
receipts and the four replicated-gradient reports. `run_model_comparison.py`
and `summarize_models.py` reproduce this diagnostic and its analysis.

### Acceptance safeguard diagnostics on newer upstream

Development commit `d02d60c` updates Inferena to Meganeura `d7a7293`, including
the subsequent Vulkan correctness fixes. The frozen cohort remains unchanged.
The acceptance safeguard is being tested separately: qualify an untuned
incumbent, reserve comparison time within the existing deadline, and require
private kernel choices (including memo reuse) to pass whole-program timing
before replacing it. Failed qualification, incomplete pairs and late samples
retain the incumbent. A new input callback reproduces uploads before every
execution, outside step timing but inside the search budget.

The initial safeguard constructs a second session for the kernel challenger.
Meganeura `98bdb57` instead retains the original dispatches and compiled
pipelines, switches choices on the same tensor allocations, and restores the
original on rejection. Eighteen search tests and the explicit training-state
crossover test pass on both local GPUs, including restoration of pruned
convolution pipelines. All-target/all-feature clippy passes.

The ten-second pilots did **not establish performance acceptance**. With independent cold
searches, ten-second held-out warmup and twenty samples, the following B570
ResNet-50 pilots pass the unchanged PyTorch output and gradient gates:

| Implementation | Meganeura | Latency (ms) |
| --- | --- | ---: |
| Fresh upstream control | `d7a7293` | 5.213 |
| Safeguard, unchanged kernel space, separate sessions | `18b46b0` | 6.306 |
| Safeguard, unchanged kernel space, original allocations | `98bdb57` | 6.203 |

These are single-process pilots, not a replicated acceptance study. The
separate-session pilot's recorded private kernel decisions match an upstream
control, so isolated-kernel reports alone do not explain the gap. Reusing the
original allocations does not eliminate it. The expanded convolution space
also remains rejected: two B570 triplets stop the initial study before RTX
timing, and adding input refresh alone still gives 6.291 ms in a later pilot.

Further controls limit interpretation of the earlier 5.224 ms ordinary-build
observation. New ordinary-build runs measure 6.321 ms on `d7a7293` and 6.316 ms
on the old `430f14b` binary, both passing correctness. This does not support
attributing the slower ordinary result to the upstream update. In one fixed
selected session, alternating input uploads off/on/off/on gives 4.993, 6.075,
5.266 and 6.351 ms; uploads affect subsequent execution even when their own
time is excluded. GPU timestamps also perturb the comparison: the timestamped
control and in-place safeguard measure 6.396 and 6.349 ms before detailed
profiling. Those instrumented results are diagnostic, not an acceptance gate.

Raw evidence is retained in `../inferena-results/search-acceptance-20261011/`
and its `-input-refresh`, `-refreshed`, `-standalone`, `-core` and `-inplace`
counterparts. Failed and superseded attempts are explicitly labelled.

A thirty-second warmup trace without GPU timestamps, in
`../inferena-results/search-acceptance-settling-20261011/`, shows substantial
drift within each fixed latency session. The control's two-second bin medians
rise from 5.326 ms initially to 6.327 ms at 28–30 seconds while its sampled GPU
clock falls from 2750 to 2150 MHz. The safeguard starts at 5.034 ms and reaches
6.177 ms at 28–30 seconds, with the GPU clock falling from 2750 to 2100 MHz.
Host submission time also varies. This is direct evidence of changing
operating conditions of sufficient size to confound the earlier comparison;
it does not reconstruct the clocks of the earlier runs. Held-out latency
after thirty seconds is 6.330 ms for the control and 6.134 ms for the safeguard.
Both complete processes pass the numerical gates.

A fresh three-process-per-variant study uses a fixed thirty-second warmup
on both GPUs, without tracing or GPU timestamps. The numerical gates, 1%
whole-program threshold, 5% private-kernel threshold and 5% material-regression
limit remain unchanged. These supplemental records use their own protocol;
they do not replace frozen cohort records.

**The safeguard passes this local acceptance gate.** All twelve processes
complete within their watchdogs and pass the preserved PyTorch output, loss
and parameter-gradient gates. All four device/variant groups pass replicated
gradients, and all six execution comparisons stay within the prespecified 5%
material-regression limit. Times are medians of three process medians:

| GPU | Phase | Upstream (ms) | Safeguard (ms) | Time change |
| --- | --- | ---: | ---: | ---: |
| B570 | Inference | 12.777 | 13.184 | +3.2% |
| B570 | Small-batch latency | 6.352 | 6.154 | −3.1% |
| B570 | Training | 53.625 | 53.381 | −0.5% |
| RTX 5070 | Inference | 5.010 | 4.989 | −0.4% |
| RTX 5070 | Small-batch latency | 2.547 | 2.538 | −0.4% |
| RTX 5070 | Training | 27.162 | 27.009 | −0.6% |

Preparation remains reported separately: 65.150 / 66.837 seconds on B570
and 57.592 / 54.989 seconds on RTX (upstream / safeguard). This includes
qualification, the untuned fallback, kernel probes and whole-program search.
The result supports retaining the safeguard for review; it does not establish
a general speedup, recover the earlier operating conditions, or promote the
expanded convolution candidate. The earlier apparent B570 latency regression
must not be presented as a demonstrated kernel effect without accounting for
the observed clock drift.

`../inferena-results/search-acceptance-steady-20261011/` contains every process,
the full summary with observed ranges, gradient reports, source patches, locks,
retained binaries and a Meganeura bundle for `98bdb57` based on upstream
`d7a7293`. The library changes remain on the local
`experiment/search-acceptance-core-2026-10-11` review branch; Inferena's normal
dependency remains the published upstream `d7a7293`. The integration patch
records the input-refresh callback used in this study without introducing an
unpublished remote dependency into the development checkout.
