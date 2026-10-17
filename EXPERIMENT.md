# P3HPC paired collection

The development collector uses **v28**, with exactly two profiles: `strict`
and `accelerated` (practical). The October 11 cohort below remains frozen at
**v18**. V19 introduced AMP; v20 changed accelerated training qualification.
V21 adds a native F32 arithmetic reference and refreshes search inputs to match
measurement. V22 applies language-model distribution/loss gates and removes the
additional curve-L2 veto. V23 introduced a model-specific SmolLM2 autocast recipe
and fixed native search coverage. Its full-model FP16 trial was stopped before
collection completed because it unnecessarily slowed already-qualified training.
V24 restricts the FP16/F32 recipe to one-token latency. V25 also checks each
non-language latency workload against F32, including ResNet's smaller batch and
SmolVLA's shorter action sequence. No quality threshold changes.
V26 replaces the one-token FP16/F32 wrapper with ordinary F32, retaining the
capability-selected dtype for prefill and training. Keep these protocols' records
separate.
V27 uses PyTorch's built-in profiler benchmarker for XPU kernel autotuning and
records the actual benchmarker in every compiled result. Other backends retain
the pinned default event-based benchmarker. Search and compilation budgets,
numerical acceptance, final step timers and the two-profile matrix are unchanged.

## Development v28: two profiles

The October 17 integration combines the approved collector from `22bd70f`
with current main's portable regression tests and tensor comparison utility.
It retains the two profiles, numerical gates, budgets and input-refresh policy.
Meganeura is pinned to merged `378d8e2` (including #261, #262 and #249), with
Blade `4845d65` and Naga `13f582dd`. BF16 diagnostics use the shared typed
cooperative kernels and report subgroup shapes separately from workgroup tiles.
The measurements below retain their original source revisions; they do not
qualify the updated stack by themselves. Collect fresh evidence before freezing
this integration for the final cohort.

V28 pins kernel search to `Race` and accepts the explicit `share_cutoff` receipt
from merged Meganeura #262: classes predicted to contribute less than 0.1% of
the whole plan may remain unvisited. The receipt must include calibration and
a finite positive plan prediction. Unexplained incomplete coverage, class caps
and changed budgets still fail validation. Whole-program timing and numerical
gates remain unchanged. Android uses its corresponding v10 protocol. The initial
integration's B570 SmolVLA run exposed this receipt incompatibility; its records
remain separate from the fresh v28 qualification.

Android v10 corrects the CPU oracle's loss arithmetic: the reference loss now
preserves the requested float64 dtype instead of using the timed AMP benchmark's
float32 loss helper. V9's SmolLM2 gradient mismatches were reproduced with that
cast and disappear when only the reference cross-entropy uses float64. Receipts
record and validate `loss_dtype` and `loss_policy=reference-dtype-v1`. Tensor
bounds, native arithmetic, search and measurement policies are unchanged. Keep
the original v9 failures; qualify the corrected oracle in a new campaign.

## Earlier v27 validation

The full local v27 collection from frozen Inferena `f587313` qualifies all
60 publication cells, 180 phase checks and 20 repeated-gradient groups across
B570 and RTX 5070. It takes 101.3 and 77.8 minutes, respectively. No phase median
exceeds the 5% review threshold against v22. Two native B570 timing flags against
the separate v26 baseline remain reported. Qualified single-choice replays
give diffusion -9.19% and ResNet +3.10%, with repeated observed-clock matches
within -1.14% to +1.27%; they expose substantial operating-state variation and
do not replace the primary timings. Every RTX comparison is
below the threshold against both baselines. Keep all original samples and
protocols separate. Use the frozen measurement revision when extending this
collection to other devices; later report-only commits are not its source ID.
The local profile is qualified for collection; other devices still need their
own runs. See the [full validation and limitations](ANALYSIS.md#v27-full-local-validation-october-15-2026).

Same-source controls reproduce an 11.19% SmolVLA inference difference solely by
changing PyTorch's selected Triton configurations. The subsequent fresh-cache
default/profiler comparison improves inference, latency and training by 8.98%,
9.84% and 7.06%, with all quality gates passing. These diagnostic controls never
replace primary samples. In the new primary collection, B570 SmolVLA latency
improves from 2.459 to 2.223 ms against v26, with all three processes agreeing
within 0.2%. See the [repair and controls](ANALYSIS.md#xpu-kernel-autotuning-repair-october-15-2026).

The wrapper resolves the actual backend during its interpreter preflight, then
sets `TORCHINDUCTOR_USE_TORCH_PROFILER_BENCHMARKER=1` for XPU before starting the
runner process. It sets that flag to zero elsewhere and explicitly retains
`TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER=1`, the pinned default. The runner
records the instantiated benchmarker class and both settings. The collector
rejects inherited overrides and compiled receipts that disagree with its
declared backend policy. Eager oracles do not use a kernel benchmarker.
Profiling is limited to compiler/autotuner microbenchmarks; the final timing
interval, fresh compiler caches, default shape padding and available kernel
choices remain intact. Both strict and accelerated XPU use the same policy.

The earlier v27 native pin was merged Meganeura `f648a45`
([PR #260](https://github.com/kvark/meganeura/pull/260)), whose complete source
tree is identical to the tested `b344d49` revision
(`851852374f1ed8102396be76e0d6acedad70d6c6`). The frozen v27 collection retains
its original Inferena `f587313` and Meganeura `b344d49` identities; the merged
pin is for subsequent builds, not a relabeling of existing measurements.
This includes `99e3de2` ([rectangular BF16 support](https://github.com/kvark/meganeura/pull/259)),
with Blade `96c1dc7` and Naga `a97559b6`. BF16
is permitted by the accelerated arithmetic contract, including explicitly
declared mixed f32/BF16 storage. Strict still forbids it. Meganeura's `Auto`
selector currently uses the existing f32/f16 paths; permission does not imply
automatic BF16 selection. Optional native dtype diagnostics do not enter the
collection or change its budgets. See the
[October 14 investigation](ANALYSIS.md#bf16-permission-and-regression-checks-october-14-2026).

Strict retains its f32 arithmetic and correctness contract. Accelerated permits
AMP and bf16/fp16 arithmetic alongside existing hardware paths. PyTorch's
`--autocast auto` selects native bf16 on CUDA/HIP/XPU when supported and fp16
otherwise; MPS uses fp16. **SmolLM2-135M one-token latency** instead uses ordinary
F32 with TF32 disabled. F32 is faster on both local GPUs than the qualified
FP16/F32 projection-wrapper recipe, so that wrapper has been removed. Prefill and
training keep the capability-selected dtype, including their original backward
arithmetic. The manifest records this as `model-phase-policy-v4`, with explicit
per-phase autocast and no module rewriting. Accelerated permits reduced precision;
it does not require it. `--autocast bf16`, `fp16` or `off` explicitly requests
ordinary autocast; `off` permits TF32/backend-high arithmetic.
Selection happens before execution
and never changes in response to a failed gate or a slow result. Strict and
the graphics-only CPU oracle remain f32. The manifest declares each model's
per-phase recipe before any benchmark process runs, and collection verifies that
actual dtypes and F32 module boundaries match it. No additional profile or timing
column is introduced.

Meganeura retains f32 storage with eligible f16 cooperative inputs. Each native
session now first builds the **original, unconverted graph in untuned strict F32**,
using the same checkpoint and inputs in a separate context on the selected GPU.
The reference context is dropped before candidate construction. Every candidate,
including the first and memoized candidates, must pass full-output quality checks
and aggregate full-gradient maximum/RMS bounds of
`1e-6 + 0.01 * reference magnitude` in accelerated mode. Strict uses `rtol=1e-4`.
For pretrained SmolLM2 prefill and one-token latency, output quality means:

- Mean `KL(reference || candidate) <= ln(1.01)` over all token distributions.
- Maximum per-token KL `<= ln(1.10)`, so averaging cannot hide one bad token.
- Mean labelled cross-entropy within `1e-6 + 0.02 * abs(reference)`.

Both engines compute these metrics over every logit in F64 on the CPU and retain
per-token evidence. Raw-logit errors are diagnostic; a constant logit offset
preserves the distribution and must not fail accelerated quality. These explicit
budgets limit distribution/loss drift on the tested inputs; they do not establish
identical generation or universal task accuracy. Non-language outputs and
training scalar losses retain the full-tensor maximum/RMS bounds of
`1e-6 + 0.02 * reference magnitude`.
The independent original-F32 check catches precision errors shared by the first
reduced-precision program and its challengers. The existing, tighter
within-search anchor check also remains and may reject a challenger even when
it satisfies the broader language-model distribution/loss budget.
Both the searched path and graph1 diagnostics enforce it before benchmark timing.
The collector requires `native-f32-full-tensor-v2` evidence for every session.

This is an arithmetic control in the native engine, not a second independent
implementation. The cross-engine check compares the two original F32 references'
sampled outputs, loss and canonical per-parameter gradient norms. Native reference
receipts must match the snapshots used in full-output/gradient qualification;
the measured accelerated output cannot supply its own F32 reference. It cannot detect every directional
gradient error shared by both native F32 and reduced implementations. The new
gate also does not replace a learning gate: the separate convergence study
compares both engines on real text.

Reference construction/readback is counted in `compile_s` and reported separately
as `arithmetic_reference.preparation_seconds`. It precedes the unchanged 30-second
search budget; there are still 30 pairs per GPU and no extra benchmark columns.
Search now uploads the representative inputs before every step, outside its
timer, matching the held-out timing loop. A rejected challenger is discarded;
a failed initial program or final incumbent aborts native construction, and
collection stops with the failure retained. It never promotes that run's timings
or changes precision.
The qualified untuned initial program may remain the winner when tuning fails to
improve it or exhausts the budget. Its selected trial need not contain a kernel
tuning receipt; attempted tuning remains recorded on the challenger trials.

V23's native pin compares complete execution plans before skipping duplicate
trials. Identical dispatch geometry does not imply identical shaders: attention
layout knobs can change generated code. Search explores attention layouts using
the incumbent's winning fusion setting before trying every remaining combination.
Both changes retain the 30-second, 16-graph and 64-program limits and all existing
candidate correctness gates.

Every accelerated PyTorch setting, **including `off`/TF32**, now has the same
untimed inference, latency and training qualification. SmolLM2 uses the distribution/loss
contract above for both prefill and one-token latency; other outputs retain their
full-tensor bounds. The eager strict-f32
oracle runs both forward workloads before candidate compilation with TF32 disabled, then restores
weights, buffers, RNG and arithmetic flags. Its initial output/loss/gradient
receipt remains the cross-engine correctness reference.
Compilation, qualification and timing share one latency input definition:
one token for SmolLM2, a zero image at batch one for ResNet, one action position
with all 16 VLM context positions for SmolVLA, and the ordinary inference inputs
for diffusion and Whisper. The collector requires separate latency quality
evidence and verifies its output shape. Replay agreement with the same AMP
computation cannot replace the F32 comparison. The extra eager forward and
candidate check are untimed and included in recorded preparation costs.

For **SmolLM2-135M**, training qualification uses the frozen
[data/smollm2-real-text-v1.json](data/smollm2-real-text-v1.json) fixture:

- 256 matched SGD updates, learning rate exactly f32(0.001), no momentum,
  weight decay or clipping; identical varying 128-token batches for both curves.
- Shakespeare's *A Midsummer Night's Dream*, the same Gutenberg snapshot,
  tokenizer, seed-17 batch order and 80/20 token split as the earlier study.
  Tokens and provenance are bundled and hash checked; qualification runs offline.
  The immutable fixture's embedded gate describes its original study; active
  acceptance is versioned separately in the campaign and qualification receipts.
- Four fixed training probes and four held-out blocks at updates 0, 64, 128,
  192 and 256. Every training, probe and held-out loss must be within
  `1e-6 + 0.02 * abs(reference)`. Curve relative L2 is retained as a diagnostic,
  without the additional 1% veto (`loss-curve-v2`).
- Both trajectories must reduce mean fixed-probe loss by at least 5%, avoid
  increasing mean held-out loss by more than 2%, and change a finite norm
  parameter. The collector recomputes these gates from raw receipts.

Other models retain the 32-update repeated-synthetic numerical gate, now applied
to TF32 as well as AMP. Receipts explicitly identify it as synthetic; it is not
real-data convergence evidence. Real datasets for those workloads remain future
work. The earlier repeated-batch SmolLM2 test remains an optional diagnostic:

```sh
.venv/bin/python scripts/amp_reference.py --backend cuda --gpu 'RTX 5070' \
  --models SmolLM2-135M --precisions off fp16 bf16 --mode eager \
  --repeated-batch --results-dir ../accelerated-repeated-diagnostic
```

The paired collector rejects the diagnostic override. There are no additional
required columns. Training and latency precision failures each leave that phase's
timing empty while retaining other qualified phases. Multiple failed phases carry
separate receipts. A failed prefill aborts the reference and supplies no timings.
V22 implements
the [acceptance audit](ANALYSIS.md#acceptance-threshold-audit-october-14-2026)
proposal. It changes the acceptance policy, not the historical evidence: earlier
raw-logit and curve-L2 failures remain failures under their original protocol.
The native full-gradient guard, strict arithmetic checks and same-policy
execution/replay repeatability checks are unchanged.

Compiled AMP uses the requested CUDA/HIP/XPU replay policy. Forward, loss and
scaled backward are captured; unscaling, finite checks and scale updates run
after replay **inside the measured step**. FP16 uses GradScaler and retries the
same batch after overflow during qualification, up to eight attempts; BF16 does
not scale. Weights and RNG are restored before timing, which performs no
optimizer updates. The timing scaler is settled on the restored benchmark input
before repeatability checks; its finite-gradient attempts are recorded separately
from the real-text scaler. Qualification rejects unexpected recompilation after the
compile budget. Failures do not substitute another dtype, compiler or replay mode.

The default matrix remains five models × two profiles × three replicates:
**30 pairs per GPU**. Native search stays 30 seconds, reference compilation
120 seconds, phase warmup ten seconds and measurement twenty samples. V20 adds
256 reference plus 256 candidate SGD updates and 80 total evaluation forwards
to each accelerated SmolLM2-135M run, replacing its former 64 repeated-batch
updates. Each other accelerated workload has 32 reference plus 32 candidate
updates. Qualification times are recorded separately from compilation and step
timings. `scripts/amp_reference.py` remains an optional dtype diagnostic;
`scripts/convergence.py` remains the separate cross-engine learning study.

### October 14 v22 local readiness run

The complete B570 and RTX 5070 matrices use frozen Inferena `1ed4df7` with the
merged Meganeura `99e3de2` and the budgets above. Both execute all 30 pairs. Each GPU passes 45/45 strict
phases, 42/45 accelerated phases, and all ten replicated-gradient groups. The
three repeated exclusions on each GPU are SmolLM2-135M BF16 one-token latency;
inference and training remain qualified. Failed latency has no timing or
substituted dtype. The common-revision report has 58/60 qualified phase cells.

These remain readiness results, not a fully qualified default-`auto` publication
cohort. V24's new recipe requires its own full collection. Do not raise the bounds
or substitute FP16 results into these BF16 records. See the
[full run and follow-up evidence](ANALYSIS.md#v22-full-local-collection-october-14-2026).
The result directories are
`../inferena-results/accelerated-v22-20261014/{b570-full-r3,rtx5070-full}`.

## Frozen October 11 v18 cohort

The October 11 cohort freezes Inferena source
`0a718b6fb9d6c990410dc274618978ad4988a36d` on branch
`experiment/p3hpc-cuda-graphs`. Use that frozen checkout for additional cohort
devices. Its Meganeura pin is
`5b37124efe405b5c11e37986fc88867b151aa351` (published search-order candidate,
[Meganeura PR #245](https://github.com/kvark/meganeura/pull/245)), with Blade
`56f05651cca73659130e924e497dda633f839d40`. Protocol
`p3hpc-paired-campaign-v18` retains the merged joint graph/schedule search:
up to 16 graph/implementation forms, interleaved physical-plan choices, and
at least two warmup pairs and 250 ms before comparing whole programs, within
the shared 30-second session deadline. Private kernel probes have no separate
two-second cap. Held-out timing warms both engines for at least ten
seconds and five calls per phase.

The pin adds incumbent-guided coverage of physical choices within the existing
64-program cap. B570 and RTX 5070 complete
local qualification and replicated collection; other backends still need
qualification. The final library commit only corrects a wall-clock test's assumed
trial count, with no change to search behavior.

After that cohort, development updated to upstream Meganeura
`d7a7293378db1b0e8ee3b94a1e6d73ef14a1a32c`, including the subsequent Vulkan
correctness, scalar-product splitting and convolution indexing changes. This
is a different library tree from the frozen cohort; the current development pin
is recorded above. The intermediate merged
revision `430f14b` had the same tree as the cohort's `5b37124`; neither its
records nor the frozen source bundle are relabelled for the new pin.

V18 extends the held-out warmup from two to ten seconds. B570 diffusion
diagnostics showed substantial timing drift after the old warmup, including
a reversed control/candidate ranking when process order changed. Both engines
use the same fixed floor; search keeps its existing deadline and comparison
warmup. Linux manifests also retain CPU frequency governor, energy preference
and frequency-limit settings. See the [search-order study](ANALYSIS.md#search-order-and-warmup-diagnostic-october-11-2026).

Development runners accept `INFERENA_WARMUP_SECONDS` for standalone experiments
(a finite positive number, default `10`). Both PyTorch and Meganeura report the
requested floor and actual warmup duration. The paired collector explicitly
sets its frozen ten-second floor and still rejects mismatched protocol records.
The separate acceptance study uses thirty seconds after observing B570 clock
drift; see the [acceptance diagnostics](ANALYSIS.md#acceptance-safeguard-diagnostics-on-newer-upstream).

V15 retains matched deterministic pseudorandom parameters, model shapes, inputs,
pinned SmolLM2 checkpoints, arithmetic classes and numerical
thresholds. It updates Meganeura and Blade, including descriptor-set reuse,
adaptive kernel timing, grouped GEMV and egglog backward-layout exploration.
V16 restores measured submission choices 1, 2, 4, 8, 16, 32 and 64 on every
GPU, including Intel. Small submission counts are probed on the current winning plan.
New layouts inherit its count and, if they win, trigger another submission
check. The first three layouts are tried before those probes, including both
backward-only and joint choices in training. Joint layout/submission alternatives remain in the
search. The same budget covers all choices. Set
`--max-submission-chunks 1` for a separately labelled single-submission control.
Whole-program selection requires a 1% improvement plus the existing timing-noise
margin; private kernel tuning retains its 5% threshold. A useful layout need not
clear a 5% floor before its submission choices can be measured.
Search candidates reuse immutable parameters only when all names, shapes,
scalar types and packed representations match. Inputs and gradients remain
private. The complete numerical gates are unchanged; native comparisons use
independent f64 accumulators, and PyTorch's CPU checks accumulate in f64 without
materializing a full f64 tensor copy. Campaign receipts now include collection
wall time, runner-build time and each paired process's elapsed time.
V17 reuses private kernel decisions within each campaign/model/precision while
retaining whole-program search and fresh qualification in every process. It
does not enable f16 weight storage. See the memo policy below before comparing
preparation times with previous cohorts.
The v15 cohort used one submission globally; it cannot establish the quality
of the restored search at 30 seconds.

SmolLM2 and ResNet training pass class indices directly to Meganeura's indexed
cross-entropy, matching PyTorch's target representation and mean loss. The
previous dense one-hot label allocation and upload are removed. The diffusion
reference now uses exact GELU to match the updated native model; native Whisper
also uses exact GELU, matching the reference. These corrections change the
executed workload relative to v14. Graph-capture accounting now excludes the
preceding uncaptured preparation; complete gradient checks remain in place,
with compact aggregate receipts.

The collector retains phase-level failures and runs a separate eager diagnostic.
The primary ROCm path uses ordinary compiled automatic SDPA with graph replay
required. Do not combine this revision with previous cohorts or relabel their
records. Existing pinned Python environments remain unchanged.

**Collection candidate: qualify each target backend before a full campaign.**
Previous acceptance results do not qualify
the updated optimizer. Linux success does not certify macOS/ROCm/Windows.

**ROCm Whisper repeatability remains unresolved (September 23):** a later RX 7900 XT
run fails repeatability with eager efficient SDPA enabled. The initial passing
control did not establish a fix. Do not treat a successful retry as qualification,
relax the tolerances, or substitute an eager timing. The failure-aware policy
below lets the planned campaign finish with this condition marked failed.

CPU contract tests check the attention policy and receipts. A passing NVIDIA
test does not qualify the AMD path, and the independent reference suite does
not replace full-model campaign qualification.
The v11 failure investigation found nearly rank-two sinusoidal weights, with
strong cancellation that also exceeded the gradient bounds in PyTorch f32.
The initializer fixes that fixture defect; #203 separately fixes the derivative.
Neither requires rollback of tuning or weaker numerical checks.

V18 adds Quest 3S through the Android collector and updates the common Meganeura
pin. The SmolVLA training mask and long-product accumulation fixes apply to every
platform. Requalify all devices at the new Inferena revision; do not mix earlier
cohorts with this one. Android uses the same native search and timing policy,
with a separate full-tensor host CPU oracle. See [Quest collection](#quest-3s).

## What to run

First qualify the frozen candidate on each backend. To transfer this local
revision without depending on a moving remote branch, use the source bundle
`../inferena-results/local-cohort-20261011/inferena-cohort-0a718b6.bundle`.
The adjacent `RUN-COHORT.txt` contains portable instructions for the remaining
machines and explains how to preserve each run's archive.
From an existing Inferena checkout, after copying the bundle to that machine:

```sh
git fetch /path/to/inferena-cohort-0a718b6.bundle HEAD
git worktree add --detach ../inferena-cohort-20261011 FETCH_HEAD
cd ../inferena-cohort-20261011
git rev-parse HEAD
/absolute/path/to/existing/.venv/bin/python scripts/p3hpc.py --qualify-only
```

Verify the complete SHA above on every machine. Existing pinned environments
do not need reinstalling; use their absolute interpreter path from the clean
worktree. Model preparation downloads the pinned checkpoint if it is not
already present. No `--no-max-autotune` is needed on any platform;
that old spelling remains accepted as a no-op.
Qualification checks all five models and both arithmetic classes once. Keep
the results, including any failure, and resolve protocol problems before
starting the common cohort. A classified numerical failure can remain an
explicit coverage result; it need not block measuring the other conditions.
Full collection is simply `python scripts/p3hpc.py`, using the interpreter below.

For a new environment, install uv and Rust, then run the appropriate setup.
Setup downloads managed Python **3.13.13**, installs the pinned requirements,
and runs a small compiled forward/backward/replay probe. It does not run
max-autotune by default. No existing venv is overwritten.

| Machine | New environment | Collection with that environment |
|---|---|---|
| NVIDIA Linux / H100 | `bash scripts/setup.sh cu130` | `python scripts/p3hpc.py --backend cuda` |
| NVIDIA Windows / 3050 | `.\scripts\setup.ps1 cu130` | `python scripts/p3hpc.py --backend cuda` |
| AMD / 7900 XT | `bash scripts/setup.sh rocm7.2` | `python scripts/p3hpc.py --backend rocm` |
| Intel Arc B570 | `bash scripts/setup.sh xpu` | `python scripts/p3hpc.py --backend xpu` |
| Apple M3 | `bash scripts/setup.sh mps` | `python scripts/p3hpc.py --backend mps` |
| Radeon 780M, CPU correctness oracle | `bash scripts/setup.sh cpu` | `python scripts/p3hpc.py --backend cpu --gpu 780M --qualify-only --eager` |
| Intel RPL-U, qualification only | `bash scripts/setup.sh cpu` | `python scripts/p3hpc.py --backend cpu --gpu RPL-U --qualify-only --eager` |
| Ryzen 5 9600X iGPU on rubik, qualification only | `bash scripts/setup.sh cpu` | `python scripts/p3hpc.py --backend cpu --gpu 'AMD Ryzen 5 9600X' --qualify-only --eager` |

Here `python` means the created environment's interpreter:
`.venv/bin/python` on Linux/macOS or
`.venv/Scripts/python.exe` on native Windows. After activation,
`python scripts/p3hpc.py` needs no arguments on a supported single-GPU
machine. CPU PyTorch is only a correctness oracle for graphics-only qualification,
not a timing competitor. Keep those native GPU results as portability evidence;
ggml or ONNX Runtime is not needed as an additional reference.
`--eager` avoids compiling this oracle; it does not disable Meganeura tuning or
relax validation. Verify the exact native name with
`cargo run --release --locked -p inferena-meganeura -- --list-devices` before
selecting an integrated GPU. The shared RADV name RAPHAEL_MENDOCINO does not
identify a separate Mendocino APU. See the [qualification instructions](https://github.com/kvark/meganeura/blob/e8d7d9e3b2192671c0a9fb1aab2e3fc505153260/paper/p3hpc/QUALIFICATION.md)
for device disambiguation and archive lifetime; older qualification results
do not certify this pin.

For example, the Mac qualification is:

```sh
.venv/bin/python scripts/check_environment.py --backend mps
.venv/bin/python scripts/p3hpc.py --backend mps --qualify-only
```

Substitute that machine's backend. The Mac check is particularly important:
this revision now compiles MPS, whereas v7 deliberately ran eager. A failure
must be investigated or explicitly reported; never silently use eager times
as compiled times. Qualification retains no publication samples. Full
collection repeats all numerical gates in every measurement process, so
qualification is not a certificate for future processes.

On the dual-GPU workstation, use separate existing environments, sequentially:

```sh
.venv/bin/python scripts/p3hpc.py --backend cuda --gpu 'RTX 5070'
.venv-xpu/bin/python scripts/p3hpc.py --backend xpu --gpu 'Arc B570'
```

To create the second environment:
`bash scripts/setup.sh xpu .venv-xpu`.
The collector selects Meganeura's matching adapter by device ID. Ambiguous
matches or a different reference GPU stop the run. Do not run both collectors
together. Avoid host suspension, competing GPU work and profiling.

Defaults collect five models, strict and accelerated arithmetic, three fresh
processes per condition, at least five warmup calls and ten seconds of actual
workload execution, then twenty samples per phase:
**30 paired processes per device**, not the former 90 CUDA pairs. Results go
to a new `../inferena-results/<host>-<UTC>-<source>/` directory outside Git.
Copy that entire directory, including `campaign.json`, logs and compilation
receipts. `status: complete` means all requested gates passed;
`complete-with-failures` means every planned condition was attempted but some
failed (exit code 1). `incomplete` means collection stopped. Neither of the
latter statuses permits retrying until favorable or hiding failures.

The collector automatically writes **`../inferena-results/latest.tgz`** for a
finished or caught-failure campaign, including qualification runs. Send that
file: it contains the current run's manifest, reports, timing records, logs and
validation receipts. Raw tensors, CPU reference arrays, source archives and
executables stay in the original result directory and are omitted from the tgz.
Each archive replaces the previous `latest.tgz`; original result directories
remain untouched. With `--results-dir`, the archive sits beside that directory.
If archiving fails, a warning is printed and the previous archive is left intact.

For the cloud GPUs only, collect the larger models separately:

```sh
.venv/bin/python scripts/p3hpc.py --models SmolLM2-360M SmolLM2-1.7B
```

This includes forward/loss/backward. Use `--inference-only` only for an
explicitly separate inference study; it does not produce training data.
The family shares batch 1, 128-token prefill and stateless one-token forward,
but differs in depth, width and attention layout. It is not a parameter-count-only
scaling law. Do not quantize, offload or shrink workloads to make them fit.
The recent cached-attention and KV-cache optimizations are not exercised by
this stateless workload. Measuring those gains needs a separate decode study;
they must not be inferred from the minimal-forward timings here.

## Separate search-budget study

The primary default is now 30 seconds per native session. Revision
0ad1a5198 and the October 9–10 archives still used 60 seconds: the earlier
30-second experiments passed an explicit override, and the default had not
been changed. Preserve that distinction when comparing archived results.
Before a full collection, compare 30 and 60 seconds with native-only diagnostics
(`run.sh -f meganeura`) on the same pinned source and environment. Start with the
cases that previously needed more search time. The diagnostic target is within
5% of the best observed 60-second held-out time in each phase and arithmetic
class; investigate and repeat outliers. This is an observed comparison, not a
global-optimum guarantee. Reference compilation need not be repeated to compare
two native budgets. All numerical checks and held-out warmups still apply.

The archived H100 1.7B training runs spent 63–75 seconds on the first candidate
alone at the old 60-second budget. Local 135M results do not qualify the shorter
budget for that case. The deadline includes initialization and qualification
but cannot interrupt in-flight work; inference, minimal forward and training
each get their own deadline.

A publication budget curve can add 10- and 20-second points, with three fresh
processes per condition. Retain search receipts, rejected candidates, elapsed
preparation time and held-out execution times. Plot latency and preparation
time separately for each model, phase and arithmetic class, and keep this
study separate from the primary paired cohort.

## The executed contract

| Component | Primary policy |
|---|---|
| Meganeura | Calibrated egglog construction enabled independently of the reference mode |
| Strict arithmetic | f32 operands/accumulators; native-f32 cooperative tiles permitted; f16-input tiles forbidden |
| Accelerated arithmetic | Reduced-input paths permitted, with f32 accumulation and the declared validation gates |
| CUDA and ROCm | Default PyTorch compilation plus qualified whole-phase `torch.cuda.CUDAGraph` replay |
| XPU | Default PyTorch compilation, public math SDPA setting, qualified whole-phase `torch.xpu.XPUGraph` replay |
| MPS | Default PyTorch compilation, including first Metal specializations; no equivalent public whole-phase replay API |
| Graphics-only qualification | Explicit eager CPU oracle, no CPU-versus-GPU performance comparison |

ROCm exposes HIP graph capture through PyTorch's CUDA-named API. It is not
disabled because the backend is AMD. Apple compilation / internal MPSGraph
execution is not the same as explicit whole-phase CUDA/HIP/XPU replay.
Capture includes inference, minimal forward, and forward/loss/backward;
the latter has no optimizer update. Compilation and capture use one dedicated
stream on every replay-capable backend. Inductor's internal
`triton.cudagraphs` is off because this harness owns whole-phase replay.

The primary reference sets `max_autotune`, `coordinate_descent_tuning`,
`max_autotune_gemm`, and `max_autotune_pointwise` false. It still uses
Inductor's ordinary compilation and library algorithm selection; “no
max-autotune” does not mean no backend heuristics or all caches disabled.
Do not call this a measurement of PyTorch's best achievable steady state.

### Preparation budgets and evidence

Meganeura uses `train::build_measured`: representative weights and inputs are
initialized before candidate execution. A **30-second soft total deadline per
session** includes graph search, lowering, GPU pipelines, initialization and
qualification. Up to **16 graph/implementation forms and 64 programs** explore
fused/unfused expressions and scalar matrix tiles, K stages, counted/unrolled
loops and split-K. This includes transposed products and packed projections.
Lowering locks an explicit schedule; an automatic alternative retains ordinary
kernel probes, including cooperative paths where legal. Gradient matrix products
inherit the forward schedule with legality checks, rather than being tuned
independently. Explicit cooperative tile dimensions are not searched yet.

Small graphs are searched together. Larger pure graphs expose a bounded joint
region of rewritable operators, including independent projections; opaque
operators form its boundaries. If that region exceeds the bound or crosses
precision domains, search falls back to one verified repeated region. It does
not cross mutations. Missing regions, truncation and rejections are recorded.
Graph and physical-plan ranks are interleaved so additional matrix schedules
do not postpone every fusion, attention-layout/split or convolution weight-split
choice. Independent scalar dQ and dK/dV layouts are explored through egglog;
submission chunk counts share the same search on all GPUs. Logical tensor
bytes order extraction; this is not
an HBM or latency model. Qualified paired whole-step wall time selects winners.

Private kernel search uses the **remaining shared session budget**, with no
separate two-second slice, no class-count cutoff and **1 GiB maximum scratch**.
The ordinary e-graph extraction is tuned first; remaining time explores graph
and physical alternatives. Completed kernel comparisons are reused across
programs. Search can still end before every class or graph is explored, but
short candidate slices no longer repeatedly interrupt costly qualification
before later kernel classes are reached. The kernel domain
includes scalar matmul/convolution shapes, legal native-f32 cooperative tiles,
GEMV widths/column groups/reductions and reduction kernels. Reduced-input cooperative paths
remain available under the accelerated arithmetic contract, but are not an
exhaustive measured search. “All” means the library's current legal kernel
domain, not every implementation or a guarantee to finish it within the budget.

The combined logical-plan/snapshot bound is **75% of the device's reported
available budget** at construction. This conservative bound precedes allocation;
padding, driver objects, staging and private probes are additional. It is not
a peak-VRAM measurement. An unavailable budget fails explicitly. Phases release
their sessions before constructing the next shape; calibration may temporarily
hold an incumbent and challenger. Most models construct three sessions; Whisper
uses two. In-flight driver work and qualification can overrun the soft deadline.

Every candidate's full outputs and canonical parameter gradients are compared
with the first checked program. V21 additionally checks every candidate against
the original graph's F32 arithmetic, as described above. Candidates with outstanding kernel probes are
checked before and after tuning; memo-only candidates are checked after installing
their kernels. Winning challengers are checked after paired measurement, and
the final incumbent is checked again if another challenger has run since its
last check. A memo hit never bypasses qualification. Packed gradients are
unpacked into the original checkpoint coordinates. Outputs use staged batch
readback, avoiding repeated mapped-read calibration for short-lived candidates.
The within-search maximum/RMS gates match the replay policy below;
accelerated gradients retain the separate 1% whole-gradient tolerance.
V22 changes only the accelerated language-model comparison to original F32
into the distribution/loss check above. The outer
PyTorch comparison remains an independent gate. No optimizer update is part of
these workloads. Construction measurements are not publication samples; the
selected session is warmed up and measured again afterwards.

Search comparisons first warm both candidate and incumbent for **two pairs and
250 ms**, bounded by the remaining session deadline. This is separate from
publication warmup. The collector checks these actual search options in every
session receipt, alongside the graph/program limits.

Since campaign protocol v17, private kernel decisions are saved under
`native-kernel-memos/<precision>/<model>/`. Runs 2/3 and compatible phase shapes
resume those decisions; they do not repeat completed kernel comparisons.
The cache is new for each campaign and keyed by Cargo.lock, device/driver,
cooperative policy and tuning options. Meganeura additionally checks its memo
schema and complete kernel-class/candidate keys. Saves are atomic and happen
only after successful construction. Incompatible or corrupt memos start fresh.
The first run is cold with respect to this memo; later runs are explicitly
warm. Receipts record loaded/saved classes, reused class visits and new comparisons;
reused visits also include reuse within a search. Preparation costs must be
interpreted with these receipts, not pooled as independent cold-start searches.
Driver shader caches are separate: a fresh kernel memo does not make shader
compilation cold. The collector records driver-cache controls but labels their
initial contents uncontrolled. Report preparation as such, or collect a separate
experiment with verified empty per-run driver caches. The native F32 reference
can warm pipelines within a process; its preparation is included in `compile_s`.
Whole-program graph/layout/submission search still runs in each fresh process,
with unchanged bounds and no patience stop. This does not persist weights,
inputs, GPU state or the selected whole program. Memo reuse never bypasses
numerical qualification.

Both engines warm up with the actual phase workload until **both** five calls
and ten seconds have completed. Actual warmup counts and elapsed seconds are
retained for each phase and checked by the collector. This addresses observed
post-construction timing drift; it does not guarantee thermal equilibrium or
constant clocks. There is no CPU busy-loop priming, forced clock setting,
adaptive selection of a favorable window, or discarded measurement sample.
Do not suspend the host or run competing work. Keep the same power policy
throughout a campaign. Search timings remain separate from these held-out
publication samples.

Receipts retain actual limits, every trial, kernel coverage/decisions, selected
program, qualification coverage, cooperative policy and time breakdowns.
The collector checks these receipts, not just an environment variable.
`compile_s` now includes weight/input initialization and full-tensor checks;
do not interpret a change from the preceding cohort as shader-compiler time.

PyTorch has a **120-second compilation deadline per process**, covering
`torch.compile` and the first specializations of all requested phases,
including backward and minimal forward. A separate supervisor kills the
compiler process tree on expiration, records `torch-preparation.json`, and
returns a failure: it never substitutes eager. Model loading and replay/
numerical qualification are outside this compile interval and have a separate
ten-minute process safety cap. The setup probe uses the same supervisor.

The collector also applies a **1,200-second wall-clock watchdog per paired
runner**, including both engines, their wrappers and compiler workers. Its
`watchdog.json` records the limit, elapsed time, exit code or timeout. Expiration
stops the campaign without retrying or starting another GPU workload. The same
cap applies to a separate eager diagnostic. `--run-seconds N` changes this
safety limit explicitly; it does not change either engine's search/compile
budget. A killed process cannot guarantee recovery of a wedged GPU driver.

Campaign manifests, per-engine records and summaries are published by writing
and syncing a temporary file in the destination directory, then replacing the
record atomically. Each completed engine is checkpointed before the next
starts, so a later hang preserves the completed half of the pair. An interrupted
pair is still incomplete. File errors stop collection instead of reporting a
successful save. The manifest records host/kernel identity before benchmarking;
native device selection records driver name/info (including Mesa's version)
before the first paired run, and native results retain their environment receipts.

Before device probing, the collector builds both the harness/native pair and
the native wrapper's single-package recipe. Cargo feature unification can select
different cached native executables for those recipes without recompiling.
The final harness and wrapper-selected native executable are copied to
`binaries/`; `runner_binaries` records their SHA-256 hashes and build commands.
Hashes are checked before and after every pair, after any separate eager
diagnostic, and at campaign completion. A changed or missing executable stops
collection with the actual hashes retained. These checks are outside timing.

Qualify the frozen source on B570 first, then collect its campaign before the
other devices. Preserve every attempt. Following a watchdog timeout, inspect
the retained records and device state before scheduling further work on that
machine. See the [October 11 checklist review](ANALYSIS.md#pre-cohort-checklist-review-october-11-2026)
for recovered library experiments and remaining decisions.

This is a bounded-startup comparison, **not an identical total wall-clock
budget for both engines**: Meganeura has a per-session construction deadline,
while PyTorch has a per-process compilation deadline. Both can overrun during
in-flight work; native qualification also counts toward its deadline. Report
measured preparation costs; do not assume every
Meganeura search finishes sooner than every PyTorch compilation.

Optional, separate studies:

- `--max-autotune`: add searched PyTorch under the same 120-second deadline.
- `--graph-ablation`: also collect the uncaptured default reference.
- `--compile-seconds N` / `--tune-seconds N`: explicitly different budgets.
- `--no-kernel-memo`: disable cross-process kernel reuse for a separate
  independent-search preparation study.
- `--no-graphs` / `--eager`: explicit reference overrides, never automatic fallbacks.
- `--sdpa auto|math|efficient`: explicit PyTorch attention backend, recorded in
  the campaign and checked against every runner receipt. This overrides all
  selected models. Defaults are math on XPU and automatic otherwise.
  Unavailable backends fail; no fallback.

Do not use these for the common primary cohort. If a primary condition times
out or fails, retain the failed record and diagnose it before choosing a
different labelled experiment. A timeout means the declared budget was not
met; an omitted opt-in condition alone is not a portability failure.

Both engines record `compile_s`; PyTorch also separates model loading and
capture/qualification. MPS no longer skips compilation or manufactures a
zero compile time. Driver caches may persist across processes; fresh Inductor
and Triton directories do not establish a completely cold driver/library.
This remains a threat to preparation-time interpretation, not a reason to
change the already declared arithmetic or numerical gates.

## Source and numerical validation

All platforms require PyTorch **2.13.0** reporting upstream source
`cf30153c4c131c8164ee7798e5022d810682e2cb`, with the appropriate vendor wheel
suffix. Python, Torch source, wheel version, package inventory, driver/device,
compiler options, selected adapter and input hashes are recorded and checked.
Common reported upstream source does not mean identical vendor libraries or
binary builds. Do not upgrade dependencies mid-cohort.

Missing selected SmolLM2 base checkpoints download automatically, using
`models/smollm2-revisions.json` and SHA-256 receipts. `--offline` requires
them already prepared. A legacy model directory is adopted only if its files
match the pins; mismatches are not overwritten.

SmolVLA, StableDiffusion, ResNet-50 and Whisper-tiny use synthetic parameters
under `name-index-uniform-v1`, recorded in both runner receipts and the campaign
manifest. A wrapping 32-bit hash of the canonical UTF-8 parameter name is XORed
with the flattened native-layout index and mixed with two integer multipliers.
Its upper 24 bits map exactly to f32 in `[-1, 1)`, then scale by `0.024494898`
(`0.012247449` for ResNet). This preserves the old sinusoid's theoretical RMS
without its rank-two matrix structure. PyTorch transposes linear storage after
generation; registration order and framework RNGs do not affect the values.
StableDiffusion's identity norms and ResNet's zero folded-BN biases are unchanged.
Inputs remain unchanged, including sinusoidal inputs. SmolLM2 loads its pinned
weights and does not use this synthetic policy. Other framework runners on this
experimental branch have not been migrated to the new fixture.

The collector requires native `inferena-paper-v4`, PyTorch
`inferena-graph-replay-v8` and matching initializer policies. Rust builds are
locked and finish before the first pair; build parallelism defaults to one job.

Both captured and uncaptured execution check every output/loss tensor separately, with exact
shapes/dtypes/participating-gradient inventory and finite values. Outputs in
both modes and strict per-parameter gradients use fixed maximum/RMS bounds:

```text
max(abs(error)) <= 1e-6 + 1e-4 * max(abs(reference))
RMS(error)      <= 1e-6 + 1e-4 * RMS(reference)
```

Both inequalities must pass; maximum error catches sparse corruption and RMS
catches diffuse drift. Loss is its own tensor, not pooled with larger outputs.
Strict gradients also pass the same full-gradient bounds. Output records retain
the count of pointwise mismatches as a diagnostic, not an acceptance gate.
`fixed-full-tensor-v4` uses the same fixed tolerance coefficients as before,
but deliberately changes the norm: small cancellation residuals near zero no
longer fail solely because their elementwise relative error is large. This
is an acceptance-rule change, not an unchanged-validation claim.

Accelerated training uses one fixed reference, eight ordinary repeats and two
replays, with complete-gradient maximum and element-weighted RMS bounds at
`atol=1e-6, rtol=0.01`. It retains per-tensor diagnostics, but never learns
tolerances from repeat samples. Strict and inference use two ordinary repeats
and, when capture is requested, two replays. Uncaptured execution has no
replays but uses the same ordinary-call checks. Numerical failures are not
discarded as timing outliers.

Cross-engine validation retains the v7 policy: strict errors must remain below
5% per process; accelerated forward must pass and each training error must
remain below a 10% safety ceiling, with medians below 5% across three independent
processes. Every sample is retained; `replicated-gradient-median-v1` is a
separate gate from within-process replay validation. These cross-engine and
gradient limits are unchanged by the output repeatability correction.

`check_pair` also rejects disabled native tuning, an incorrect strict policy,
missing session evidence, unexpected class caps, uncompiled requested modes,
missing budget supervision, wrong graph APIs, revision/device mismatches,
diagnostic samples and invalid timing series. A successful subprocess exit
alone is insufficient.

### Failed references and the separate eager diagnostic

A numerical repeatability failure or a narrowly identified rejected graph
capture is a result, not permission to weaken the gate. The primary JSON and
preparation receipt keep the failure phase, check, tensor and available error
metrics. An already qualified inference result survives a later training
failure as `status: partial`; failed or unattempted phases have no timing
samples. Subsequent planned process conditions still run, exactly once each.
Cross-engine mismatches are recorded without assuming which engine is wrong.
Unknown exceptions, timeouts, OOM, device loss and configuration/identity
errors still stop collection. An arbitrary traceback is not classified as a
numerical failure by string matching.

For each failed compiled condition, the controller runs **one** additional
PyTorch process in `diagnostic-eager/`: eager execution, no graph capture,
math SDPA, the same GPU, weights, inputs and precision. It must pass the same
full-tensor ordinary-repeat checks. Its receipt sets `protocol.diagnostic`
and its command is recorded. When it succeeds, the existing native result
is compared with it in a separate `comparison.json`; the primary files are
not rewritten and Meganeura is not rerun. This can provide a correctness
reference when the compiled path fails, not proof of the underlying fault.
If eager also fails, that failure remains visible. Neither path is retried.

Eager diagnostic timings **never** enter compiled-reference ratios. Inspect
`runs[].phases` and `phase_coverage`, including failed and unattempted phases.
Aggregate a primary phase only with all declared process replicates present
and valid; training additionally requires its replicated-gradient gate. That
gate reports incomplete groups instead of accepting a smaller, lucky subset.
The automatic diagnostic does not change `--eager`: an explicitly requested
eager primary is a different labelled experiment and has no eager fallback.

## Platform notes

### Quest 3S

Quest is a native Vulkan measurement target with a host CPU correctness oracle.
It has no paired PyTorch GPU timing. Use the same committed Inferena revision as
the other devices. The Android campaign is labelled `p3hpc-android-campaign-v1`
and records its common `p3hpc-paired-campaign-v18` native policy.

Install Android SDK platform-tools (ADB), Android NDK **r27**, and the Rust target
`rustup target add aarch64-linux-android`. Use the pinned host Python environment
above; the existing NVIDIA environment also provides the CPU oracle. No Android
PyTorch installation or patched benchmark sources are required.

Connect USB once to authorize debugging. To move to a strong charger, run
`adb -s <USB-serial> tcpip 5555`, find the headset's Wi-Fi address, then
`adb connect <IP>:5555`. Use that explicit address as `--adb-serial`. The collector
checks the Quest 3S identity, external power, battery >=30%, and battery
temperature <45 C. It warns on weak charging and records battery/thermal state before and after every
process. It enables stay-awake/proximity automation for the campaign and restores
the original stay-awake setting and normal proximity behavior in `finally`.
If the connection is lost, `campaign.json` records any failed restoration;
reconnect and run `adb -s <serial> shell am broadcast -a
com.oculus.vrpowermanager.automation_disable` and restore the recorded
`original_stay_on_while_plugged_in` with `settings put global`.

From this checkout, first qualify both arithmetic classes and all five models:

```sh
python scripts/p3hpc.py --backend android --adb-serial <IP>:5555 --android-ndk <SDK>/ndk/27.0.12077973 --qualify-only
```

Then start collection by omitting `--qualify-only`. Defaults retain three fresh
device processes per model/class, five warmup calls and at least two seconds,
twenty measured calls per phase, and the common 30-second calibrated search.
The runner is built from Cargo.lock, pushed, and hash-verified. Pinned SmolLM2
assets are prepared on the host, pushed, and hash-verified on the headset.
`--offline` requires already prepared host weights. `--android-timeout` defaults
to 1800 seconds per device or CPU process. Device timeouts also kill the remote
runner; they do not retry it. Each campaign uses a new remote directory and new
kernel memo; replicas share only that campaign's kernel decisions.

When the Android driver reports no GPU memory budget, the search uses 75% of
`min(MemAvailable, 3 GiB)` as its logical-plan bound. Each session records the
actual Vulkan budget/usage, system availability, cap source and effective bound.
This is a conservative shared-memory search limit, not a GPU allocation or peak
measurement. Drivers with a usable Vulkan budget keep the common budget policy.

The pinned Meganeura serializes Android Adreno driver pipeline creation after
an intermittent crash was observed inside Qualcomm's LLVM compiler during
parallel pipeline construction. Shader generation stays parallel, and this
workaround does not serialize measured GPU execution.

Every process exports the selected program's exact full parameters, inputs,
inference/latency outputs, training loss, and all canonical gradients outside
the timing regions. A fresh host PyTorch CPU **float64** model checks every
output/loss/gradient tensor using unchanged maximum and RMS bounds
`atol=1e-6, rtol=1e-4`, plus the aggregate gradient vector. The higher-precision
oracle avoids the previously demonstrated SmolLM2 CPU f32 accumulation error.
Both requested arithmetic classes currently execute f32 on Adreno 740 because
it exposes no cooperative matrices; the collector verifies this and uses strict
bounds for both. No reduced-arithmetic benefit is claimed. Native candidate
qualification also remains enabled throughout search.

The result directory contains `campaign.json`, `results.md`, exact source and
binary snapshots, raw device/CPU logs, timing samples, tensor blobs, full CPU
reference arrays and comparison receipts. Reserve tens of GB on both host and
headset for a full campaign. `latest.tgz` is produced as on desktop. Remote
evidence remains at the recorded `remote_directory` until explicitly removed.
Failed native processes also retain the Android crash log buffer (which may
include older device crashes). Failed processes and mismatches remain visible; only complete, qualified
replicate groups produce timings in the report. Qualification produces coverage
but no publication timings. Host CPU duration never enters performance ratios.

Combine same-revision desktop and Quest campaigns with:

```sh
python scripts/p3hpc_report.py <desktop-campaign> <quest-campaign>
```

Add `--json` for machine-readable rows. Existing paper artifact cohorts remain
historical and are not rewritten by this collector.

Windows uses native x64 Python, Git for Windows / Git Bash (not WSL), Rust MSVC,
and Visual Studio C++ Build Tools with a Windows SDK. Setup selects
`requirements-pytorch-cu130-windows.txt`, including pinned
`triton-windows==3.7.1.post27`. Importing Torch alone does not establish that
Inductor works. The watchdog uses Windows process-tree termination; preserve
both the receipt and runner log on failure.

Intel needs Vulkan for Meganeura and Level Zero / IGC / offline compilation
support for XPU. A discrete GPU is not inherently required, but older iGPU
branding does not establish support in the pinned PyTorch build. On this
Ubuntu workstation, `intel-opencl-icd` supplies a shared IGC dependency;
the benchmark itself does not execute OpenCL. The XPU causal-LM embedding
backward probe and qualified `index_add` workaround remain recorded under
`execution.embedding_backward`; this must not be described as an unmodified
native XPU result.
Full-model replay qualification also exposes an event wait inside the pinned
fused attention operator, reproduced by an isolated grouped-query attention
call. The public `sdpa_kernel(SDPBackend.MATH)` setting passes isolated
forward/backward capture. XPU uses this explicitly reported configuration
for both replay and its uncaptured ablation; other backends retain automatic
SDPA selection. The full-model workaround passes
the same gates on B570.
`execution.sdpa_policy` and `sdpa_enabled_backends` record the active setting.

The October 10 SmolLM2-360M captured run still fails inside XPU's compiled
embedding backward with `wait method cannot be used for an event associated
with a command graph`. An October 11 diagnostic with replay disabled passes
all three uncaptured phase checks in both arithmetic classes (one process
each). Forward/loss/backward takes 54.501 ms strict and 54.558 ms accelerated.
Both also pass the sampled-output and gradient-norm comparison against the
archived strict native record. Results, source snapshot and commands are in
`../inferena-results/precohort-20261011/b570-360m-no-graphs/` and
`../inferena-results/precohort-20261011/rerun_b570.py`. These are diagnostic
results from a dirty checkout; neither a replay fix nor replicated qualification
of the current native pin is established. For a separate frozen-source study,
use `--backend xpu --models SmolLM2-360M --no-graphs`; preserve the captured
failure and keep the override labelled.

The September 23 RX 7900 XT campaign at Inferena `2a8cbf52` still failed strict
Whisper after SDPA was excluded from compilation. Inference capture passed;
two uncaptured training calls disagreed on the encoder output (max error
0.00197 against a 9.1e-6 bound). This fails PyTorch's own repeatability check,
before comparison with Meganeura or training HIP graph capture.

Initial reported controls on that GPU varied attention policy and compilation:

| Encoder / SDPA execution | SDPA backend | Training output repeatability |
|---|---|---|
| Compiled / compiled | auto | Failed, about 0.002 max error |
| Compiled / eager | math | Failed, about 0.002 max error |
| Compiled / eager | efficient only | Initially passed bit-exact; a later run failed |
| Compiled / eager | auto | Isolated pass, campaign failure |
| Entirely eager | math | Passed, bit-exact |

Campaign `rubik-20260923T163520494362Z-757f6a80` confirms the later failure at
`757f6a80`, with `SDPA: efficient, eager`, not a stale checkout. Strict Whisper
fails at `uncaptured repeat 1 output 0`: maximum error 0.00235241 versus
9.10142e-6, RMS error 9.13354e-5 versus 3.07931e-6, with 157,226 of 576,000
pointwise mismatches. Compilation completed in 7.91 seconds; inference capture
passed. The first four strict pairs passed, with bit-exact ordinary output
repeats in all three phases; no accelerated pair was reached.
This invalidates the policy as an established workaround; the earlier pass does
not isolate the defect to math SDPA. An enabled-backend list also does not prove
which kernel an automatic call selected.

V13 combined `torch.compiler.disable` for ROCm Whisper SDPA with
`sdpa_kernel(EFFICIENT_ATTENTION)`. That configuration did not restore
repeatability and is no longer the default. V14 records the ordinary compiled
failure, if it occurs, and attempts the separate whole-model eager/math
diagnostic described above. This is not a claim that the ROCm defect is fixed.

For further diagnosis, distinguish training forward output **before backward**
from the same tensor **after backward**, while checking that parameters and
inputs did not change. Preserve the failing process's generated code and first
divergent intermediate. These are diagnostic controls, not benchmark results:
extra copies or synchronization can themselves hide a lifetime/race bug. A pass
under instrumentation is not a fix either.

The campaign records `reference_sdpa_policy` and `reference_sdpa_compile` as
per-model maps. Each runner records `execution.sdpa_policy`,
`sdpa_enabled_backends` and `sdpa_compile`; the collector checks all three.
No manual `--sdpa` override is needed to reproduce the current policy. The
following checks the current policy and preserves both outcomes, without
retrying the compiled condition:

```sh
.venv/bin/python scripts/p3hpc.py --backend rocm --models Whisper-tiny --qualify-only
```

Every numerical gate and the compilation deadline remain enabled. `--sdpa`
still selects the attention backend, independently of whether SDPA is compiled.
This is a declared backend modification, not an unmodified Inductor result or
a demonstrated fix. A fix must address the observed failure and pass unchanged
gates in both precision classes across independently started processes; all
attempts count. Keep the failed campaigns without treating them as paired
timings. Do not change the common-cohort protocol on the strength of an isolated
passing control.

The runner selects Triton's backend explicitly from the requested Torch
device, avoiding ambiguous auto-detection when NVIDIA and Intel drivers are
both installed. It records the selection and rejects a conflicting override.

On B570 with Mesa 26.0.3, the advertised f16-input/f32-accumulate matrix shape
is 8x16x16. Blade now reports full shapes, but Meganeura's WGSL kernels still
accept only square 8x8x8 or 16x16x16 tiles, so it reports no usable tiles and
uses scalar kernels even under the accelerated policy. This is a stack shape
limitation, not missing matrix hardware or a setup failure. Rectangular-tile
support would benefit the accelerated path; it would not make f16 operands
eligible for the strict contract.

For a separate ROCm diagnostic on the Radeon 780M configuration previously reported in
[issue #61](https://github.com/kvark/inferena/issues/61), retain the needed
runtime workarounds:

```sh
export HSA_OVERRIDE_GFX_VERSION=11.0.2
export HSA_ENABLE_SDMA=0
.venv/bin/python scripts/p3hpc.py --backend rocm
```

The cohort uses the CPU correctness oracle above on this device while its
ROCm failure remains unresolved. Do not add those workarounds on machines that do not need them. Their values
are recorded. The missing gfx1103 rocBLAS library, SDMA failure and failing
max-autotune kernels were observed portability problems, not numerical
exceptions invented by the harness. The new common primary condition omits
max-autotune everywhere, so it is not an AMD-only fallback. Graph replay must
now pass its own qualification on ROCm.

Numerical success does not certify VRAM residency. Large discrete-GPU
workloads can spill into host-accessible memory; unified-memory devices need a
different interpretation. Native planned/allocated storage, Torch allocator
peaks and MPS end-of-phase allocation (no peak API) remain separately labelled.
Do not compare them as identical peak-VRAM measurements.

The acknowledged Naga Workgroup ArrayStride diagnostic is not a reason to
disable Vulkan validation. Distinguish it from new validation or numerical
failures.

## Local acceptance evidence

At `0a718b6`, all five models in both arithmetic classes pass B570 and RTX 5070
qualification on the current library pin: ten valid conditions and thirty
valid inference/latency/training phase comparisons per GPU. Each subsequent
collection also completes all thirty paired processes and ninety phase
comparisons, including all ten replicated-gradient gates. PyTorch graph
replay is enabled, numerical gates and budgets are unchanged, and each new
campaign starts with an empty kernel memo before recorded reuse. Other
backends require their own receipts. See the
[October 11 diagnosis and qualification](ANALYSIS.md#b570-amp-diagnosis-and-kernel-review-october-11-2026)
and the [replicated local cohort](ANALYSIS.md#frozen-local-cohort-october-11-2026).

For v12, CPU checks cover shared Rust/Python f32 bit patterns, transposed storage,
matrix rank and scale, and refusal of missing or stale initializer receipts.
An all-model CPU audit matches all 508 canonical parameter layouts/counts and
SHA-256 digests (144,516,216 f32 values), including the norm/bias exceptions.
Eleven Rust tests and six Python campaign-contract tests pass, as do Clippy and
formatting for the two collection packages. These are not GPU qualification.

The following evidence predates v12 and does not qualify its changed fixture.
At Inferena `689f924`, all ten RTX 5070 model/precision pairs
pass against PyTorch on Meganeura `71c202c`. Candidate qualification exposed an
RMSNorm fusion bug; merged fix `dbb4364` has the identical source tree to `c74ea82`,
which passes strict and accelerated full-model SmolLM2 checks with every formerly
invalid candidate now qualifying. Its focused GPU regressions pass on both the
5070 and B570, and upstream CI passes. At Inferena `028a414`, all ten B570
model/precision pairs pass against GPU PyTorch on the merged pin. Six strict
StableDiffusion training alternatives exceed the full-gradient gate and are
discarded; the selected program passes. No accelerated candidates are rejected.
These are qualification runs, not a new publication timing cohort.

A same-pin strict SmolLM2 pilot measured staged output readback and single-pass
gradient checking: inference qualification fell from 38.4 to 4.7 seconds while
completed program trials increased from 24 to 28; total three-session preparation
fell from 144.2 to 113.6 seconds. All 134.5M gradient elements are still checked.
These are preparation diagnostics, not replicated publication speedups. Training
search can still exhaust its deadline before visiting every kernel class.
Raw evidence stays outside Git. Previous-cohort acceptance is available in Git
history; it is not a certificate for this construction path on other backends.

## Separate diagnostics

No profilers, compiler-debug overrides or local dependency overrides belong
in publication collection. For diagnostics use a new output directory:

```sh
INFERENA_TORCH_MODE=default INFERENA_GRAPH_REPLAY=1 \
  ./run.sh -f pytorch,meganeura -m ResNet-50 --strict --profile \
  --results-dir ../resnet-profile
```

PyTorch exports a CPU/GPU Chrome timeline; MPS needs native Metal tooling for
GPU events. Meganeura exports per-pass GPU timing sidecars.
For XPU, PTI may omit child-kernel events inside command-buffer replay even
when ordinary kernel profiling works; API events alone are not GPU timings.
Use `scripts/profile_report.py` to rank families. Instrumentation can change the
schedule and timing; synthetic host placement of duration slices is not a
calibrated GPU start time. `wall - sum(kernel medians)` is neither CPU time nor
barrier cost. Prefer vendor timelines for causal attribution.

For a validated NVIDIA CUDA/Vulkan Nsight Systems pair on Linux:

```sh
.venv/bin/python scripts/limited.py --memory-mib 6144 --seconds 600 -- \
  .venv/bin/python scripts/nsys.py --gpu 'RTX 5070' \
  --torch-version 2.13.0+cu130 --model ResNet-50 --precision strict \
  --nsys /path/to/nsys --results-dir ../resnet-nsight
```

Use Nsight Systems 2026.4.1 or newer for this workstation's Vulkan capture.
The Linux wrapper verifies cgroup memory/swap/time limits and leaves a host
RAM reserve; it does not bound VRAM or all driver-pinned allocations. Nsight
uses diagnostic single-thread compilation and its times are not publication
preparation times. `scripts/nsys_report.py` summarizes the resulting SQLite
events; [ANALYSIS.md](ANALYSIS.md) documents attribution limits and earlier
pinned findings. Keep traces and raw results outside Git. Git history preserves
the superseded protocol notes and experiment revisions without binary artifacts.

## AMP reference and real-data convergence

The October 11 evidence in this section consists of supplementary studies,
separate from the paired v18 cohort. V19 integrated the repeated-input AMP
gate into accelerated; v20 replaces it for SmolLM2-135M as described above.
The historical observations below retain their original gates.
`amp-bf16` and `amp-fp16` keep f32 parameters, use PyTorch autocast for eligible
operations, and reduce losses in f32. Remaining f32 matmuls use `highest` with
TF32 disabled. fp16 uses `torch.amp.GradScaler`; bf16 disables scaling.
Native f16 activation storage is not required to collect these reference columns.

```sh
.venv-xpu/bin/python scripts/amp_reference.py --backend xpu --gpu B570 \
  --models SmolLM2-135M Whisper-tiny --precisions bf16 fp16 \
  --results-dir ../amp-b570
```

The optional dtype diagnostic is compiled, explicitly uncaptured, and uses the cohort's
attention policy (math on XPU, auto elsewhere). It retains the
same 120 s compilation deadline and five-call/ten-second warmup followed by
twenty samples. One fresh process per condition is the diagnostic default; dtype
order alternates. `--mode eager` is a labelled diagnostic. No capture or
compiler failure substitutes another timing. `columns.csv` names each dtype,
compiler mode, capture setting and replicate. The development paired collector supports
AMP replay with scaler bookkeeping outside capture and inside timing. The
historical v3 AMP diagnostic used entirely uncaptured execution.

Before timing, the requested compiled/eager candidate must match eager strict
f32 on the entire forward output: per-tensor RMS and maximum error bounded by
`1e-6 + 0.02 * reference magnitude`. The historical training gate performs 32 SGD updates
at learning rate 0.001 on the fixed benchmark input. Every loss must be finite,
no update may be skipped, per-step loss error is at most
`1e-6 + 0.02 * abs(reference)`, and curve relative L2 is at most 0.01.
Weights, buffers and random state are restored afterward. This precision gate
does not establish learning on real data. Ordinary full-tensor repeatability
qualification still applies before each timed phase.

AMP v3 preserves qualified inference and latency when only the training curve
fails. The checkpoint is restored before measurement, the training cell is
empty, and the record/campaign is marked partial. Its failed training gate
and complete trajectories remain in the receipt; no training timing is substituted.

AMP v2 counts completed optimizer updates. If GradScaler detects overflow,
it lowers the scale and retries the **same batch with unchanged weights**,
up to eight attempts. Every attempted loss, scale and update decision is
retained; a batch is never silently dropped. The timed backward starts with
the qualified scale. The initial v1 diagnostic counted attempts, which left
only 255 updates after one fp16 overflow; that failed receipt is retained.

Timed training retains the existing forward/loss/backward scope. The fp16
reference includes scaling, unscaling, finite checks and scale maintenance.
GradScaler uses an optimizer whose step performs no update; real SGD appears
only in the untimed gate and convergence study. See PyTorch's
[AMP guidance](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

The real-data study uses the pinned SmolLM2-135M checkpoint and tokenizer with
Project Gutenberg's [A Midsummer Night's Dream](https://www.gutenberg.org/ebooks/1514).
The corpus and tokenizer SHA-256 values are frozen in `scripts/convergence.py`;
a changed download is rejected. Gutenberg headers/footers are stripped before
tokenization, no special tokens are added, and the first 80% of tokens train
while the last 20% are held out. All engines receive the same saved 128-token
next-token batches and seeded order, for 256 SGD updates at the same f32
learning rate, without momentum, weight decay or clipping.

```sh
curl --fail --location https://www.gutenberg.org/ebooks/1514.txt.utf-8 \
  --output ../pg1514.txt
.venv-xpu/bin/python scripts/convergence.py --backend xpu --gpu B570 \
  --corpus ../pg1514.txt --results-dir ../convergence-b570
```

Start with B570, then repeat on the other devices. Each worker has a 1,200 s
process-tree watchdog. Receipts are replaced atomically after every update;
the study keeps source archives, the corpus, tokenizer, ordered token batches,
search preparation and raw losses outside Git. Native construction uses a
fresh kernel memo and the 30 s search policy. PyTorch is explicitly eager,
math SDPA and uncaptured. Real-text XPU runs qualify embedding backward on
the first batch, including repeated tokens, and record any selected
mathematically equivalent `index_add` workaround.

The learning gate requires every update, a changed finite `model.norm.weight`,
at least 5% lower loss on four fixed training probe sequences, and no more
than 2% higher loss on four held-out sequences. Probe and held-out losses are
recorded at steps 0, 64, 128, 192 and 256. Native f32 and each AMP trajectory
must also pass the same per-step/relative-L2 curve bounds against PyTorch f32,
including every evaluation sequence. Failed numerical gates remain failures.
This establishes a short matched training trajectory for one model, corpus
and seed; it does not establish general convergence or supply throughput data.

SmolLM2-135M currently fails the fixed synthetic-input AMP training gate on
both B570 and RTX 5070, including eager math SDPA. The corresponding real-data
study has different outcomes. Preserve these as separate results: do not
substitute the real-data gate for failed benchmark training cells, or block
the primary f32 cohort on an assumed XPU-specific AMP defect. The
[diagnostic and kernel review](ANALYSIS.md#b570-amp-diagnosis-and-kernel-review-october-11-2026)
records the evidence. GEMV unrolling and convolution interleaving prototypes
remain excluded from the frozen `0a718b6` candidate, which corrects the
collector's handling of omitted empty reuse lists from `0c78dff`.
