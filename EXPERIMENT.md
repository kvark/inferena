# P3HPC CUDA Graph comparison

Source branch: `experiment/p3hpc-cuda-graphs`, based on Inferena main.
The submitted source remains tagged `paper-arxiv-1`. Git records both bases;
do not copy binaries or experimental raw records into this branch or main.
The Meganeura dependency is pinned to merged main `fcdd76d1` (0.3.0); it is not a
floating sibling checkout.

## Current collection readiness, September 12

The full RTX 5070 campaign at `f4255c4b` passed all 30 qualification pairs and
52 measurement pairs before the next accelerated Stable Diffusion max-autotune
pair crossed the old one-process gradient-norm gate. Forward relative L2 was
0.373%, total-gradient error 2.11%, and the 181-parameter gradient-norm error
5.184%, just beyond the fixed 5% cutoff. Across the three fresh PyTorch
processes available for that condition, PyTorch's own parameter-gradient-norm
vectors differ by as much as 5.575%; Meganeura is identical across every
process. This is measured reference variation, not a runner crash or a
Meganeura regression. The raw incomplete campaign was discarded after this
diagnosis; it is reproducible from `f4255c4b`.

`replicated-gradient-median-v1` therefore evaluates the distribution that the
campaign already collects. Strict mode keeps the 5% per-process gate.
Accelerated mode retains every sample below a 10% safety ceiling and requires
the median cross-engine total-gradient and parameter-gradient errors to remain
below 5% across three independent processes. No observed value enlarges a
limit, and no sample is discarded or retried. The v6 manifest carries every raw
error and the small aggregate report; the paired records retain both engines'
outputs for offline diagnosis. A fresh complete campaign is required for the
revised policy; completed v4 campaigns already satisfied the stricter
per-process gate.

Earlier attempts established the within-process replay policy:

At source `db048638`, all 30 paired CUDA conditions passed on RTX 5070 / driver
595.71.05.
A later collection attempt at source `6531bdee` stopped before measurement after
22 valid pairs: accelerated diffusion failed while comparing two ordinary
PyTorch runs, not a CUDA Graph replay or cross-engine result. Its local gradient
error exceeded the maximum learned from the preceding repeats while remaining
inside the independent 1% whole-gradient limit. This disproves the old
sample-fitted policy as a stable collection gate.

The incomplete manifest is preserved at
`../inferena-results/zork-20260911T041114972531Z-6531bdee/campaign.json`; it is
not performance data and must not be resumed. `fixed-full-gradient-v3` below is
the replacement candidate. The provisional collection tags were removed; the
moving `experiment/p3hpc-cuda-graphs` branch is authoritative until the
protocol and data settle, at which point one final revision can be tagged.
Other machines must still pass their own preflight.

Reproduce a short preflight with `python scripts/p3hpc.py --qualify-only`;
omit that option to collect three fresh-process replicates per condition. Each
measurement process performs the full numerical and replay validation before
retaining timing samples, so a separate fourth cold-compile process would add
cost without validating the later processes. The old complete qualification
remains at
`../inferena-results/zork-20260910T161443333460Z-db048638/campaign.json`, but it
does not qualify the revised policy. Failed attempts remain evidence rather
than retries selected for a favorable result.

The old/new Meganeura screen below found no large steady-state regression.
Getting the reference qualified required addressing two pre-existing problems:

1. At `59d4cab`, 13 strict pairs passed before Whisper training capture failed
   because an autograd node retained the default compilation stream. `50d10f2`
   gives all CUDA conditions one preparation/run stream. A direct Whisper
   reproduction now passes; all three broad Python and nine harness tests pass.
2. Fresh qualification at `50d10f2` passed seven strict pairs, then diffusion's
   default/graph condition failed the full-gradient replay gate: 3/2304 values
   in `conv_in.weight`, maximum absolute difference 1.933e-6. A separate
   eight-call uncaptured-repeat diagnostic also exceeds that same gate, so
   this is not evidence of a capture-only error or a Meganeura regression.

The default diagnostic fails; cuDNN determinism alone also fails. Full PyTorch
determinism plus `CUBLAS_WORKSPACE_CONFIG=:4096:8` passes the diagnostic's eight
repeat checks and all capture phases, including 181 gradient tensors. This is
a possible **separate** reference condition, not a silently adopted baseline:
it can change algorithms and performance. We did not adopt this restricted
baseline or automatically exclude diffusion. Instead, the replay acceptance
policy was revised using the reference-repeat controls below; original failures
remain preserved, not retried unchanged until lucky.
The first tensor-scale policy at `f2bb048` qualified all 15 strict conditions
and seven accelerated conditions before accelerated diffusion failed an
**uncaptured** gradient check. Its 16-call diagnostic shows ordinary versus
captured variability, with worst full-gradient relative L2 differences of
0.224% and 0.305%. Full deterministic mode has zero measured difference in both sets of
16 calls. Local small-gradient tensors can vary more than the full vector;
merely applying one larger percentage to every tensor is inappropriate.
The sample-fitted policy then passed one fresh full qualification, but the
September 11 failure above demonstrates that one pass did not make it stable.

Diagnostic source and concise results are preserved on
[`experiment/p3hpc-replay-stability-2026-09-10`](https://github.com/kvark/inferena/tree/experiment/p3hpc-replay-stability-2026-09-10)
(`231bd1a`). It is an evidence branch, **not a collection-ready tag**. The three
failed campaign manifests remain outside Git at
`../inferena-results/zork-20260910T040555218867Z-59d4cab0/campaign.json`,
`../inferena-results/zork-20260910T042613175806Z-50d10f28/campaign.json` and
`../inferena-results/zork-20260910T051440799518Z-f2bb048a/campaign.json`.
Their qualification samples are not publication timings. No Nsight or RAM
investigation was resumed, and no driver/sysctl setting was changed.

### Replay qualification policy

PyTorch explicitly permits nondeterministic CUDA backward implementations;
NVIDIA documents atomic rounding variability in some cuDNN backward algorithms.
The uncaptured/default versus full-deterministic controls above agree with
this documented behavior. They do not identify a particular offending kernel.
[PyTorch reproducibility](https://docs.pytorch.org/docs/main/notes/randomness.html),
[cuDNN determinism](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/misc.html#reproducibility-determinism).

`fixed-full-gradient-v3` retains elementwise output/loss comparison at
`rtol=1e-4, atol=1e-6`. For **each** participating parameter gradient, let
`d = actual - reference`. Strict-f32 retains these per-tensor bounds:

```
max(abs(d)) <= 1e-6 + 1e-4 * max(abs(reference))
RMS(d)      <= 1e-6 + 1e-4 * RMS(reference)
```

Shapes, dtypes, finiteness and the complete participating-gradient set must
match. Strict mode also applies the same bounds to the complete gradient.

Accelerated-f32 allows reduced input precision, including TF32's ten input
mantissa bits. Small preceding roundoff can cross later quantization boundaries;
the much larger observed variation is consistent with this arithmetic, but
the controls do not localize the responsible kernel.
[PyTorch TF32 accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html#tensorfloat-32-tf32-on-nvidia-ampere-and-later-devices).
Accelerated training uses one fixed reference, eight ordinary repeats and two
captured replays. Every comparison must satisfy both complete-gradient bounds:

```
max(abs(d)) <= 1e-6 + 0.01 * max(abs(reference))
RMS(d)      <= 1e-6 + 0.01 * RMS(reference)
```

The maximum spans every element and catches sparse corruption; RMS is weighted
by element count across every participating parameter and catches diffuse
drift. Per-tensor metrics remain recorded, but observed repeats do not enlarge
or otherwise fit the pass threshold. This avoids both near-zero local relative
errors and a finite-sample maximum masquerading as a statistical guarantee.
The declared 1% is an experiment policy limit, not a PyTorch guarantee or proof
of correctness. The replicated cross-engine policy above is a separate final
campaign gate.

Errors, reference scales and full-gradient checks are saved in execution
metadata. Readback/CPU validation is outside ordinary timings and charged to
`validation_s`. Full deterministic algorithms remain off; effective settings
are recorded. The prospective campaign is not a retry of unchanged policy,
and all earlier failed campaigns remain incomplete.
Diagnostic source is on
[`experiment/p3hpc-precision-stability-2026-09-10`](https://github.com/kvark/inferena/tree/experiment/p3hpc-precision-stability-2026-09-10)
at `0734dd7`; its ungated gradient measurements are not qualification records.

## What was missing

The `paper-arxiv-1` runner's normal `bench_v2` path uses default
`torch.compile(model)` and never calls the explicit CUDA Graph helpers.
Those helpers exist only in the legacy path, and there only when compilation
is unavailable. A compile request is not evidence of CUDA Graph replay.
The unused legacy benchmark and its duplicate capture helpers have now been
removed from this branch. Reproduce that historical path at `paper-arxiv-1`.

The September 8 profile handoff also exposed a Meganeura API migration:
convenience builders now use pure defaults, so the runner must opt into
`SessionConfig::from_env` for diagnostics, then apply the protocol's typed
precision options explicitly. Strict uses `CoopPolicy::Disabled` and scalar
attention; accelerated uses `Auto` and eligible cooperative forward attention,
with full-precision derivative regions protected. `AllowF16` is inappropriate
here because it permits raw f16 derivative operands. Strict also disables
native-f32 cooperative tiles where available: it is a declared scalar control,
not the fastest possible full-f32 configuration. This restriction must remain
visible in the new cohort; do not assume equivalence to old strict timings.
Earlier pilot records do not qualify this corrected Meganeura configuration.

This branch adds a generic whole-phase capture wrapper, not model-specific
kernels. Each inference, minimal-shape and forward/loss/backward phase gets its
own `torch.cuda.CUDAGraph`. The replay callable retains static output storage;
inputs and parameters remain resident at stable addresses. Training allocates
gradients during capture, and replay overwrites those buffers. No optimizer
is included, just as in the frozen workload contract.

Before any phase is timed, compare **all output and participating gradient
elements** with the same uncaptured implementation over two consecutive
replays under the [declared noise policy](#replay-qualification-policy), with
finite values required. This validates the
capture transformation, not PyTorch-versus-Meganeura accuracy; the existing
cross-engine sampled-output/loss/gradient-norm gates remain unchanged.
The single broad regression also mutates inputs and parameters in place and
checks repeated forward/backward results against an independent eager model.

## Controls and reporting

`INFERENA_TORCH_MODE` selects `eager` or a PyTorch compiler mode (`default`,
`reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`).
`INFERENA_CUDA_GRAPHS=0|1` controls explicit whole-phase capture, defaulting to
1 on NVIDIA CUDA and 0 elsewhere. Explicit capture is not yet qualified on
ROCm. Inductor's own `triton.cudagraphs` option is disabled in every condition:
the explicit switch owns replay, including the no-graph control. Other
options of the requested compiler mode remain active. The exact resolved
options, compile status and per-phase capture/validation reports are in
`execution`, not inferred from the mode's name. Requested compilation or
capture failures abort the runner; no eager timings replace them.

All CUDA conditions use one dedicated stream for model preparation,
compilation, warmup, capture and execution, including the no-graph control.
This avoids retained autograd nodes referring to a different/default warmup
stream. `execution.stream_policy` records it. The September 10 qualification
at `59d4cab` passed 13 strict pairs, then exposed this pre-existing capture
failure on Whisper training; the failed campaign is retained, not relabelled.
The corrected source needs fresh qualification. Capture checks still compare
every participating output/gradient element over consecutive replays, with
the original tolerances. Full failed-run tracebacks now survive in records and
logs; capture errors cannot be classified as unsupported models by substring.

Timings remain synchronized **host wall time** around one full call/replay,
with resident inputs, no readback and no optimizer update. Compilation,
capture and qualification are reported separately. Each process gets an empty
private Inductor/Triton cache; shared developer caches are not deleted. Graph
pool residency remains included in allocator/device memory accounting. No
phase memory value is an isolated incremental allocation measurement.

The PyTorch record uses `inferena-cuda-graphs-v2`; the harness requires that
name on this experiment branch. Meganeura retains the v1 matched workload and
validation contract. Do not mix these new records into the submitted matrix.

## Collect a new cohort

Run `bash scripts/setup.sh cu130` (NVIDIA), `xpu` (Intel), `rocm7.2`, `cpu`, or
`mps`. With [uv](https://docs.astral.sh/uv/pip/environments/) installed, this
downloads managed Python **3.13.13** and installs `requirements-p3hpc.txt` in a
new `.venv-p3hpc`; pass a second argument for a different new directory. No
existing venv is overwritten. Use `.venv-p3hpc/bin/python` below; on Windows
use `scripts/setup.ps1` and `.venv-p3hpc/Scripts/python.exe` from PowerShell,
or the shell setup from Git Bash.
Setup also probes the requested backend with a tiny forward/backward workload
in every selected reference condition, including CUDA Graph replay. Missing drivers,
compiler support or failed numerical checks stop setup; it never reports an
eager fallback as successful compilation. Rerun the check in an existing venv
with `python scripts/check_environment.py --backend cuda` (or `xpu`, etc.).
This is an installation check, not model qualification or a timing result.

The v6 campaign collector requires that Python version, PyTorch 2.13.0 and reported source commit
`cf30153c4c131c8164ee7798e5022d810682e2cb` on **every** platform. It checks
both before collection and in every paired result; unknown/different commits
fail closed. The exact installed platform wheel suffix is detected and recorded;
optional `--torch-version` additionally asserts a particular build.
The campaign and PyTorch records retain the commit and `torch.__config__.show()`.
This fixes the earlier release-only check; old v1 campaigns did not enforce a
common source commit; v2 did not pin Python. Neither may be relabelled retroactively.

This is a common **reported upstream source**, not identical binaries or
vendor libraries, and not an attestation of an unmodified build. CUDA, ROCm,
Metal and CPU builds necessarily differ. A vendor fork requiring another
commit/release needs its own labelled source ref and availability cohort,
not an override inside this controlled cohort. Do not upgrade during collection.

The collector downloads missing selected SmolLM2 checkpoints before running;
`--offline` instead requires them to exist already. To prepare another machine
ahead of time, use `python scripts/prepare_models.py SmolLM2-135M` (or select the
larger sizes). All are **base**
checkpoints, pinned in `models/smollm2-revisions.json`. This writes ignored
weights/configs and a source/hash receipt. A legacy directory without a receipt
is adopted only when both files match the tracked hashes; mismatched or partial
inputs are never replaced implicitly. Both engines read those files, including
the actual 1.7B RoPE configuration. The collector verifies receipts and hashes inputs and
Cargo.lock before/after the offline campaign. The other four workloads use
the unchanged deterministic initialization in source.

Use an idle device with no concurrent builds, profiles or experiments, a clean
source revision, and a **new directory outside the checkout**:

```sh
# A correctness check, not a benchmark.
.venv-p3hpc/bin/python -m unittest discover -s frameworks/pytorch -p test_execution.py -v

# Optional short paired qualification. No publication performance samples.
.venv-p3hpc/bin/python scripts/p3hpc.py --qualify-only \
  --models ResNet-50 --precisions strict

# All five models, both precision classes; validate and measure every pair.
.venv-p3hpc/bin/python scripts/p3hpc.py
```

After environment activation the last command is `python scripts/p3hpc.py`, with
no required arguments. Defaults are the five common models, both precision
classes, three measurement replicates, and a new
`../inferena-results/<host>-<UTC>-<source>/` directory outside Git. Backend and
GPU are detected from the installed reference build. The native device is
enumerated and selected by device ID; software adapters, ambiguous matches and
cross-engine name mismatches fail the preflight. `--backend`, `--gpu`, and
`--results-dir` remain available as explicit assertions/overrides. CPU requires
an explicit `--backend cpu`; it is never an automatic fallback.

Collection is the default (`--collect` remains accepted). Every measurement
process compiles from an empty PyTorch cache, runs the full correctness and
CUDA-replay gates, then retains 5 warmups and 20 timing samples. This gives 90
validated pairs rather than repeating the same work in 30 separate preflight
processes. Optional `--qualify-only` remains a short hardware check, not a
prerequisite that certifies later processes. Compiler configurations rotate
across replicates and engine order alternates. After all three measurement
processes, the replicated gradient report must also pass. Each
configuration gets its own Meganeura control, not an old or fastest control
reused across unrelated runs. Rust builds finish before the
first pair and later wrapper checks use the locked dependency resolution.
Build parallelism defaults to one job to limit preparation RAM. Missing weights
are prepared once; existing receipts and file hashes must still match exactly.
Copy the complete printed result directory, including logs and `campaign.json`.
The manifest must finish with `status: complete`; an interrupted campaign is
not a complete data point. Use the same clean source revision on every machine.

| Declared reference backend | Collected configurations |
|---|---|
| CUDA | default/no-graph, default/whole-phase-graph, max-autotune/whole-phase-graph |
| ROCm | default/no-graph, max-autotune/no-graph; explicit capture still unqualified |
| XPU | default/no-graph, max-autotune/no-graph; no claim of whole-phase replay |
| MPS | declared eager reference |
| CPU | declared eager availability reference, separate from GPU comparisons |

Every pair must retain both engines, pass the requested forward/backward gates,
and match the declared revisions, torch build version, backend, GPU and execution
mode. A successful harness exit alone is insufficient. Invalid or missing
records stop the campaign; logs and `campaign.json` mark it incomplete. Do not
retry until favorable: diagnose, record the reason, and use a new source ref
and output directory when the protocol changes. Unsupported/oracle-disputed
pairs need an explicit scientific disposition, not an automatic exclusion.

`campaign.json` records the complete source SHA, pinned Meganeura dependency,
Python package versions, input hashes, device-selection overrides and run order.
It also records the common PyTorch source, platform-specific build settings,
selected reference-condition coverage and relevant runtime overrides.
The per-engine records retain driver/device, preparation, memory, execution and
validation details. Do not run two collectors on the same device concurrently.

### Windows NVIDIA

Use **native Windows x64**, Git for Windows / Git Bash (not WSL), uv, Rust's
MSVC toolchain, Visual Studio C++ Build Tools with a Windows SDK, and a current
NVIDIA driver supporting the CUDA 13.0 wheel. Rust builds require the MSVC
linker even though Triton's wheel bundles its own minimal CUDA/C toolchain.
Run from PowerShell or Git Bash; Python is downloaded automatically. Git Bash is located
from Git's installation and propagated to the Rust harness; `INFERENA_BASH`
can name its `bash.exe` explicitly for a nonstandard installation. Paths with
spaces are kept as individual arguments; scripts use LF and Python UTF-8 I/O.

```powershell
.\scripts\setup.ps1 cu130
.\.venv-p3hpc\Scripts\python.exe scripts\p3hpc.py
# Optional short preflight before the full campaign:
# .\.venv-p3hpc\Scripts\python.exe scripts\p3hpc.py --qualify-only --models ResNet-50 --precisions strict
```

CUDA setup selects `requirements-p3hpc-cu130-windows.txt`, adding exactly
`triton-windows==3.7.1.post27`. Upstream PyTorch only depends on `triton==3.7.1`
on Linux; importing torch on Windows does **not** establish that Inductor can
compile. This [Windows Triton port](https://github.com/triton-lang/triton-windows)
supports Ampere/RTX 30xx and the 3.7 compiler family required by PyTorch 2.13;
the [pinned wheel](https://pypi.org/project/triton-windows/3.7.1.post27/) supports
Python 3.13. It is a platform-specific compiler distribution, **not identical
Linux/Windows compiler binaries**. The collector records installed package
versions and the common PyTorch source independently. Do not replace it with
the newest Triton minor version. XPU setup does not install this CUDA port.

For an environment made by the earlier setup, install the corrected file with
`uv pip install --python .venv-p3hpc/Scripts/python.exe --torch-backend cu130 -r requirements-p3hpc-cu130-windows.txt`,
then run `.venv-p3hpc/Scripts/python.exe scripts/check_environment.py --backend cuda`.
Successful Linux probes and Windows wheel resolution are not Windows hardware
qualification; the first completed paired campaign must establish that.
Record the exact 3050 model, VRAM, driver, Windows version and laptop power mode
where applicable. Keep 1.7B separate until memory placement/capacity is checked;
do not silently shrink precision, batch size or sequence length to fit it.

RTX 3050 / Windows 11 completed all 120 pairs at `5103dedf`, but one PyTorch
Stable Diffusion compile interval is 17,249 seconds and spans the archive's
4.8-hour activity gap. The records cannot distinguish host suspension from a
compiler stall, so that interval is not compile-time evidence. Even after
removing it, PyTorch compilation totals 3.83 hours across the old campaign's
120 isolated processes. Campaign v6 removes the redundant 30-process preflight;
the remaining 90 cold compilations are still expected to dominate wall time.
A later campaign at `f4255c4b`, with the same pinned packages, inputs, driver and device,
passed qualification and 61 measurement pairs before PyTorch failed while
capturing strict SmolLM2 max-autotune training. The exact GEMM had succeeded in
three synchronized ordinary calls immediately before capture; cuBLAS then
returned `CUBLAS_STATUS_EXECUTION_FAILED` only inside the second CUDA Graph.
Six other processes across the two campaigns captured this condition
successfully. This single event makes the later campaign incomplete, but is not
yet a repeatable platform exclusion. The XPU embedding probe added between
those revisions had also been running unnecessarily on CUDA; later revisions
restrict it to XPU so the next Windows campaign does not carry that unrelated
state perturbation.

Historical preparation source: `experiment/p3hpc-portability-2026-09-09`
(`819b7d2`). On Linux/RTX 5070 it passes all three strict ResNet-50 paired
qualification cases (forward + latency + backward), the 3 broad Python checks
and 9 harness tests, and the CPU/CUDA setup probes. The Windows requirements
resolve for Python 3.13 x64; **Windows and B570 hardware qualification remain
pending at that ref**. This is not the updated collection pin above; old
measurements keep their original refs. Untimed qualification evidence is outside Git at
`/mnt/data/inferena portability.8Hc6y5/resnet qualification`.

### Intel, including mobile GPUs

A discrete GPU is **not required**. For the pinned PyTorch 2.13 build, supported
mobile targets include Core Ultra Meteor Lake-H and Series 2 Lunar Lake /
Arrow Lake-H; Series 3 Panther Lake has additional OS-version requirements.
Arc A/B discrete GPUs are also useful. Older Intel UHD/Xe branding alone does
not establish support. Follow Intel's [2.13 OS/driver matrix](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-13.html);
Windows compilation additionally needs the Level Zero SDK. Install a Vulkan
driver for Meganeura as well as the XPU prerequisites.

```sh
bash scripts/setup.sh xpu
.venv-p3hpc/bin/python scripts/p3hpc.py --qualify-only --models ResNet-50 --precisions strict
.venv-p3hpc/bin/python scripts/p3hpc.py
```

An explicit XPU request performs a numerical matmul/backward probe and fails
without CPU fallback; requested compilation failures likewise stop collection.
On Ubuntu, the tested Arc stack additionally needs `libze-intel-gpu1`, `libze1`,
`intel-opencl-icd`, and `intel-ocloc`. The benchmark does not use OpenCL:
Ubuntu's OpenCL package is the dependency vehicle for IGC, which the Level Zero
runtime also needs, while `intel-ocloc` supplies Triton's offline compiler.
The harness synchronizes XPU, reports its device/allocator metadata, and selects
XPU activity for diagnostic PyTorch profiles. Hardware qualification remains
required: successful wheel resolution or a mocked test is not an Intel result.
Arc B570 qualification found a separate correctness defect in the pinned XPU
build: native dense `nn.Embedding` backward populated only 19,968 of 73,728
expected elements for a unique-index `[128, 576]` probe. CPU produced all
73,728, and XPU's standard dense `index_add` produced all 73,728. The runner
therefore performs this small shape-derived probe before compiling a causal LM.
If native backward fails, it substitutes an equivalent custom autograd backward
built from `index_add`, qualifies that path, and records both outcomes plus the
selection under `execution.embedding_backward`. This is not a relaxed gate;
the corrected full SmolLM2 gradient agrees with CPU and Meganeura. Training
results that select the workaround must retain that metadata and be described
as a qualified PyTorch workaround, not an unmodified native XPU result.
On hybrid machines the collector selects the native match for the reference
GPU, including trademark spelling differences. If selection is ambiguous,
restrict visibility with `ONEAPI_DEVICE_SELECTOR` / `ZE_AFFINITY_MASK` and
Vulkan loader controls (`VK_ICD_FILENAMES` or `MESA_VK_DEVICE_SELECT`);
both reported names are checked. Record laptop power
mode, AC power and shared-memory capacity; do not treat memory budget as VRAM.
For the planned B570 + RTX 5070 machine, use a **separate** `xpu` venv (pass a
new path to setup). That environment's reference probe selects XPU and the
collector selects the matching native adapter; `--gpu 'Arc B570'` can additionally
assert the model. An Intel-only Vulkan ICD is needed only if ordinary enumeration
cannot identify one available matching device. Installing the card alone does
not qualify its driver or compiler. Start with ResNet-50 strict qualification,
then add accelerated and larger-model cases after those gates pass.

ROCm's official wheel index and AMD's APU-specific requirements may offer
different source builds or supported devices. Setup does not promise every
APU can execute the common-source cohort: an incompatible vendor build needs
a separately labelled availability experiment, not a relaxed source check.

### Radeon 780M availability case

The Radeon 780M report exposes three independent portability failures in the
pinned ROCm/PyTorch stack:

1. The ROCm wheel has no gfx1103 rocBLAS/Tensile library, so the process must
   impersonate the nearby gfx1102 target with `HSA_OVERRIDE_GFX_VERSION=11.0.2`.
2. With that override, asynchronous SDMA copies reproducibly fail during the
   real training backward pass, requiring `HSA_ENABLE_SDMA=0`.
3. PyTorch max-autotune tries a generated Triton matmul configuration that
   faults the HIP context. Synchronous launch diagnostics identify the failure
   more cleanly but do not make that condition runnable.

Keep both runtime workarounds active for setup and collection, and explicitly
omit the broken condition:

```sh
export HSA_OVERRIDE_GFX_VERSION=11.0.2
export HSA_ENABLE_SDMA=0
bash scripts/setup.sh rocm7.2 --no-max-autotune
.venv-p3hpc/bin/python scripts/p3hpc.py --no-max-autotune
```

For an already-created environment, rerun the minimal probe with
`python scripts/check_environment.py --backend rocm --no-max-autotune` before
the collector. The v6 manifest retains both HSA values, `args.max_autotune=false`,
the omitted condition and `reference_conditions.coverage=availability-subset`.
A `complete` status therefore means the declared subset completed; it is not a
full ROCm condition matrix and must remain a labelled availability result.

This is a concrete portability test for Meganeura's hardware-driven design:
its side uses the Vulkan device and capability-derived implementations rather
than ROCm architecture impersonation, Tensile target tables or Triton kernels.
The architectural independence is real, but the paired qualification remains
the evidence gate; do not claim success on the 780M until that record completes.
Full reproducer details and failure logs are tracked in
[issue #61](https://github.com/kvark/inferena/issues/61).

### Matched SmolLM2 scaling

```sh
.venv-p3hpc/bin/python scripts/p3hpc.py --models SmolLM2-135M SmolLM2-360M SmolLM2-1.7B \
  --inference-only --precisions strict
```

All sizes use batch 1, 128-token prefill and stateless one-token forward, with
f32 persistent weights. `--inference-only` creates no training graph/gradients
and reports training as absent, not zero or validated. It currently applies
only to the paired SmolLM2 runners. The same forward and CUDA replay gates
remain in force. Meganeura drops each shape's session before creating the
next, avoiding duplicate resident weights. Omit the flag for a **separate**
F+L+B cohort on sizes that fit; do not silently quantize/offload/shrink workloads.
1.7B f32 weights plus gradients alone exceed this machine's 12 GB device.
Depth, width and grouped-query attention differ across the family, so this is
an observed family scaling curve, not a parameter-count-only law.

Each native session owns its GPU context, which is destroyed after use; its
creation is included in preparation time. This avoids the library's immortal
default context. On this machine that default caused Nsight 2025.5.2 and
2026.4.1 to crash during exit and omit Vulkan GPU records. Owned contexts
complete and flush with **2026.4.1**; 2025.5.2 still fails during context
recreation, so use the newer tool for this host. The lifetime rule applies to both profiled and
ordinary runs, not just to a diagnostic workaround.

Numerical qualification does **not** certify all-VRAM residency. On discrete
GPUs, check actual memory placement when the large-model timing or memory
counters change abruptly. Blade's `Shared` allocation prefers device-local
memory but can fall back to a host heap. The plan's `device_local_bytes` counts
the unmappable allocation class, not all actual device-local storage; shared
buffers may also be device-local. Total device budget alone does not establish
their placement. The 1.7B result below is a concrete example. Unified-memory
devices need their own memory interpretation, not a discrete-VRAM test.

## Diagnose host and device costs separately

After ordinary collection, run a **separate** representative profile with the
same revisions and configuration:

```sh
INFERENA_TORCH_MODE=max-autotune INFERENA_CUDA_GRAPHS=1 \
  ./run.sh -f pytorch,meganeura -m ResNet-50 --strict --profile \
  --results-dir /mnt/data/p3hpc-resnet-profile
```

`--profile` now retains PyTorch host/device Chrome traces with synchronized
`inferena.phase` regions, alongside Meganeura's per-dispatch GPU sidecars.
Inspect `cudaGraphLaunch`, kernels, launch gaps and waits in the timeline;
compare instrumented wall durations with ordinary samples to disclose overhead.
Do not call `wall minus sum(kernel medians)` CPU time **or barrier cost**: overlap, gaps and
instrumentation defeat that decomposition. CUDA/ROCm profiler activity needs
the device tracing runtime; MPS currently yields CPU traces only and requires
Metal tooling for its GPU timeline. Missing GPU events are not zero GPU cost.
For compiler diagnosis, use `TORCH_LOGS=graph_breaks,recompiles,perf_hints` in a
separate run. The publication collector rejects profiling/debug overrides.

Use `scripts/profile_report.py <meganeura-sidecar.json>` to rank families and
dispatches, or pass two sidecars for a revision comparison. The report prints
the timestamp contract: Meganeura's Vulkan pass intervals include inter-pass
barriers, and one-pass-per-dispatch capture changes the normal grouped schedule.
Its Perfetto GPU slices are durations laid out at a host submission timestamp,
**not calibrated device start/end times**. Do not infer queue gaps or overlap
from that synthetic alignment. The Inferena wrapper currently captures GPU
sidecars, not Meganeura CPU spans; a `profiler`-enabled caller can collect those
separately. Vendor timelines are needed for correlated host/device attribution.

### NVIDIA paper-analysis captures

On Linux, bound the **entire** profiler/runner tree. Choose a host-memory and
time budget that fits the workload and leaves headroom; these are safety
limits, not hardware-specific performance gates:

```sh
.venv-p3hpc/bin/python scripts/limited.py --memory-mib 6144 --seconds 600 -- \
  .venv-p3hpc/bin/python scripts/nsys.py --gpu 'RTX 5070' --torch-version 2.13.0+cu130 \
  --model ResNet-50 --precision strict --mode max-autotune \
  --nsys /path/to/nsys --results-dir ../resnet-nsight
```

`limited.py` requires a systemd user manager and cgroup v2. It checks available
RAM, reserves at least 1 GiB / 10% of host RAM (whichever is larger), then
verifies the actual memory cap, zero swap allowance and group OOM termination
before starting the command. Environment and working directory are inherited;
core dumps are disabled. The printed scope name lets you stop the whole job.
There is no unbounded fallback. It does **not** bound VRAM or guarantee that
all driver-pinned memory is charged to the cgroup. Check profiler buffer sizes
and device/host memory separately, and never attach to an unbounded detached
target. A killed/timed-out run is incomplete, not a smaller valid sample.
This wrapper is Linux-only; `nsys.py` itself remains usable on Windows, where
equivalent process-tree resource containment needs separate qualification.

Use Nsight **2026.4.1** for the qualified Linux setup here. Alternatively put
`nsys` on PATH or set `NSYS` (including its Windows `.exe`
path). `--no-graphs` captures the launch-overhead control; `--inference-only`
supports the larger SmolLM2 workloads. The wrapper builds before capture,
retains both engines and applies the same numerical/source/device gates.
It records the exact command, tool version and input hashes, plus compiler
diagnostics, a `.nsys-rep` and queryable SQLite export **per engine**. Only the
engine process is profiled, with CUDA tracing for PyTorch and Vulkan tracing
for Meganeura; builds and wrapper/device-discovery processes stay outside.
The Vulkan capture omits OSRT interposition: adding it reproduced a crash
during training-session creation on this host; Vulkan API calls and NVTX still
provide the host timeline. CUDA uses explicit software tracing (`cuda-sw`)
without OSRT, serial in-process compilation, and waits only for the primary
process; the default collector configuration hung after PyTorch exited here.
Compilation under these diagnostic settings is not a preparation benchmark.
The compile-thread override is recorded and ordinary campaigns do not set it.
These are explicit profiler
limitations, not ignored engine failures.
Missing GPU events
fail capture qualification. All artifacts stay outside Git.

For low-memory, offline analysis, run
`python scripts/nsys_report.py <capture-dir>/pytorch.sqlite` (or
`meganeura.sqlite`). It restricts analysis to complete measured samples,
reports host spans and ranks CUDA kernels with launch counts/register/shared
memory metadata. Vulkan rows remain grouped submissions. This is a diagnostic
summary, not a replacement for `capture.json` qualification. See
[current findings and their limits](ANALYSIS.md).
Add `--launches` for CUDA graph-node rankings with grid/block geometry; without
it, identically named kernels remain aggregated. Launch-level attribution
requires graph-node IDs, which identify launches within that trace, not stable
operators across runs. Reported local-memory bytes are not measured spill traffic.

Open the reports with the same or a newer Nsight GUI; an older installed GUI
may not read them. No driver or system-wide profiler upgrade is required by
the wrapper: `--nsys` can point to an isolated CLI installation.

[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/) captures
CUDA graph **nodes**, Vulkan individual GPU workloads and host NVTX regions
for each phase's warmup/measurement. Meganeura's measured samples additionally
mark `step` and `wait`; the CPU/API/GPU timeline provides actual correlation.
The Vulkan runner keeps its production grouped schedule: this is **not**
`--profile` / one-pass-per-dispatch mode. Records are labelled diagnostic and
cannot pass ordinary campaign checks. Tool instrumentation changes timings;
compare ordinary controls separately, never publish these as speed results.
On the qualified Linux captures, Systems resolves Meganeura's GPU work per
grouped submission, not per shader dispatch. Use the separate dispatch sidecars
for family rankings and Graphics for finer Vulkan analysis; do not claim native
per-kernel Systems timings from these reports.

Systems locates launch/queue gaps, synchronization and active workloads; it
does not make every gap a barrier cost. For Meganeura's Vulkan barrier/stall
analysis, open the same release runner/configuration in **Nsight Graphics GPU
Trace** (compute workload, no swapchain): working directory `frameworks/meganeura`,
executable `<checkout>/target/release/inferena-meganeura` (absolute path), argument the model name. Set
`INFERENA_STRICT`, `INFERENA_INFERENCE_ONLY`, warmup/sample counts and device
selection exactly as in `capture.json`; do not enable `MEGANEURA_GPU_TIMING`.
Native source/name correlation is being prepared in
[Meganeura PR #165](https://github.com/kvark/meganeura/pull/165), with a separate
`MEGANEURA_GPU_CAPTURE` switch and descriptive pipeline keys. This is not yet in
the pinned collection revision: setting that variable on the old runner does
not enable capture support. Record and qualify any diagnostic revision separately.
For this direct executable launch, set `INFERENA_NSYS=1` to enable host NVTX
markers. Choose **Submit Count** or **Elapsed Time** as the start condition and
**Max Submits** or **None** as the limit; there are no present/frame boundaries.
Record clock-lock and capture settings. This Graphics workflow follows the
[headless application instructions](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-ui.html).
The initial traces below have explicit completeness limits.
Use its [barrier/occupancy timeline](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-ui.html)
and shader profiler to examine the expensive regions. **Nsight Compute**
is for the PyTorch CUDA kernels, not Meganeura's Vulkan shaders. Workgroup
barrier stalls are distinct from Vulkan resource barriers. A causal removable
barrier-cost number still needs the legal schedule A/B described below.

Counter access is now enabled (`RmProfilingAdminOnly: 0`), following the user's
September 9 reboot. Graphics **2026.3.1.0** collects counters with driver
595.71.05. No further driver reload, security-policy change or reboot is part
of this workflow. The earlier counter-denied attempts produced no metrics.
[NVIDIA's counter-access documentation](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

**Do not repeat the former 1.7B recipe or detached retry.** At 06:11 UTC the profiler's
4 GB sampling buffer plus the large runner exhausted host RAM and swap; the
kernel's global OOM killer killed the runner. There was no subsequent reboot.
The failed run has no complete runner JSON and is excluded. Later offline
viewing hit a 900 MiB cgroup limit and terminated only that analysis job.
New GPU captures are on hold while host-memory headroom remains low.

For the next bounded Graphics qualification, start with resident 135M and a
short window (about 100 ms), unaltered clocks and screenshots disabled. Set
sampling bandwidth **at launch**, not at a later attach. Inspect the actual
`GPU PMA Buffer Size` in the launch log: a 256 bandwidth request still allocated
2500 MB with a five-second window here. The default allocated 4000 MB.
[GPU Trace memory/sampling controls](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-overview.html).
Keep the profiler and target together under `limited.py`. Shorten the window
or lower sampling density before increasing any buffer budget.

Build the clean, recorded source first. On this Linux CLI, use
`QT_QPA_PLATFORM=offscreen` and omit `--platform`, which Qt misinterprets.
Keep `--time-every-action` off. With five warmups, start-after-submits 5 and
limit-to-submits 4 captured two complete prefill measurement markers on 135M;
verify the markers rather than assuming counts transfer between workloads.
Context destruction between phases can end collection before the token phase.
`--keep-going` requests more traces; it does not guarantee runner completion.
Retain complete runner stdout separately and verify hashes/numerics against an
ordinary control. Reject hardware-event overflow for per-dispatch attribution;
the first 135M trace passed output checks but reported this overflow.

Do not substitute frame capture/replay for a placement experiment:
`ngfx-capture` defaults to demoting host-visible video memory to system memory
(`--hvvm-demote`). Successful attachment, exported counters or a partial trace
alone do not establish a qualified, correctly placed run.

### Bounded gap-analysis plan

| Quantity | Instrument | Interpretation / remaining work |
|---|---|---|
| End-to-end latency | Ordinary paired process samples | The primary comparison; no profiling enabled |
| Host encode/submit/wait | CPU spans plus OS/vendor timeline | Elapsed wait is not busy CPU time; current Meganeura spans need finer encode/submit boundaries |
| Kernel/family cost | Meganeura dispatch sidecars; PyTorch device events | Different instrumentation contracts; compare ranks and variants, not an additive cross-engine decomposition |
| Queue gaps and overlap | Correlated vendor CPU/GPU timeline | Not recoverable from Meganeura's reconstructed Perfetto GPU track |
| Barrier overpayment | Validated, interleaved legal-schedule A/B | **Unmeasured on the current paired cohort**; counts and pass intervals alone do not establish removable cost |
| Memory and preparation | Resident/allocator counters; build/capture timers | Keep their scopes and instrumentation overhead visible |

Start with ResNet-50 training, Whisper training and SmolLM2 minimal forward:
they cover convolution derivatives, attention and short-dispatch latency.
Localize the largest families, then inspect only their register/spill,
occupancy and memory counters with the applicable vendor tool. Retain generated
shaders and traces outside Git, identified by source/configuration in a short
results summary. Use the paired Nsight entry point above; the older Meganeura
shell helpers still contain workstation-specific Windows paths.

For barriers, first capture the **production** grouped schedule with Blade 0.9.
Normal steps already use inline compute-to-compute barriers; the old broad
inter-pass barrier experiment is not a current application overhead estimate.
Any challenger must preserve RAW/WAR/WAW ordering, physical aliases, external
buffer visibility and full output/gradient correctness. Compare the same kernels,
geometry, data and precision with interleaved unprofiled trials; report the
paired latency delta and uncertainty, including a null result. Never disable
all barriers on a dependent graph or multiply a no-op barrier cost by its count.

For scaling, use the matched SmolLM2 family above before introducing another
architecture. Hold batch/context/precision fixed, then
vary batch and context separately. Separate prefill, KV-cache decode and F+L+B;
the current minimal forward is **not** cached autoregressive decode. More weights
can amortize fixed costs, but deeper networks also add dependencies, and memory
bandwidth, cache and matrix efficiency can widen the gap. Two different model
sizes/architectures do not justify extrapolating a scaling law.

Gemma 4's large dense model is 31B, not 32B; it is not a current Meganeura builder
or Inferena workload. F32 weights alone are about 124 GB, and F32 weights plus
gradients about 248 GB before activations/workspace. Google's approximate BF16
loading budget is 69.9 GB, also before context-dependent KV storage. Reduced
weights, quantization, offload or a new training method require their own matched
protocol, not comparison with the existing f32 cells.
[Model sizes and memory planning](https://ai.google.dev/gemma/docs/core).

## Collection plan (not yet a new paper matrix)

Initial harness pilot: one strict ResNet-50 process each for default compiled
without and with whole-phase CUDA Graphs, in that order, 5 warmups and 20
samples. This is a functional/measurement pilot, not independent replication
or a PyTorch-versus-Meganeura speed claim. Keep failures and do not retry a
configuration merely to improve its timing. The max-autotune condition and
paired Meganeura/full-model/device campaign remain separate required work.

The collector implements the validation and process-rotation plan above;
it does not establish that all workloads/platforms have already passed it.
We request automatic max-autotune everywhere. When PyTorch rejects or times out
its own candidates, that is measured compiler availability evidence; retain its
diagnostics rather than overriding its hardware policy per model.
Keep compilation/capture/search costs and graph-pool memory alongside timings.

In the pinned PyTorch source, Inductor's `is_big_gpu` gates some NVIDIA GEMM
search at 68 SMs. RTX 5070 has 48 SMs and RTX 5080 has 84: the latter is a useful
automatic-search control despite sharing the Blackwell architecture. Clearing
that gate does not guarantee a different selected kernel or a timing gain.
[NVIDIA specifications](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

This is **PyTorch's upstream policy**, not a new Inferena or Meganeura device
threshold. We do not override it to privilege either GPU. Meganeura's actual
measured kernel search is `SessionConfig.tune` (off in the default cohort):
it probes bounded legal f32 matmul/convolution alternatives, not every dtype,
attention variant or graph representation. Baseline shape/occupancy heuristics
remain; supported timed challengers are not vetoed by a card-name/SM cutoff.
The runner reports `optimizer.measured_kernel_search`. Its former misleading
"auto-tune" preflight only queried capabilities and has been removed; builds
already query the selected GPU. A tuned-vs-default study must declare that
setting and include preparation costs rather than pretend all selection is
already measured.

The Naga Workgroup `ArrayStride` validation diagnostic is an acknowledged
upstream limitation for this revision, not a release blocker or a reason to
disable Vulkan validation. Preserve diagnostics; distinguish this known message
from new validation failures and from numerical qualification.

Recollect Meganeura at the declared revision in the same campaign. Do not
compare current PyTorch times against old Meganeura timings or select a winner
independently for every reported sample. Report configurations and process
replicates, not a fastest-run table. Revisit other devices explicitly; success
on NVIDIA does not establish ROCm, Metal or CPU behavior.

See [PyTorch CUDA Graph semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)
and [compiler modes](https://docs.pytorch.org/docs/stable/generated/torch.compile).

## September 8 pilot result

Measured source: `experiment/p3hpc-cuda-graphs-pilot-2026-09-08` (`d1bd3e6`).
RTX 5070, driver 595.71.05, PyTorch 2.13.0+cu130, Python 3.14; strict f32.
Both declared processes succeeded; no retries. CUDA Graph qualification passed
for all three phases, including all elements of 108 training-gradient tensors
on two consecutive replays. The final full inference hashes agree across the
two processes. The reported parameter-norm vectors differ by relative L2
`1.43e-8`; they are not claimed bit-identical or full cross-process gradients.

| ResNet-50 phase | Default compile, no graph | Default compile, whole-phase graph |
|---|---:|---:|
| Inference, batch 4 | 8.044 ms | 7.886 ms |
| Minimal forward, batch 1 | 4.877 ms | 4.721 ms |
| Forward + loss + backward | 16.239 ms | 16.006 ms |

These are medians of 20 synchronized calls in **one process per condition in
fixed order**, not a replicated speedup estimate. There is no paired Meganeura
result and nothing here replaces a paper cell. Compiler preparation was about
14 seconds per process; a short local Rust build overlapped control process
startup, so these are not controlled compile-cost measurements.

Additional graph preparation plus qualification totalled about 0.397 s across
the three phases. Here `capture_s` includes warmup, the reference snapshot and
capture; `validation_s` includes replay, readback and CPU comparisons. It is
not a measurement of GPU capture or CPU comparison alone. Final per-process
NVML residency was 888 MiB without graphs and 1,946 MiB with graphs. Both
allocator reservation and driver residency must accompany performance results;
the smaller live-allocation counter alone would obscure this cost.

Local records/logs are outside Git at
`/mnt/data/inferena-cuda-pilot.kK9H06/`. Reproduce the procedure from the tagged
source with the documented mode switches; binaries are not retained in Git.
The branch subsequently removed the unused legacy runner and made the replay
object explicitly own its callable/model as well as its graph/output storage.
Max-autotune and the full replicated campaign remain unmeasured.

## September 10 dependency regression check

The one-command collector is implemented at Inferena `59d4cab`, updating
Meganeura `43b606ff` → `e59bd32d` while retaining Blade 0.9 and the common
PyTorch source. The Windows PowerShell setup commit is preserved.

The source review covers the softplus negative-tail correction, added multimodal
primitives, and weighted epilogues. The common workloads keep f32 storage, but
the epilogue change also repairs small-tile f32 shader/grid consistency, so it
cannot be dismissed as quantization-only.

On the RTX 5070 / driver 595.71.05, a native dependency regression screen ran
all five common models in strict and accelerated modes: three paired fresh
processes per condition, 5 warmups and 20 samples per phase, no profiling or
measured tuning. All 30 old/new pairs have identical full-output hashes, losses,
total gradient norms and per-parameter gradient-norm records. This is not an
elementwise-gradient or convergence check.

Median paired change in ordinary wall time (new/old − 1; positive is slower):

| Model | Precision | Inference | Minimal shape | F+L+B |
|---|---|---:|---:|---:|
| SmolLM2-135M | strict | −0.1% | +2.6% | +0.0% |
| SmolVLA | strict | −1.2% | −1.2% | −0.5% |
| Diffusion U-Net | strict | −3.3% | −0.6% | −0.7% |
| ResNet-50 | strict | +0.0% | −0.1% | +0.0% |
| Whisper-tiny | strict | +0.0% | +0.0% | +0.0% |
| SmolLM2-135M | accelerated | −0.0% | +0.2% | +0.1% |
| SmolVLA | accelerated | +0.2% | −0.6% | +0.0% |
| Diffusion U-Net | accelerated | +2.0% | −1.2% | +1.2% |
| ResNet-50 | accelerated | +0.0% | −0.0% | −0.1% |
| Whisper-tiny | accelerated | +0.1% | −0.2% | −0.0% |

No large steady-state regression appears in this screen; this is not a
statistical equivalence test or a new PyTorch performance comparison. Order
alternates across model/precision pairs, not within each condition's three
repeats. Retain the noisy single-token SmolLM2 observations (one pair is +9.2%),
not only the median.

Preparation is separate: Whisper's medians rise 0.694→0.732 s strict and
1.223→1.312 s accelerated. Driver caches were left as found, so these are
startup-cost observations, not controlled cold-compile attribution or a reason
to undo the correctness fixes.

A predeclared reverse-order follow-up (three additional pairs per precision)
does **not** reproduce the Whisper preparation increase: 0.617→0.616 s strict,
1.174→1.171 s accelerated. All six pairs' output records match exactly.
Records are at `../inferena-results/whisper-uprev-reverse.OtFit4/`. Do not pool
the two blocks into a controlled cold-start claim; driver caches were not reset.

Rebuild the native binaries at Inferena `6fcdbc4` and `59d4cab` with
`cargo build --release --locked -j1 -p inferena-harness -p inferena-meganeura`.
Run each binary from the prepared checkout for the five common models, strict
then accelerated, repeated three times in that order. Set `INFERENA_STRICT`
to 1/0, `INFERENA_WARMUP_RUNS=5`, `INFERENA_MEASUREMENT_RUNS=20`,
`INFERENA_REQUIRE_LOCAL_WEIGHTS=1`, `HF_HUB_OFFLINE=1`,
`MEGANEURA_DEVICE_ID` to the enumerated adapter ID and `FRAMEWORK_REV` to
the corresponding pin. Start old/new, reverse order for each following pair.
This screen used a 6144 MiB/no-swap process-tree limit; GPU/driver allocations
are not all charged to it. Rust builds finished before measurement. Native
records and the local comparison script are outside Git at
`/mnt/data/inferena-uprev-2026-09-10.qj6ufb/`; binaries are reproducible from
the two revisions and do not belong in the source archive.

## Collection-handoff qualification

Source `experiment/p3hpc-methodology-2026-09-08` (`ed77335`) includes the
corrected typed Meganeura configuration. ResNet-50 passes all six paired
qualification conditions: both precision classes, each with default/no-graph,
default/graph and max-autotune/graph. Each phase retains one qualification call,
not a publication timing sample. The separate strict max-autotune profile also
passes both cross-engine gates and emits all three Meganeura GPU sidecars plus
PyTorch traces containing GPU kernels and `cudaGraphLaunch` events. The broad
compiled replay/profile regression and nine harness checks pass.

Records remain outside Git at `/mnt/data/inferena-methodology.fEQO0q/resnet`
and `/mnt/data/inferena-profile-handoff.bzLgOQ`. Earlier handoff attempts are
retained thereabouts as diagnostics, not pooled with these checks. No full
matrix, independent performance replication or other-platform qualification is
claimed by that handoff. The later source-pin and scaling checks below extend
it; other-platform qualification and the remaining workloads are still needed.

The source-identity follow-up is tagged `experiment/p3hpc-source-pin-2026-09-08`
(`47462c1`). The CPU identity check, broad CUDA replay/profile check and all
three strict ResNet-50 qualification pairs pass under campaign v2. Records are
outside Git at `/mnt/data/inferena-source-pin.GJxFR7/resnet`; these are not a
replicated timing study or evidence that other platform wheels share the pin.

## September 8 SmolLM2 scaling and vendor traces

Measured source: `experiment/p3hpc-scaling-2026-09-08` (`c7d9e79`). Check out
that tag to reproduce or join this cohort; later documentation/console fixes
do not relabel its records. RTX 5070, driver 595.71.05, Python 3.13.13,
PyTorch 2.13.0+cu130 at the common source pin, Meganeura `43b606ff`, Blade 0.9.
Both precision classes pass all nine paired forward-only qualification cases.
Strict additionally completes all 27 measurement pairs: three fresh processes
per model/configuration, five warmups and twenty retained calls per phase.
There were no numerical exclusions or timing retries. No larger-model training
or replicated accelerated timing result is claimed.

The table uses the declared **max-autotune + whole-phase CUDA Graph** reference,
not a fastest configuration selected per model. Times are medians of the three
process medians. Ratios are medians of paired process ratios; brackets give
their min–max range, **not** a confidence interval. M/P above one means Meganeura
is slower. This is the strict scalar-control configuration, with Meganeura's
measured kernel search off, not its practical accelerated configuration.

| SmolLM2 | Prefill P / M (ms) | Prefill M/P [range] | Stateless token P / M (ms) | Token M/P [range] |
|---|---:|---:|---:|---:|
| 135M | 6.064 / 12.723 | 2.100 [2.098–2.111] | 1.346 / 2.643 | 1.962 [1.960–1.966] |
| 360M | 11.817 / 21.823 | 1.850 [1.842–1.855] | 2.816 / 5.371 | 1.909 [1.870–1.931] |
| 1.7B, placement-limited | 28.716 / 945.587 | 32.945 [32.902–32.968] | 11.090 / 1940.140 | 174.937 [174.487–177.460] |

The smaller models show some prefill-gap narrowing, not a universal scaling
law. With default compilation plus graphs, their token ratios instead rise
from 1.575 to 1.693. The no-graph controls also show why capture matters:
135M PyTorch token time is 2.743 ms without graphs and 1.632 ms with graphs
under the same default compiler mode. All controls and process variation are
retained, including a faster 135M Meganeura prefill replicate; none was chosen
as a replacement for the paired controls.

**1.7B is not an all-VRAM scaling point on this setup.** Its planned buffers
occupy 9.414 GiB for prefill and 9.376 GiB for a token, but device-local heap
usage is about 4.5 GiB. A separate qualified Nsight capture confirms 4.930 GiB
and 4.906 GiB of bindings, respectively, on the non-device-local heap (property
flags 14: host-visible, coherent, cached). The token bindings include 24 each
of 128 MiB and 64 MiB matrix-sized buffers. NVIDIA's kernel log also reports
BAR1 mapping-allocation failures. `Shared` permits this fallback; passing the
total-budget preflight and numerical gates does not rule it out. Resolve or
explicitly account for placement before drawing an in-core scaling conclusion;
do not interpret this row as evidence for a barrier fraction or extrapolate it
to Gemma. No offload or reduced-weight option was requested.
Offline byte accounting also finds exactly **3 GiB of extra packed gate/up
weights** at 1.7B: the original named parameters remain allocated alongside their
derived copy. This explains the plan footprint, not the fallback threshold or
an unpacked speedup. See [representation and placement analysis](ANALYSIS.md#weight-representation-and-placement-first-for-17b).

Nsight Systems 2026.4.1 captures qualify for strict ResNet-50 F+L+B and 1.7B
forward-only, with GPU events inside every measured phase for both engines.
For the latter's token phase, the diagnostic host `step` averages about 2.5 ms
while grouped GPU work lasts about 1.95 s. This localizes the elapsed time to
device execution; it does not separate shader work, memory stalls and barriers.
The current removable barrier cost remains **unmeasured**.

Evidence is outside Git under `/mnt/data/inferena-native-analysis.nC5jZj/`:
`smollm-strict`, `smollm-accelerated-qualification`, `resnet-nsys-final` and
`smollm-1.7b-nsys`. Earlier incomplete profiler attempts are diagnostic failures,
not samples in these completed campaigns. The tagged source reproduces the
procedure; no weights, binaries, traces or raw measurement arrays are in Git.
Some tagged human-readable logs say comparison was skipped when Meganeura ran
first; the JSON validator always locates PyTorch independently of order, and
all declared gates passed. The subsequent console fix removes that misleading
message without changing the validator or measured engine code.
