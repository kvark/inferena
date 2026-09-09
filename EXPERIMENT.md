# P3HPC CUDA Graph comparison

Source branch: `experiment/p3hpc-cuda-graphs`, based on Inferena main.
The submitted source remains tagged `paper-arxiv-1`. Git records both bases;
do not copy binaries or experimental raw records into this branch or main.
The Meganeura dependency is pinned to merged main `43b606ff`; it is not a
floating sibling checkout.

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
replays (`rtol=1e-4`, `atol=1e-6`, finite values required). This validates the
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
use Git Bash and `.venv-p3hpc/Scripts/python.exe`.
Setup also probes the requested backend with a tiny forward/backward workload
in every reference condition, including CUDA Graph replay. Missing drivers,
compiler support or failed numerical checks stop setup; it never reports an
eager fallback as successful compilation. Rerun the check in an existing venv
with `python scripts/check_environment.py --backend cuda` (or `xpu`, etc.).
This is an installation check, not model qualification or a timing result.

The v3 campaign collector requires that Python version, PyTorch 2.13.0 and reported source commit
`cf30153c4c131c8164ee7798e5022d810682e2cb` on **every** platform. It checks
both before collection and in every paired result; unknown/different commits
fail closed. `--torch-version` still declares the exact platform wheel suffix.
The campaign and PyTorch records retain the commit and `torch.__config__.show()`.
This fixes the earlier release-only check; old v1 campaigns did not enforce a
common source commit; v2 did not pin Python. Neither may be relabelled retroactively.

This is a common **reported upstream source**, not identical binaries or
vendor libraries, and not an attestation of an unmodified build. CUDA, ROCm,
Metal and CPU builds necessarily differ. A vendor fork requiring another
commit/release needs its own labelled source ref and availability cohort,
not an override inside this controlled cohort. Do not upgrade during collection.

For SmolLM2, run `python scripts/prepare_models.py SmolLM2-135M SmolLM2-360M SmolLM2-1.7B`
with this environment (or select only the sizes you need). All are **base**
checkpoints, pinned in `models/smollm2-revisions.json`. This writes ignored
weights/configs and a source/hash receipt; it refuses to replace an existing
model directory. Both engines read those files, including the actual 1.7B
RoPE configuration. The collector verifies receipts and hashes inputs and
Cargo.lock before/after the offline campaign. The other four workloads use
the unchanged deterministic initialization in source.

Use an idle device with no concurrent builds, profiles or experiments, a clean
source revision, and a **new directory outside the checkout**:

```sh
# A correctness check, not a benchmark.
.venv-p3hpc/bin/python -m unittest discover -s frameworks/pytorch -p test_execution.py -v

# Short paired qualification. No publication performance samples.
.venv-p3hpc/bin/python scripts/p3hpc.py --backend cuda --gpu 'RTX 5070' \
  --torch-version 2.13.0+cu130 --models ResNet-50 --precisions strict \
  --results-dir /mnt/data/p3hpc-resnet-qualification

# All five models, both precision classes; qualify ALL pairs, then measure.
.venv-p3hpc/bin/python scripts/p3hpc.py --backend cuda --gpu 'RTX 5070' \
  --torch-version 2.13.0+cu130 --collect --replicates 3 \
  --results-dir /mnt/data/p3hpc-nvidia-campaign
```

The first stage retains one call per phase for each pair to exercise the full
runner and validity gates; these are qualification records, not publishable
timings. `--collect` starts the 5-warmup/20-sample campaign only after every
selected qualification pair passes. Each pair uses fresh PyTorch and Meganeura
processes; compiler configurations rotate across replicates and engine order
alternates. Each configuration gets its own Meganeura control, not an old or
fastest control reused across unrelated runs. Rust builds finish before the
first pair and later wrapper checks use the locked dependency resolution.

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
It also records the common PyTorch source and platform-specific build settings.
The per-engine records retain driver/device, preparation, memory, execution and
validation details. Do not run two collectors on the same device concurrently.

### Windows NVIDIA

Use **native Windows x64**, Git for Windows / Git Bash (not WSL), uv, Rust's
MSVC toolchain, Visual Studio C++ Build Tools with a Windows SDK, and a current
NVIDIA driver supporting the CUDA 13.0 wheel. Rust builds require the MSVC
linker even though Triton's wheel bundles its own minimal CUDA/C toolchain.
Run from Git Bash; Python is downloaded automatically. Git Bash is located
from Git's installation and propagated to the Rust harness; `INFERENA_BASH`
can name its `bash.exe` explicitly for a nonstandard installation. Paths with
spaces are kept as individual arguments; scripts use LF and Python UTF-8 I/O.

```sh
bash scripts/setup.sh cu130
.venv-p3hpc/Scripts/python.exe scripts/prepare_models.py SmolLM2-135M SmolLM2-360M
.venv-p3hpc/Scripts/python.exe scripts/p3hpc.py --backend cuda --gpu 'RTX 3050' \
  --torch-version 2.13.0+cu130 --models SmolLM2-135M SmolLM2-360M \
  --inference-only --precisions strict --results-dir ../rtx3050-windows-qualification
# After qualification, repeat with --collect and a new results directory.
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
.venv-p3hpc/bin/python scripts/p3hpc.py --backend xpu --gpu 'Intel' \
  --torch-version 2.13.0+xpu --allow-integrated-gpu \
  --models ResNet-50 --precisions strict --results-dir ../intel-qualification
```

Use the actual wheel suffix printed by setup and a specific GPU substring.
An explicit XPU request performs a numerical matmul/backward probe and fails
without CPU fallback; requested compilation failures likewise stop collection.
The harness synchronizes XPU, reports its device/allocator metadata, and selects
XPU activity for diagnostic PyTorch profiles. Hardware qualification remains
required: successful wheel resolution or a mocked test is not an Intel result.
On hybrid machines select the same device using `ONEAPI_DEVICE_SELECTOR` /
`ZE_AFFINITY_MASK` and Vulkan loader selection (`VK_ICD_FILENAMES` or
`MESA_VK_DEVICE_SELECT`); both reported names are checked. Record laptop power
mode, AC power and shared-memory capacity; do not treat memory budget as VRAM.
For the planned B570 + RTX 5070 machine, use a **separate** `xpu` venv (pass a
new path to setup), select Intel's Vulkan ICD explicitly, and use
`--gpu 'Arc B570'`. The backend flag selects PyTorch, not a Vulkan adapter;
the two reported GPU names must both match. Installing the card alone does
not qualify its driver or compiler. Start with ResNet-50 strict qualification,
then add accelerated and larger-model cases after those gates pass.

ROCm's official wheel index and AMD's APU-specific requirements may offer
different source builds or supported devices. Setup does not promise every
APU can execute the common-source cohort: an incompatible vendor build needs
a separately labelled availability experiment, not a relaxed source check.

### Matched SmolLM2 scaling

```sh
.venv-p3hpc/bin/python scripts/p3hpc.py --backend cuda --gpu 'RTX 5070' \
  --torch-version 2.13.0+cu130 --models SmolLM2-135M SmolLM2-360M SmolLM2-1.7B \
  --inference-only --precisions strict --collect --results-dir ../smollm-scaling
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

```sh
.venv-p3hpc/bin/python scripts/nsys.py --gpu 'RTX 5070' --torch-version 2.13.0+cu130 \
  --model ResNet-50 --precision strict --mode max-autotune \
  --nsys /path/to/nsys --results-dir ../resnet-nsight
```

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
For this direct executable launch, set `INFERENA_NSYS=1` to enable host NVTX
markers. Choose **Submit Count** or **Elapsed Time** as the start condition and
**Max Submits** or **None** as the limit; there are no present/frame boundaries.
Record clock-lock and capture settings. This Graphics workflow follows the
[headless application instructions](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-ui.html)
but has not yet been qualified on this host.
Use its [barrier/occupancy timeline](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-ui.html)
and shader profiler to examine the expensive regions. **Nsight Compute**
is for the PyTorch CUDA kernels, not Meganeura's Vulkan shaders. Workgroup
barrier stalls are distinct from Vulkan resource barriers. A causal removable
barrier-cost number still needs the legal schedule A/B described below.

The September 9 headless attempt with Graphics **2026.3.1.0** launches and
attaches, but fails before collection: `GPU Performance Counters unavailable`.
The installed 595.71.05 driver reports `RmProfilingAdminOnly: 1` in
`/proc/driver/nvidia/params`; interactive administrative authentication is
required on this host. No Graphics metrics or barrier attribution resulted.
See [NVIDIA's counter-access instructions](https://developer.nvidia.com/ERR_NVGPUCTRPERM):
an administrator can run the profiling application elevated, or deliberately
enable non-admin counters. The persistent driver option may require rebuilding
initramfs and rebooting. Do not unload the live display driver or change this
security policy from the collection script.

Once access is enabled, this reproduces the attempted **diagnostic** launch
from Inferena's root; set `NGFX` to the installed `host/.../ngfx` executable:

```sh
NGFX_OUT=$(mktemp -d ../inferena-ngfx.XXXXXX)
QT_QPA_PLATFORM=offscreen INFERENA_NSYS=1 INFERENA_STRICT=1 \
INFERENA_INFERENCE_ONLY=1 INFERENA_WARMUP_RUNS=5 INFERENA_MEASUREMENT_RUNS=3 \
"$NGFX" --activity 'GPU Trace Profiler' \
  --exe "$PWD/target/release/inferena-meganeura" \
  --dir "$PWD/frameworks/meganeura" --args SmolLM2-1.7B \
  --output-dir "$NGFX_OUT" --start-after-submits 15 --limit-to-submits 1 \
  --max-duration-ms 6000 --collect-screenshot 0 --set-gpu-clocks unaltered \
  --auto-export --trace-timeout 180 > "$NGFX_OUT/launcher.log" 2>&1
```

Build the clean, recorded source first. Omit `--platform` on this Linux CLI:
Qt misinterprets that option as its window-system plugin and aborts before
launch. The submit count 15 comes from the qualified 1.7B Systems capture
(one initialization submission and five warmups per phase, three prefill
samples), not a portable timing heuristic. Verify the captured region against
`meganeura/latency/measure`; other workloads/counts need their own trigger.
Do not use `--time-every-action` for the initial production-schedule trace.
First inspect system-memory traffic/long-latency loads in this placement-limited
case, then repeat on a resident smaller model for kernel/barrier analysis.
Account for the profiler's own memory allocations. Do not substitute frame
capture/replay for this placement experiment: `ngfx-capture` defaults to
demoting host-visible video memory to system memory (`--hvvm-demote`).
Preserve the full runner's numerical qualification and an ordinary control;
successful attachment or an exported file alone is not capture qualification.

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

The collector implements the qualification and process-rotation plan above;
it does not establish that all workloads/platforms have already passed it.
Automatic max-autotune may reject a kernel family on hardware/capacity grounds;
retain its diagnostics rather than overriding its hardware policy per model.
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
