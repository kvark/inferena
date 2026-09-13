# P3HPC paired collection

Use branch `experiment/p3hpc-cuda-graphs`. Meganeura is pinned to merged
`428fc2d2322229e5338f5d80a10d700340d593cd`. Protocol
`p3hpc-paired-campaign-v8` is a new cohort: do not combine its results with
v7 / `efb1e520` or relabel older results as always-tuned or native-f32 strict.

**Collection candidate: qualify the new MPS and ROCm paths before launching
the common cohort or renting H100 again.** Local CUDA and XPU full-model checks
pass as detailed below; Linux success does not certify macOS/ROCm/Windows.

## What to run

First qualify the candidate on macOS, ROCm and Windows, using an existing
checkout/environment:

```sh
git switch experiment/p3hpc-cuda-graphs
git pull --ff-only
git rev-parse HEAD
.venv-p3hpc/bin/python scripts/p3hpc.py --qualify-only
```

Use the same complete source SHA on every machine. Existing pinned environments
do not need reinstalling. No `--no-max-autotune` is needed on any platform;
that old spelling remains accepted as a no-op.
Qualification checks all five models and both arithmetic classes once. Keep
the results, including any failure, and resolve protocol problems before
starting the common cohort. Once those backend checks pass, full collection
is simply `python scripts/p3hpc.py`, using the interpreter below.

For a new environment, install uv and Rust, then run the appropriate setup.
Setup downloads managed Python **3.13.13**, installs the pinned requirements,
and runs a small compiled forward/backward/replay probe. It does not run
max-autotune by default. No existing venv is overwritten.

| Machine | New environment | Collection with that environment |
|---|---|---|
| NVIDIA Linux / H100 | `bash scripts/setup.sh cu130` | `python scripts/p3hpc.py --backend cuda` |
| NVIDIA Windows / 3050 | `.\scripts\setup.ps1 cu130` | `python scripts/p3hpc.py --backend cuda` |
| AMD / 7900 XT, 780M | `bash scripts/setup.sh rocm7.2` | `python scripts/p3hpc.py --backend rocm` |
| Intel Arc B570 | `bash scripts/setup.sh xpu` | `python scripts/p3hpc.py --backend xpu` |
| Apple M3 | `bash scripts/setup.sh mps` | `python scripts/p3hpc.py --backend mps` |
| Intel iGPU with unsupported XPU | `bash scripts/setup.sh cpu` | `python scripts/p3hpc.py --backend cpu` |

Here `python` means the created environment's interpreter:
`.venv-p3hpc/bin/python` on Linux/macOS or
`.venv-p3hpc/Scripts/python.exe` on native Windows. After activation,
`python scripts/p3hpc.py` needs no arguments on a supported single-GPU
machine. CPU comparison is always explicit; it is GPU-versus-CPU availability
evidence, not part of GPU speed-ratio aggregates.

For example, the Mac qualification is:

```sh
.venv-p3hpc/bin/python scripts/check_environment.py --backend mps
.venv-p3hpc/bin/python scripts/p3hpc.py --backend mps --qualify-only
```

Substitute that machine's backend. The Mac check is particularly important:
this revision now compiles MPS, whereas v7 deliberately ran eager. A failure
must be investigated or explicitly reported; never silently use eager times
as compiled times. Qualification retains no publication samples. Full
collection repeats all numerical gates in every measurement process, so
qualification is not a certificate for future processes.

On the dual-GPU workstation, use separate existing environments, sequentially:

```sh
.venv-p3hpc/bin/python scripts/p3hpc.py --backend cuda --gpu 'RTX 5070'
.venv-p3hpc-xpu/bin/python scripts/p3hpc.py --backend xpu --gpu 'Arc B570'
```

To create the second environment:
`bash scripts/setup.sh xpu .venv-p3hpc-xpu`.
The collector selects Meganeura's matching adapter by device ID. Ambiguous
matches or a different reference GPU stop the run. Do not run both collectors
together. Avoid host suspension, competing GPU work and profiling.

Defaults collect five models, strict and accelerated arithmetic, three fresh
processes per condition, five warmups and twenty samples per phase:
**30 paired processes per device**, not the former 90 CUDA pairs. Results go
to a new `../inferena-results/<host>-<UTC>-<source>/` directory outside Git.
Copy that entire directory, including `campaign.json`, logs and compilation
receipts. Only `status: complete` establishes completed declared coverage.
An incomplete run may contain valid partial records and a failure finding;
it is not a complete aggregate or permission to retry until favorable.

For the cloud GPUs only, collect the larger models separately:

```sh
.venv-p3hpc/bin/python scripts/p3hpc.py --models SmolLM2-360M SmolLM2-1.7B
```

This includes forward/loss/backward. Use `--inference-only` only for an
explicitly separate inference study; it does not produce training data.
The family shares batch 1, 128-token prefill and stateless one-token forward,
but differs in depth, width and attention layout. It is not a parameter-count-only
scaling law. Do not quantize, offload or shrink workloads to make them fit.

## The executed contract

| Component | Primary policy |
|---|---|
| Meganeura | Empirical tuning enabled independently of the reference mode |
| Strict arithmetic | f32 operands/accumulators; native-f32 cooperative tiles permitted; f16-input tiles forbidden |
| Accelerated arithmetic | Reduced-input paths permitted, with f32 accumulation and the declared validation gates |
| CUDA and ROCm | Default PyTorch compilation plus qualified whole-phase `torch.cuda.CUDAGraph` replay |
| XPU | Default PyTorch compilation, public math SDPA setting, qualified whole-phase `torch.xpu.XPUGraph` replay |
| MPS | Default PyTorch compilation, including first Metal specializations; no equivalent public whole-phase replay API |
| CPU availability control | Explicit default-compiled CPU reference, no GPU replay |

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

Meganeura searches all classes in its **current legal tuning domain**, with
no eight-class cutoff, a **60-second soft deadline per session**, and at most
**1 GiB private scratch**, including staging. This is a ceiling, not a
reservation; the engine also checks the available device-memory budget.
The larger ceiling admits substantial language-model matrix bindings that
the old 64 MiB limit excluded. Most workloads create three sessions; Whisper
reuses its inference session for minimal forward and creates two.
Search returns as soon as the legal candidates finish; 60 seconds is not a
mandatory wait. A local ResNet check reached all 71 strict / 59 accelerated
training classes in 29 / 24 seconds, while a 10-second cap reached only
22 / 25. These are qualification observations, not a replicated speed claim.

The legal domain includes scalar f32 matmul tiles, compatible native-f32
cooperative alternatives, and scalar convolution forward/dX/dW shapes and
staging choices. It does not yet include arbitrary graph rewrites, GEMV,
reduced-input cooperative kernels, complex fused prologues/epilogues,
overlapping bindings or split-K. “All” is not an exhaustive search of all
possible programs. Every candidate retains the existing numerical
qualification and paired timing gates. An incomplete or invalid comparison
keeps its incumbent.

Each session records its actual cooperative policy, eligible/visited classes,
excluded dispatches, comparison decisions, scratch use, time consumed and
whether the deadline was hit. The collector checks these executed receipts,
not just an environment variable. Hardware capability fields distinguish
permission to use native f32 from its availability and actual use.

PyTorch has a **120-second compilation deadline per process**, covering
`torch.compile` and the first specializations of all requested phases,
including backward and minimal forward. A separate supervisor kills the
compiler process tree on expiration, records `torch-preparation.json`, and
returns a failure: it never substitutes eager. Model loading and replay/
numerical qualification are outside this compile interval and have a separate
ten-minute process safety cap. The setup probe uses the same supervisor.

This is a bounded-startup comparison, **not an identical total wall-clock
budget for both engines**: native graph/pipeline construction is timed but
outside the per-session search deadline, which can overrun by an in-flight
driver operation. Report measured preparation costs; do not assume every
Meganeura search finishes sooner than every PyTorch compilation.

Optional, separate studies:

- `--max-autotune`: add searched PyTorch under the same 120-second deadline.
- `--graph-ablation`: also collect the uncaptured default reference.
- `--compile-seconds N` / `--tune-seconds N`: explicitly different budgets.
- `--no-graphs` / `--eager`: explicit reference overrides, never automatic fallbacks.

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
match the pins; mismatches are not overwritten. Other workloads retain the
source-defined deterministic initialization. Rust builds are locked and
finish before the first pair; build parallelism defaults to one job.

Whole-phase replay retains full output/loss comparison at `rtol=1e-4,
atol=1e-6`, exact shapes/dtypes/participating-gradient inventory and finite
values. Strict gradients pass per-tensor and full-gradient maximum/RMS bounds:

```text
max(abs(error)) <= 1e-6 + 1e-4 * max(abs(reference))
RMS(error)      <= 1e-6 + 1e-4 * RMS(reference)
```

Accelerated training uses one fixed reference, eight ordinary repeats and two
replays, with complete-gradient maximum and element-weighted RMS bounds at
`atol=1e-6, rtol=0.01`. It retains per-tensor diagnostics, but never learns
tolerances from repeat samples. Strict and inference use two ordinary repeats
and two replays. Numerical failures are not discarded as timing outliers.

Cross-engine validation retains the v7 policy: strict errors must remain below
5% per process; accelerated forward must pass and each training error must
remain below a 10% safety ceiling, with medians below 5% across three independent
processes. Every sample is retained; `replicated-gradient-median-v1` is a
separate gate from within-process replay validation. No limit has been loosened
for the new execution paths.

`check_pair` also rejects disabled native tuning, an incorrect strict policy,
missing session evidence, unexpected class caps, uncompiled requested modes,
missing budget supervision, wrong graph APIs, revision/device mismatches,
diagnostic samples and invalid timing series. A successful subprocess exit
alone is insufficient.

## Platform notes

Windows uses native x64 Python, Git for Windows / Git Bash (not WSL), Rust MSVC,
and Visual Studio C++ Build Tools with a Windows SDK. Setup selects
`requirements-p3hpc-cu130-windows.txt`, including pinned
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
SDPA selection. The full-model workaround passes the same gates on B570.
`execution.sdpa_policy` and `sdpa_enabled_backends` record the active setting.
The runner selects Triton's backend explicitly from the requested Torch
device, avoiding ambiguous auto-detection when NVIDIA and Intel drivers are
both installed. It records the selection and rejects a conflicting override.

On the Radeon 780M configuration previously reported in
[issue #61](https://github.com/kvark/inferena/issues/61), retain the needed
runtime workarounds:

```sh
export HSA_OVERRIDE_GFX_VERSION=11.0.2
export HSA_ENABLE_SDMA=0
.venv-p3hpc/bin/python scripts/p3hpc.py --backend rocm
```

Do not add those workarounds on machines that do not need them. Their values
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

On September 13, RTX 5070 passed all ten model/arithmetic pairs at `7d671a0`;
the expanded ResNet search passed again with the final default budgets and
receipt checks at `1430d0d`. B570 passed all ten pairs at `d8335a8`, using
the declared math SDPA policy; all receipts also pass the `1430d0d` checker.
Both installed vendor wheels pass the six broad execution/contract tests,
both setup probes pass, and the nine Rust harness tests pass. Raw qualification
outputs stay outside Git; these single-pair checks are not publication data.

The B570 ResNet training search can reach its 60-second deadline before
visiting all classes; this is recorded and retains untested incumbents.
macOS/ROCm/Windows hardware qualification and cloud-model coverage remain
outstanding. In particular, local tests do not prove MPS compilation or actual
native-f32 cooperative use on a Mac. Do not launch the paid cloud campaign
until the new backend paths have qualified.

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
.venv-p3hpc/bin/python scripts/limited.py --memory-mib 6144 --seconds 600 -- \
  .venv-p3hpc/bin/python scripts/nsys.py --gpu 'RTX 5070' \
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
