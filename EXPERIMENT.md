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

Use a dedicated environment with `requirements-p3hpc.txt` and the appropriate
vendor wheel index; it contains only this comparison's Python dependencies.
For NVIDIA, install using `--extra-index-url https://download.pytorch.org/whl/cu130`.
The v2 campaign collector requires PyTorch 2.13.0 and reported source commit
`cf30153c4c131c8164ee7798e5022d810682e2cb` on **every** platform. It checks
both before collection and in every paired result; unknown/different commits
fail closed. `--torch-version` still declares the exact platform wheel suffix.
The campaign and PyTorch records retain the commit and `torch.__config__.show()`.
This fixes the earlier release-only check; old v1 campaigns did not enforce a
common source commit and must not be relabelled retroactively.

This is a common **reported upstream source**, not identical binaries or
vendor libraries, and not an attestation of an unmodified build. CUDA, ROCm,
Metal and CPU builds necessarily differ. A vendor fork requiring another
commit/release needs its own labelled source ref and availability cohort,
not an override inside this controlled cohort. Do not upgrade during collection.

For SmolLM2, prepare `models/SmolLM2-135M/config.json` and `model.safetensors`
from a declared immutable Hugging Face revision before running. Both engines
must use those files. The collector hashes them and Cargo.lock before/after
the campaign and disables downloads/random-weight fallback. The other four
workloads use the unchanged deterministic initialization in source.

Use an idle device with no concurrent builds, profiles or experiments, a clean
source revision, and a **new directory outside the checkout**:

```sh
# A correctness check, not a benchmark.
.venv/bin/python -m unittest discover -s frameworks/pytorch -p test_execution.py -v

# Short paired qualification. No publication performance samples.
.venv/bin/python scripts/p3hpc.py --backend cuda --gpu 'RTX 5070' \
  --torch-version 2.13.0+cu130 --models ResNet-50 --precisions strict \
  --results-dir /mnt/data/p3hpc-resnet-qualification

# All five models, both precision classes; qualify ALL pairs, then measure.
.venv/bin/python scripts/p3hpc.py --backend cuda --gpu 'RTX 5070' \
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
| MPS | declared eager reference |
| CPU | declared eager availability reference, separate from GPU comparisons |

Every pair must retain both engines, pass forward **and** backward validation,
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
Do not call `wall minus sum(kernel medians)` CPU time: overlap, gaps and
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
results summary. The existing NVIDIA shell helper is Windows-path-specific;
do not mistake it for a qualified Linux profiling entry point.

For barriers, first capture the **production** grouped schedule with Blade 0.9.
Normal steps already use inline compute-to-compute barriers; the old broad
inter-pass barrier experiment is not a current application overhead estimate.
Any challenger must preserve RAW/WAR/WAW ordering, physical aliases, external
buffer visibility and full output/gradient correctness. Compare the same kernels,
geometry, data and precision with interleaved unprofiled trials; report the
paired latency delta and uncertainty, including a null result. Never disable
all barriers on a dependent graph or multiply a no-op barrier cost by its count.

For scaling, extend the matched SmolLM2 family from 135M to 360M and 1.7B before
introducing another architecture; the latter two still need native-runner and
collector wiring and qualification. Hold batch/context/precision fixed, then
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
claimed. Next: prepare the immutable SmolLM2 files, qualify the remaining models,
then run the declared `--collect` campaign on each available machine.
