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
The collector checks the exact installed torch version including its vendor
suffix. Use the same release across machines where available; a vendor channel
requiring another release is a separately labelled availability cohort, not a
controlled cross-backend comparison. Do not upgrade packages during a campaign.

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
and match the declared revision, torch version, backend, GPU and execution
mode. A successful harness exit alone is insufficient. Invalid or missing
records stop the campaign; logs and `campaign.json` mark it incomplete. Do not
retry until favorable: diagnose, record the reason, and use a new source ref
and output directory when the protocol changes. Unsupported/oracle-disputed
pairs need an explicit scientific disposition, not an automatic exclusion.

`campaign.json` records the complete source SHA, pinned Meganeura dependency,
Python package versions, input hashes, device-selection overrides and run order.
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
