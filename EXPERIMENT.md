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
ROCm. When it is enabled, Inductor's own `triton.cudagraphs` option is disabled
to avoid nesting partial graph trees inside the whole-phase graph; other
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

## Reproduce

```sh
# A correctness check, not a benchmark.
.venv/bin/python -m unittest discover -s frameworks/pytorch -p test_execution.py -v

# Compiled whole-phase CUDA Graph baseline with automatic kernel search.
INFERENA_TORCH_MODE=max-autotune INFERENA_CUDA_GRAPHS=1 \
  ./run.sh -f pytorch,meganeura -m ResNet-50 --strict \
  --results-dir results/p3hpc-cuda-graphs/max-autotune-graph-01

# Matched compiled control, no explicit graph replay.
INFERENA_TORCH_MODE=default INFERENA_CUDA_GRAPHS=0 \
  ./run.sh -f pytorch,meganeura -m ResNet-50 --strict \
  --results-dir results/p3hpc-cuda-graphs/default-01
```

For diagnosis, set `TORCH_LOGS=graph_breaks,recompiles,perf_hints`. Capture is
untimed and qualification errors name the output or parameter. Use PyTorch's
profiler/Nsight separately to inspect `cudaGraphLaunch`, kernels and transfers;
instrumented durations must not replace ordinary timing samples.

## Collection plan (not yet a new paper matrix)

Initial harness pilot: one strict ResNet-50 process each for default compiled
without and with whole-phase CUDA Graphs, in that order, 5 warmups and 20
samples. This is a functional/measurement pilot, not independent replication
or a PyTorch-versus-Meganeura speed claim. Keep failures and do not retry a
configuration merely to improve its timing. The max-autotune condition and
paired Meganeura/full-model/device campaign remain separate required work.

Freeze source and dependency versions, then qualify all five matched workloads
before the publication cohort. Preserve strict and accelerated classes
separately. Compare default/no-graphs, default/whole-phase-graphs, and
max-autotune/whole-phase-graphs; eager/graphs is a useful separate diagnosis,
not a silently substituted compiled baseline. Use at least three fresh
processes per configuration, rotate order, keep 5 warmups and 20 retained
samples per phase, and retain failures as well as successes outside Git.
Collect on an idle GPU, without concurrent builds/profiles, recording driver,
PyTorch/CUDA versions and source revision. Keep compilation/capture/search
costs and graph-pool memory alongside steady-state times.

Recollect Meganeura at the declared revision in the same campaign. Do not
compare current PyTorch times against old Meganeura timings or select a winner
independently for every reported sample. Report configurations and process
replicates, not a fastest-run table. Revisit other devices explicitly; success
on NVIDIA does not establish ROCm, Metal or CPU behavior.

See [PyTorch CUDA Graph semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)
and [compiler modes](https://docs.pytorch.org/docs/stable/generated/torch.compile).
