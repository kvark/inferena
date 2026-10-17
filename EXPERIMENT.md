# PR #261 measurement controls

Diagnostic branch based on Inferena `7cecdc6`; production input-refresh support
is independently proposed against current main in Inferena #67.

The paired controller runs independently searched or frozen selected programs
in two resident processes. GPU work is serial. It warms both arms, alternates
ABBA/BAAB blocks, and records execution, CPU encoding, and wait durations.
`--timing-cpu 2` pins only the main execution threads while timing and restores
the original affinity before the next search. `--cpu 2` pins the whole process,
including compilation, and is used only for frozen-plan/compiler controls.

Frozen replay asserts exact plans, pipelines, barrier groups, and output/gradient
hashes. The ordinary session constructor can reselect tuned kernels; replay
aborts if that happens. In particular, the recorded ResNet replay aborted after
inference. Do not relax that assertion or treat its partial record as complete.

```sh
export INFERENA_ARTIFACT_DIR=/path/to/new/artifacts
export INFERENA_MODEL_CWD=/path/to/inferena-with-models
python3 experiments/pr261_followup/build.py
python3 experiments/pr261_followup/paired.py resnet --gpu rtx5070 --model ResNet-50 --profile accelerated --timing-cpu 2
python3 experiments/pr261_followup/analyze.py
```

`cache-controls.py` compares cold and warm NVIDIA caches in separate newly
created artifact directories, without changing the user's existing cache.
Use B570 with `--gpu b570`. The 1,800-second watchdog terminates owned child
process groups on failure. Keep benchmarks serial and do not build alongside
timing. The default protocol uses 20 rounds of 32 samples per block: 1,280
samples per arm and phase, following 20 seconds of balanced warmup.

Measurements predating this source checkpoint are retained with original
patches, locks and binary hashes under
`/x/Code/inferena-results/pr261-noloss-20261017/`. The published builder uses
exact Git pins for the same Meganeura/Blade sources; the original measurements
used local detached checkouts. Raw data, frozen plans, shader dumps, telemetry
and binaries remain outside Git. Compiler-instrumentation patches are retained
there separately. This branch's temporary replay API is not a production feature.
