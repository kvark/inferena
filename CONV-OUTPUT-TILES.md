# Smaller convolution weight-gradient tiles on RTX 5070

The measured search reduces ResNet-50 training time by about 7%. Nsight confirms
that the saving is on the GPU. The price is about 12 seconds of extra search;
neither execution-plan memory nor synchronization groups increase. These are
post-submission diagnostics, not replacement P3HPC results.

## Revisions and method

- Baseline Meganeura: `428fc2d2322229e5338f5d80a10d700340d593cd`.
- Measured candidate: `493326075d76da900be10876a0297305682eec5b`.
  PR [185](https://github.com/kvark/meganeura/pull/185) adds only shader comments
  and a changelog edit after that measured revision.
- Inferena baseline runner: `eda3384bec0402328615c56b9dd40b2287c14ae5`;
  candidate: `26e88aa14f6931ee7c9a39568e0d85b6e4ffa04d` on this branch.
  Runner source is identical. Only the
  Meganeura entry differs in the dependency lockfile; Blade remains
  `f6f2729e850cc0aefdc0bb18523da58a72765169`.
- RTX 5070, 48 SMs, driver 595.91.07, Vulkan. No clock, driver or memory-policy
  changes; no competing GPU job. Caches were not cleared, so preparation times
  do not represent cold compiler caches.
- ResNet-50 batch four, deterministic synthetic inputs/weights, forward + loss
  + backward, no optimizer update. Five warmups, 20 step-and-wait samples per
  fresh process; input uploads are outside these timers.
- Same full-domain 60-second soft limit, 1 GiB scratch cap, qualification and
  decision policy. No split reductions or reduced-precision derivatives.
- Strict pairs r2/r3/r4 alternate candidate-first, baseline-first,
  candidate-first. Accelerated r1/r2 use opposite orders. Separate seven-sample
  pass profiles and 20-sample Nsight captures do not enter the timing table.

## Whole-model result

Values are medians of process medians, milliseconds except preparation:

| Quantity | Strict baseline | Strict candidate | Accelerated baseline | Accelerated candidate |
|---|---:|---:|---:|---:|
| Training | 31.783 | 29.624 | 32.324 | 30.049 |
| Inference | 5.855 | 5.839 | 6.047 | 6.065 |
| Single-image latency | 3.153 | 3.154 | 3.335 | 3.332 |
| Preparation, seconds | 33.655 | 45.635 | 29.912 | 41.944 |

Each strict pair reduces training time by 6.79–7.06%; the accelerated pairs
reduce it by 6.82–7.25%. This is a small local replication, not a confidence
interval or a multi-device performance claim. Roughly 5,000–6,000 training
steps amortize the added search under this workload and search configuration.
The library's shorter default search budget is unchanged and was not evaluated
for a whole-model speedup here.

Every matched pair has identical retained outputs: full-logit hashes, sampled
logits, loss, total gradient norm and all parameter gradient norms. ResNet
records also pass the original numerical gates against the retained, validated
PyTorch 2.13.0+cu130 references from the preceding Nsight study. This does not
compare unrecorded full gradient arrays or a long training trajectory.

The candidate visits all 71 strict / 59 accelerated classes. Training search
takes about 38 / 35 seconds, versus 26 / 23 seconds. The same two synthetic
baseline classes still fail qualification; their repeated rejections rise from
10 to 22 because more challengers are proposed. Those challengers are not run
after a failed control, and no threshold was relaxed.

A separate strict Whisper-tiny encoder pair is unchanged: 33.466 → 33.464 ms
training, identical retained outputs and plan memory. Preparation is
4.544 → 5.483 seconds. This is one holdout pair, not evidence of a Whisper gain.

## Why it helps

The common generator now has independent row/column register tiles and A/B
staging extents. Weight-gradient search adds 16×16, 16×32 and 32×16 output
tiles at K=16/32, using the existing 256-thread workgroup. Device capabilities,
f32 precision, bindings and allocation bounds remain legality checks; there
is no card/model-specific performance threshold.

The strict pass profiles select nine smaller weight-gradient dispatches:

| Measured pass intervals | Baseline ms | Candidate ms |
|---|---:|---:|
| Stem weight gradient | 2.782 | 1.451 |
| All 53 weight gradients | 16.087 | 13.435 |
| All 52 input gradients | 7.610 | 7.664 |
| Median sum of GPU pass intervals | 32.879 | 30.287 |

The stem changes from 10 to 40 workgroups, using a 16×16 output tile and K=32.
Most remaining savings come from smaller/rectangular 1×1 weight-gradient tiles.
The plan still has 512 dispatches, 273 barrier groups and 633,028,308 allocated
buffer bytes. Profile wall overhead is 1.13× for both arms. Pass intervals
include instrumentation and inter-pass work; they are not kernel-only time.

Nsight Systems 2026.4.1 confirms the effect with the normal grouped schedule:

| Training, 20-call mean | Baseline ms | Candidate ms |
|---|---:|---:|
| Host sample, including input preparation | 34.565 | 32.342 |
| Host step (record + submit) | 4.557 | 4.673 |
| Grouped Vulkan GPU execution | 29.473 | 27.137 |

Each measured sample contains one complete GPU interval. Both captures visit
all search classes and reproduce their corresponding unprofiled retained
outputs. The GPU saving explains the result; this experiment does not measure
removable Vulkan-barrier cost. Host recording/wait ranges are elapsed time,
not busy CPU time.

## Rejected search expansion

Meganeura `e775c120d56e27142e90892489e370378df69cbc` tried the additional tiles
for all convolution directions. The pilot exhausted 60 seconds after 61/71
training classes, versus 71/71 in the baseline, and gave no whole-step gain:
31.676 → 31.669 ms. Total preparation rose from 44.771 to 85.494 seconds.
This pilot is excluded from the table. Its source and records are retained.

The final experiment narrows the extra candidates to weight gradients rather
than increasing the budget. Broader search needs better scheduling or pruning
to avoid starving existing choices; more candidate combinations alone are not
an improvement.

## Checks and reproduction

Existing tests cover Naga validation/SPIR-V emission, tile geometry, legacy
report decoding, full-f64 GPU convolution oracles, live-state/scratch
preservation and subsequent optimizer updates after exchanging selections.
No test function or executable was added. Local library tests and all-target,
all-feature Clippy pass. PR CI records Rust host coverage; it does not measure
WGSL execution coverage.

Raw records, profiles, captures and the serial invocation helper are outside
Git in `/mnt/data/conv-output-tiles-20260914.BAWatK/`. Each invocation records
its binary hash, engine revision, command and diagnostic environment. The
submitted archives and ordinary benchmarking checkout were not changed.

Build the locked runner at each listed Inferena revision with
`cargo build --release --locked -p inferena-meganeura`. Keep the resulting
binaries separately. From a checkout with the standard model definitions,
create a new output directory and run each binary serially:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
MEGANEURA_DEVICE_ID=12036 MEGANEURA_TUNE=1 INFERENA_TUNE_SECONDS=60 \
INFERENA_STRICT=1 INFERENA_WARMUP_RUNS=5 INFERENA_MEASUREMENT_RUNS=20 \
INFERENA_NSYS=1 FRAMEWORK_REV=493326075d76da900be10876a0297305682eec5b \
RUST_LOG=warn,meganeura::runtime::tuning=info \
.venv-p3hpc/bin/python scripts/limited.py --memory-mib 4096 --seconds 240 -- \
bash scripts/ngfx_target.sh /absolute/new-output-directory /absolute/candidate-binary ResNet-50
```

For the baseline, change both the binary and `FRAMEWORK_REV`. Use
`INFERENA_STRICT=0` for accelerated mode, or `Whisper-tiny` for the holdout.
For separate pass profiles, also set `MEGANEURA_GPU_TIMING=1`,
`INFERENA_PROFILE_SAMPLES=7` and `INFERENA_PROFILE_DIR=<new-directory>`.
Inspect them with `scripts/profile_report.py`; match dispatch origins when
comparing changed pipeline names.

For Nsight, use the same environment and put
`nsys profile --trace=vulkan,nvtx --vulkan-gpu-workload=individual --sample=none
--cpuctxsw=none --wait=primary --output=<new-prefix>` before the target Bash
command inside a 6 GiB/240-second limit. Export with `nsys export --type sqlite`,
then use `scripts/nsys_report.py`. No GUI, clock locking or per-action Graphics
serialization is needed for this follow-up.
