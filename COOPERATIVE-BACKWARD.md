# Cooperative convolution and the NVIDIA gap

RTX 5070, driver 595.91.07, 2026-09-15. This is a post-submission experiment,
not a new P3HPC cohort. The cooperative backward prototype is not a production
speedup: none of its new kernels won the completed ResNet search. The ordinary
benchmark checkout, submitted archives and paper are unchanged.

## What was missing

ResNet already uses cooperative operations in some forward work. The accelerated
training plan retains 25 cooperative dispatches. Its missing tensor path is
principally convolution derivatives, not all convolution.

- Autodiff marks derivative work as requiring full-precision operands. Converting
  tiny gradients directly to FP16 can turn them into zero. Compensated FP16
  improves mantissa precision but does not restore the exponent range.
- The driver's fixed-size KHR cooperative-matrix query advertises 16x16 FP16
  and BF16 input / f32 accumulator tiles, but no f32-input tile on this GPU.
  This is a statement about this queried interface and driver, not a hardware
  impossibility or a result for all NVIDIA devices.
- Meganeura had a generated cooperative input-gradient kernel, but no
  cooperative weight-gradient kernel. Simply overriding the precision guard
  would neither implement the missing kernel nor make training safe.

The retained prototype reuses the three scalar convolution indexing templates.
For each K tile it normalizes operands by powers of two, splits them into high
and low FP16 components, computes three cooperative products, and rescales the
f32 result. Four subgroups own the four 16x16 pieces of a 32x32 output tile.
The subgroup size comes from the device; there is no GPU/model performance gate.
It is offered to the existing measured search under a half-input policy, never
strict `NativeF32`. Incompatible device/precision settings discard a stored choice.

This handles the tested tiny gradients, not arbitrary f32 dynamic range within
one tile. Wide mixed-magnitude operands, other devices and longer training
trajectories remain unqualified. Do not promote the prototype on the strength
of small-oracle success alone.

## Results and numerical checks

The retained engine is
[`037c209`](https://github.com/kvark/meganeura/commit/037c209e2574c4364c581f522ce9668586741207),
on the previous scalar-tile experiment. Its only Blade changes are subgroup-size
reporting and enabling Naga's FP16-in-f32 rounding intrinsics:
[`7bba679`](https://github.com/kvark/blade/commit/7bba67943db18fffc218290f78984f36b9e08ce4).
The clean dependency build uses published Naga 30.0.1, without a Naga fork.

The first clean ResNet run (`e57bb4d`, cooperative challenger first) visits all
59 eligible training classes and makes 477
comparisons within the unchanged 60-second soft budget. Training search takes
42.91 seconds, total preparation 50.03 seconds, and training 29.989 ms. There
are zero new cooperative winners; selected execution remains scalar for those
derivatives. These are one process's 20-sample medians, not a replicated gain.
Earlier one-subgroup 16x16/32x32 and four-subgroup trials also had no wins.

The final revision moves this challenger after scalar tuning, without changing
qualification. It still visits 59 classes / 477 comparisons, takes 38.55 seconds
for training search and 44.77 seconds overall preparation, and runs training in
30.014 ms. Across inference, latency and training, both orders have 55 qualified
cooperative comparisons that lose on time and 21 rejected comparisons. Moving
the challenger does not unblock selection. Preparation differences are not a
claimed improvement: these runs do not control compiler-cache warmth.
Both orders retain identical recorded outputs and memory summaries.

The full f64 convolution oracle passes forward, input gradients and weight
gradients for four awkward channel/kernel/stride/padding cases, at ordinary
and `1e-12` gradient scales. The test forces all three cooperative kernels;
success cannot come from a scalar fallback. Existing scalar/generated/tuned
convolution regressions also pass. There are 275 passing library tests, three
ignored library tests, six passing opt-in GPU convolution tests, and passing
library/test Clippy with warnings denied. Only one GPU test function was added,
within the existing regression executable.

Qualification is unchanged: finite/full-output agreement plus sampled independent
f64 dots on both input patterns, before timing. New diagnostics expose the first
disagreement. A rejection is not necessarily an inaccurate challenger. For a
256-channel, 14x14, 3x3 forward convolution, output 108950 is:

| Scalar control | Cooperative | Independent f64 |
|---:|---:|---:|
| 0.0021697981 | 0.0021582246 | 0.002158616292991411 |

Here the scalar control is outside the f64 tolerance at that unsampled element;
the cooperative answer is inside it. The existing pairwise gate still rejects
the comparison. Other failures include pre-existing control failures. The 40
rejected training comparisons must not all be described as cooperative errors.
Unqualified choices are not timed, so zero wins does not prove every proposed
kernel would be slower.

Explicit `quantizeToF16` is important in the scaled implementation. Ordinary
cast/residual staging lost its correction in the optimized convolution test;
explicit rounding restores the full-oracle result. A separate scalar cast
microtest passes, so this is not an isolated/proven general driver bug.
The intended rounding operation is specified by
[WGSL](https://www.w3.org/TR/WGSL/#quantizetof16-builtin).

A separate BF16 prototype used three components and six products. It passed
the small convolution oracle but found no ResNet or SmolLM2 winners. It needs
experimental Naga/Blade BF16 plumbing and is not a production dependency.
Its local source snapshots are retained with the raw diagnostics.
It has not been validated across all Naga backends or the full test suite.

## What Nsight says

A forced, isolated ResNet-shaped weight gradient (batch 4, 256 input/output
channels, 14x14 spatial size, 3x3 kernel) passes 128 f64 spots for both arms.
Unprofiled 100-call medians are 0.179 ms scalar and 0.368 ms cooperative.
These are diagnostic single-process timings, not whole-model results.

Separate Nsight Graphics captures cover 32 warmed submissions:

| Counter, percent of its reported peak | Scalar | Cooperative |
|---|---:|---:|
| Sampled active compute warps | 38.45 | 41.41 |
| Tensor-pipeline active cycles | 0 | 11.66 |
| Heavy ALU instruction throughput | 33.23 | 37.94 |
| Sampled short-scoreboard stalled warps | 3.05 | 6.38 |
| Sampled fixed-latency wait warps | 8.93 | 13.39 |

Tensor operations really execute. Similar occupancy and substantial non-tensor
instruction activity are consistent with conversion, indexing and staging
overhead. These aggregate counters do not isolate the cost of each operation.
The repeated isolated kernel has a different cache working set from the model.
The next design question is how to amortize packing/scaling across more matrix
work, not how to force selection of this slower kernel.

The investigation also covers other workloads. Paired Nsight Systems captures
use Meganeura `4933260` and PyTorch 2.13.0+cu130, strict f32, five warmups and three
measured calls; both models pass the existing cross-framework checks:

| Training diagnostic, mean ms | SmolLM2-135M | StableDiffusion |
|---|---:|---:|
| Meganeura grouped Vulkan GPU interval | 38.336 | 6.292 |
| PyTorch sum of CUDA kernel intervals | 15.039 | 3.831 |
| Meganeura host sample | 51.865 | 9.862 |
| PyTorch host sample | 15.975 | 4.105 |

All 1,543 / 840 PyTorch training launches per call are captured CUDA graph nodes.
SmolLM2's top four kernel families consume 8.45 ms and are **SIMT f32 GEMMs**.
The strict gap is therefore not explained solely by missing tensor operations.
The interval rows have different aggregation semantics, include profiler effects,
and must not be subtracted to estimate removable barrier or busy CPU time.

An accelerated SmolLM2 training capture with Nsight Graphics shows 97.3% compute
queue activity, 28.0% sampled occupancy and 1.64% tensor-pipeline activity over
36.47 ms. Barrier-stalled warp samples are 6.03% of peak occupancy, approximately
21.5% of active-warp observations, **not 21.5% of elapsed time or Vulkan-barrier
cost**. This also points to GPU kernel work, beyond CPU submission overhead.
Its shader hashes have not yet been mapped to exact source lines.

## Reproduction and exclusions

This Inferena branch pins the engine and Blade revisions. Build with
`cargo build --release --locked -p inferena-meganeura --bins --examples`.
The runner's measurement code is unchanged. For an unprofiled forced dW check:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  python scripts/limited.py --memory-mib 2048 --seconds 60 -- \
  target/release/examples/conv_probe cooperative
```

Repeat with `scalar`. This example is a diagnostic, not an accepted tuner result.
For a native capture, use the installed Nsight Graphics 2026.3.1 CLI with
`--exe` pointing to that example and `--args cooperative` (or `scalar`). Set
`MEGANEURA_GPU_CAPTURE=1 INFERENA_NSYS=1 INFERENA_NGFX_PHASE=training
NV_AGORA_FORCE_BREAKPAD=-1`. Use `--activity "GPU Trace Profiler"
--start-with-ngfx-sdk --max-duration-ms 200 --limit-to-submits 32
--allocated-hes-buffer-memory-kb 8192 --pm-bandwidth-limit 100
--set-gpu-clocks unaltered --collect-screenshot 0 --real-time-shader-profiler
--auto-export --trace-timeout 120`, a fresh output directory, and a 3 GiB/no-swap
cgroup. Do not pass `--platform` through the Qt launcher on this installation.
The SDK trigger occurs after warmup. Nsight deliberately terminates the target
after capture: a saved capture is not a completed benchmark JSON record.

Raw records, per-invocation source diffs/hashes, compiler prototypes and native
captures remain outside Git in `/mnt/data/coop-backward-20260915.BBmjCm/`.
Useful captures are `smollm2-strict-r2`, `diffusion-strict`,
`smollm-native-graphics-r2`, and `conv-dw-{scalar,scaled-f16}-graphics`.
The last three use earlier source snapshots, not the cleaned dependency build;
their exact provenance is recorded alongside them. They had no missing-event
warning. No driver reset, reboot, clock change or cache flush was performed.

Accelerated Nsight Systems captures with separate contexts hit a process crash
in `libGLX_nvidia.so` and are excluded. Sharing one context allowed the later
Nsight Graphics captures. SmolLM2's per-pass training profile exceeded Blade's
1,000-pass timestamp limit (1,093 dispatches); its inference/latency sidecars
are partial evidence, not a complete training profile. Neither issue is hidden
by fallback timing. The ordinary benchmark was not changed to work around them.
