# CUDA repeat noise, September 10

Measured source: `0734dd7`, based on collector `f2bb048`. Source only; generated
records and binaries do not belong in this branch. This diagnostic deliberately
records gradients without a gradient acceptance gate. It is **not** a paired
qualification or a publication timing campaign; the ordinary collector never
uses this override.

On RTX 5070 / driver 595.71.05, Python 3.13.13, torch 2.13.0+cu130 at the
campaign's pinned source, each fresh process compiles the unchanged diffusion
workload, then measures 16 uncaptured training calls and 16 captured replays
against the first uncaptured result. All 181 participating parameter-gradient
tensors are compared; output/loss finite and elementwise checks remain active.

Worst full-gradient relative L2 error across the calls (percent):

| Control | Uncaptured | Captured |
|---|---:|---:|
| strict, default algorithms | 0.000194% | 0.000206% |
| accelerated, default algorithms | 0.224% | 0.305% |
| accelerated, full deterministic control | 0 | 0 |

Aggregate squared RMS errors/reference scales are weighted by each parameter's
element count, not averaged across parameters. Some small-gradient tensors
have much larger relative variation: roughly 10% for an attention projection
whose RMS gradient is only 4.46e-5. This is why one larger relative tolerance
per tensor is not an adequate description. A retrospective split of the
default accelerated trace (first eight ordinary calls for calibration, last
eight held out) fits a twice-observed-noise allowance plus the strict base:
worst allowance fractions 0.858 held out and 0.802 captured. This motivates a
**prospective** qualification policy; it is not an independent validation of it.

These controls establish ordinary variability, not a capture-only defect. They
are consistent with documented CUDA backward nondeterminism and TF32's reduced
input precision. They do not identify a specific atomic kernel or prove the
quantization-amplification mechanism. Deterministic mode changes algorithms;
it is a diagnostic, not a silently restricted performance baseline.
[PyTorch reproducibility](https://docs.pytorch.org/docs/main/notes/randomness.html),
[TF32 precision](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html#tensorfloat-32-tf32-on-nvidia-ampere-and-later-devices).

Run serially with the pinned CUDA environment. Linux launches here used
`scripts/limited.py --memory-mib 6144 --seconds 360 --` around each process:

```sh
python scripts/replay_precision.py --precision accelerated
python scripts/replay_precision.py --precision accelerated --deterministic
python scripts/replay_precision.py --precision strict
```

The script prints a new output directory outside Git. Original local folders
are `../inferena-precision-noise-62mpscft`, `../inferena-precision-noise-aqrjs5rz`
and `../inferena-precision-noise-g9a0xisy`, respectively. All three runs finished;
none was repeated or selected for a favorable result. RAM/Nsight investigation
was not resumed, and no driver or sysctl setting was changed.
