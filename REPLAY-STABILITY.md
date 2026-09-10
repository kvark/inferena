# Replay stability diagnostic, September 10

This branch retains source, not binary/data artifacts. Its base is the
one-command collector and stream-ownership repair at `50d10f2`. It is **not a
collection-ready tag**, a new PyTorch performance result, or a change to the
default numerical policy.

The full CUDA campaign first stopped at Whisper's stale default-stream
autograd node; the shared preparation/capture stream fixes that. The next
campaign stopped at diffusion's full-gradient replay comparison (3/2304
elements in the input convolution weight, maximum absolute difference
1.933e-6). The gate remains `rtol=1e-4, atol=1e-6`, with finite/full-tensor
checks and consecutive replays. No rejected campaign was retried until lucky.

The diagnostic adds eight uncaptured training calls before ordinary capture,
compares every participating parameter's gradient with the first call using
that same gate, then runs the unchanged full capture checks. Each of these
three controls ran once, in a fresh process on the RTX 5070 / driver 595.71.05,
Python 3.13.13, torch 2.13.0+cu130 at the common source pin:

| Control | Uncaptured repeat check | Subsequent capture |
|---|---|---|
| Default | One repeat fails; maximum absolute difference in the failing tensor 1.356e-6 | Fails |
| cuDNN deterministic only | Seven repeats fail; up to 8.583e-6 in affected tensors | Fails |
| Full PyTorch deterministic mode plus `CUBLAS_WORKSPACE_CONFIG=:4096:8` | All eight checks pass | All phases pass, including all 181 participating gradient tensors |

Ordinary variability already exceeds the replay gate. This does not localize
the responsible kernel, prove a capture implementation wrong, or justify
increasing tolerance until a run passes. Full deterministic mode is a possible
separate reference condition, not an invisible repair: it can change algorithm
selection and performance. The production collector still uses the original
policy and stops on failure. [PyTorch reproducibility controls](https://docs.pytorch.org/docs/stable/notes/randomness.html).

From this checkout, using the pinned CUDA environment (Linux can wrap each
command with `scripts/limited.py --memory-mib 6144 --seconds 300 --`):

```sh
python scripts/replay_stability.py
python scripts/replay_stability.py --deterministic
python scripts/replay_stability.py --all-deterministic
```

Outputs go to new directories outside Git. Expected numerical failures exit
nonzero; keep their summaries too. The emitted one-sample benchmark-shaped
JSON is diagnostic only, not paired qualification or publishable timing.
The original local summaries are at
`/x/Code/inferena-results/capture-stability.ThAAAQ/`. This retained script
changes their local path/provenance handling, not the diagnostic loop.
