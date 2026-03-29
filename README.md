# infermark

[![CI](https://github.com/kvark/infermark/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/infermark/actions/workflows/ci.yml)

ML framework inference benchmark. Compares inference and training performance of
the same models across different ML frameworks on single-GPU hardware.

Inspired by [meganeura's bench/compare.sh](https://github.com/kvark/meganeura/tree/main/bench) pipeline.

## Frameworks

All Rust frameworks use bleeding-edge git dependencies pinned to specific revisions.

| Framework | Language | GPU Backend | Rev |
|-----------|----------|-------------|-----|
| [PyTorch](https://pytorch.org/) | Python | CUDA / ROCm / MPS | latest pip |
| [Burn](https://github.com/tracel-ai/burn) | Rust | wgpu (Vulkan / Metal / DX12) | `ed72d2b` |
| [Luminal](https://github.com/luminal-ai/luminal) | Rust | CUDA / Metal / CPU | `f32161d` |
| [Meganeura](https://github.com/kvark/meganeura) | Rust | blade (Vulkan / Metal) | `550bb6c` |

## Platform support

| Framework | Linux | macOS | Windows |
|-----------|:-----:|:-----:|:-------:|
| PyTorch | CUDA, ROCm, CPU | MPS, CPU | CPU |
| Burn | Vulkan, CPU | Metal, CPU | DX12, Vulkan, CPU |
| Luminal | CUDA, CPU | Metal, CPU | CPU |
| Meganeura | Vulkan | Metal | Vulkan |

Frameworks that can't run on a given platform are reported as `✗` in the results.

## Models

| Model | Parameters | Architecture | HuggingFace |
|-------|-----------|--------------|-------------|
| SmolLM2-135M | 135M | LLaMA | [HuggingFaceTB/SmolLM2-135M](https://hf.co/HuggingFaceTB/SmolLM2-135M) |
| SmolLM2-360M | 360M | LLaMA | [HuggingFaceTB/SmolLM2-360M-Instruct](https://hf.co/HuggingFaceTB/SmolLM2-360M-Instruct) |
| SmolLM2-1.7B | 1.7B | LLaMA | [HuggingFaceTB/SmolLM2-1.7B](https://hf.co/HuggingFaceTB/SmolLM2-1.7B) |
| SmolVLM-256M | 256M | Idefics3 | [HuggingFaceTB/SmolVLM-256M-Instruct](https://hf.co/HuggingFaceTB/SmolVLM-256M-Instruct) |

## What it measures

Each framework runs a fake training step on the selected model:

1. **Compile/Init** — Time to build, compile/optimize, and prepare the model.
2. **Forward** — Single forward pass with a fixed dummy input (seq_len=128).
3. **Backward** — Backpropagation from a cross-entropy loss.

Outputs (logits, loss) are compared across frameworks with error metrics
(max error, MAE, RMSE, relative error) to verify semantic equivalence.

## Results

Results are populated dynamically by running the benchmark on different hardware.
Each run saves JSON to `results/` and the table below shows the latest.

### SmolLM2-135M training step (seq_len=128, float32, random weights)

| CPU / GPU | Framework | Compile (s) | Forward (ms) | Backward (ms) | Loss |
|-----------|-----------|:-----------:|:------------:|:--------------:|:----:|
| Intel Xeon @ 2.10GHz (Lavapipe) | PyTorch 2.11.0+cu130 (torch.compile) | 58.53 | 39580 | 21276 | 10.94 |
| Intel Xeon @ 2.10GHz (Lavapipe) | Burn `ed72d2b` (wgpu) | 0.00 | 1879 | 3583 | 11.49 |
| Intel Xeon @ 2.10GHz | Luminal `f32161d` (CPU) | 3.98 | 11695 | 11662 | 10.81 |
| Intel Xeon @ 2.10GHz (Lavapipe) | Meganeura `550bb6c` (blade) | 2.07 | 3301 | 2944 | 10.98 |

> **Note:** PyTorch and Meganeura run the real SmolLM2 architecture (GQA, RoPE, RMSNorm,
> causal attention). Burn and Luminal currently use a simplified LLaMA-style model (single-head
> attention, no RoPE/RMSNorm) — forward times are not directly comparable until those
> implementations are upgraded. Backward for Luminal is estimated as a second forward pass
> (training graph not yet wired). All frameworks use random-init weights.

Run `./run.sh` on your machine and share results via a PR to populate this table!

## Quick start

```bash
# Prerequisites: Rust toolchain, Python 3, GPU drivers
./run.sh                                 # all frameworks, SmolLM2-135M
./run.sh -m SmolLM2-135M -f pytorch      # just PyTorch
./run.sh -f meganeura,pytorch            # compare two frameworks
./run.sh --json                          # machine-readable output
./run.sh --download -f pytorch           # download model first, then run
```

### Download pre-trained weights

All frameworks that load real weights (PyTorch, Meganeura) use the standard
HuggingFace cache at `~/.cache/huggingface/hub/`. Pre-download with:

```bash
pip install huggingface-hub
./models/download.sh SmolLM2-135M
```

## Output format

Each framework runner produces a JSON object:

```json
{
  "framework": "meganeura",
  "model": "SmolLM2-135M",
  "device": "blade-gpu",
  "gpu_name": "blade-gpu",
  "timings": {
    "compile_s": 1.23,
    "forward_ms": 56.7,
    "backward_ms": 89.0
  },
  "outputs": {
    "logits_hash": "sha256:...",
    "logits_sample": [0.1, 0.2, "..."],
    "loss": 2.345
  }
}
```

The harness collects these, prints a comparison table, and checks output
consistency with error metrics (modeled after meganeura's compare.sh).
Results are saved to `results/<model>_<framework>.json`.

## Project structure

```
infermark/
├── run.sh                    # Main entry point
├── .github/workflows/ci.yml  # CI: build check + smoke test with Lavapipe
├── harness/                  # Rust: orchestration, timing, output comparison
├── frameworks/
│   ├── pytorch/              # Python + bash wrapper (HF transformers)
│   ├── burn/                 # Rust (wgpu backend, LLaMA-style model)
│   ├── luminal/              # Rust (graph-compiled, e-graph optimized)
│   └── meganeura/            # Rust (blade-graphics, e-graph optimized)
├── models/
│   └── download.sh           # HuggingFace model downloader (shared cache)
└── results/                  # Benchmark output (gitignored, per-run)
```

## License

MIT
