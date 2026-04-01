# inferena

[![CI](https://github.com/kvark/inferena/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/inferena/actions/workflows/ci.yml)

ML framework inference benchmark. Compares inference and training performance of
the same models across different ML frameworks on single-GPU hardware.

Inspired by [meganeura's bench/compare.sh](https://github.com/kvark/meganeura/tree/main/bench) pipeline.

## Frameworks

All Rust frameworks use bleeding-edge git dependencies pinned to specific
revisions in `Cargo.toml`. The harness reads these at build time — framework
links in the results tables always point to the exact revision tested.

| Framework | Language | GPU Backend |
|-----------|----------|-------------|
| [PyTorch](https://pytorch.org/) | Python | CUDA / ROCm / MPS |
| [Candle](https://github.com/huggingface/candle) | Rust | CUDA / Metal / CPU |
| [Burn](https://github.com/tracel-ai/burn) | Rust | wgpu (Vulkan / Metal / DX12) |
| [Luminal](https://github.com/luminal-ai/luminal) | Rust | CUDA / Metal / CPU |
| [Meganeura](https://github.com/kvark/meganeura) | Rust | blade (Vulkan / Metal) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C++ | CUDA / Metal / Vulkan / CPU |

## Platform support

| Framework | Linux | macOS | Windows |
|-----------|:-----:|:-----:|:-------:|
| PyTorch | CUDA, ROCm, CPU | MPS, CPU | CPU |
| Candle | CUDA, CPU | Metal, CPU | CPU |
| Burn | Vulkan, CPU | Metal, CPU | — |
| Luminal | CUDA, CPU | Metal, CPU | — |
| Meganeura | Vulkan | Metal | Vulkan |
| llama.cpp | CUDA, Vulkan, CPU | Metal, CPU | CUDA, Vulkan, CPU |

Frameworks that can't run on a given platform are reported as `✗` in the results.

## Models

Each model has its own page with architecture details, benchmark caveats,
and results tables.

| Model | Type | Params | Results |
|-------|------|-------:|---------|
| [SmolLM2-135M](models/SmolLM2-135M.md) | Text LLM | 135M | [results](models/SmolLM2-135M.md#results) |
| [SmolVLA](models/SmolVLA.md) | Robotics Action Expert | ~14M | [results](models/SmolVLA.md#results) |
| [Stable Diffusion 1.5](models/StableDiffusion.md) | Image Diffusion (UNet) | ~860M | [results](models/StableDiffusion.md#results) |

## What it measures

Each framework runs a fake training step on the selected model:

1. **Compile** — Time to build, compile/optimize, and prepare the model (seconds).
2. **Forward** — Single forward pass with a fixed dummy input (milliseconds).
3. **Backward** — Backpropagation from a cross-entropy loss (milliseconds).

Outputs (logits, loss) are compared across frameworks to verify they run
the same model — flagged as **PASS**, **CLOSE**, or **DIFFERENT MODEL**.

## Quick start

```bash
# Prerequisites: Rust toolchain, Python 3, GPU drivers
./run.sh                                 # all models, all frameworks
./run.sh -m SmolLM2-135M                 # single model
./run.sh -m SmolLM2-135M -f pytorch      # single model + framework
./run.sh --json                          # machine-readable output
```

### Download pre-trained weights

```bash
pip install huggingface-hub
./models/download.sh SmolLM2-135M
```

Or generate random-init weights locally (no network needed):

```bash
python3 models/generate_weights.py SmolLM2-135M
```

## Project structure

```
./
├── run.sh                    # Main entry point
├── .github/workflows/ci.yml  # CI: build check + smoke test with Lavapipe
├── harness/                  # Rust: orchestration, timing, output comparison
├── frameworks/
│   ├── pytorch/              # Python + bash wrapper (HF transformers)
│   ├── burn/                 # Rust (wgpu backend, LLaMA-style model)
│   ├── luminal/              # Rust (graph-compiled, e-graph optimized)
│   └── meganeura/            # Rust (blade-graphics, e-graph optimized)
├── models/
│   ├── SmolLM2-135M.md       # Model description + results
│   ├── SmolVLA.md            # Model description + results
│   ├── download.sh           # HuggingFace model downloader
│   └── generate_weights.py   # Generate random-init weights locally
└── results/                  # Benchmark output (gitignored, per-run)
```

## License

MIT
