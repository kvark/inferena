# infermark

ML framework inference benchmark. Compares inference and training performance of
the same models across different ML frameworks on single-GPU hardware.

Inspired by [meganeura's bench/compare.sh](https://github.com/kvark/meganeura/tree/main/bench) pipeline.

## Frameworks

All Rust frameworks use bleeding-edge git dependencies pinned to specific revisions.

| Framework | Language | Backend | Status |
|-----------|----------|---------|--------|
| [PyTorch](https://pytorch.org/) | Python | CUDA/ROCm/MPS | Implemented |
| [Burn](https://github.com/tracel-ai/burn) @ `ed72d2b` | Rust | wgpu (Vulkan/Metal/DX12) | Implemented |
| [Luminal](https://github.com/luminal-ai/luminal) @ `f32161d` | Rust | NativeRuntime (CPU) / CUDA | Implemented |
| [Meganeura](https://github.com/kvark/meganeura) @ `550bb6c` | Rust | blade-graphics (Vulkan/Metal) | Implemented |

## Models

| Model | Parameters | Architecture | HuggingFace |
|-------|-----------|--------------|-------------|
| SmolLM2-135M | 135M | LLaMA | [HuggingFaceTB/SmolLM2-135M](https://hf.co/HuggingFaceTB/SmolLM2-135M) |
| SmolLM2-360M | 360M | LLaMA | [HuggingFaceTB/SmolLM2-360M-Instruct](https://hf.co/HuggingFaceTB/SmolLM2-360M-Instruct) |
| SmolLM2-1.7B | 1.7B | LLaMA | [HuggingFaceTB/SmolLM2-1.7B](https://hf.co/HuggingFaceTB/SmolLM2-1.7B) |
| SmolVLM-256M | 256M | Idefics3 | [HuggingFaceTB/SmolVLM-256M-Instruct](https://hf.co/HuggingFaceTB/SmolVLM-256M-Instruct) |

## What it measures

Each framework runs a fake training step on the selected model:

1. **Compile/Init** — Time to build, compile, and prepare the model on the GPU.
2. **Forward** — Single forward pass with a fixed dummy input (seq_len=128).
3. **Backward** — Backpropagation from a cross-entropy loss.

Outputs (logits, loss) are compared across frameworks with error metrics
(max error, MAE, RMSE, relative error) to verify semantic equivalence.

## Quick start

```bash
# Prerequisites: Rust toolchain, Python 3, GPU drivers
./run.sh                              # all frameworks, SmolLM2-135M
./run.sh -m SmolLM2-135M -f pytorch   # just PyTorch
./run.sh -f meganeura,pytorch         # compare two frameworks
./run.sh --json                       # machine-readable output
```

### Download pre-trained weights (for PyTorch / Meganeura)

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
    "compile_ms": 1234.5,
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
consistency with error metrics modeled after meganeura's compare.sh.

## Project structure

```
infermark/
├── run.sh                  # Main entry point
├── harness/                # Rust: orchestration, timing, output comparison
├── frameworks/
│   ├── pytorch/            # Python + bash wrapper (HF transformers)
│   ├── burn/               # Rust (wgpu backend, LLaMA-style model)
│   ├── luminal/            # Rust (graph-compiled, e-graph optimized)
│   └── meganeura/          # Rust (blade-graphics, e-graph optimized)
└── models/
    └── download.sh         # HuggingFace model downloader
```

## License

MIT
