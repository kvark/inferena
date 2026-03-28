# infermark

ML framework inference benchmark. Compares inference and training performance of
the same models across different ML frameworks on single-GPU hardware.

## Frameworks

| Framework | Language | Status |
|-----------|----------|--------|
| [PyTorch](https://pytorch.org/) | Python | Implemented |
| [Burn](https://burn.dev/) | Rust | Implemented |
| [Luminal](https://github.com/jafioti/luminal) | Rust | Scaffold |
| [Meganeura](frameworks/meganeura/) | Rust | Scaffold |

## Models

| Model | Parameters | Architecture | HuggingFace |
|-------|-----------|--------------|-------------|
| SmolLM2-135M | 135M | LLaMA | [HuggingFaceTB/SmolLM2-135M](https://hf.co/HuggingFaceTB/SmolLM2-135M) |
| SmolLM2-360M | 360M | LLaMA | [HuggingFaceTB/SmolLM2-360M-Instruct](https://hf.co/HuggingFaceTB/SmolLM2-360M-Instruct) |
| SmolLM2-1.7B | 1.7B | LLaMA | [HuggingFaceTB/SmolLM2-1.7B](https://hf.co/HuggingFaceTB/SmolLM2-1.7B) |
| SmolVLM-256M | 256M | Idefics3 | [HuggingFaceTB/SmolVLM-256M-Instruct](https://hf.co/HuggingFaceTB/SmolVLM-256M-Instruct) |

## What it measures

Each framework runs a fake training step on the selected model:

1. **Compile/Init** — Time to load and prepare the model on the GPU.
2. **Forward** — Single forward pass with a fixed dummy input (seq_len=128).
3. **Backward** — Backpropagation from a cross-entropy loss.

Outputs (logits, loss) are compared across frameworks to verify semantic equivalence.

## Quick start

```bash
# Prerequisites: Rust toolchain, Python 3, GPU drivers
./run.sh                              # all frameworks, SmolLM2-135M
./run.sh -m SmolLM2-135M -f pytorch   # just PyTorch
./run.sh --json                       # machine-readable output
```

### Download pre-trained weights (for PyTorch)

```bash
pip install huggingface-hub
./models/download.sh SmolLM2-135M
```

## Output format

Each framework runner produces a JSON object:

```json
{
  "framework": "pytorch",
  "model": "SmolLM2-135M",
  "device": "cuda:0",
  "gpu_name": "NVIDIA RTX 4090",
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

The harness collects these, prints a comparison table, and checks output consistency.

## Project structure

```
infermark/
├── run.sh                  # Main entry point
├── harness/                # Rust: orchestration, timing, comparison
├── frameworks/
│   ├── pytorch/            # Python + bash wrapper
│   ├── burn/               # Rust (wgpu backend)
│   ├── luminal/            # Rust (scaffold)
│   └── meganeura/          # Rust (scaffold)
└── models/
    └── download.sh         # HuggingFace model downloader
```

## License

MIT
