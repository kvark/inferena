---
layout: default
title: Home
---

# infermark

ML framework inference benchmark — comparing training step performance of the
same models across different ML frameworks on single-GPU hardware.

## Models

| Model | Type | Params | |
|-------|------|-------:|---|
| [SmolLM2-135M](models/SmolLM2-135M) | Text LLM | 135M | [results →](models/SmolLM2-135M#results) |
| [SmolVLA](models/SmolVLA) | Robotics Action Expert | ~14M | [results →](models/SmolVLA#results) |

## How to read the tables

- **Bold** — best among frameworks running the **same model** as PyTorch.
- ~~Struck through~~ — framework runs a simplified/different model, not comparable.
- **✗** — framework doesn't support this model yet.
- Framework names link to the exact git revision tested.

## Frameworks

| Framework | Language | GPU Backend | Rev |
|-----------|----------|-------------|-----|
| [PyTorch](https://pytorch.org/) | Python | CUDA / ROCm / MPS | latest pip |
| [Burn](https://github.com/tracel-ai/burn) | Rust | wgpu (Vulkan / Metal / DX12) | [`ed72d2b`](https://github.com/tracel-ai/burn/tree/ed72d2b) |
| [Luminal](https://github.com/luminal-ai/luminal) | Rust | CUDA / Metal / CPU | [`f32161d`](https://github.com/luminal-ai/luminal/tree/f32161d) |
| [Meganeura](https://github.com/kvark/meganeura) | Rust | blade (Vulkan / Metal) | [`550bb6c`](https://github.com/kvark/meganeura/tree/550bb6c) |

## Run it yourself

```bash
git clone https://github.com/kvark/infermark && cd infermark
./run.sh                    # all models, all frameworks
./run.sh -m SmolLM2-135M    # single model
```

Results print as markdown tables — paste into the model page and submit a PR!

Source: [github.com/kvark/infermark](https://github.com/kvark/infermark)
