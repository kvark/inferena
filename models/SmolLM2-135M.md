---
layout: default
title: SmolLM2-135M
permalink: /models/SmolLM2-135M
---

# SmolLM2-135M

[HuggingFaceTB/SmolLM2-135M](https://hf.co/HuggingFaceTB/SmolLM2-135M) — 134.5M parameter decoder-only language model.

## Results

Benchmark config: seq_len=128, float32, input=[0,1,...,127].

| Platform | Framework | Compile (s) | Forward (ms) | Backward (ms) | Loss |
|----------|-----------|:-----------:|:------------:|:--------------:|:----:|
| Intel Xeon @ 2.10GHz | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CPU) | 54.51 | 34961 | 19049 | 10.98 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ~~0.28~~ | ~~325~~ | ~~—~~ | ~~11.11~~ |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/Lavapipe) | ~~0.00~~ | ~~2288~~ | ~~4143~~ | ~~11.70~~ |
| | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | ~~3.31~~ | ~~14585~~ | ~~14299~~ | ~~10.81~~ |
| | [Meganeura](https://github.com/kvark/meganeura/tree/550bb6c) (Vulkan/Lavapipe) | **2.01** | **3097** | **3032** | 10.98 |

**Correctness:** PyTorch vs Meganeura: **PASS** (max error 1.7e-6, loss diff 8.6e-4).
Struck-through values are from frameworks running a different (simplified) model.

## Architecture

LLaMA-family transformer with Grouped Query Attention:

| Parameter | Value |
|-----------|-------|
| Hidden size | 576 |
| Layers | 30 |
| Attention heads | 9 (3 KV heads, GQA) |
| FFN intermediate | 1536 |
| Vocab size | 49152 |
| Context length | 2048 |
| Activations | SiLU / SwiGLU |
| Normalization | RMSNorm |
| Position encoding | RoPE |

## What this exercises

- Matrix multiplications (Q/K/V projections, FFN up/gate/down)
- Grouped Query Attention with causal masking
- Rotary Position Embeddings
- RMSNorm (2 per layer, 60 total)
- SwiGLU activation fusion
- Embedding lookup + tied lm_head

## Caveats

- **PyTorch** and **Meganeura** load real model weights and run the full
  architecture — their outputs match.
- **Candle** runs the real LLaMA architecture but uses zero-init weights
  (not the same safetensors). Returns only last-position logits.
  Backward not yet wired.
- **Burn** and **Luminal** use a simplified model (single-head attention,
  no RoPE/RMSNorm) with random weights.
- Luminal backward is estimated as a second forward pass.
