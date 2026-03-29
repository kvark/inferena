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
| Intel Xeon @ 2.10GHz | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CPU) | 59.26 | 38907 | 21049 | 10.98 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ~~0.31~~ | ~~373~~ | ~~—~~ | ~~11.11~~ |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/Lavapipe) | ~~0.00~~ | ~~2288~~ | ~~4143~~ | ~~11.70~~ |
| | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | ~~3.31~~ | ~~14585~~ | ~~14299~~ | ~~10.81~~ |
| | [Meganeura](https://github.com/kvark/meganeura/tree/c43315d) (Vulkan/Lavapipe) | **2.80** | **3972** | **3434** | 10.98 |
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | **52.86** | **63** | **113** | 8.35 |
|  | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ~~0.00~~ | ~~247~~ | ~~—~~ | ~~10.80~~ |
|  | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu) | ~~0.00~~ | ~~171~~ | ~~206~~ | ~~11.68~~ |
|  | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | ~~1.59~~ | ~~7621~~ | ~~7561~~ | ~~10.81~~ |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/c43315d) (Vulkan) | — | — | — | *re-run needed* |
|  | [llama.cpp](https://github.com/ggml-org/llama.cpp) (CPU) | ✗ | ✗ | ✗ | |

**Correctness:** PyTorch vs Meganeura: **PASS** (max error 1.7e-6, loss diff 6.4e-4).
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
- **Candle** runs the real LLaMA architecture with the same safetensors
  weights, but its `forward()` returns last-position logits only (private
  fields prevent getting all-position logits). Loss is computed on 1
  position vs 128 for others — hence DIFFERENT MODEL in correctness check.
  Timing is valid. Backward not yet wired.
- **Burn** and **Luminal** use a simplified model (single-head attention,
  no RoPE/RMSNorm) with random weights.
- Luminal backward is estimated as a second forward pass.
