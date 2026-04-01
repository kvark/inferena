---
layout: default
title: SmolLM2-135M
permalink: /models/SmolLM2-135M
---

# SmolLM2-135M

[HuggingFaceTB/SmolLM2-135M](https://hf.co/HuggingFaceTB/SmolLM2-135M) — 134.5M parameter decoder-only language model.

## Results

Benchmark config: seq_len=128, float32, input=[0,1,...,127].

| Platform | Framework | Compile (s) | Inference (ms) | Latency (ms) | Training (ms) | Loss |
|----------|-----------|:-----------:|:--------------:|:------------:|:-------------:|:----:|
| Intel Xeon @ 2.10GHz | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CPU) | 107.69 | **133** | — | **323** | 10.98 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ~~0.37~~ | ~~352~~ | — | ~~—~~ | ~~11.11~~ |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/Lavapipe) | ~~0.00~~ | ~~2288~~ | — | ~~4143~~ | ~~11.70~~ |
| | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | ~~3.31~~ | ~~14585~~ | — | ~~14299~~ | ~~10.81~~ |
| | [Meganeura](https://github.com/kvark/meganeura/tree/3d34aad) (Vulkan/Lavapipe) | **8.33** | 4943 | — | 3283 | 10.99 |
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | **52.71** | **66** | — | **111** | 8.35 |
|  | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ~~0.00~~ | ~~261~~ | — | ~~—~~ | ~~10.80~~ |
|  | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu) | ~~0.00~~ | ~~189~~ | — | ~~197~~ | ~~11.67~~ |
|  | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | ~~1.55~~ | ~~7452~~ | — | ~~7323~~ | ~~10.81~~ |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/77ceb78) (Vulkan) | ~~1.09~~ | ~~77~~ | — | ~~79~~ | ~~8.64~~ |
| Apple M3 | [PyTorch 2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (MPS) | 0.00 | **216** | — | **354** | 8.35 |
|  | [MLX](https://github.com/ml-explore/mlx) (MLX) | ~~0.00~~ | ~~82~~ | — | ~~188~~ | ~~8.64~~ |
|  | [Candle](https://github.com/huggingface/candle) (CPU) | ~~0.05~~ | ~~254~~ | — | ~~—~~ | ~~10.80~~ |
|  | [Burn](https://github.com/tracel-ai/burn) (wgpu) | ~~0.00~~ | ~~925~~ | — | ~~797~~ | ~~11.37~~ |
|  | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | ~~1.95~~ | ~~14650~~ | — | ~~14704~~ | ~~10.81~~ |
|  | [Meganeura](https://github.com/kvark/meganeura) (Metal) | ~~1.60~~ | ~~99~~ | — | ~~80~~ | ~~9.46~~ |
| Intel Graphics (RPL-U) | [PyTorch](https://github.com/pytorch/pytorch) | ✗ | ✗ | ✗ | ✗ | |
|  | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | 0.00 | **605** | — | — | 10.80 |
|  | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu) | ~~0.00~~ | ~~1502~~ | — | ~~5248~~ | ~~11.82~~ |
|  | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | 3.70 | 15959 | — | **15948** | 10.81 |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/77ceb78) (Vulkan) | ~~2.34~~ | ~~227~~ | — | ~~190~~ | ~~8.64~~ |
|  | [llama.cpp](https://github.com/ggml-org/llama.cpp) (CPU) | ✗ | ✗ | ✗ | ✗ | |

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
