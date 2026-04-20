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
| Intel Xeon @ 2.10GHz | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CPU) | 135.63 | 188 | **18** | **486** | 10.98 |
| | [ONNX Runtime 1.24.4](https://github.com/microsoft/onnxruntime) (CPU) | 65.50 | **118** | 20 | — | 10.98 |
| | [JAX 0.9.2](https://github.com/jax-ml/jax) (CPU) | 6.79 | 194 | 31 | 2107 | 10.98 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | **0.31** | 453 | 61 | — | 11.11 |
| | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | 3.37 | 17006 | — | 14459 | 10.81 |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/Lavapipe) | ~~0.00~~ | ~~2369~~ | ~~320~~ | ~~5700~~ | ~~11.73~~ |
| | [Meganeura](https://github.com/kvark/meganeura/tree/0b91e08) (Vulkan/Lavapipe) | 7.29 | 3933 | 852 | 3651 | 10.99 |
| | [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/5a0ed51) (CPU) | 0.10 | 221 | 24 | — | 10.98 |
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | 51.07 | 64 | 27 | 119 | 8.35 |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/vulkan) | ~~0.00~~ | ~~182~~ | ~~31~~ | ~~206~~ | ~~11.55~~ |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Vulkan) | 0.59 | **26** | **9.1** | **92** | 8.64 |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) | ✗ | ✗ | ✗ | ✗ | |
| Apple M3 | [PyTorch 2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (MPS) | 0.00 | 356 | 71 | 699 | 8.35 |
| | [MLX](https://github.com/ml-explore/mlx) (MLX) | 0.00 | 97 | — | **253** | 8.64 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (Metal) | ~~0.02~~ | ~~22~~ | ~~2.8~~ | ~~—~~ | ~~10.80~~ |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/metal) | ~~0.00~~ | ~~873~~ | ~~39~~ | ~~905~~ | ~~11.59~~ |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | ✗ | ✗ | ✗ | ✗ | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Metal) | 1.50 | 201 | **9.1** | 464 | 8.65 |
| | [GGML](https://github.com/ggerganov/ggml/tree/0.3.20) (Metal) | 0.38 | **49** | 11 | — | 8.69 |
| | [JAX](https://github.com/jax-ml/jax) (METAL) | ~~3.13~~ | ~~47~~ | ~~21~~ | ~~253~~ | ~~5.79~~ |
| NVIDIA GeForce RTX 5080 | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 13.0) | 16.83 | **4.0** | 2.8 | **6.5** | 8.35 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CUDA) | ~~0.05~~ | ~~48~~ | ~~2.4~~ | ~~—~~ | ~~10.80~~ |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/vulkan) | ~~0.00~~ | ~~154~~ | ~~26~~ | ~~86~~ | ~~11.69~~ |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/8042e00) (Vulkan) | 0.87 | 5.2 | 2.2 | 17 | 8.64 |
| | [GGML](https://github.com/ggerganov/ggml/tree/0.3.20) (CUDA) | **0.25** | 25 | **1.5** | — | 8.69 |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | ~~20.54~~ | ~~5.2~~ | ~~3.1~~ | ~~—~~ | ~~6.01~~ |
| | [MAX](https://github.com/modular/modular) (GPU) | ~~18.79~~ | ~~3.5~~ | ~~0.1~~ | ~~—~~ | ~~10.80~~ |
| NVIDIA GeForce RTX 3050 (Windows) | [PyTorch 2.11.0+cu128](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 12.8) | 0.00 | **11** | 5.1 | **51** | 8.35 |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/vulkan) | ~~0.00~~ | ~~125~~ | ~~28~~ | ~~138~~ | ~~11.76~~ |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Vulkan/DX12) | 1.40 | 13 | **3.6** | 58 | 8.63 |
| | [GGML](https://github.com/ggerganov/ggml/tree/0.3.4) (CUDA) | 0.31 | 132 | 5.9 | — | 8.69 |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | ~~39.92~~ | ~~18~~ | ~~14~~ | ~~—~~ | ~~6.01~~ |
| | [JAX](https://github.com/jax-ml/jax) | ✗ | ✗ | ✗ | ✗ | |
| Intel(R) Graphics (RPL-U) | [PyTorch](https://github.com/pytorch/pytorch) | ✗ | ✗ | ✗ | ✗ | |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | 0.00 | 599 | 24 | — | 10.80 |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/vulkan) | 0.00 | 1029 | 78 | 2411 | 11.58 |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | 3.49 | 16125 | — | 16174 | 10.81 |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Vulkan) | 1.82 | **170** | **24** | **675** | 8.64 |
| | [GGML](https://github.com/ggerganov/ggml) | ✗ | ✗ | ✗ | ✗ | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPUExecutionProvider) | 72.24 | 408 | 64 | — | 6.01 |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | 10.90 | 579 | 177 | 1426 | 5.79 |

**Correctness:** PyTorch vs ONNX Runtime: **PASS** (loss diff 3.2e-3).
PyTorch vs JAX: **PASS** (loss diff 3.2e-3).
PyTorch vs Meganeura: **PASS** (max error 1.7e-6, loss diff 5.3e-3).
PyTorch vs llama.cpp: **PASS** (loss diff 4.5e-3). Candle, Luminal: **CLOSE**.
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
