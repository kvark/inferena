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
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | 53.54 | **148** | 21 | **128** | 8.35 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ~~0.01~~ | ~~258~~ | ~~11~~ | ~~—~~ | ~~10.80~~ |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/vulkan) | ~~0.00~~ | ~~187~~ | ~~30~~ | ~~207~~ | ~~11.50~~ |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | ~~1.51~~ | ~~7465~~ | ~~—~~ | ~~7497~~ | ~~10.81~~ |
| | [Meganeura](https://github.com/kvark/meganeura) | ✗ | ✗ | ✗ | ✗ | |
| | [GGML](https://github.com/ggerganov/ggml/tree/0.3.16) (CPU) | **0.18** | 1406 | **18** | — | 8.69 |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) | ✗ | ✗ | ✗ | ✗ | |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | ~~4.49~~ | ~~162~~ | ~~33~~ | ~~376~~ | ~~5.79~~ |
| Apple M3 | [PyTorch 2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (MPS) | 0.00 | 193 | **18** | 398 | 8.35 |
| | [MLX](https://github.com/ml-explore/mlx) (MLX) | 0.00 | **35** | — | **66** | 8.64 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (Metal) | ~~6.45~~ | ~~7519~~ | ~~2~~ | ~~—~~ | ~~10.80~~ |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/metal) | ~~0.00~~ | ~~421~~ | ~~40~~ | ~~859~~ | ~~11.78~~ |
| | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/5ddf5e5) (Metal) | 2.83 | 79 | 32 | 1955 | 8.64 |
| | [GGML](https://github.com/ggerganov/ggml) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | ~~4.91~~ | ~~362~~ | ~~136~~ | ~~934~~ | ~~5.79~~ |
| NVIDIA GeForce RTX 5080 | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 13.0) | 17.66 | **4** | 3 | **7** | 8.35 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CUDA) | ~~0.04~~ | ~~28~~ | ~~3~~ | ~~—~~ | ~~10.80~~ |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/vulkan) | ~~0.00~~ | ~~107~~ | ~~26~~ | ~~87~~ | ~~11.57~~ |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | ~~1.30~~ | ~~5793~~ | ~~—~~ | ~~5801~~ | ~~10.81~~ |
| | [Meganeura](https://github.com/kvark/meganeura/tree/33dcf42) (Vulkan) | 1.89 | 7 | 3 | 23 | 8.64 |
| | [GGML](https://github.com/ggerganov/ggml/tree/0.3.20) (CUDA) | **0.19** | 26 | **2** | — | 8.69 |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | ~~19.47~~ | ~~6~~ | ~~3~~ | ~~—~~ | ~~6.01~~ |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | ~~2.73~~ | ~~97~~ | ~~19~~ | ~~230~~ | ~~5.79~~ |
| NVIDIA GeForce RTX 3050 (Windows) | [PyTorch 2.11.0+cu128](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 12.8) | 0.00 | 331 | 36 | **122** | 8.35 |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu/vulkan) | ~~0.00~~ | ~~614~~ | ~~53~~ | ~~1048~~ | ~~11.73~~ |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/33dcf42) (Vulkan/DX12) | 4.22 | **32** | 35 | 134 | 8.59 |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) | ✗ | ✗ | ✗ | ✗ | |
| | [JAX](https://github.com/jax-ml/jax) | ✗ | ✗ | ✗ | ✗ | |
| Intel Graphics (RPL-U) | [PyTorch](https://github.com/pytorch/pytorch) | ✗ | ✗ | ✗ | ✗ | |
|  | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | 0.00 | **605** | — | — | 10.80 |
|  | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu) | ~~0.00~~ | ~~1502~~ | — | ~~5248~~ | ~~11.82~~ |
|  | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | 3.70 | 15959 | — | **15948** | 10.81 |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/77ceb78) (Vulkan) | ~~2.34~~ | ~~227~~ | — | ~~190~~ | ~~8.64~~ |
|  | [llama.cpp](https://github.com/ggml-org/llama.cpp) (CPU) | ✗ | ✗ | ✗ | ✗ | |

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
