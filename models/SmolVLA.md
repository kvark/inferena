---
layout: default
title: SmolVLA
permalink: /models/SmolVLA
---

# SmolVLA Action Expert

[lerobot/smolvla_base](https://hf.co/lerobot/smolvla_base) — SmolVLA action expert decoder for robotics.

## Results

Benchmark config: chunk_size=50, vlm_seq_len=16, float32, random weights, MSE loss.

| Platform | Framework | Compile (s) | Inference (ms) | Latency (ms) | Training (ms) | Loss |
|----------|-----------|:-----------:|:--------------:|:------------:|:-------------:|:----:|
| Intel Xeon @ 2.10GHz | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CPU) | 51.63 | **40** | **11** | **116** | 0.00 |
| | [Meganeura](https://github.com/kvark/meganeura/tree/0b91e08) (Vulkan/Lavapipe) | **2.75** | 696 | — | 3850 | 0.01 |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| | [Candle](https://github.com/huggingface/candle) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| | [Burn](https://github.com/tracel-ai/burn) (wgpu) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | 20.65 | 26 | 15 | 45 | 0.00 |
| | [Candle](https://github.com/huggingface/candle) (CPU) | — | — | — | — | |
| | [Burn](https://github.com/tracel-ai/burn) (wgpu) | — | — | — | — | |
| | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/5ddf5e5) (Vulkan) | **0.48** | **20** | **15** | **39** | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) (CPU) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPUExecutionProvider) | 4.35 | 36 | — | — | 0.00 |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | 1.81 | 32 | — | 122 | 0.00 |
| Apple M3 | [PyTorch 2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (MPS) | 0.00 | 232 | **9** | 117 | 0.00 |
| | [MLX](https://github.com/ml-explore/mlx) (MLX) | 0.00 | **12** | — | **24** | 0.00 |
| | [Candle](https://github.com/huggingface/candle) (CPU) | — | — | — | — | |
| | [Burn](https://github.com/tracel-ai/burn) (wgpu) | — | — | — | — | |
| | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/5ddf5e5) (Metal) | 0.68 | 53 | 22 | 42 | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) (CPU) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CoreMLExecutionProvider) | 6.62 | 118 | — | — | 0.00 |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | 1.89 | 57 | — | 180 | 0.00 |
| NVIDIA GeForce RTX 5080 | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 13.0) | 8.06 | **3** | **1** | **3** | 0.00 |
| | [Candle](https://github.com/huggingface/candle) (CPU) | — | — | — | — | |
| | [Burn](https://github.com/tracel-ai/burn) (wgpu) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) (Vulkan) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/6b76b5e) (Vulkan) | **0.55** | 5 | 5 | 14 | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) (CPU) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPUExecutionProvider) | 2.25 | 13 | — | — | 0.00 |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | 0.98 | 16 | — | 85 | 0.00 |
| Intel Graphics (RPL-U) | [PyTorch](https://github.com/pytorch/pytorch) | ✗ | ✗ | ✗ | ✗ | |
|  | [Candle](https://github.com/huggingface/candle) (CPU) | ✗ | ✗ | ✗ | ✗ | |
|  | [Burn](https://github.com/tracel-ai/burn) (wgpu) | ✗ | ✗ | ✗ | ✗ | |
|  | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | ✗ | ✗ | ✗ | ✗ | |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/77ceb78) (Vulkan) | **2.62** | **114** | — | **188** | 0.01 |
|  | [llama.cpp](https://github.com/ggml-org/llama.cpp) (CPU) | ✗ | ✗ | ✗ | ✗ | |

**Correctness:** PyTorch vs Meganeura: **CLOSE** (loss diff 1e-5, max error 4.6e-3).

## Architecture

Transformer action expert with alternating self-attention and cross-attention:

| Parameter | Value |
|-----------|-------|
| Hidden size | 720 |
| Layers | 16 (even: self-attn, odd: cross-attn) |
| Query heads | 15 |
| KV heads | 5 (GQA) |
| Head dim | 64 |
| FFN intermediate | 2048 |
| Activations | SiLU / SwiGLU |
| Normalization | RMSNorm |
| Action dim | 32 |
| Chunk size | 50 |

## What this exercises

Compared to [SmolLM2-135M](SmolLM2-135M):

- **Cross-attention** (action tokens attending to VLM context) — not just self-attention
- **Alternating attention patterns** (self-attn on even layers, cross-attn on odd)
- **Timestep conditioning** via sinusoidal embeddings concatenated to input
- **MSE loss** (regression) instead of cross-entropy (classification)
- **GQA with 15/5 heads** (different ratio than SmolLM2's 9/3)
- Exercises the same SwiGLU, RMSNorm, and matmul kernels as SmolLM2

This is the model that [Meganeura benchmarks against PyTorch](https://github.com/kvark/meganeura/tree/main/bench)
in its `compare.sh` pipeline.

## Caveats

- **PyTorch** and **Meganeura** implement the full action expert architecture
  and should produce matching outputs.
- **Burn** and **Luminal** do not implement this architecture yet (reported as ✗).
- Inputs are synthetic: random noisy actions, sinusoidal timestep, random VLM context.
