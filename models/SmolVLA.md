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
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | 21.63 | 25 | 15 | 41 | 0.00 |
| | [Candle](https://github.com/huggingface/candle) | — | — | — | — | |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/33dcf42) (Vulkan) | **0.51** | 15 | **6** | **35** | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIGraphXExecutionProvider) | 22.10 | **12** | — | — | 0.00 |
| Apple M3 | [PyTorch 2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (MPS) | 0.00 | 172 | 11 | 119 | 0.00 |
| | [MLX](https://github.com/ml-explore/mlx) (MLX) | 0.00 | **13** | — | **24** | 0.00 |
| | [Candle](https://github.com/huggingface/candle) | — | — | — | — | |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/33dcf42) (Metal) | 0.77 | 37 | **8** | 149 | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CoreMLExecutionProvider) | 7.13 | 84 | — | — | 0.00 |
| | [JAX](https://github.com/jax-ml/jax) (METAL) | 2.06 | 23 | — | 96 | 0.00 |
| NVIDIA GeForce RTX 5080 | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 13.0) | 9.47 | 2.7 | 2.9 | **5.5** | 0.00 |
| | [Candle](https://github.com/huggingface/candle) | — | — | — | — | |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/33dcf42) (Vulkan) | 1.23 | 3.6 | **2.6** | 22 | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | 2.67 | **1.6** | — | — | 0.00 |
| NVIDIA GeForce RTX 3050 (Windows) | [PyTorch 2.11.0+cu128](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 12.8) | 0.00 | **4.5** | 3.5 | 53 | 0.00 |
| | [Candle](https://github.com/huggingface/candle) | — | — | — | — | |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/1a06c02) (Vulkan/DX12) | 0.71 | 5.0 | **2.9** | **23** | 0.00 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | 4.03 | 5.3 | — | — | 0.00 |
| | [JAX](https://github.com/jax-ml/jax) | ✗ | ✗ | ✗ | ✗ | |
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
