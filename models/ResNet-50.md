---
layout: default
title: ResNet-50
permalink: /models/ResNet-50
---

# ResNet-50

Classic convolutional neural network for image classification. 25.6M parameters.

## Results

Benchmark config: batch=4, 3x224x224, float32, random weights, cross-entropy loss.

| Platform | Framework | Compile (s) | Inference (ms) | Latency (ms) | Training (ms) | Loss |
|----------|-----------|:-----------:|:--------------:|:------------:|:-------------:|:----:|
| Intel Xeon @ 2.10GHz | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CPU) | 60.61 | 141 | 40 | **284** | 10.10 |
| | [ONNX Runtime 1.24.4](https://github.com/microsoft/onnxruntime) (CPU) | **0.28** | **76** | **18** | — | 10.37 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ~~0.00~~ | ~~782~~ | ~~311~~ | ~~—~~ | ~~6.91~~ |
| | [Meganeura](https://github.com/kvark/meganeura/tree/2ef151e) (Vulkan/Lavapipe) | ~~0.98~~ | ~~3906~~ | ~~1192~~ | ~~—~~ | ~~∞~~ |
| | [Burn](https://github.com/tracel-ai/burn) (wgpu) | ✗ | ✗ | ✗ | ✗ | |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | 36.46 | 48 | 16 | **97** | 6.92 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Vulkan) | 0.31 | 58 | 19 | 125 | 6.92 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIGraphXExecutionProvider) | 3.17 | **29** | **9.3** | — | 6.92 |
| Apple M3 | [PyTorch 2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (MPS) | 0.00 | 166 | 21 | **274** | 6.92 |
| | [MLX](https://github.com/ml-explore/mlx) | — | — | — | — | |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (Metal) | 0.01 | 16 | 5.3 | — | 6.91 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Metal) | 0.32 | 63 | 23 | 965 | 6.92 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CoreMLExecutionProvider) | 4.87 | **6.5** | **2.1** | — | 6.92 |
| | [JAX](https://github.com/jax-ml/jax) (METAL) | 0.90 | 139 | 9.4 | 409 | 6.92 |
| NVIDIA GeForce RTX 5080 | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 13.0) | 8.83 | 2.4 | 1.5 | **4.5** | 6.92 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CUDA) | **0.17** | 55 | 2.4 | — | 6.92 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/8042e00) (Vulkan) | 0.58 | 3.9 | 2.5 | 22 | 6.92 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | 1.52 | **2.4** | **1.3** | — | 6.92 |
| | [MAX](https://github.com/modular/modular) | — | — | — | — | |
| NVIDIA GeForce RTX 3050 (Windows) | [PyTorch 2.11.0+cu128](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 12.8) | 0.00 | **12** | **4.1** | **36** | 6.92 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Vulkan/DX12) | 1.04 | 19 | 6.5 | 52 | 6.92 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | 3.20 | 13 | 4.5 | — | 6.92 |
| | [JAX](https://github.com/jax-ml/jax) | ✗ | ✗ | ✗ | ✗ | |
| Intel(R) Graphics (RPL-U) | [PyTorch 2.11.0+xpu](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CPU) | 0.00 | 511 | 127 | **1048** | 6.92 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | 0.36 | 1187 | 368 | — | 6.92 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | — | — | — | — | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/8042e00) (Vulkan) | 0.77 | 225 | 73 | 1049 | 6.92 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPUExecutionProvider) | 5.75 | **204** | **53** | — | 6.92 |
| | [MAX](https://github.com/modular/modular) | ✗ | ✗ | ✗ | ✗ | |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | 6.99 | 436 | 176 | 2813 | 6.92 |

**Correctness:** PyTorch vs ONNX Runtime: **CLOSE** (loss diff 0.27, rel error 8.8%).

## Architecture

| Parameter | Value |
|-----------|-------|
| Input | 3x224x224 (ImageNet) |
| Conv layers | 53 (1x1, 3x3, 1x1 bottleneck) |
| Residual blocks | 16 (3+4+6+3) |
| Batch normalization | After every conv |
| Activation | ReLU |
| Global average pool | 7x7 -> 1x1 |
| Classifier | 2048 -> 1000 |
| Parameters | 25.6M |

## What this exercises

Completely different compute profile from transformer models:

- **Conv2D** — the dominant operation, not present in LLM benchmarks
- **Batch normalization** (not LayerNorm/RMSNorm/GroupNorm)
- **Residual connections** with dimension-matching 1x1 convolutions
- **Global average pooling** — spatial reduction
- No attention, no embedding lookup, no positional encoding
- Tests how well frameworks optimize spatial convolution kernels
