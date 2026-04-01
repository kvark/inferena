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
