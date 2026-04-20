---
layout: default
title: Stable Diffusion 1.5
permalink: /models/StableDiffusion
---

# Stable Diffusion 1.5

[stable-diffusion-v1-5/stable-diffusion-v1-5](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5) — Latent diffusion model for text-to-image generation.

## Results

Most frameworks (PyTorch, Meganeura, ONNX Runtime, JAX, MLX) run the **simplified U-Net** — Conv + GroupNorm + skip connections, no cross-attention or timestep embedding. Batch 2, 32×32×4 latent, base\_channels=64, 3 levels, ~2M params. Shared architecture, but each framework uses its own random-init parameters, so losses don't match across frameworks and several end up marked DIFFERENT MODEL even on identical structure.

**Candle** runs the **full SD 1.5 U-Net** (~860M params, 64×64×4 latent, cross-attention + timestep) — the real thing, marked DIFFERENT MODEL by design.

| Platform | Framework | Compile (s) | Inference (ms) | Latency (ms) | Training (ms) | Loss |
|----------|-----------|:-----------:|:--------------:|:------------:|:-------------:|:----:|
| Intel Xeon @ 2.10GHz | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CPU) | 53.02 | **14** | **11** | **28** | 0.57 |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/0b91e08) (Vulkan/Lavapipe) | **2.75** | 379 | — | 666 | 0.57 |
|  | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ~~0.00~~ | ~~10777~~ | ~~—~~ | ~~—~~ | ~~0.00~~ |
|  | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPU) | ✗ | ✗ | ✗ | ✗ | |
|  | [JAX](https://github.com/jax-ml/jax) (CPU) | ✗ | ✗ | ✗ | ✗ | |
|  | [Burn](https://github.com/tracel-ai/burn) (wgpu) | ✗ | ✗ | ✗ | ✗ | |
|  | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | 12.68 | **2.6** | **3.0** | **5.4** | 0.50 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Vulkan) | **0.09** | 10 | 11 | 15 | 0.53 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIGraphXExecutionProvider) | ~~29.97~~ | ~~3.7~~ | ~~—~~ | ~~—~~ | ~~0.05~~ |
| Apple M3 | [PyTorch 2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (MPS) | 0.00 | 504 | 11 | 222 | 0.50 |
| | [MLX](https://github.com/ml-explore/mlx) (MLX) | 0.00 | **6.9** | — | **9.3** | 0.51 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (Metal) | ~~0.01~~ | ~~233~~ | ~~—~~ | ~~—~~ | ~~0.00~~ |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Metal) | 0.49 | 8.9 | **8.9** | 68 | 0.53 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CoreMLExecutionProvider) | ~~2.38~~ | ~~12~~ | ~~—~~ | ~~—~~ | ~~0.05~~ |
| | [JAX](https://github.com/jax-ml/jax) (METAL) | ~~0.72~~ | ~~6.0~~ | ~~—~~ | ~~25~~ | ~~0.05~~ |
| NVIDIA GeForce RTX 5080 | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 13.0) | 6.35 | **1.0** | **0.9** | **1.4** | 0.50 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CUDA) | ~~0.01~~ | ~~108~~ | ~~—~~ | ~~—~~ | ~~0.00~~ |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/8042e00) (Vulkan) | **0.42** | **1.0** | 0.9 | 5.5 | 0.53 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | ~~0.84~~ | ~~0.8~~ | ~~—~~ | ~~—~~ | ~~0.05~~ |
| | [MAX](https://github.com/modular/modular) | — | — | — | — | |
| NVIDIA GeForce RTX 3050 (Windows) | [PyTorch 2.11.0+cu128](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 12.8) | 0.00 | **1.4** | **1.0** | **4.7** | 0.50 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Vulkan/DX12) | 0.50 | 3.1 | 3.1 | 7.8 | 0.52 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | ~~1.70~~ | ~~4.0~~ | ~~—~~ | ~~—~~ | ~~0.05~~ |
| | [JAX](https://github.com/jax-ml/jax) | ✗ | ✗ | ✗ | ✗ | |
| Intel(R) Graphics (RPL-U) | [PyTorch](https://github.com/pytorch/pytorch) | ✗ | ✗ | ✗ | ✗ | |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | 0.00 | 19889 | — | — | 0.00 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Vulkan) | 0.34 | **21** | **15** | **87** | 0.53 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPUExecutionProvider) | 2.42 | 31 | — | — | 0.05 |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | 6.09 | 94 | — | 245 | 0.05 |

*Run `./run.sh -m StableDiffusion` to populate this table.*

## Architecture

The benchmark measures the **UNet denoising backbone** — the compute-intensive core of Stable Diffusion.

### Full SD 1.5 (Candle)

| Component | Parameter | Value |
|-----------|-----------|-------|
| **UNet** | Input channels | 4 (latent space) |
| | Base channels | 320 |
| | Channel multipliers | [1, 2, 4, 4] |
| | Layers per block | 2 |
| | Attention resolutions | 32×32, 16×16, 8×8 |
| | Attention heads | 8 |
| | Cross-attention dim | 768 (CLIP text encoder) |
| | Parameters | ~860M |
| **Input** | Latent | 64×64×4 (512×512 image) |
| | Text embedding | 77×768 |
| | Timestep | 500 |

### Simplified U-Net (PyTorch, Meganeura)

| Component | Parameter | Value |
|-----------|-----------|-------|
| **UNet** | Input channels | 4 (latent space) |
| | Base channels | 64 |
| | Channel multipliers | [1, 2, 4] |
| | Levels | 3 |
| | GroupNorm groups | 16 |
| | Parameters | ~2M |
| **Input** | Latent | 32×32×4, batch 2 |
| | Loss | MSE (predict noise) |

## What this exercises

Completely different compute profile from SmolLM2 / SmolVLA:

- **Conv2D** (not present in transformer-only models) — spatial convolutions in UNet
- **Skip connections** (U-Net architecture) — memory-intensive residual paths
- **GroupNorm** (not RMSNorm/LayerNorm)
- **Spatial downsampling / upsampling** at multiple resolutions
- Tests Conv2D kernel performance, a major framework differentiator
- Full SD 1.5 (Candle) additionally exercises cross-attention and timestep embedding

## Caveats

- Only the **UNet** is benchmarked (not VAE encode/decode or text encoding).
- Input is deterministic synthetic data — no actual image generation.
- PyTorch and Meganeura use a simplified architecture for fair comparison.
- Candle runs the full SD 1.5 UNet but on CPU only (DIFFERENT MODEL vs others).
