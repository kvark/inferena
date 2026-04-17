---
layout: default
title: Stable Diffusion 1.5
permalink: /models/StableDiffusion
---

# Stable Diffusion 1.5

[stable-diffusion-v1-5/stable-diffusion-v1-5](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5) — Latent diffusion model for text-to-image generation.

## Results

Two benchmark configurations are used depending on framework capabilities:

**Simplified U-Net** (PyTorch, Meganeura): Conv-only U-Net without cross-attention or timestep embedding. Batch 2, 32×32×4 latent, base_channels=64, 3 levels (~2M params). Enables apples-to-apples comparison of Conv2D + GroupNorm + skip connection performance.

**Full SD 1.5 U-Net** (Candle): Complete architecture with cross-attention, timestep embedding, ~860M params, 64×64×4 latent. Marked DIFFERENT MODEL.

| Platform | Framework | Compile (s) | Inference (ms) | Latency (ms) | Training (ms) | Loss |
|----------|-----------|:-----------:|:--------------:|:------------:|:-------------:|:----:|
| Intel Xeon @ 2.10GHz | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CPU) | 53.02 | **14** | **11** | **28** | 0.57 |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/0b91e08) (Vulkan/Lavapipe) | **2.75** | 379 | — | 666 | 0.57 |
|  | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ~~0.00~~ | ~~10777~~ | ~~—~~ | ~~—~~ | ~~0.00~~ |
|  | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPU) | ✗ | ✗ | ✗ | ✗ | |
|  | [JAX](https://github.com/jax-ml/jax) (CPU) | ✗ | ✗ | ✗ | ✗ | |
|  | [Burn](https://github.com/tracel-ai/burn) (wgpu) | ✗ | ✗ | ✗ | ✗ | |
|  | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | 13.42 | **3** | **2** | **6** | 0.57 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ~~0.00~~ | ~~5781~~ | ~~—~~ | ~~—~~ | ~~0.00~~ |
| | [Burn](https://github.com/tracel-ai/burn) (wgpu) | — | — | — | — | |
| | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/5ddf5e5) (Vulkan) | **1.29** | 19 | 20 | 29 | 0.57 |
| | [GGML](https://github.com/ggerganov/ggml) (CPU) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPUExecutionProvider) | ~~1.19~~ | ~~14~~ | ~~—~~ | ~~—~~ | ~~0.05~~ |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| Apple M3 | [PyTorch 2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (MPS) | 0.00 | 526 | **10** | 199 | 0.57 |
| | [MLX](https://github.com/ml-explore/mlx) (MLX) | ~~0.00~~ | ~~8~~ | ~~—~~ | ~~10~~ | ~~0.51~~ |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (Metal) | ~~0.01~~ | ~~1541~~ | ~~—~~ | ~~—~~ | ~~0.00~~ |
| | [Burn](https://github.com/tracel-ai/burn) (wgpu) | — | — | — | — | |
| | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/5ddf5e5) (Metal) | 2.77 | **42** | 19 | **6** | 0.57 |
| | [GGML](https://github.com/ggerganov/ggml) (CPU) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CoreMLExecutionProvider) | ~~2.48~~ | ~~9~~ | ~~—~~ | ~~—~~ | ~~0.05~~ |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| NVIDIA GeForce RTX 5080 | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 13.0) | 6.45 | **1** | **1** | **1** | 0.57 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CUDA) | ~~0.01~~ | ~~102~~ | ~~—~~ | ~~—~~ | ~~0.00~~ |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/3e0b489) (Vulkan) | **2.11** | 1 | 1 | 8 | 0.52 |
| | [GGML](https://github.com/ggerganov/ggml) | — | — | — | — | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | ~~0.74~~ | ~~1~~ | ~~—~~ | ~~—~~ | ~~0.05~~ |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | ~~1.15~~ | ~~14~~ | ~~—~~ | ~~31~~ | ~~0.05~~ |
| Intel Graphics (RPL-U) | [PyTorch](https://github.com/pytorch/pytorch) | ✗ | ✗ | ✗ | ✗ | |
|  | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | 0.00 | **16876** | — | — | 0.00 |
|  | [Burn](https://github.com/tracel-ai/burn) (wgpu) | ✗ | ✗ | ✗ | ✗ | |
|  | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | ✗ | ✗ | ✗ | ✗ | |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/77ceb78) (Vulkan) | ~~3.56~~ | ~~44~~ | — | ~~38~~ | ~~0.57~~ |
|  | [llama.cpp](https://github.com/ggml-org/llama.cpp) (CPU) | ✗ | ✗ | ✗ | ✗ | |

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
