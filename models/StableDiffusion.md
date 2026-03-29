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

| Platform | Framework | Compile (s) | Forward (ms) | Backward (ms) | Loss |
|----------|-----------|:-----------:|:------------:|:--------------:|:----:|
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | 5.53 | 4578 | 6586 | 0.57 |
|  | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ~~0.00~~ | ~~5305~~ | ~~—~~ | ~~0.00~~ |
|  | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu) | ✗ | ✗ | ✗ | |
|  | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | ✗ | ✗ | ✗ | |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/a480c6d) (Vulkan) | **1.17** | **54** | **271** | 0.57 |
|  | [llama.cpp](https://github.com/ggml-org/llama.cpp) (CPU) | ✗ | ✗ | ✗ | |


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
