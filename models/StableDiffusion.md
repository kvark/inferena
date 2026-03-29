---
layout: default
title: Stable Diffusion 1.5
permalink: /models/StableDiffusion
---

# Stable Diffusion 1.5

[stable-diffusion-v1-5/stable-diffusion-v1-5](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5) — Latent diffusion model for text-to-image generation.

## Results

Benchmark config: UNet single denoising step, 512×512 latent (64×64×4), float32, random weights.

| Platform | Framework | Compile (s) | Forward (ms) | Backward (ms) | Loss |
|----------|-----------|:-----------:|:------------:|:--------------:|:----:|
| | | | | | |

*Run `./run.sh -m StableDiffusion` to populate this table.*

## Architecture

The benchmark measures the **UNet denoising backbone** — the compute-intensive core of Stable Diffusion.

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
| **Text encoder** | Type | CLIP ViT-L/14 |
| | Hidden size | 768 |
| **VAE** | Not benchmarked | (encode/decode is cheap) |

## What this exercises

Completely different compute profile from SmolLM2 / SmolVLA:

- **Conv2D** (not present in transformer-only models) — spatial convolutions in UNet
- **Cross-attention** between image features and text conditioning
- **Skip connections** (U-Net architecture) — memory-intensive residual paths
- **GroupNorm** (not RMSNorm/LayerNorm)
- **Time embedding** projection (sinusoidal → MLP)
- **Spatial self-attention** at multiple resolutions
- Tests Conv2D kernel performance, a major framework differentiator

## Caveats

- Only the **UNet** is benchmarked (not VAE encode/decode or text encoding).
- Input is random latent noise + random text embedding — no actual image generation.
- Framework support varies: PyTorch (diffusers), Candle (in-repo), Burn (community port).
