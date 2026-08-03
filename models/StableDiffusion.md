---
layout: default
title: Scaled Stable Diffusion 1.x U-Net
permalink: /models/StableDiffusion
---

# Scaled Stable Diffusion 1.x U-Net

The CLI retains the historical `StableDiffusion` workload name. The matched
PyTorch/Meganeura workload is now a scaled, conditioned latent-diffusion U-Net,
not the earlier generic convolutional U-Net and not the complete Stable
Diffusion 1.5 pipeline.

It retains the denoiser features that materially change the compiler workload:
timestep conditioning in every residual block, spatial self-attention,
77-token text cross-attention, transformer feed-forward blocks, convolutional
down/up paths, GroupNorm, and U-Net skip connections.

## Results

The previous table measured the superseded 5.29M-parameter convolution-only
workload and has been removed to prevent accidental comparison. This empty
table preserves the validated results schema and insertion point for the next
frozen run.

| Platform | Framework | Compile (s) | Inference (ms) | Latency (ms) | Training (ms) | Loss |
|----------|-----------|:-----------:|:--------------:|:------------:|:-------------:|:----:|

Freeze one Meganeura revision and rerun both practical-default and strict
configurations on every platform before publishing new results:

```bash
./run.sh -m StableDiffusion -f pytorch,meganeura
./run.sh -m StableDiffusion -f pytorch,meganeura --strict
```

During development, strict-f32 cross-engine validation on an NVIDIA RTX 5080
reached approximately `1.2e-5` forward relative L2 error and `8.1e-5`
per-parameter gradient-vector relative L2 error. These are correctness smoke
results from a dirty development revision, not frozen performance results.

The legacy Candle runner uses a much larger SD 1.5-like U-Net and is not
comparable to this matched workload.

## Matched architecture

| Component | Parameter | Value |
|-----------|-----------|-------|
| **U-Net** | Input/output channels | 4 (latent space) |
| | Base channels | 64 |
| | Channel multipliers | [1, 2, 4] |
| | Levels | 3 |
| | GroupNorm groups | 16 |
| | Parameters | 10,928,768 |
| **Conditioning** | Timestep input / MLP width | 64 / 256 |
| | Text context | 77 × 768 |
| **Attention** | Spatial resolutions | 16×16 and 8×8, plus middle block |
| | Head width | 32 |
| | Blocks | self-attention + text cross-attention + GELU FFN |
| **Input** | Latent | batch 1, 4 × 32 × 32 |
| | Training objective | MSE noise prediction |

Batch 1 is currently required because Meganeura's differentiable attention
primitive represents one sequence per operation. For this workload, inference
and latency therefore use the same shape.

## What this exercises

- 3×3 and 1×1 Conv2D, including stride-2 downsampling
- nearest-neighbor upsampling and channel-wise U-Net skip concatenation
- GroupNorm and token-wise LayerNorm
- timestep projection and spatial broadcast into residual blocks
- non-causal self-attention and 77-token cross-attention
- dense GELU feed-forward expansion and contraction
- end-to-end autodiff through the combined convolution/attention graph

This operator mix is intentionally complementary to the transformer-only
SmolLM2 and SmolVLA workloads.

## Scope and caveats

- This is a scaled research workload, not an SD 1.5 checkpoint-compatible
  architecture. SD 1.5's U-Net alone is roughly two orders of magnitude larger
  and contains more blocks.
- Only the denoising U-Net is represented. The VAE, CLIP text encoder,
  scheduler, classifier-free guidance, and iterative denoising loop are out of
  scope.
- The feed-forward activation is GELU rather than SD 1.x's GEGLU because the
  reduced workload stays within Meganeura's current primitive set.
- Inputs and parameters are deterministic synthetic values. The benchmark
  checks matched computation and gradients; it does not generate an image or
  make a quality claim.
- Normalization scales use identity initialization and normalization biases
  use zero initialization. Other parameters use matched canonical-name-seeded
  values at scale 0.02.
