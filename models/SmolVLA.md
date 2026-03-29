---
layout: default
title: SmolVLA
permalink: /models/SmolVLA
---

# SmolVLA Action Expert

[lerobot/smolvla_base](https://hf.co/lerobot/smolvla_base) — SmolVLA action expert decoder for robotics.

## Results

Benchmark config: chunk_size=50, vlm_seq_len=16, float32, random weights, MSE loss.

| Platform | Framework | Compile (s) | Forward (ms) | Backward (ms) | Loss |
|----------|-----------|:-----------:|:------------:|:--------------:|:----:|
| Intel Xeon @ 2.10GHz | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CPU) | 37.26 | 15475 | 8184 | 0.00 |
| | [Meganeura](https://github.com/kvark/meganeura/tree/550bb6c) (Vulkan) | **0.16** | **903** | **4280** | 0.00 |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) | ✗ | ✗ | ✗ | |
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | 20.10 | 25 | **43** | 0.00 |
|  | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ✗ | ✗ | ✗ | |
|  | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) (wgpu) | ✗ | ✗ | ✗ | |
|  | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) (CPU) | ✗ | ✗ | ✗ | |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/a480c6d) (Vulkan) | **0.34** | **20** | 47 | 0.00 |
|  | [llama.cpp](https://github.com/ggml-org/llama.cpp) (CPU) | ✗ | ✗ | ✗ | |

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
