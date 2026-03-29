---
layout: default
title: SmolLM2-135M
permalink: /models/SmolLM2-135M
---

# SmolLM2-135M

[HuggingFaceTB/SmolLM2-135M](https://hf.co/HuggingFaceTB/SmolLM2-135M) — 134.5M parameter decoder-only language model.

## Results

Benchmark config: seq_len=128, float32, input=[0,1,...,127].

| CPU / GPU | Framework | Compile (s) | Forward (ms) | Backward (ms) | Loss |
|-----------|-----------|:-----------:|:------------:|:--------------:|:----:|
| Intel Xeon @ 2.10GHz (Lavapipe) | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch) | 124.23 | 58081 | 31163 | 10.98 |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) | ~~0.00~~ | ~~2779~~ | ~~5814~~ | ~~11.82~~ |
| | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) | ~~4.17~~ | ~~18028~~ | ~~17848~~ | ~~10.81~~ |
| | [Meganeura](https://github.com/kvark/meganeura/tree/550bb6c) | **2.20** | **4127** | **3920** | 10.98 |

**Correctness:** PyTorch vs Meganeura: **PASS** (max error 1.7e-6, loss diff 8.6e-4).
Struck-through values are from frameworks running a different (simplified) model.

## Architecture

LLaMA-family transformer with Grouped Query Attention:

| Parameter | Value |
|-----------|-------|
| Hidden size | 576 |
| Layers | 30 |
| Attention heads | 9 (3 KV heads, GQA) |
| FFN intermediate | 1536 |
| Vocab size | 49152 |
| Context length | 2048 |
| Activations | SiLU / SwiGLU |
| Normalization | RMSNorm |
| Position encoding | RoPE |

## What this exercises

- Matrix multiplications (Q/K/V projections, FFN up/gate/down)
- Grouped Query Attention with causal masking
- Rotary Position Embeddings
- RMSNorm (2 per layer, 60 total)
- SwiGLU activation fusion
- Embedding lookup + tied lm_head

## Caveats

- **PyTorch** and **Meganeura** load real model weights and run the full
  architecture — their outputs match.
- **Burn** and **Luminal** use a simplified model (single-head attention,
  no RoPE/RMSNorm) with random weights. Their timings are struck through.
- Luminal backward is estimated as a second forward pass (training graph
  not yet wired).
