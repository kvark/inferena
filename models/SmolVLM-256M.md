---
layout: default
title: SmolVLM-256M
permalink: /models/SmolVLM-256M
---

# SmolVLM-256M

[HuggingFaceTB/SmolVLM-256M-Instruct](https://hf.co/HuggingFaceTB/SmolVLM-256M-Instruct) — 256M parameter vision-language model.

## Results

Benchmark config: seq_len=128, float32, 1 dummy 384×384 image, input=[0,1,...,127].

| CPU / GPU | Framework | Compile (s) | Forward (ms) | Backward (ms) | Loss |
|-----------|-----------|:-----------:|:------------:|:--------------:|:----:|
| Intel Xeon @ 2.10GHz | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch) | 68.18 | 54492 | 30852 | 10.89 |
| | [Burn](https://github.com/tracel-ai/burn/tree/ed72d2b) | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal/tree/f32161d) | ✗ | ✗ | ✗ | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/550bb6c) | ✗ | ✗ | ✗ | |

Only PyTorch supports this model so far.

## Architecture

Idefics3: SigLIP vision encoder + SmolLM2-135M language backbone + MLP connector.

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Vision encoder** (SigLIP-B/16) | Patch size | 16×16 |
| | Hidden size | 768 |
| | Layers | 12 |
| | Heads | 12 |
| | Image resolution | 384×384 |
| **Connector** | Type | MLP + pixel-shuffle (9× token reduction) |
| **Language model** (SmolLM2-135M) | Hidden size | 576 |
| | Layers | 30 |
| | Heads | 9 (3 KV) |

## What this exercises

Everything in [SmolLM2-135M](SmolLM2-135M.md), plus:

- **Vision encoder**: patch embedding (Conv2D), ViT self-attention, LayerNorm
- **Pixel-shuffle compression**: space-to-depth (384×384 → 81 tokens per patch)
- **MLP connector**: projects vision embeddings to language model dimension
- **Cross-modal fusion**: vision tokens concatenated with text tokens in the LM

## Caveats

- Only **PyTorch** supports this architecture (via HF transformers).
- **Meganeura** has `smolvlm2` in its repo but isn't wired in the runner yet.
- **Burn** and **Luminal** do not implement this architecture.
- Vision input is a dummy random tensor — no real image preprocessing.
