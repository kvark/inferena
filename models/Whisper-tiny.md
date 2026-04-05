---
layout: default
title: Whisper-tiny
permalink: /models/Whisper-tiny
---

# Whisper-tiny

[openai/whisper-tiny](https://hf.co/openai/whisper-tiny) — Encoder-decoder transformer for speech recognition. ~39M parameters.

Uses a custom tiny configuration (4 encoder + 4 decoder layers) for fast benchmarking.

## Results

Benchmark config: 30s mel spectrogram (80x3000), 4-token decoder input, float32, random weights.

| Platform | Framework | Compile (s) | Inference (ms) | Latency (ms) | Training (ms) | Loss |
|----------|-----------|:-----------:|:--------------:|:------------:|:-------------:|:----:|
| Intel Xeon @ 2.10GHz | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CPU) | 39.88 | **150** | — | **371** | 11.80 |
| | [ONNX Runtime 1.24.4](https://github.com/microsoft/onnxruntime) (CPU) | **0.84** | 212 | — | — | 11.80 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | ~~0.01~~ | ~~616~~ | ~~—~~ | ~~—~~ | ~~0.00~~ |
| | [Meganeura](https://github.com/kvark/meganeura/tree/2ef151e) (Vulkan/Lavapipe) | ~~7.84~~ | ~~53467~~ | ~~—~~ | ~~—~~ | ~~0.01~~ |
| | [Burn](https://github.com/tracel-ai/burn) (wgpu) | ✗ | ✗ | ✗ | ✗ | |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | ✗ | ✗ | ✗ | ✗ | |
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | 16.54 | 77 | **62** | **220** | 0.01 |
|  | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | **0.01** | 317 | — | — | 0.00 |
|  | [Burn](https://github.com/tracel-ai/burn) (wgpu) | — | — | — | — | |
|  | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | — | — | — | — | |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/5ddf5e5) (Vulkan) | 0.17 | 261 | 244 | — | 0.01 |
|  | [GGML](https://github.com/ggerganov/ggml) (faster-whisper (CTranslate2)) | 7.89 | 235 | 209 | — | 0.00 |
|  | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPUExecutionProvider) | 3.32 | **68** | — | — | 0.01 |
|  | [JAX](https://github.com/jax-ml/jax) (CPU) | 1.66 | 288 | 284 | 594 | 0.01 |
| Apple M3 | [PyTorch 2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (MPS) | 0.00 | **224** | **46** | **127** | 0.01 |
|  | [MLX](https://github.com/ml-explore/mlx) (MLX) | — | — | — | — | |
|  | [Candle](https://github.com/huggingface/candle) (CPU) | 0.02 | 359 | — | — | 0.00 |
|  | [Burn](https://github.com/tracel-ai/burn) (wgpu) | — | — | — | — | |
|  | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | — | — | — | — | |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/5ddf5e5) (Metal) | 0.25 | 665 | 673 | — | 0.01 |
|  | [GGML](https://github.com/ggerganov/ggml) (faster-whisper (CTranslate2)) | 18.61 | 250 | 230 | — | 0.00 |
|  | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPUExecutionProvider) | 4.25 | 261 | — | — | 0.01 |
|  | [JAX](https://github.com/jax-ml/jax) (CPU) | 2.36 | 356 | 352 | 1249 | 0.01 |
| NVIDIA GeForce RTX 5080 | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 13.0) | 3.79 | **2** | **2** | **9** | 0.01 |
|  | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | 0.00 | 201 | — | — | 0.00 |
|  | [Burn](https://github.com/tracel-ai/burn) (wgpu) | — | — | — | — | |
|  | [Luminal](https://github.com/luminal-ai/luminal) (CPU) | — | — | — | — | |
|  | [Meganeura](https://github.com/kvark/meganeura/tree/5ddf5e5) (Vulkan) | 1.55 | 35 | 35 | — | 0.01 |
|  | [GGML](https://github.com/ggerganov/ggml) (faster-whisper (CTranslate2)) | 6.46 | 153 | 156 | — | 0.00 |
|  | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPUExecutionProvider) | 1.64 | 35 | — | — | 0.01 |
|  | [JAX](https://github.com/jax-ml/jax) (CPU) | 1.14 | 140 | 123 | 385 | 0.01 |

**Correctness:** PyTorch vs ONNX Runtime: **PASS** (loss diff 0.0).

## Architecture

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Encoder** | Conv1D frontend | 2 layers (80->384, stride 2->2) |
| | Transformer layers | 4 |
| | Attention heads | 6 |
| | Model dim | 384 |
| | FFN dim | 1536 |
| **Decoder** | Transformer layers | 4 |
| | Cross-attention | encoder->decoder at each layer |
| | Attention heads | 6 |
| | Model dim | 384 |
| | FFN dim | 1536 |
| | Vocab size | 51865 |
| **Input** | Mel spectrogram | 80 bins x 3000 frames (30s) |
| **Parameters** | Total | ~39M |

## What this exercises

Exercises several operations absent from text-only LLMs:

- **Conv1D** — audio frontend (mel spectrogram -> encoder input)
- **Encoder-decoder cross-attention** — not just self-attention
- **Sinusoidal positional encoding** (encoder) + learned positions (decoder)
- **Encoder-decoder architecture** — separate compute graphs with cross-attention bridge
- Tests framework support for multi-modal input processing

## Caveats

- Uses a custom tiny config (4+4 layers, d=384), not the full whisper-tiny from OpenAI
- Input is synthetic mel spectrogram, not real audio
- Decoder runs with a 4-token input (language/task tokens), not full transcription
