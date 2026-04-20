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
| AMD Radeon 890M Graphics | [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.2.53210) | 17.09 | 79 | 63 | 220 | 0.00 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Vulkan) | 0.20 | 34 | **33** | **101** | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) | ✗ | ✗ | ✗ | ✗ | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIGraphXExecutionProvider) | 23.58 | **32** | — | — | 0.01 |
| Apple M3 | [PyTorch 2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (MPS) | 0.00 | 318 | **41** | **127** | 0.00 |
| | [MLX](https://github.com/ml-explore/mlx) | — | — | — | — | |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (Metal) | 0.01 | **22** | — | — | 0.00 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Metal) | 0.15 | 406 | 415 | 1062 | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) | ✗ | ✗ | ✗ | ✗ | |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CoreMLExecutionProvider) | 7.93 | 440 | — | — | 0.01 |
| | [JAX](https://github.com/jax-ml/jax) (METAL) | 2.17 | 128 | 315 | 445 | 0.01 |
| NVIDIA GeForce RTX 5080 | [PyTorch 2.11.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 13.0) | 4.04 | **2.3** | **2.1** | 13 | 0.00 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CUDA) | **0.01** | 44 | — | — | 0.00 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/8042e00) (Vulkan) | 0.48 | 2.9 | 2.8 | **12** | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) (faster-whisper (CTranslate2, CUDA)) | 7.05 | 19 | 19 | — | 0.00 |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | 1.94 | 3.5 | — | — | 0.01 |
| | [MAX](https://github.com/modular/modular) | — | — | — | — | |
| NVIDIA GeForce RTX 3050 (Windows) | [PyTorch 2.11.0+cu128](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 12.8) | 0.00 | **13** | **13** | **43** | 0.00 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Vulkan/DX12) | 0.66 | 19 | 19 | 43 | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) (faster-whisper (CTranslate2, CUDA)) | 7.00 | 40 | 45 | — | 0.00 |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | 4.82 | 20 | — | — | 0.01 |
| | [JAX](https://github.com/jax-ml/jax) | ✗ | ✗ | ✗ | ✗ | |

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
