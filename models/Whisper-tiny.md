---
layout: default
title: Whisper-tiny
permalink: /models/Whisper-tiny
---

# Whisper-tiny encoder

[openai/whisper-tiny](https://hf.co/openai/whisper-tiny) supplies the
configuration, but the audited workload is the four-layer **encoder only**:
8,208,384 total parameters, of which 7,632,384 are trainable. It is not a
full speech-to-text pipeline.

## Results

Benchmark config: batch=1, synthetic 30-second mel spectrogram (80×3000),
float32, deterministic matched weights, and mean-square encoder-output loss.
The positional embedding is frozen in both implementations.

The table below predates the current timing and validation metadata and
includes runners that measured older or different Whisper workloads.
Regenerate it before using the numbers in a publication.

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
| NVIDIA GeForce RTX 5080 | [PyTorch 2.13.0+cu130](https://github.com/pytorch/pytorch/releases/tag/v2.13.0) (CUDA 13.0) | 4.14 | 2.3 | 2.4 | 9.7 | 0.00 |
| | [Candle](https://github.com/huggingface/candle/tree/31f35b1) (CUDA) | **0.02** | 45 | — | — | 0.00 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/a7ced10) (Vulkan) | 0.21 | 3.4 | 3.4 | 12 | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) (faster-whisper (CTranslate2, CUDA)) | 7.98 | 15 | 15 | — | 0.00 |
| | [ONNX Runtime 1.27.0](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | 2.04 | 3.6 | — | — | 0.01 |
| | [MAX](https://github.com/modular/modular) | — | — | — | — | |
| | [JAX 0.11.0](https://github.com/jax-ml/jax) (GPU) | 17.39 | **2.1** | **1.6** | **5.6** | 0.01 |
| NVIDIA GeForce RTX 3050 (Windows) | [PyTorch 2.11.0+cu128](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CUDA 12.8) | 0.00 | **13** | **13** | **43** | 0.00 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/ef9c251) (Vulkan/DX12) | 0.66 | 19 | 19 | 43 | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) (faster-whisper (CTranslate2, CUDA)) | 7.00 | 40 | 45 | — | 0.00 |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CUDAExecutionProvider) | 4.82 | 20 | — | — | 0.01 |
| | [JAX](https://github.com/jax-ml/jax) | ✗ | ✗ | ✗ | ✗ | |
| Intel(R) Graphics (RPL-U) | [PyTorch 2.11.0+xpu](https://github.com/pytorch/pytorch/releases/tag/v2.11.0) (CPU) | 0.00 | 477 | **420** | **899** | 0.00 |
| | [Candle](https://github.com/huggingface/candle/tree/6b4d8a1) (CPU) | 0.02 | 795 | — | — | 0.00 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | — | — | — | — | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/8042e00) (Vulkan) | 0.39 | 467 | 466 | 1594 | 0.01 |
| | [GGML](https://github.com/ggerganov/ggml) (faster-whisper (CTranslate2, CPU)) | 14.76 | 1036 | 1104 | — | 0.00 |
| | [ONNX Runtime](https://github.com/microsoft/onnxruntime) (CPUExecutionProvider) | 6.18 | **333** | — | — | 0.01 |
| | [MAX](https://github.com/modular/modular) | ✗ | ✗ | ✗ | ✗ | |
| | [JAX](https://github.com/jax-ml/jax) (CPU) | 5.59 | 717 | 686 | 2681 | 0.01 |
| AMD Radeon RX 7900 XT | [PyTorch 2.10.0+rocm7.1](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) (ROCm 7.1.25424) | 5.38 | 12 | 6.5 | 44 | 0.00 |
| | [Burn](https://github.com/tracel-ai/burn) | — | — | — | — | |
| | [Inferi](https://github.com/dimforge/inferi) | ✗ | ✗ | ✗ | ✗ | |
| | [Luminal](https://github.com/luminal-ai/luminal) | — | — | — | — | |
| | [Meganeura](https://github.com/kvark/meganeura/tree/8042e00) (Vulkan) | 0.82 | **4.8** | **4.8** | **21** | 0.01 |
| | [MAX](https://github.com/modular/modular) | — | — | — | — | |

**Historical correctness:** the old table's PyTorch/ONNX Runtime `PASS`
applies to the prior workload and validator, not the audited encoder
comparison.

## Architecture

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Encoder** | Conv1D frontend | 2 layers (80->384, stride 2->2) |
| | Transformer layers | 4 |
| | Attention heads | 6 |
| | Model dim | 384 |
| | FFN dim | 1536 |
| **Input** | Mel spectrogram | 80 bins x 3000 frames (30s) |
| **Objective** | Training loss | Mean square of encoder output |
| **Parameters** | Total / trainable | 8,208,384 / 7,632,384 |

## What this exercises

Exercises several operations absent from text-only LLMs:

- **Conv1D** — audio frontend (mel spectrogram -> encoder input)
- **Full bidirectional self-attention** over 1,500 encoded positions
- **Learned positional embedding**, frozen as in the reference encoder
- **LayerNorm and GELU** in an audio-transformer workload
- Tests the same training-capable compiler on a long-sequence speech encoder

## Caveats

- Uses the Whisper-tiny encoder shape (4 layers, d=384), not its decoder
- Input is synthetic mel spectrogram, not real audio
- The objective is a synthetic differentiable systems workload, not a
  transcription-quality evaluation
