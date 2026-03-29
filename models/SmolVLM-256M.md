# SmolVLM-256M

[HuggingFaceTB/SmolVLM-256M-Instruct](https://hf.co/HuggingFaceTB/SmolVLM-256M-Instruct) — 256M parameter vision-language model.

## Architecture

Idefics3 architecture: SigLIP vision encoder + SmolLM2-135M language backbone + MLP connector.

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
- **Pixel-shuffle compression**: space-to-depth rearrangement (384×384 → 81 tokens per patch)
- **MLP connector**: projects vision embeddings to language model dimension
- **Cross-modal fusion**: vision tokens concatenated with text tokens in the LM

This model stresses framework support for vision pipelines, Conv2D, and
heterogeneous input modalities — areas not tested by text-only SmolLM2.

## Benchmark caveats

- Only **PyTorch** currently supports this model fully (via HF transformers).
- **Meganeura** has `smolvlm2` support but is not yet wired in the benchmark runner.
- **Burn** and **Luminal** do not implement this architecture.
- Frameworks that don't support a model are reported as `✗`.
- Vision input is a dummy tensor (no real image preprocessing benchmarked yet).

## Results

Benchmark config: seq_len=128, float32, 1 dummy 384x384 image, input=[0,1,...,127].

| CPU / GPU | Framework | Compile (s) | Forward (ms) | Backward (ms) | Loss |
|-----------|-----------|:-----------:|:------------:|:--------------:|:----:|
| Intel Xeon @ 2.10GHz | PyTorch 2.11.0+cu130 (torch.compile) | 68.18 | 54492 | 30852 | 10.89 |
| | Burn | ✗ | ✗ | ✗ | |
| | Luminal | ✗ | ✗ | ✗ | |
| | Meganeura | ✗ | ✗ | ✗ | |
