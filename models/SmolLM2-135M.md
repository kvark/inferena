# SmolLM2-135M

[HuggingFaceTB/SmolLM2-135M](https://hf.co/HuggingFaceTB/SmolLM2-135M) — 134.5M parameter decoder-only language model.

## Architecture

LLaMA-family transformer with Grouped Query Attention:

| Parameter | Value |
|-----------|-------|
| Hidden size | 576 |
| Layers | 30 |
| Attention heads | 9 (3 KV heads) |
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

This is a text-only LLM — no vision or cross-modal components.

## Benchmark caveats

- **PyTorch** and **Meganeura** load the real model weights and run the full
  architecture. Their outputs should match (verified by the harness).
- **Burn** and **Luminal** currently use a simplified model (single-head
  attention, no RoPE/RMSNorm) with random weights. Their forward times are
  **not comparable** to PyTorch/Meganeura until the implementations are upgraded.
  The harness flags these as **DIFFERENT MODEL**.
- Backward for Luminal is estimated as a second forward pass (training graph
  not yet wired).

## Results

Benchmark config: seq_len=128, float32, input=[0,1,...,127].

| CPU / GPU | Framework | Compile (s) | Forward (ms) | Backward (ms) | Loss |
|-----------|-----------|:-----------:|:------------:|:--------------:|:----:|
| Intel Xeon @ 2.10GHz (Lavapipe) | PyTorch 2.11.0+cu130 (torch.compile) | 79.23 | 33968 | 18159 | 10.98 |
| Intel Xeon @ 2.10GHz (Lavapipe) | Burn `ed72d2b` (wgpu) | 0.00 | 2148 | 4191 | 11.69 |
| Intel Xeon @ 2.10GHz | Luminal `f32161d` (CPU) | 3.80 | 11267 | 11202 | 10.81 |
| Intel Xeon @ 2.10GHz (Lavapipe) | Meganeura `550bb6c` (blade) | 1.58 | 3374 | 3009 | 10.98 |

**Correctness:** PyTorch vs Meganeura: **PASS** (max error 1.7e-6, loss diff 8.6e-4).
