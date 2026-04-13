#!/usr/bin/env python3
"""Convert safetensors LLaMA model to GGUF format for llama-cpp-python.

Minimal converter that bypasses llama.cpp's tokenizer recognition.
Only supports LLaMA-family models (SmolLM2, etc.) in f32.
"""

import json
import os
import struct
import sys

import numpy as np


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.gguf>", file=sys.stderr)
        sys.exit(1)

    model_dir = sys.argv[1]
    out_path = sys.argv[2]

    # Load config.
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    intermediate_size = config["intermediate_size"]
    max_pos = config.get("max_position_embeddings", 2048)
    rms_eps = config.get("rms_norm_eps", 1e-5)
    rope_theta = config.get("rope_theta", 10000.0)
    head_dim = hidden_size // n_heads

    # Load safetensors.
    import torch
    from safetensors.torch import load_file
    st_path = os.path.join(model_dir, "model.safetensors")
    raw = load_file(st_path)
    tensors = {k: v.float().numpy() for k, v in raw.items()}

    # Map HF tensor names to GGUF names.
    gguf_tensors = {}
    gguf_tensors["token_embd.weight"] = tensors["model.embed_tokens.weight"]
    if "model.norm.weight" in tensors:
        gguf_tensors["output_norm.weight"] = tensors["model.norm.weight"]
    if "lm_head.weight" in tensors:
        gguf_tensors["output.weight"] = tensors["lm_head.weight"]

    for i in range(n_layers):
        pfx = f"model.layers.{i}"
        bpfx = f"blk.{i}"
        mapping = {
            f"{pfx}.self_attn.q_proj.weight": f"{bpfx}.attn_q.weight",
            f"{pfx}.self_attn.k_proj.weight": f"{bpfx}.attn_k.weight",
            f"{pfx}.self_attn.v_proj.weight": f"{bpfx}.attn_v.weight",
            f"{pfx}.self_attn.o_proj.weight": f"{bpfx}.attn_output.weight",
            f"{pfx}.mlp.gate_proj.weight": f"{bpfx}.ffn_gate.weight",
            f"{pfx}.mlp.up_proj.weight": f"{bpfx}.ffn_up.weight",
            f"{pfx}.mlp.down_proj.weight": f"{bpfx}.ffn_down.weight",
            f"{pfx}.input_layernorm.weight": f"{bpfx}.attn_norm.weight",
            f"{pfx}.post_attention_layernorm.weight": f"{bpfx}.ffn_norm.weight",
        }
        for hf_name, gguf_name in mapping.items():
            if hf_name in tensors:
                gguf_tensors[gguf_name] = tensors[hf_name]

    # Write GGUF using the gguf-py library.
    try:
        from gguf import GGUFWriter, GGMLQuantizationType
    except ImportError:
        print("[llama-cpp] gguf package not available — pip install gguf", file=sys.stderr)
        sys.exit(1)

    _write_with_gguf_py(out_path, config, gguf_tensors, vocab_size,
                        hidden_size, n_layers, n_heads, n_kv_heads,
                        intermediate_size, max_pos, rms_eps, rope_theta,
                        head_dim)


def _write_with_gguf_py(out_path, config, gguf_tensors, vocab_size,
                         hidden_size, n_layers, n_heads, n_kv_heads,
                         intermediate_size, max_pos, rms_eps, rope_theta,
                         head_dim):
    from gguf import GGUFWriter, GGMLQuantizationType

    writer = GGUFWriter(out_path, "llama")

    # Architecture metadata.
    writer.add_context_length(max_pos)
    writer.add_embedding_length(hidden_size)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_layer_norm_rms_eps(rms_eps)
    writer.add_rope_freq_base(rope_theta)
    writer.add_file_type(0)  # f32

    # Minimal vocab — llama.cpp needs tokens but we only care about weights.
    tokens = [f"<tok_{i}>".encode() for i in range(vocab_size)]
    scores = [0.0] * vocab_size
    token_types = [1] * vocab_size  # NORMAL
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    # At least one merge needed — llama.cpp refuses to load GPT2 vocab without merges.
    writer.add_token_merges(["<tok_0> <tok_1>"])

    # Tensors.
    for name, data in gguf_tensors.items():
        data = data.astype(np.float32)
        writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"[llama-cpp] wrote {out_path} ({len(gguf_tensors)} tensors)", file=sys.stderr)


if __name__ == "__main__":
    main()
