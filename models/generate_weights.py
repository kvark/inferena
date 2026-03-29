#!/usr/bin/env python3
"""Generate random-init model weights for benchmarking without network access.

Usage: python3 models/generate_weights.py SmolLM2-135M [SmolVLM-256M ...]
"""

import os
import sys

MODELS = {
    "SmolLM2-135M": {
        "config_cls": "LlamaConfig",
        "model_cls": "LlamaForCausalLM",
        "config_args": dict(
            vocab_size=49152, hidden_size=576, num_hidden_layers=30,
            num_attention_heads=9, num_key_value_heads=3,
            intermediate_size=1536, max_position_embeddings=2048,
            rms_norm_eps=1e-5, tie_word_embeddings=True,
        ),
    },
    "SmolLM2-360M": {
        "config_cls": "LlamaConfig",
        "model_cls": "LlamaForCausalLM",
        "config_args": dict(
            vocab_size=49152, hidden_size=960, num_hidden_layers=32,
            num_attention_heads=15, num_key_value_heads=5,
            intermediate_size=2560, max_position_embeddings=2048,
            rms_norm_eps=1e-5, tie_word_embeddings=True,
        ),
    },
    "SmolLM2-1.7B": {
        "config_cls": "LlamaConfig",
        "model_cls": "LlamaForCausalLM",
        "config_args": dict(
            vocab_size=49152, hidden_size=2048, num_hidden_layers=24,
            num_attention_heads=32, num_key_value_heads=32,
            intermediate_size=8192, max_position_embeddings=2048,
            rms_norm_eps=1e-5, tie_word_embeddings=True,
        ),
    },
}


def generate(name: str):
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models as tok_models, pre_tokenizers

    spec = MODELS.get(name)
    if spec is None:
        print(f"Unknown model: {name}. Available: {list(MODELS.keys())}", file=sys.stderr)
        return False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    # Import the right config/model classes.
    import transformers
    config_cls = getattr(transformers, spec["config_cls"])
    model_cls = getattr(transformers, spec["model_cls"])

    # Generate config.
    config = config_cls(**spec["config_args"])
    config.save_pretrained(out_dir)
    print(f"  {name}/config.json", file=sys.stderr)

    # Generate random-init weights.
    torch.manual_seed(0)
    model = model_cls(config)
    state = {k: v.contiguous() for k, v in model.state_dict().items()
             if k != "lm_head.weight"}  # Tied with embed_tokens.
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name}/model.safetensors ({len(state)} tensors, {n_params:,} params)", file=sys.stderr)

    # Generate minimal tokenizer.
    vocab_size = spec["config_args"]["vocab_size"]
    tok = Tokenizer(tok_models.BPE(
        vocab={f"<tok_{i}>": i for i in range(vocab_size)}, merges=[],
    ))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    print(f"  {name}/tokenizer.json", file=sys.stderr)

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model> [model ...]", file=sys.stderr)
        print(f"Available: {list(MODELS.keys())}", file=sys.stderr)
        sys.exit(1)

    ok = True
    for name in sys.argv[1:]:
        print(f"Generating {name}...", file=sys.stderr)
        if not generate(name):
            ok = False

    sys.exit(0 if ok else 1)
