#!/usr/bin/env python3
"""JAX benchmark runner for inferena.

Implements SmolLM2 (LLaMA-family) in pure JAX with the same deterministic
weights as other frameworks, then measures JIT-compiled inference and
gradient computation.
"""

import hashlib
import json
import os
import struct
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


# ── Model implementation (LLaMA-family) ──────────────────────────────

def rms_norm(x, weight, eps=1e-5):
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps) * weight


def rope_freqs(head_dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (seq, n_heads, head_dim)
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    # cos, sin: (seq, d2)
    cos = cos[:, None, :]  # (seq, 1, d2)
    sin = sin[:, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def attention(q, k, v):
    # q: (seq, n_heads, head_dim), k: (seq, n_kv_heads, head_dim)
    n_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    head_dim = q.shape[2]

    if n_kv_heads != n_heads:
        rep = n_heads // n_kv_heads
        k = jnp.repeat(k, rep, axis=1)
        v = jnp.repeat(v, rep, axis=1)

    scale = head_dim ** -0.5
    # (n_heads, seq, seq)
    scores = jnp.einsum('shd,thd->hst', q, k) * scale
    # Causal mask
    seq_len = q.shape[0]
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    scores = jnp.where(mask[None, :, :], scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    # (n_heads, seq, head_dim) -> (seq, n_heads, head_dim)
    out = jnp.einsum('hst,thd->shd', weights, v)
    return out


def transformer_block(x, params, config, cos, sin):
    dim = config.dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = dim // n_heads

    # Attention
    h = rms_norm(x, params['attn_norm'])
    q = h @ params['wq']  # (seq, n_heads * head_dim)
    k = h @ params['wk']  # (seq, n_kv_heads * head_dim)
    v = h @ params['wv']

    q = q.reshape(-1, n_heads, head_dim)
    k = k.reshape(-1, n_kv_heads, head_dim)
    v = v.reshape(-1, n_kv_heads, head_dim)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    out = attention(q, k, v)
    out = out.reshape(-1, dim)
    out = out @ params['wo']
    x = x + out

    # FFN (SwiGLU)
    h = rms_norm(x, params['ffn_norm'])
    gate = jax.nn.silu(h @ params['w_gate'])
    up = h @ params['w_up']
    x = x + (gate * up) @ params['w_down']

    return x


def forward(params, config, input_ids):
    seq_len = input_ids.shape[0]
    dim = config.dim
    head_dim = dim // config.n_heads

    x = params['embed'][input_ids]
    cos, sin = rope_freqs(head_dim, seq_len)

    for layer_params in params['layers']:
        x = transformer_block(x, layer_params, config, cos, sin)

    x = rms_norm(x, params['final_norm'])
    logits = x @ params['lm_head']
    return logits


def cross_entropy_loss(logits, labels):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, labels[:, None], axis=-1)
    return -jnp.mean(target_log_probs)


def loss_fn(params, config, input_ids, labels):
    logits = forward(params, config, input_ids)
    return cross_entropy_loss(logits, labels)


# ── Weight loading ───────────────────────────────────────────────────

def load_weights_safetensors(model_dir, config):
    """Load real weights from safetensors."""
    from safetensors import safe_open
    path = os.path.join(model_dir, "model.safetensors")
    tensors = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            tensors[key] = jnp.array(f.get_tensor(key))

    n_layers = config.n_layers
    layers = []
    for i in range(n_layers):
        pfx = f"model.layers.{i}"
        layer = {
            'attn_norm': tensors[f"{pfx}.input_layernorm.weight"],
            'wq': tensors[f"{pfx}.self_attn.q_proj.weight"].T,
            'wk': tensors[f"{pfx}.self_attn.k_proj.weight"].T,
            'wv': tensors[f"{pfx}.self_attn.v_proj.weight"].T,
            'wo': tensors[f"{pfx}.self_attn.o_proj.weight"].T,
            'ffn_norm': tensors[f"{pfx}.post_attention_layernorm.weight"],
            'w_gate': tensors[f"{pfx}.mlp.gate_proj.weight"].T,
            'w_up': tensors[f"{pfx}.mlp.up_proj.weight"].T,
            'w_down': tensors[f"{pfx}.mlp.down_proj.weight"].T,
        }
        layers.append(layer)

    embed = tensors["model.embed_tokens.weight"]
    # Tied weights
    lm_head = embed.T if "lm_head.weight" not in tensors else tensors["lm_head.weight"].T

    return {
        'embed': embed,
        'layers': layers,
        'final_norm': tensors["model.norm.weight"],
        'lm_head': lm_head,
    }


# ── Config ───────────────────────────────────────────────────────────

from collections import namedtuple

ModelConfig = namedtuple('ModelConfig', ['vocab_size', 'dim', 'n_layers', 'n_heads', 'n_kv_heads', 'intermediate'])

CONFIGS = {
    "SmolLM2-135M": ModelConfig(
        vocab_size=49152, dim=576, n_layers=30,
        n_heads=9, n_kv_heads=3, intermediate=1536,
    ),
}


# ── Benchmark ────────────────────────────────────────────────────────

def sha256_f32(data):
    flat = np.asarray(data, dtype=np.float32).flatten()
    raw = struct.pack(f"<{flat.size}f", *flat.tolist())
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def bench(model_name):
    config = CONFIGS.get(model_name)
    if config is None:
        print(f"Unknown model: {model_name}. Available: {list(CONFIGS.keys())}", file=sys.stderr)
        sys.exit(1)

    seq_len = 128
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    model_dir = os.path.join(root_dir, "models", model_name)

    # --- Load weights ---
    print(f"[jax] loading {model_name}...", file=sys.stderr)
    t0 = time.perf_counter()
    local_path = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(local_path):
        params = load_weights_safetensors(model_dir, config)
    else:
        # Fall back to HF hub cache.
        HF_IDS = {"SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M"}
        hf_id = HF_IDS.get(model_name)
        if hf_id is None:
            print(f"[jax] model weights not found at {model_dir}", file=sys.stderr)
            sys.exit(1)
        try:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(hf_id, allow_patterns=["*.safetensors", "*.json"])
            print(f"[jax] using HF cache: {model_dir}", file=sys.stderr)
            params = load_weights_safetensors(model_dir, config)
        except Exception as e:
            print(f"[jax] model weights not found at {model_dir}: {e}", file=sys.stderr)
            sys.exit(1)
    load_s = time.perf_counter() - t0
    print(f"[jax] loaded in {load_s:.2f}s", file=sys.stderr)

    # --- Prepare inputs ---
    input_ids = jnp.arange(seq_len, dtype=jnp.int32)
    labels = jnp.array([(i + 1) % config.vocab_size for i in range(seq_len)], dtype=jnp.int32)

    # --- JIT compile ---
    print("[jax] JIT compiling...", file=sys.stderr)
    compile_t0 = time.perf_counter()
    jit_forward = jax.jit(forward, static_argnums=(1,))
    jit_grad = jax.jit(jax.grad(loss_fn, argnums=0), static_argnums=(1,))

    # Warm-up (triggers XLA compilation)
    logits = jit_forward(params, config, input_ids)
    logits.block_until_ready()
    grads = jit_grad(params, config, input_ids, labels)
    jax.tree.map(lambda x: x.block_until_ready(), grads)
    compile_s = time.perf_counter() - compile_t0
    print(f"[jax] compiled in {compile_s:.2f}s", file=sys.stderr)

    # --- Inference ---
    t0 = time.perf_counter()
    logits = jit_forward(params, config, input_ids)
    logits.block_until_ready()
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # --- Loss ---
    loss = cross_entropy_loss(logits, labels)

    # --- Training (backward) ---
    t0 = time.perf_counter()
    grads = jit_grad(params, config, input_ids, labels)
    jax.tree.map(lambda x: x.block_until_ready(), grads)
    training_ms = (time.perf_counter() - t0) * 1000.0

    # --- Latency (single-token) ---
    lat_input = jnp.array([0], dtype=jnp.int32)
    jit_forward_lat = jax.jit(forward, static_argnums=(1,))
    # Warm-up for new shape
    out = jit_forward_lat(params, config, lat_input)
    out.block_until_ready()
    t0 = time.perf_counter()
    out = jit_forward_lat(params, config, lat_input)
    out.block_until_ready()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # --- Outputs ---
    logits_np = np.asarray(logits, dtype=np.float32)
    logits_hash = sha256_f32(logits_np)
    logits_flat = logits_np.flatten()
    logits_sample = [round(float(v), 6) for v in logits_flat[:16]]

    backend = str(jax.default_backend()).upper()
    devices = jax.devices()
    gpu_name = str(devices[0]) if devices else "cpu"

    result = {
        "framework": "jax",
        "model": model_name,
        "device": gpu_name,
        "gpu_name": gpu_name,
        "jax_version": jax.__version__,
        "backend": backend,
        "timings": {
            "compile_s": round(compile_s, 2),
            "inference_ms": round(inference_ms, 3),
            "latency_ms": round(latency_ms, 3),
            "training_ms": round(training_ms, 3),
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": logits_sample,
            "loss": round(float(loss), 6),
        },
    }
    print(json.dumps(result))


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M"
    bench(model_name)
