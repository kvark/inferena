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

ALL_MODELS = ["SmolLM2-135M", "ResNet-50", "Whisper-tiny"]


# ── Benchmark ────────────────────────────────────────────────────────

def sha256_f32(data):
    flat = np.asarray(data, dtype=np.float32).flatten()
    raw = struct.pack(f"<{flat.size}f", *flat.tolist())
    return "sha256:" + hashlib.sha256(raw).hexdigest()


# ── Name-seeded init (matches PyTorch/meganeura) ─────────────────

def _name_seed(name: str) -> float:
    h = 0
    for c in name.encode('ascii'):
        h = ((h * 31) + c) & 0xFFFFFFFF
    return float(h % 10000)


def _init_param(name, shape, scale=0.1):
    seed = _name_seed(name)
    n = 1
    for s in shape:
        n *= s
    return (jnp.sin(jnp.arange(n, dtype=jnp.float32) * 0.01 + seed) * scale).reshape(shape)


def _init_transposed(name, out_dim, in_dim, scale=0.1):
    """Init in PyTorch [out, in] layout, return as [in, out] for JAX matmul."""
    seed = _name_seed(name)
    w_flat = jnp.sin(jnp.arange(out_dim * in_dim, dtype=jnp.float32) * 0.01 + seed) * scale
    return w_flat.reshape(out_dim, in_dim).T


# ── ResNet-50 in pure JAX ────────────────────────────────────────

def _conv2d(x, w, stride=1, padding=0):
    """Conv2d: x=[N,C,H,W], w=[Co,Ci,kH,kW]."""
    # JAX conv expects [N,H,W,C] — transpose
    x = jnp.transpose(x, (0, 2, 3, 1))
    w = jnp.transpose(w, (2, 3, 1, 0))  # [kH,kW,Ci,Co]
    out = jax.lax.conv_general_dilated(
        x, w, window_strides=(stride, stride),
        padding=[(padding, padding), (padding, padding)],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return jnp.transpose(out, (0, 3, 1, 2))


def _max_pool(x, k=3, stride=2, pad=1):
    x = jnp.transpose(x, (0, 2, 3, 1))
    x = jnp.pad(x, ((0,0),(pad,pad),(pad,pad),(0,0)), constant_values=-jnp.inf)
    out = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max,
                                 (1,k,k,1), (1,stride,stride,1), 'VALID')
    return jnp.transpose(out, (0, 3, 1, 2))


def _bottleneck(x, params, stride, has_ds):
    h = _conv2d(x, params['conv1'], stride=1, padding=0)
    h = jax.nn.relu(h)
    h = _conv2d(h, params['conv2'], stride=stride, padding=1)
    h = jax.nn.relu(h)
    h = _conv2d(h, params['conv3'], stride=1, padding=0)
    shortcut = _conv2d(x, params['ds'], stride=stride, padding=0) if has_ds else x
    return jax.nn.relu(h + shortcut)


# Static block config: (stride, has_downsample) per block — not traced by JIT.
_RESNET50_BLOCK_CFG = []
_stages = [(64,64,256,1,3), (256,128,512,2,4), (512,256,1024,2,6), (1024,512,2048,2,3)]
for _si, (_ic, _mc, _oc, _fs, _n) in enumerate(_stages):
    for _i in range(_n):
        _s = _fs if _i == 0 else 1
        _c = _ic if _i == 0 else _oc
        _RESNET50_BLOCK_CFG.append((_s, _s > 1 or _c != _oc))
_RESNET50_BLOCK_CFG = tuple(_RESNET50_BLOCK_CFG)


def _resnet50_forward(params, images):
    x = _conv2d(images, params['conv1'], stride=2, padding=3)
    x = jax.nn.relu(x)
    x = _max_pool(x, 3, 2, 1)
    for bp, (stride, has_ds) in zip(params['blocks'], _RESNET50_BLOCK_CFG):
        x = _bottleneck(x, bp, stride, has_ds)
    x = x.mean(axis=(2, 3))  # global avg pool -> [N, C]
    x = x @ params['fc_w'] + params['fc_b']
    return x


def _build_resnet50_params():
    scale = 0.01
    params = {'conv1': _init_param('conv1.weight', (64, 3, 7, 7), scale)}
    blocks = []
    stages = [(64,64,256,1,3), (256,128,512,2,4), (512,256,1024,2,6), (1024,512,2048,2,3)]
    for stage_idx, (in_c, mid_c, out_c, first_stride, n) in enumerate(stages):
        for i in range(n):
            stride = first_stride if i == 0 else 1
            ic = in_c if i == 0 else out_c
            prefix = f"layer{stage_idx+1}.{i}"
            bp = {
                'conv1': _init_param(f'{prefix}.conv1.weight', (mid_c, ic, 1, 1), scale),
                'conv2': _init_param(f'{prefix}.conv2.weight', (mid_c, mid_c, 3, 3), scale),
                'conv3': _init_param(f'{prefix}.conv3.weight', (out_c, mid_c, 1, 1), scale),
            }
            if stride > 1 or ic != out_c:
                bp['ds'] = _init_param(f'{prefix}.downsample.0.weight', (out_c, ic, 1, 1), scale)
            blocks.append(bp)
    params['blocks'] = blocks
    params['fc_w'] = _init_transposed('fc.weight', 1000, 2048, scale)
    params['fc_b'] = _init_param('fc.bias', (1000,), scale)
    return params


# ── Whisper encoder in pure JAX ──────────────────────────────────

def _layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * w + b


def _whisper_attention(x, params, n_heads):
    seq_len, d = x.shape
    head_dim = d // n_heads
    q = (x @ params['wq'] + params['q_b']).reshape(seq_len, n_heads, head_dim)
    k = (x @ params['wk']).reshape(seq_len, n_heads, head_dim)
    v = (x @ params['wv'] + params['v_b']).reshape(seq_len, n_heads, head_dim)
    scores = jnp.einsum('shd,thd->hst', q, k) * (head_dim ** -0.5)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum('hst,thd->shd', weights, v).reshape(seq_len, d)
    return out @ params['wo'] + params['o_b']


def _whisper_encoder_forward(params, mel):
    # Conv stem: mel is [1, 80, 3000]
    x = _conv2d(mel.reshape(1, 80, 3000, 1),
                params['conv1_w'].reshape(-1, 80, 3, 1), stride=1, padding=1)
    x = x + params['conv1_b'].reshape(1, -1, 1, 1)
    x = jax.nn.gelu(x, approximate=False)
    x = _conv2d(x, params['conv2_w'].reshape(-1, 384, 3, 1), stride=2, padding=1)
    x = x + params['conv2_b'].reshape(1, -1, 1, 1)
    x = jax.nn.gelu(x, approximate=False)
    # x is [1, d, seq_len, 1] -> [seq_len, d]
    x = x[0, :, :, 0].T  # [d, seq_len] -> [seq_len, d]
    x = x + params['pos_embed']
    for lp in params['layers']:
        # Self-attention + residual
        h = _layer_norm(x, lp['ln1_w'], lp['ln1_b'])
        x = x + _whisper_attention(h, lp, 6)
        # FFN + residual
        h = _layer_norm(x, lp['ln2_w'], lp['ln2_b'])
        h = jax.nn.gelu(h @ lp['fc1_w'] + lp['fc1_b'], approximate=False)
        x = x + (h @ lp['fc2_w'] + lp['fc2_b'])
    return _layer_norm(x, params['final_ln_w'], params['final_ln_b'])


def _build_whisper_params():
    d, ffn, n_layers = 384, 1536, 4
    seq_len = 1500
    p = {
        'conv1_w': _init_param('conv1.weight', (d * 80 * 3,)),
        'conv1_b': _init_param('conv1.bias', (d,)),
        'conv2_w': _init_param('conv2.weight', (d * d * 3,)),
        'conv2_b': _init_param('conv2.bias', (d,)),
        'pos_embed': _init_param('embed_positions.weight', (seq_len, d)),
        'final_ln_w': _init_param('layer_norm.weight', (d,)),
        'final_ln_b': _init_param('layer_norm.bias', (d,)),
    }
    layers = []
    for i in range(n_layers):
        pf = f'layers.{i}'
        lp = {
            'ln1_w': _init_param(f'{pf}.self_attn_layer_norm.weight', (d,)),
            'ln1_b': _init_param(f'{pf}.self_attn_layer_norm.bias', (d,)),
            'wq': _init_transposed(f'{pf}.self_attn.q_proj.weight', d, d),
            'q_b': _init_param(f'{pf}.self_attn.q_proj.bias', (d,)),
            'wk': _init_transposed(f'{pf}.self_attn.k_proj.weight', d, d),
            'wv': _init_transposed(f'{pf}.self_attn.v_proj.weight', d, d),
            'v_b': _init_param(f'{pf}.self_attn.v_proj.bias', (d,)),
            'wo': _init_transposed(f'{pf}.self_attn.out_proj.weight', d, d),
            'o_b': _init_param(f'{pf}.self_attn.out_proj.bias', (d,)),
            'ln2_w': _init_param(f'{pf}.final_layer_norm.weight', (d,)),
            'ln2_b': _init_param(f'{pf}.final_layer_norm.bias', (d,)),
            'fc1_w': _init_transposed(f'{pf}.fc1.weight', ffn, d),
            'fc1_b': _init_param(f'{pf}.fc1.bias', (ffn,)),
            'fc2_w': _init_transposed(f'{pf}.fc2.weight', d, ffn),
            'fc2_b': _init_param(f'{pf}.fc2.bias', (d,)),
        }
        layers.append(lp)
    p['layers'] = layers
    return p


# ── Emit helper ──────────────────────────────────────────────────

def _emit(model_name, compile_s, inference_ms, latency_ms, training_ms, logits_np, loss):
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


def bench_resnet():
    batch = 4
    print("[jax] building ResNet-50...", file=sys.stderr)
    params = _build_resnet50_params()

    images = jnp.sin(jnp.arange(batch * 3 * 224 * 224, dtype=jnp.float32) * 0.001).reshape(batch, 3, 224, 224)
    labels = jnp.arange(batch, dtype=jnp.int32) % 1000

    jit_fwd = jax.jit(_resnet50_forward)
    def loss_fn(params, images, labels):
        logits = _resnet50_forward(params, images)
        return -jnp.mean(jax.nn.log_softmax(logits, axis=-1)[jnp.arange(batch), labels])
    jit_grad = jax.jit(jax.grad(loss_fn))

    # Compile
    print("[jax] JIT compiling...", file=sys.stderr)
    t0 = time.perf_counter()
    logits = jit_fwd(params, images); logits.block_until_ready()
    grads = jit_grad(params, images, labels); jax.tree.map(lambda x: x.block_until_ready(), grads)
    compile_s = time.perf_counter() - t0

    # Inference
    t0 = time.perf_counter()
    logits = jit_fwd(params, images); logits.block_until_ready()
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # Loss
    logits_np = np.asarray(logits, dtype=np.float32)
    loss_val = float(-np.mean([
        np.log(np.exp(logits_np[b] - logits_np[b].max()) / np.exp(logits_np[b] - logits_np[b].max()).sum())[labels[b]]
        for b in range(batch)
    ]))

    # Training
    t0 = time.perf_counter()
    grads = jit_grad(params, images, labels); jax.tree.map(lambda x: x.block_until_ready(), grads)
    training_ms = (time.perf_counter() - t0) * 1000.0

    # Latency
    lat_img = jnp.zeros((1, 3, 224, 224), dtype=jnp.float32)
    jit_fwd(params, lat_img).block_until_ready()
    t0 = time.perf_counter()
    jit_fwd(params, lat_img).block_until_ready()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    _emit("ResNet-50", compile_s, inference_ms, latency_ms, training_ms, logits_np, loss_val)


def bench_whisper():
    print("[jax] building Whisper-tiny encoder...", file=sys.stderr)
    params = _build_whisper_params()

    mel = jnp.sin(jnp.arange(80 * 3000, dtype=jnp.float32) * 0.001).reshape(1, 80, 3000)

    jit_fwd = jax.jit(_whisper_encoder_forward)
    def loss_fn(params, mel):
        out = _whisper_encoder_forward(params, mel)
        return jnp.mean(out ** 2)
    jit_grad = jax.jit(jax.grad(loss_fn))

    print("[jax] JIT compiling...", file=sys.stderr)
    t0 = time.perf_counter()
    out = jit_fwd(params, mel); out.block_until_ready()
    grads = jit_grad(params, mel); jax.tree.map(lambda x: x.block_until_ready(), grads)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = jit_fwd(params, mel); out.block_until_ready()
    inference_ms = (time.perf_counter() - t0) * 1000.0

    out_np = np.asarray(out, dtype=np.float32)
    loss_val = float(np.mean(out_np ** 2))

    t0 = time.perf_counter()
    grads = jit_grad(params, mel); jax.tree.map(lambda x: x.block_until_ready(), grads)
    training_ms = (time.perf_counter() - t0) * 1000.0

    # Latency (same input — encoder is not autoregressive)
    jit_fwd(params, mel).block_until_ready()
    t0 = time.perf_counter()
    jit_fwd(params, mel).block_until_ready()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    _emit("Whisper-tiny", compile_s, inference_ms, latency_ms, training_ms, out_np, loss_val)


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
    if model_name not in ALL_MODELS:
        print(f"Unknown model: {model_name}. Available: {ALL_MODELS}", file=sys.stderr)
        sys.exit(1)
    if os.environ.get("INFERENA_DRY_RUN") == "1":
        print(f"[jax] dry-run OK: {model_name}", file=sys.stderr)
        sys.exit(0)
    if model_name == "ResNet-50":
        bench_resnet()
    elif model_name == "Whisper-tiny":
        bench_whisper()
    else:
        bench(model_name)
