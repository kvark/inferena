#!/usr/bin/env python3
"""MLX benchmark runner for infermark.

Runs a forward + backward training step on a given model using Apple's MLX
framework and prints a JSON result to stdout matching the BenchResult schema.

MLX is Apple's array framework for Apple Silicon, using Metal for GPU compute.
Models are built from scratch using mlx.nn (no transformers library).
"""

import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.utils


# ---------------------------------------------------------------------------
# SmolVLA Action Expert
# ---------------------------------------------------------------------------

class SmolVLARMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps) * self.weight


class SmolVLASwiGLU(nn.Module):
    def __init__(self, dim, intermediate):
        super().__init__()
        self.gate = nn.Linear(dim, intermediate, bias=False)
        self.up = nn.Linear(dim, intermediate, bias=False)
        self.down = nn.Linear(intermediate, dim, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class SmolVLAGQAttention(nn.Module):
    def __init__(self, dim, num_heads=15, num_kv_heads=5, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_repeat = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

    def __call__(self, q_input, kv_input=None):
        if kv_input is None:
            kv_input = q_input
        b, sq, _ = q_input.shape
        sk = kv_input.shape[1]

        q = self.q_proj(q_input).reshape(b, sq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(kv_input).reshape(b, sk, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(kv_input).reshape(b, sk, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Repeat KV heads for GQA
        if self.kv_repeat > 1:
            k = mx.repeat(k, self.kv_repeat, axis=1)
            v = mx.repeat(v, self.kv_repeat, axis=1)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, sq, self.num_heads * self.head_dim)
        return self.o_proj(out)


class ExpertLayer(nn.Module):
    def __init__(self, dim, intermediate, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.norm1 = SmolVLARMSNorm(dim)
        self.attn = SmolVLAGQAttention(dim, num_heads, num_kv_heads, head_dim)
        self.norm2 = SmolVLARMSNorm(dim)
        self.mlp = SmolVLASwiGLU(dim, intermediate)

    def __call__(self, x, kv=None):
        x = x + self.attn(self.norm1(x), kv)
        x = x + self.mlp(self.norm2(x))
        return x


class ActionExpert(nn.Module):
    def __init__(self, action_dim=32, expert_hidden=720, intermediate=2048,
                 num_layers=16, num_heads=15, num_kv_heads=5, head_dim=64,
                 vlm_kv_dim=320, self_attn_every_n=2):
        super().__init__()
        self.self_attn_every_n = self_attn_every_n
        self.action_proj = nn.Linear(action_dim, expert_hidden, bias=False)
        self.time_proj = nn.Linear(expert_hidden * 2, expert_hidden, bias=False)
        self.kv_proj = nn.Linear(vlm_kv_dim, expert_hidden, bias=False)
        self.layers = [
            ExpertLayer(expert_hidden, intermediate, num_heads, num_kv_heads, head_dim)
            for _ in range(num_layers)
        ]
        self.norm = SmolVLARMSNorm(expert_hidden)
        self.head = nn.Linear(expert_hidden, action_dim, bias=False)

    def __call__(self, noisy_actions, timestep, vlm_kv):
        x = self.action_proj(noisy_actions) + self.time_proj(timestep)
        kv = self.kv_proj(vlm_kv)
        for i, layer in enumerate(self.layers):
            if i % self.self_attn_every_n == 0:
                x = layer(x)  # self-attention
            else:
                x = layer(x, kv)  # cross-attention
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# SD U-Net (simplified, matching meganeura's sd_unet module)
# ---------------------------------------------------------------------------

class SDResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 + residual."""
    def __init__(self, in_c, out_c, num_groups=16, eps=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_c, eps=eps)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_c, eps=eps)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.res_conv = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else None

    def __call__(self, x):
        h = nn.silu(self.norm1(x))
        h = self.conv1(h)
        h = nn.silu(self.norm2(h))
        h = self.conv2(h)
        res = self.res_conv(x) if self.res_conv is not None else x
        return h + res


class SDUNet(nn.Module):
    """Simplified SD U-Net matching meganeura's SDUNetConfig::small()."""
    def __init__(self, in_channels=4, base_channels=64, num_levels=3,
                 num_groups=16, eps=1e-5):
        super().__init__()
        self.num_levels = num_levels
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False)

        ch_mults = [1 << i for i in range(num_levels)]
        self.encoder_blocks = []
        self.downsamples = []

        prev_c = base_channels
        for level, mult in enumerate(ch_mults):
            out_c = base_channels * mult
            self.encoder_blocks.append(SDResBlock(prev_c, out_c, num_groups, eps))
            if level < num_levels - 1:
                self.downsamples.append(nn.Conv2d(out_c, out_c, 3, stride=2, padding=1, bias=False))
            else:
                self.downsamples.append(None)
            prev_c = out_c

        self.middle = SDResBlock(prev_c, prev_c, num_groups, eps)

        self.decoder_blocks = []
        for level in reversed(range(num_levels)):
            out_c = base_channels * ch_mults[level]
            self.decoder_blocks.append(SDResBlock(prev_c + out_c, out_c, num_groups, eps))
            prev_c = out_c

        self.norm_out = nn.GroupNorm(num_groups, base_channels, eps=eps)
        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1, bias=False)

    def __call__(self, x):
        # MLX uses NHWC layout
        x = self.conv_in(x)

        skips = []
        for level in range(self.num_levels):
            x = self.encoder_blocks[level](x)
            skips.append(x)
            if level < self.num_levels - 1 and self.downsamples[level] is not None:
                x = self.downsamples[level](x)

        x = self.middle(x)

        for i, level in enumerate(reversed(range(self.num_levels))):
            if level < self.num_levels - 1:
                # Nearest-neighbor upsample (2x)
                b, h, w, c = x.shape
                x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
            x = mx.concatenate([x, skips[level]], axis=-1)
            x = self.decoder_blocks[i](x)

        x = nn.silu(self.norm_out(x))
        return self.conv_out(x)


# ---------------------------------------------------------------------------
# SmolLM2 (LLaMA architecture)
# ---------------------------------------------------------------------------

class LlamaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps) * self.weight


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
        self._inv_freq = inv_freq

    def __call__(self, x, offset=0):
        """Apply rotary embeddings. x shape: (B, H, S, D)."""
        seq_len = x.shape[2]
        t = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self._inv_freq)  # (S, D/2)
        cos_vals = mx.cos(freqs)  # (S, D/2)
        sin_vals = mx.sin(freqs)  # (S, D/2)

        # Split x into pairs for rotation
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]

        # Broadcast: cos/sin are (S, D/2), x1/x2 are (B, H, S, D/2)
        rotated = mx.concatenate([
            x1 * cos_vals - x2 * sin_vals,
            x2 * cos_vals + x1 * sin_vals,
        ], axis=-1)
        return rotated


class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, max_position_embeddings=2048):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_repeat = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(head_dim, max_position_embeddings)

    def __call__(self, x, mask=None):
        b, s, _ = x.shape

        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rotary_emb(q)
        k = self.rotary_emb(k)

        if self.kv_repeat > 1:
            k = mx.repeat(k, self.kv_repeat, axis=1)
            v = mx.repeat(v, self.kv_repeat, axis=1)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size)

    def __call__(self, x, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers,
                 num_attention_heads, num_key_value_heads, intermediate_size,
                 max_position_embeddings=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        head_dim = hidden_size // num_attention_heads

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            LlamaDecoderLayer(hidden_size, num_attention_heads, num_key_value_heads,
                              head_dim, intermediate_size)
            for _ in range(num_hidden_layers)
        ]
        self.norm = LlamaRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids, mask=None):
        x = self.embed_tokens(input_ids)

        # Causal mask
        if mask is None:
            seq_len = input_ids.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(x.dtype)

        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha256_f32_array(a):
    """Compute sha256 of an mx.array as f32 little-endian bytes."""
    flat = a.astype(mx.float32).reshape(-1)
    mx.eval(flat)
    vals = flat.tolist()
    raw = struct.pack(f"<{len(vals)}f", *vals)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def device_name():
    """Return a human-readable Apple Silicon chip name."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return f"Apple {platform.processor()}"


def gpu_name():
    """Return the GPU/chip name for Apple Silicon."""
    try:
        import plistlib
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-xml"],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            data = plistlib.loads(result.stdout)
            for item in data:
                for entry in item.get("_items", []):
                    name = entry.get("sppci_model", "")
                    if name:
                        return name
    except Exception:
        pass
    return device_name()


def mlx_version():
    try:
        import importlib.metadata
        return importlib.metadata.version("mlx")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Model registry and loading
# ---------------------------------------------------------------------------

SMOLLM2_135M_CONFIG = dict(
    vocab_size=49152,
    hidden_size=576,
    num_hidden_layers=30,
    num_attention_heads=9,
    num_key_value_heads=3,
    intermediate_size=1536,
)

MODEL_REGISTRY = {
    "SmolLM2-135M": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "type": "causal_lm",
        "config": SMOLLM2_135M_CONFIG,
    },
    "SmolVLA": {
        "hf_id": "lerobot/smolvla_base",
        "type": "smolvla",
    },
    "StableDiffusion": {
        "hf_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "type": "sd_unet",
    },
}


def _weight_name_map_llama(mlx_key):
    """Map MLX model parameter names to HuggingFace safetensors names."""
    # MLX uses: layers.N.self_attn.q_proj.weight -> model.layers.N.self_attn.q_proj.weight
    # embed_tokens.weight -> model.embed_tokens.weight
    # norm.weight -> model.norm.weight
    # lm_head.weight -> lm_head.weight
    if mlx_key == "lm_head.weight":
        return "lm_head.weight"
    return "model." + mlx_key


def _load_safetensors_weights(model, model_dir):
    """Load safetensors weights from a directory into the MLX model."""
    import glob as globmod
    safetensor_files = sorted(globmod.glob(os.path.join(model_dir, "*.safetensors")))
    if not safetensor_files:
        return False

    try:
        weights = {}
        for sf_path in safetensor_files:
            w = mx.load(sf_path)
            weights.update(w)
    except Exception as e:
        print(f"[mlx] failed to load safetensors: {e}", file=sys.stderr)
        return False

    # Build mapping from MLX param names to loaded weights
    param_map = {}
    for mlx_key, _ in mlx.utils.tree_flatten(model.parameters()):
        hf_key = _weight_name_map_llama(mlx_key)
        if hf_key in weights:
            param_map[mlx_key] = weights[hf_key]
        elif mlx_key in weights:
            param_map[mlx_key] = weights[mlx_key]
        elif mlx_key == "lm_head.weight" and "model.embed_tokens.weight" in weights:
            # Weight tying: lm_head shares embed_tokens weight
            param_map[mlx_key] = weights["model.embed_tokens.weight"]

    if not param_map:
        print("[mlx] no matching weight keys found", file=sys.stderr)
        return False

    total_params = len(list(mlx.utils.tree_flatten(model.parameters())))
    print(f"[mlx] loaded {len(param_map)}/{total_params} parameters from safetensors", file=sys.stderr)

    # Unflatten and load
    model.load_weights(list(param_map.items()), strict=False)
    return True


def load_model(model_name, spec):
    """Load model: try local dir -> HF download -> random-init fallback."""
    model_type = spec["type"]

    if model_type == "sd_unet":
        print(f"[mlx] {model_name}: deterministic init (SD U-Net)", file=sys.stderr)
        model = SDUNet(in_channels=4, base_channels=64, num_levels=3)
        # Deterministic init matching meganeura
        params = model.parameters()
        flat = mlx.utils.tree_flatten(params)
        new_weights = []
        for i, (key, p) in enumerate(flat):
            n = p.size
            shape = p.shape
            vals = mx.sin(mx.arange(n, dtype=mx.float32) * 0.01 + i) * 0.1
            new_weights.append((key, vals.reshape(shape)))
        model.load_weights(new_weights)
        return model

    if model_type == "smolvla":
        print(f"[mlx] {model_name}: deterministic init (custom architecture)", file=sys.stderr)
        model = ActionExpert()
        # Match meganeura's deterministic init: sin(j * 0.01 + i) * 0.1
        params = model.parameters()
        flat = mlx.utils.tree_flatten(params)
        new_weights = []
        for i, (key, p) in enumerate(flat):
            n = p.size
            shape = p.shape
            vals = mx.sin(mx.arange(n, dtype=mx.float32) * 0.01 + i) * 0.1
            new_weights.append((key, vals.reshape(shape)))
        model.load_weights(new_weights)
        return model

    # Causal LM (LLaMA)
    config = spec.get("config", SMOLLM2_135M_CONFIG)
    model = LlamaModel(**config)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    local_dir = os.path.join(root_dir, "models", model_name)

    loaded = False
    # Try local directory first
    if os.path.isdir(local_dir):
        print(f"[mlx] trying local model at {local_dir}", file=sys.stderr)
        loaded = _load_safetensors_weights(model, local_dir)

    # Try HF cache / download
    if not loaded:
        try:
            from huggingface_hub import snapshot_download
            hf_dir = snapshot_download(spec["hf_id"])
            print(f"[mlx] trying HuggingFace model from {hf_dir}", file=sys.stderr)
            loaded = _load_safetensors_weights(model, hf_dir)
        except Exception as e:
            print(f"[mlx] HF download failed ({e}), using random-init", file=sys.stderr)

    if not loaded:
        print(f"[mlx] using random-init for {model_name}", file=sys.stderr)

    return model


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def prepare_smolvla_inputs():
    """Build deterministic SmolVLA inputs matching meganeura."""
    chunk_size = 50
    action_dim = 32
    expert_hidden = 720
    vlm_seq_len = 16
    vlm_kv_dim = 320

    noisy_actions = mx.sin(mx.arange(chunk_size * action_dim, dtype=mx.float32) * 0.01).reshape(1, chunk_size, action_dim)
    timestep = mx.sin(mx.arange(expert_hidden * 2, dtype=mx.float32) * 0.005).reshape(1, 1, expert_hidden * 2)
    vlm_kv = mx.cos(mx.arange(vlm_seq_len * vlm_kv_dim, dtype=mx.float32) * 0.01).reshape(1, vlm_seq_len, vlm_kv_dim)

    return noisy_actions, timestep, vlm_kv


def prepare_sd_unet_inputs():
    """Build deterministic SD U-Net inputs matching meganeura."""
    batch, in_c, res = 2, 4, 32
    in_size = batch * in_c * res * res
    # MLX uses NHWC layout
    noisy = mx.sin(mx.arange(in_size, dtype=mx.float32) * 0.01).reshape(batch, res, res, in_c)
    target = mx.cos(mx.arange(in_size, dtype=mx.float32) * 0.007).reshape(batch, res, res, in_c)
    return noisy, target


def prepare_causal_lm_inputs(vocab_size, seq_len=128):
    """Build deterministic causal LM inputs."""
    input_ids = mx.arange(seq_len).reshape(1, seq_len)
    labels = (mx.arange(1, seq_len + 1) % vocab_size).reshape(1, seq_len)
    return input_ids, labels


def cross_entropy_loss(logits, labels, vocab_size):
    """Cross-entropy loss (HF-compatible: shifted labels).

    HuggingFace internally shifts: logits[0..seq-1] predict labels[1..seq].
    """
    # Shift: logits[:-1] predicts labels[1:]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    b, s, v = shift_logits.shape
    logits_flat = shift_logits.reshape(b * s, v)
    labels_flat = shift_labels.reshape(b * s)
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
    loss = -mx.mean(mx.take_along_axis(log_probs, labels_flat[:, None], axis=1))
    return loss


def mse_loss(pred, target):
    """Mean squared error loss."""
    return mx.mean((pred - target) ** 2)


def bench(model_name, spec):
    model_type = spec["type"]
    dev = device_name()
    gpu = gpu_name()
    ver = mlx_version()

    print(f"[mlx] device: {dev}, gpu: {gpu}, mlx {ver}", file=sys.stderr)

    # --- Load model ---
    print(f"[mlx] loading {model_name}...", file=sys.stderr)
    t0 = time.perf_counter()
    model = load_model(model_name, spec)
    mx.eval(model.parameters())
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[mlx] loaded in {load_ms:.0f}ms", file=sys.stderr)

    # --- Build forward/loss functions ---
    if model_type == "smolvla":
        noisy_actions, timestep, vlm_kv = prepare_smolvla_inputs()
        mx.eval(noisy_actions, timestep, vlm_kv)

        def fwd_fn(model):
            return model(noisy_actions, timestep, vlm_kv)

        def loss_fn(model):
            out = model(noisy_actions, timestep, vlm_kv)
            return mse_loss(out, mx.zeros_like(out))

    elif model_type == "sd_unet":
        noisy_latent, noise_target = prepare_sd_unet_inputs()
        mx.eval(noisy_latent, noise_target)

        def fwd_fn(model):
            return model(noisy_latent)

        def loss_fn(model):
            return mse_loss(model(noisy_latent), noise_target)

    else:
        config = spec.get("config", SMOLLM2_135M_CONFIG)
        vocab_size = config["vocab_size"]
        input_ids, labels = prepare_causal_lm_inputs(vocab_size)
        mx.eval(input_ids, labels)

        def fwd_fn(model):
            return model(input_ids)

        def loss_fn(model):
            return cross_entropy_loss(model(input_ids), labels, vocab_size)

    # --- Warmup (JIT compile) ---
    print(f"[mlx] warmup...", file=sys.stderr)
    grad_fn = nn.value_and_grad(model, loss_fn)
    warmup_loss, warmup_grads = grad_fn(model)
    mx.eval(warmup_loss, warmup_grads)

    # --- Forward (timed, post-JIT) ---
    t0 = time.perf_counter()
    logits = fwd_fn(model)
    mx.eval(logits)
    forward_ms = (time.perf_counter() - t0) * 1000.0

    if model_type == "smolvla":
        loss_val = mse_loss(logits, mx.zeros_like(logits))
    elif model_type == "sd_unet":
        loss_val = mse_loss(logits, noise_target)
    else:
        loss_val = cross_entropy_loss(logits, labels, vocab_size)
    mx.eval(loss_val)

    # --- Backward (timed, post-JIT) ---
    t0 = time.perf_counter()
    _, grads = grad_fn(model)
    mx.eval(grads)
    fwd_bwd_ms = (time.perf_counter() - t0) * 1000.0
    backward_ms = max(0.0, fwd_bwd_ms - forward_ms)

    # --- Collect outputs ---
    logits_hash = sha256_f32_array(logits)
    logits_flat = logits.astype(mx.float32).reshape(-1)
    mx.eval(logits_flat)
    logits_sample = logits_flat[:16].tolist()

    result = {
        "framework": "mlx",
        "model": model_name,
        "device": dev,
        "gpu_name": gpu,
        "mlx_version": ver,
        "backend": "MLX",
        "timings": {
            "compile_s": 0.0,
            "forward_ms": round(forward_ms, 3),
            "backward_ms": round(backward_ms, 3),
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": [round(v, 6) for v in logits_sample],
            "loss": round(loss_val.item(), 6),
        },
    }
    print(json.dumps(result))


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M"
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)
    bench(model_name, spec)
