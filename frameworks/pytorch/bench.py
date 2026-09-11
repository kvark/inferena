#!/usr/bin/env python3
"""PyTorch benchmark runner for inferena.

Runs a fake training step (forward + backward) on a given model and prints
a JSON result to stdout matching the BenchResult schema.

Features inspired by meganeura's bench/compare.sh (PR #30):
- torch.compile with fresh inductor cache for fair compile-time measurement
- torch.set_float32_matmul_precision("high") for TF32 on Ampere+
- Device name reporting (not just "cuda:0")
- torch version in output
"""

from contextlib import nullcontext
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from execution import capture_phase, profile_phase, synchronize, nsys_range


# --- Conditioned latent-diffusion U-Net (matches meganeura::models::sd_unet) ---

class ResBlock(nn.Module):
    """Diffusion ResBlock with projected timestep conditioning."""
    def __init__(self, in_c, out_c, time_dim, num_groups=16, eps=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_c, eps=eps)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.time_proj = nn.Linear(time_dim, out_c)
        self.norm2 = nn.GroupNorm(num_groups, out_c, eps=eps)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.res_conv = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x, time_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.res_conv(x)


class DiffusionAttention(nn.Module):
    """Explicit projections around PyTorch SDPA, matching Meganeura layout."""
    def __init__(self, query_dim, context_dim, head_dim):
        super().__init__()
        assert query_dim % head_dim == 0
        self.num_heads = query_dim // head_dim
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=False)

    def forward(self, query, context):
        def heads(x):
            return x.reshape(x.shape[0], self.num_heads, self.head_dim).transpose(0, 1)
        q = heads(self.q_proj(query))
        k = heads(self.k_proj(context))
        v = heads(self.v_proj(context))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(0, 1).reshape(query.shape[0], -1)
        return self.out_proj(out)


class SpatialTransformerBlock(nn.Module):
    def __init__(self, channels, context_dim, head_dim, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=eps)
        self.self_attn = DiffusionAttention(channels, channels, head_dim)
        self.norm2 = nn.LayerNorm(channels, eps=eps)
        self.cross_attn = DiffusionAttention(channels, context_dim, head_dim)
        self.norm3 = nn.LayerNorm(channels, eps=eps)
        self.ff = nn.Module()
        self.ff.fc1 = nn.Linear(channels, 4 * channels)
        self.ff.fc2 = nn.Linear(4 * channels, channels)

    def forward(self, x, context):
        norm = self.norm1(x)
        x = x + self.self_attn(norm, norm)
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.ff.fc2(F.gelu(self.ff.fc1(self.norm3(x)), approximate="tanh"))
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, channels, context_dim, head_dim, num_groups, eps):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, eps=eps)
        self.proj_in = nn.Conv2d(channels, channels, 1, bias=False)
        self.transformer = SpatialTransformerBlock(channels, context_dim, head_dim, eps)
        self.proj_out = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x, context):
        residual = x
        h = self.proj_in(self.norm(x))
        # Meganeura's differentiable attention primitive represents one
        # sequence per op, so the matched workload intentionally uses batch 1.
        assert h.shape[0] == 1
        tokens = h.flatten(2).transpose(1, 2)[0]
        tokens = self.transformer(tokens, context)
        h = tokens.transpose(0, 1).reshape(1, h.shape[1], h.shape[2], h.shape[3])
        return residual + self.proj_out(h)


class SDUNet(nn.Module):
    """Scaled conditioned U-Net matching SDUNetConfig::small().

    Architecture: Conv_in → [ResBlock + Downsample]×N → Middle ResBlock
                  → [Upsample + CatSkip + ResBlock]×N → GroupNorm → SiLU → Conv_out

    ResBlocks receive timestep conditioning; levels below 32×32 and the
    bottleneck include self-attention and 77-token text cross-attention.
    """
    def __init__(self, in_channels=4, base_channels=64, num_levels=3,
                 num_groups=16, eps=1e-5, time_input_dim=64,
                 time_embed_dim=256, context_dim=768, attention_head_dim=32):
        super().__init__()
        self.num_levels = num_levels
        self.time_mlp = nn.Module()
        self.time_mlp.fc1 = nn.Linear(time_input_dim, time_embed_dim)
        self.time_mlp.fc2 = nn.Linear(time_embed_dim, time_embed_dim)

        # Input conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False)

        # Encoder
        ch_mults = [1 << i for i in range(num_levels)]
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        prev_c = base_channels
        for level, mult in enumerate(ch_mults):
            out_c = base_channels * mult
            self.encoder_blocks.append(
                ResBlock(prev_c, out_c, time_embed_dim, num_groups, eps)
            )
            self.encoder_attn.append(
                SpatialTransformer(
                    out_c, context_dim, attention_head_dim, num_groups, eps
                ) if level > 0 else nn.Identity()
            )
            if level < num_levels - 1:
                self.downsamples.append(
                    nn.Conv2d(out_c, out_c, 3, stride=2, padding=1, bias=False)
                )
            else:
                self.downsamples.append(nn.Identity())
            prev_c = out_c

        # Middle
        self.middle_resblock = ResBlock(
            prev_c, prev_c, time_embed_dim, num_groups, eps
        )
        self.middle_attn = SpatialTransformer(
            prev_c, context_dim, attention_head_dim, num_groups, eps
        )

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attn = nn.ModuleList()

        for level in reversed(range(num_levels)):
            out_c = base_channels * ch_mults[level]
            skip_c = out_c  # from encoder
            if level < num_levels - 1:
                self.upsamples.append(nn.Upsample(scale_factor=2, mode='nearest'))
            else:
                self.upsamples.append(nn.Identity())
            self.decoder_blocks.append(
                ResBlock(prev_c + skip_c, out_c, time_embed_dim, num_groups, eps)
            )
            self.decoder_attn.append(
                SpatialTransformer(
                    out_c, context_dim, attention_head_dim, num_groups, eps
                ) if level > 0 else nn.Identity()
            )
            prev_c = out_c

        # Output
        self.norm_out = nn.GroupNorm(num_groups, base_channels, eps=eps)
        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1, bias=False)

    def forward(self, x, timestep_embedding, text_context):
        time_emb = self.time_mlp.fc2(F.silu(self.time_mlp.fc1(timestep_embedding)))
        x = self.conv_in(x)

        # Encoder
        skips = []
        for level in range(self.num_levels):
            x = self.encoder_blocks[level](x, time_emb)
            if level > 0:
                x = self.encoder_attn[level](x, text_context)
            skips.append(x)
            if level < self.num_levels - 1:
                x = self.downsamples[level](x)

        # Middle
        x = self.middle_resblock(x, time_emb)
        x = self.middle_attn(x, text_context)

        # Decoder
        for i, level in enumerate(reversed(range(self.num_levels))):
            if level < self.num_levels - 1:
                x = self.upsamples[i](x)
            x = torch.cat([x, skips[level]], dim=1)
            x = self.decoder_blocks[i](x, time_emb)
            if level > 0:
                x = self.decoder_attn[i](x, text_context)

        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


# --- SmolVLA Action Expert (matches meganeura's bench_smolvla_train_pytorch.py) ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class GQAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        kv_input_dim,
        num_heads=15,
        num_kv_heads=5,
        head_dim=64,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(
            kv_input_dim, num_kv_heads * head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            kv_input_dim, num_kv_heads * head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            num_heads * head_dim, query_dim, bias=False
        )
        self.kv_repeat = num_heads // num_kv_heads

    def forward(self, q_input, kv_input, causal):
        if kv_input is None:
            kv_input = q_input
        b, sq, _ = q_input.shape
        sk = kv_input.shape[1]
        q = self.q_proj(q_input).view(b, sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).view(b, sk, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(b, sk, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if causal:
            causal_mask = torch.ones(
                sq, sk, dtype=torch.bool, device=attn.device
            ).triu(diagonal=1)
            attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, sq, self.num_heads * self.head_dim)
        return self.o_proj(out)


class ExpertLayer(nn.Module):
    def __init__(
        self,
        dim,
        kv_dim,
        intermediate,
        num_heads,
        num_kv_heads,
        head_dim,
        is_cross_attention,
    ):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        self.input_layernorm = RMSNorm(dim)
        self.self_attn = GQAttention(
            dim,
            kv_dim if is_cross_attention else dim,
            num_heads,
            num_kv_heads,
            head_dim,
        )
        self.post_attention_layernorm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, intermediate)

    def forward(self, x, vlm_kv):
        kv_input = vlm_kv if self.is_cross_attention else x
        x = x + self.self_attn(
            self.input_layernorm(x),
            kv_input,
            causal=not self.is_cross_attention,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class ActionExpert(nn.Module):
    def __init__(self, action_dim=32, expert_hidden=720, intermediate=2048,
                 num_layers=16, num_heads=15, num_kv_heads=5, head_dim=64,
                 vlm_kv_dim=320, self_attn_every_n=2):
        super().__init__()
        self.action_in_proj = nn.Linear(
            action_dim, expert_hidden, bias=True
        )
        self.action_time_mlp_in = nn.Linear(
            expert_hidden * 2, expert_hidden, bias=True
        )
        self.action_time_mlp_out = nn.Linear(
            expert_hidden, expert_hidden, bias=True
        )
        self.layers = nn.ModuleList([
            ExpertLayer(
                expert_hidden,
                vlm_kv_dim,
                intermediate,
                num_heads,
                num_kv_heads,
                head_dim,
                is_cross_attention=(i % self_attn_every_n != 0),
            )
            for i in range(num_layers)
        ])
        self.action_out_proj = nn.Linear(
            expert_hidden, action_dim, bias=True
        )

    def forward(self, noisy_actions, timestep, vlm_kv):
        x = self.action_in_proj(noisy_actions)
        time_embed = self.action_time_mlp_out(
            F.silu(self.action_time_mlp_in(timestep))
        )
        x = x + time_embed
        for layer in self.layers:
            x = layer(x, vlm_kv)
        return self.action_out_proj(x)


class FusedBatchNormBias(nn.Module):
    """BatchNorm folded to an identity scale plus trainable channel bias."""

    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return x + self.bias.view(1, -1, 1, 1)


def _replace_resnet_batch_norm(module):
    """Match Meganeura's inference-folded BatchNorm representation."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, FusedBatchNormBias(child.num_features))
        else:
            _replace_resnet_batch_norm(child)


def _xpu_actually_works() -> bool:
    """XPU may report available but fail at kernel-launch time on older Intel
    iGPUs (Gen12 Raptor/Alder Lake UHD) — JIT compilation aborts with
    "program was built for 1 devices". Probe with a trivial matmul."""
    try:
        x = torch.ones(4, 4, device="xpu", requires_grad=True)
        output = x @ x.t()
        output.sum().backward()
        torch.testing.assert_close(output.cpu(), torch.full((4, 4), 4.0))
        torch.testing.assert_close(x.grad.cpu(), torch.full((4, 4), 8.0))
        return True
    except Exception as e:
        print(f"[pytorch] XPU compute probe failed: {e}", file=sys.stderr)
        return False


class _IndexAddEmbeddingBackward(torch.autograd.Function):
    """Dense embedding gradient through a widely supported PyTorch primitive."""

    @staticmethod
    def forward(ctx, indices, weight):
        ctx.save_for_backward(indices)
        ctx.weight_shape = weight.shape
        return F.embedding(indices, weight)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_weight = grad_output.new_zeros(ctx.weight_shape)
        grad_weight.index_add_(
            0,
            indices.reshape(-1),
            grad_output.reshape(-1, ctx.weight_shape[1]),
        )
        return None, grad_weight


def _index_add_embedding_forward(embedding, indices):
    return _IndexAddEmbeddingBackward.apply(indices, embedding.weight)


def qualify_embedding_backward(model, dev: str):
    """Probe the native dense backward and select an equivalent safe path."""
    embedding = model.get_input_embeddings()
    if not isinstance(embedding, nn.Embedding):
        return {"status": "not-applicable", "reason": "input module is not nn.Embedding"}

    rows = min(128, embedding.num_embeddings)
    width = embedding.embedding_dim

    def probe(forward):
        weight = torch.zeros(
            rows, width, dtype=embedding.weight.dtype, device=dev, requires_grad=True
        )
        indices = torch.arange(rows, device=dev)
        forward(indices, weight).sum().backward()
        synchronize(dev)
        grad = weight.grad.detach()
        matching = int((grad == 1).sum().cpu())
        finite = bool(torch.isfinite(grad).all().cpu())
        return matching, finite

    total = rows * width
    matching, finite = probe(lambda indices, weight: F.embedding(indices, weight))
    native_matching = matching
    report = {
        "probe_shape": [rows, width],
        "native_dense": {
            "status": "pass" if matching == total and finite else "fail",
            "matching_elements": matching,
            "total_elements": total,
            "finite": finite,
        },
        "selected": "native-dense",
    }
    if matching == total and finite:
        return report

    if (
        embedding.padding_idx is not None
        or embedding.max_norm is not None
        or embedding.scale_grad_by_freq
        or embedding.sparse
    ):
        raise RuntimeError(
            "native embedding backward failed its probe and the input embedding "
            "uses options unsupported by the dense index_add workaround"
        )

    matching, finite = probe(_IndexAddEmbeddingBackward.apply)
    if matching != total or not finite:
        raise RuntimeError("both native and index_add embedding backward failed qualification")
    embedding.forward = _index_add_embedding_forward.__get__(embedding, type(embedding))
    report["selected"] = "dense-index-add-autograd"
    report["workaround"] = {
        "status": "pass",
        "matching_elements": matching,
        "total_elements": total,
        "finite": finite,
    }
    print(
        f"[pytorch] native dense embedding backward matched {native_matching}/{total} "
        "contract elements; selected qualified dense index_add backward",
        file=sys.stderr,
    )
    return report


def detect_device() -> str:
    requested = os.environ.get("INFERENA_TORCH_BACKEND")
    if requested:
        if requested == "cpu":
            return "cpu"
        if requested in ("cuda", "rocm") and torch.cuda.is_available():
            actual = "rocm" if torch.version.hip else "cuda"
            if actual == requested:
                return "cuda:0"
        if requested == "xpu" and torch.xpu.is_available() and _xpu_actually_works():
            return "xpu:0"
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError(f"requested {requested} backend is unavailable or failed its probe; no fallback")
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and torch.xpu.is_available() and _xpu_actually_works():
        return "xpu:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name(dev: str) -> str:
    if dev.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    if dev.startswith("xpu"):
        return torch.xpu.get_device_name(0)
    if dev == "mps":
        return f"Apple {platform.processor()}"
    return "cpu"


def backend_name(dev: str) -> str:
    """Return the GPU API backend name for the framework column."""
    if dev.startswith("cuda"):
        version = torch.version.cuda or ""
        if torch.version.hip:
            return f"ROCm {torch.version.hip}"
        if version:
            return f"CUDA {version}"
        return "CUDA"
    if dev.startswith("xpu"):
        return "XPU"
    if dev == "mps":
        return "MPS"
    return "CPU"


def torch_release_url(version: str) -> str:
    """GitHub release URL for a PyTorch version."""
    # Strip build metadata like +cu130 or +rocm6.2
    base = version.split("+")[0]
    return f"https://github.com/pytorch/pytorch/releases/tag/v{base}"


def sha256_f32_tensor(t: torch.Tensor) -> str:
    flat = t.detach().float().cpu().contiguous().flatten()
    raw = flat.numpy().astype("<f4", copy=False).tobytes()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def clear_compile_cache():
    """Own an empty cache for this run; never delete the developer's cache."""
    torch._dynamo.reset()
    # Windows may keep a loaded JIT DLL locked until process exit.
    cache = tempfile.TemporaryDirectory(prefix="inferena-inductor-", ignore_cleanup_errors=os.name == "nt")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache.name
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache.name, "triton")
    return cache


# --- Model registry ---

MODEL_REGISTRY = {
    "SmolLM2-135M": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "type": "causal_lm",
    },
    "SmolLM2-360M": {
        "hf_id": "HuggingFaceTB/SmolLM2-360M",
        "type": "causal_lm",
    },
    "SmolLM2-1.7B": {
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B",
        "type": "causal_lm",
    },
    "SmolVLA": {
        "hf_id": "lerobot/smolvla_base",
        "type": "smolvla",
    },
    "StableDiffusion": {
        "hf_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "type": "sd_unet",
    },
    "ResNet-50": {
        "hf_id": "torchvision/resnet50",
        "type": "resnet",
    },
    "Whisper-tiny": {
        "hf_id": "openai/whisper-tiny",
        "type": "whisper",
    },
}


def load_model(model_name: str, spec: dict, dev: str):
    """Load model, trying: local dir -> HF download -> random-init fallback."""
    hf_id = spec["hf_id"]
    model_type = spec["type"]

    # Custom architectures (SmolVLA, SD U-Net) are always random-init.
    if model_type in ("smolvla", "sd_unet"):
        print(f"[pytorch] {model_name}: random-init (custom architecture)", file=sys.stderr)
        return _random_init(model_type, model_name)

    if model_type == "resnet":
        import torchvision.models as tv_models
        model = tv_models.resnet50(weights=None)
        _replace_resnet_batch_norm(model)
        return _resnet_init(model)

    if model_type == "whisper":
        from transformers import WhisperForConditionalGeneration, WhisperConfig
        config = WhisperConfig(
            d_model=384, encoder_layers=4, decoder_layers=4,
            encoder_attention_heads=6, decoder_attention_heads=6,
            encoder_ffn_dim=1536, decoder_ffn_dim=1536,
            vocab_size=51865, max_source_positions=1500,
            max_target_positions=448, num_mel_bins=80,
        )
        full_model = WhisperForConditionalGeneration(config)
        encoder = full_model.get_encoder()
        return _whisper_encoder_init(encoder)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    local_dir = os.path.join(root_dir, "models", model_name)

    model = None

    # Try local dir first.
    if os.path.isfile(os.path.join(local_dir, "config.json")):
        print(f"[pytorch] found local model at {local_dir}", file=sys.stderr)
        try:
            model = _load_pretrained(model_type, local_dir)
        except Exception as e:
            if os.environ.get("INFERENA_REQUIRE_LOCAL_WEIGHTS") == "1":
                raise
            print(f"[pytorch] local load failed ({e})", file=sys.stderr)

    if model is None and os.environ.get("INFERENA_REQUIRE_LOCAL_WEIGHTS") == "1":
        raise FileNotFoundError(f"required local checkpoint could not be loaded: {local_dir}")

    # Try HF download.
    if model is None:
        try:
            model = _load_pretrained(model_type, hf_id)
        except Exception as e:
            print(f"[pytorch] HF load failed ({e}), using random-init", file=sys.stderr)
            model = _random_init(model_type, model_name)

    return model


def _load_pretrained(model_type: str, path_or_id: str):
    if model_type == "smolvla":
        # SmolVLA is a custom architecture — always random-init.
        return None
    else:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(path_or_id, torch_dtype=torch.float32)


def _name_seed(name: str) -> float:
    """Deterministic seed from parameter name — framework-independent init."""
    h = 0
    for c in name.encode('ascii'):
        h = ((h * 31) + c) & 0xFFFFFFFF
    return float(h % 10000)


def _deterministic_init(model):
    """Match meganeura's deterministic init: sin(j * 0.01 + i) * 0.1."""
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            n = p.numel()
            p.copy_(torch.sin(torch.arange(n, dtype=torch.float32) * 0.01 + i).view_as(p) * 0.1)
    return model


# Linear weight suffixes that meganeura stores as [in, out] (transposed vs PyTorch [out, in]).
_TRANSPOSED_SUFFIXES = frozenset([
    'q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'out_proj.weight',
    'fc1.weight', 'fc2.weight',
])


_INIT_SCALE = 0.02  # Standard transformer init scale (GPT-2/LLaMA convention)


def _name_seeded_init(p, name, scale=_INIT_SCALE):
    """Fill a parameter from its cross-framework canonical name."""
    seed = _name_seed(name)
    n = p.numel()
    p.copy_(
        torch.sin(
            torch.arange(n, dtype=torch.float32) * 0.01 + seed
        ).view_as(p)
        * scale
    )


def _transposed_init(p, name, scale=_INIT_SCALE):
    """Init as [in, out] (meganeura layout), store as [out, in] (PyTorch layout).

    Ensures x @ W_torch.T == x @ W_mega for matching forward passes.
    """
    seed = _name_seed(name)
    out_f, in_f = p.shape
    w = (
        torch.sin(
            torch.arange(in_f * out_f, dtype=torch.float32) * 0.01 + seed
        ).view(in_f, out_f)
        * scale
    )
    p.copy_(w.T)


def _sd_parameter_name(name):
    if name.startswith("encoder_blocks."):
        _, level, suffix = name.split(".", 2)
        return f"encoder.{level}.resblock.{suffix}"
    if name.startswith("encoder_attn."):
        _, level, suffix = name.split(".", 2)
        return f"encoder.{level}.attn.{suffix}"
    if name.startswith("downsamples."):
        _, level, suffix = name.split(".", 2)
        return f"encoder.{level}.downsample.{suffix}"
    if name.startswith("middle_resblock."):
        return "middle.resblock." + name.removeprefix("middle_resblock.")
    if name.startswith("middle_attn."):
        return "middle.attn." + name.removeprefix("middle_attn.")
    if name.startswith("decoder_blocks."):
        _, decoder_index, suffix = name.split(".", 2)
        level = 2 - int(decoder_index)
        return f"decoder.{level}.resblock.{suffix}"
    if name.startswith("decoder_attn."):
        _, decoder_index, suffix = name.split(".", 2)
        level = 2 - int(decoder_index)
        return f"decoder.{level}.attn.{suffix}"
    if name.startswith("norm_out."):
        return "conv_out.norm." + name.removeprefix("norm_out.")
    return name


def _smolvla_parameter_name(name):
    if name.startswith("layers."):
        _, layer, suffix = name.split(".", 2)
        return (
            "model.vlm_with_expert.lm_expert.layers."
            f"{layer}.{suffix}"
        )
    return f"model.{name}"


def _gradient_parameter_name(model_type, name):
    """Canonicalize trainable parameter names for cross-engine validation."""
    name = name.removeprefix("_orig_mod.")
    if model_type == "sd_unet":
        return _sd_parameter_name(name)
    if model_type == "smolvla":
        return _smolvla_parameter_name(name)
    if model_type == "whisper":
        canonical = f"model.encoder.{name}"
        if name in ("conv1.bias", "conv2.bias"):
            canonical = canonical.replace(".bias", ".fused_bias")
        return canonical
    if model_type == "resnet":
        is_folded_bn = (
            name.startswith("bn")
            or ".bn" in name
            or ".downsample.1." in name
        )
        if is_folded_bn and name.endswith(".bias"):
            return name.removesuffix(".bias") + ".fused_bias"
    return name


def _smolvla_init(model):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            canonical_name = _smolvla_parameter_name(name)
            if parameter.ndim == 2 and name.endswith(".weight"):
                _transposed_init(parameter, canonical_name)
            else:
                _name_seeded_init(parameter, canonical_name)
    return model


def _resnet_init(model):
    """Deterministic ResNet init matching meganeura's fused-BN approach.

    Folded-BN channel biases → zero; conv/fc → name-seeded sin values.
    FC weight uses transposed init to match meganeura's [in, out] matmul.

    Scale 0.01 (not 0.1) prevents activation explosion through 50+ layers
    with identity BN while preserving a non-trivial residual path.
    """
    scale = 0.01
    with torch.no_grad():
        for name, p in model.named_parameters():
            if (
                name.startswith("bn")
                or ".bn" in name
                or ".downsample.1." in name
            ):
                p.zero_()
            elif name == 'fc.weight':
                seed = _name_seed(name)
                out_f, in_f = p.shape
                w = torch.sin(torch.arange(in_f * out_f, dtype=torch.float32) * 0.01 + seed).view(in_f, out_f) * scale
                p.copy_(w.T)
            else:
                seed = _name_seed(name)
                n = p.numel()
                p.copy_(torch.sin(torch.arange(n, dtype=torch.float32) * 0.01 + seed).view_as(p) * scale)
    model.eval()
    return model


def _whisper_encoder_init(encoder):
    """Deterministic Whisper encoder init matching meganeura convention.

    Linear weights that meganeura stores as [in, out] get transposed init.
    """
    with torch.no_grad():
        for name, p in encoder.named_parameters():
            if any(name.endswith(s) for s in _TRANSPOSED_SUFFIXES):
                _transposed_init(p, name, scale=0.02)
            else:
                _name_seeded_init(p, name, scale=0.02)
    return encoder


def _random_init(model_type: str, model_name: str):
    if model_type == "sd_unet":
        # Match meganeura's SDUNetConfig::small()
        model = SDUNet(
            in_channels=4, base_channels=64, num_levels=3,
            num_groups=16, eps=1e-5, time_input_dim=64,
            time_embed_dim=256, context_dim=768, attention_head_dim=32,
        ).to(torch.float32)
        # Canonical name-seeded init matches Meganeura regardless of module
        # registration order. Norms use standard identity initialization so
        # deep activations and gradients remain representative.
        with torch.no_grad():
            for name, p in model.named_parameters():
                canonical = _sd_parameter_name(name)
                if ".norm" in canonical and canonical.endswith(".weight"):
                    p.fill_(1.0)
                elif ".norm" in canonical and canonical.endswith(".bias"):
                    p.zero_()
                elif p.ndim == 2 and name.endswith(".weight"):
                    _transposed_init(p, canonical, scale=0.02)
                else:
                    _name_seeded_init(p, canonical, scale=0.02)
        return model
    if model_type == "smolvla":
        model = ActionExpert().to(torch.float32)
        return _smolvla_init(model)
    else:
        from transformers import LlamaConfig, LlamaForCausalLM
        configs = {
            "SmolLM2-135M": LlamaConfig(
                vocab_size=49152, hidden_size=576, num_hidden_layers=30,
                num_attention_heads=9, num_key_value_heads=3,
                intermediate_size=1536, max_position_embeddings=2048,
            ),
            "SmolLM2-360M": LlamaConfig(
                vocab_size=49152, hidden_size=960, num_hidden_layers=32,
                num_attention_heads=15, num_key_value_heads=5,
                intermediate_size=2560, max_position_embeddings=2048,
            ),
            "SmolLM2-1.7B": LlamaConfig(
                vocab_size=49152, hidden_size=2048, num_hidden_layers=24,
                num_attention_heads=32, num_key_value_heads=32,
                intermediate_size=8192, max_position_embeddings=2048,
            ),
        }
        config = configs.get(model_name)
        if config is None:
            print(f"[pytorch] no fallback config for {model_name}", file=sys.stderr)
            sys.exit(1)
        model = LlamaForCausalLM(config).to(torch.float32)
        with torch.no_grad():
            for name, p in model.named_parameters():
                _name_seeded_init(p, name)
        return model


def prepare_inputs(model_type: str, model, dev: str, seq_len: int = 128):
    """Build deterministic dummy inputs matching the model type."""
    if model_type == "sd_unet":
        # Match the conditioned SDUNetConfig::small().
        batch, in_c, res = 1, 4, 32
        in_size = batch * in_c * res * res
        noisy = torch.tensor(
            [(i * 0.01) for i in range(in_size)],
            dtype=torch.float32, device=dev,
        ).sin().reshape(batch, in_c, res, res)
        target = torch.tensor(
            [(i * 0.007) for i in range(in_size)],
            dtype=torch.float32, device=dev,
        ).cos().reshape(batch, in_c, res, res)
        timestep_embedding = torch.tensor(
            [(i * 0.005) for i in range(batch * 64)],
            dtype=torch.float32, device=dev,
        ).sin().reshape(batch, 64)
        text_context = (
            torch.tensor(
                [(i * 0.003) for i in range(77 * 768)],
                dtype=torch.float32, device=dev,
            ).cos().reshape(77, 768) * 0.1
        )
        return {
            "noisy_latent": noisy,
            "noise_target": target,
            "timestep_embedding": timestep_embedding,
            "text_context": text_context,
        }

    if model_type == "smolvla":
        # SmolVLA action expert inputs — deterministic, matching meganeura.
        chunk_size = 50
        action_dim = 32
        expert_hidden = 720
        vlm_seq_len = 16
        vlm_kv_dim = 320
        noisy_actions = torch.sin(torch.arange(chunk_size * action_dim, dtype=torch.float32) * 0.01).view(1, chunk_size, action_dim).to(dev)
        timestep = torch.sin(torch.arange(expert_hidden * 2, dtype=torch.float32) * 0.005).view(1, 1, expert_hidden * 2).to(dev)
        vlm_kv = torch.cos(torch.arange(vlm_seq_len * vlm_kv_dim, dtype=torch.float32) * 0.01).view(1, vlm_seq_len, vlm_kv_dim).to(dev)
        return {
            "noisy_actions": noisy_actions,
            "timestep": timestep,
            "vlm_kv": vlm_kv,
        }

    if model_type == "resnet":
        # ImageNet-style input: batch=4, 3×224×224.
        batch, c, h, w = 4, 3, 224, 224
        in_size = batch * c * h * w
        images = torch.tensor(
            [(i * 0.001) for i in range(in_size)],
            dtype=torch.float32, device=dev,
        ).sin().reshape(batch, c, h, w)
        labels = torch.arange(batch, device=dev, dtype=torch.long) % 1000
        return {"images": images, "labels": labels}

    if model_type == "whisper":
        # 30s mel spectrogram: (1, 80, 3000).  Encoder-only (matches meganeura).
        mel_len = 3000
        n_mels = 80
        mel_size = n_mels * mel_len
        mel = torch.tensor(
            [(i * 0.001) for i in range(mel_size)],
            dtype=torch.float32, device=dev,
        ).sin().reshape(1, n_mels, mel_len)
        return {"input_features": mel}

    vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else model.config.text_config.vocab_size
    input_ids = torch.arange(seq_len, device=dev, dtype=torch.long).unsqueeze(0)
    labels = (torch.arange(1, seq_len + 1, device=dev, dtype=torch.long) % vocab_size).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=dev)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _sd_forward(model, inputs: dict):
    return model(
        inputs["noisy_latent"],
        inputs["timestep_embedding"],
        inputs["text_context"],
    )


def _benchmark_forward(model_type: str, model, inputs: dict):
    if model_type == "sd_unet":
        return _sd_forward(model, inputs)
    if model_type == "resnet":
        return model(inputs["images"])
    if model_type == "whisper":
        return model(inputs["input_features"])
    if model_type == "causal_lm":
        kwargs = {k: v for k, v in inputs.items() if k != "labels"}
        return model(**kwargs, use_cache=False)
    return model(**inputs)


def _benchmark_logits(model_type: str, outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if model_type == "whisper":
        return outputs.last_hidden_state
    if model_type == "causal_lm":
        return outputs.logits
    return outputs


def _benchmark_loss(model_type: str, outputs, inputs: dict):
    logits = _benchmark_logits(model_type, outputs)
    if model_type == "sd_unet":
        return F.mse_loss(logits, inputs["noise_target"])
    if model_type == "smolvla":
        return F.mse_loss(logits, torch.zeros_like(logits))
    if model_type == "resnet":
        return F.cross_entropy(logits, inputs["labels"])
    if model_type == "whisper":
        return logits.pow(2).mean()
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size), inputs["labels"].reshape(-1)
    )


def _benchmark_latency_call(model_type: str, model, inputs: dict, dev: str):
    if model_type == "causal_lm":
        token = torch.tensor([[0]], device=dev, dtype=torch.long)
        mask = torch.ones(1, 1, dtype=torch.long, device=dev)
        return lambda: model(input_ids=token, attention_mask=mask, use_cache=False)
    if model_type == "resnet":
        image = torch.zeros(1, 3, 224, 224, device=dev, dtype=torch.float32)
        return lambda: model(image)
    if model_type == "sd_unet":
        return lambda: _sd_forward(model, inputs)
    if model_type == "smolvla":
        latency_inputs = {
            "noisy_actions": inputs["noisy_actions"][:, :1],
            "timestep": inputs["timestep"],
            # Keep all 16 VLM context positions. Slicing every tensor to one
            # position (the legacy runner's behavior) changed the workload.
            "vlm_kv": inputs["vlm_kv"],
        }
        return lambda: model(**latency_inputs)
    if model_type == "whisper":
        return lambda: model(inputs["input_features"])
    raise AssertionError(f"no latency workload for {model_type}")


def _quantile(samples, q):
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _timing_summary(samples):
    return {
        "median": _quantile(samples, 0.5),
        "p25": _quantile(samples, 0.25),
        "p75": _quantile(samples, 0.75),
        "min": min(samples),
        "max": max(samples),
    }


def _measure_call(fn, warmup_runs: int, measurement_runs: int, device, phase, before=None):
    last = None
    samples = []
    for stage, count in (("warmup", warmup_runs), ("measure", measurement_runs)):
        with nsys_range(f"pytorch/{phase}/{stage}"):
            for _ in range(count):
                if before is not None:
                    before()
                synchronize(device)
                with nsys_range("pytorch/sample"):
                    start = time.perf_counter()
                    last = fn()
                    synchronize(device)
                    elapsed = (time.perf_counter() - start) * 1000.0
                if stage == "measure":
                    samples.append(elapsed)
    return last, samples


ALLOCATOR_MEMORY_BASIS = "caching-allocator peak allocated bytes for the phase"
MPS_MEMORY_BASIS = "MPS allocated bytes at end of phase (no peak API)"


def _reset_peak_memory(dev: str) -> None:
    """Clear the allocator high-water mark so a phase measures only itself."""
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    elif dev.startswith("xpu"):
        torch.xpu.reset_peak_memory_stats()


def _phase_memory(dev: str):
    """Allocator accounting for the phase that just ran.

    Returns None where the backend has no allocator statistics, so that an
    unavailable measurement is recorded as missing rather than as zero.
    """
    if dev.startswith("cuda"):
        return {
            "allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "basis": ALLOCATOR_MEMORY_BASIS,
            # Reserved is what the process holds from the driver; allocated
            # is only the live tensor high-water mark inside that pool.
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    if dev.startswith("xpu"):
        return {
            "allocated_bytes": int(torch.xpu.max_memory_allocated()),
            "basis": ALLOCATOR_MEMORY_BASIS,
            "peak_reserved_bytes": int(torch.xpu.max_memory_reserved()),
        }
    if dev == "mps":
        return {
            "allocated_bytes": int(torch.mps.current_allocated_memory()),
            "basis": MPS_MEMORY_BASIS,
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        }
    return None


def _nvml_process_bytes():
    """Per-process device memory through the NVML Python binding."""
    try:
        import pynvml
    except ImportError:
        return None
    handle = None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if process.pid == os.getpid() and process.usedGpuMemory is not None:
                return int(process.usedGpuMemory)
        return None
    except Exception as exc:
        print(f"[pytorch] pynvml unavailable: {exc}", file=sys.stderr)
        return None
    finally:
        if handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _nvidia_smi_process_bytes():
    """Per-process device memory through `nvidia-smi`.

    The binding is not always installed, but the driver always ships this
    tool, so the frozen device matrix can report the figure on every NVIDIA
    machine without adding a runtime dependency.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    import subprocess

    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except Exception as exc:
        print(f"[pytorch] nvidia-smi query failed: {exc}", file=sys.stderr)
        return None
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid, used_mib = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if pid == os.getpid():
            return used_mib * 1024 * 1024
    return None


_MEMORY_UNIT_BYTES = {
    "B": 1, "BYTE": 1, "BYTES": 1,
    "KB": 1024, "KIB": 1024,
    "MB": 1024 ** 2, "MIB": 1024 ** 2,
    "GB": 1024 ** 3, "GIB": 1024 ** 3,
    "TB": 1024 ** 4, "TIB": 1024 ** 4,
}


def _to_bytes(value, unit):
    """Convert an amd-smi {value, unit} pair to bytes.

    amd-smi reports each memory field in its own unit -- VRAM in GB, GTT in
    MB, CPU in B -- so the unit travels with the value and cannot be assumed.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    scale = _MEMORY_UNIT_BYTES.get(str(unit).strip().upper())
    return int(number * scale) if scale else None


def _vram_bytes_in(node):
    """Find a VRAM-keyed memory value anywhere in a subtree.

    Used only after the subtree has been identified as this process's, so it
    deliberately does not re-check the PID: amd-smi nests the figure inside a
    `memory_usage` object that carries no PID of its own.
    """
    if isinstance(node, list):
        for item in node:
            found = _vram_bytes_in(item)
            if found is not None:
                return found
        return None
    if not isinstance(node, dict):
        return None
    for key, value in node.items():
        if "vram" not in str(key).lower():
            continue
        if isinstance(value, dict):
            found = _to_bytes(value.get("value"), value.get("unit"))
        else:
            # rocm-smi writes a bare count, with the unit in the key.
            found = _to_bytes(value, "B")
        if found is not None:
            return found
    for value in node.values():
        found = _vram_bytes_in(value)
        if found is not None:
            return found
    return None


def _node_pid(node):
    value = None
    for key in ("pid", "process_id"):
        for actual, candidate in node.items():
            if str(actual).lower() == key:
                value = candidate
                break
        if value is not None:
            break
    if isinstance(value, dict):
        value = value.get("value")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _key_is_pid(key, pid):
    """Whether a dict key names this PID, as rocm-smi's `PID1234` entries do."""
    text = str(key).strip().upper().replace(" ", "").replace("_", "")
    return text in (str(pid), f"PID{pid}")


def _find_pid_vram_bytes(node, pid):
    """Recursively find this process's VRAM figure in an SMI JSON document.

    The schema differs across amd-smi and rocm-smi versions, so rather than
    hard-coding a nesting that would silently return nothing on a version we
    did not anticipate, this locates the subtree belonging to our PID --
    carried as a field by amd-smi and as a key by rocm-smi -- and then reads
    whatever VRAM figure that subtree holds.
    """
    if isinstance(node, list):
        for item in node:
            found = _find_pid_vram_bytes(item, pid)
            if found is not None:
                return found
        return None
    if not isinstance(node, dict):
        return None

    if _node_pid(node) == pid:
        found = _vram_bytes_in(node)
        if found is not None:
            return found
    for key, value in node.items():
        if _key_is_pid(key, pid):
            found = _vram_bytes_in(value)
            if found is not None:
                return found
    for value in node.values():
        found = _find_pid_vram_bytes(value, pid)
        if found is not None:
            return found
    return None


def _amd_smi_process_bytes():
    """Per-process VRAM through `amd-smi`, the current ROCm tool."""
    return _smi_process_bytes(
        ["amd-smi", "process", "--json", "--pid", str(os.getpid())], "amd-smi"
    )


def _rocm_smi_process_bytes():
    """Per-process VRAM through `rocm-smi`, which amd-smi supersedes."""
    return _smi_process_bytes(["rocm-smi", "--showpids", "--json"], "rocm-smi")


def _smi_process_bytes(command, tool):
    if shutil.which(command[0]) is None:
        return None
    import subprocess

    try:
        output = subprocess.run(
            command, capture_output=True, text=True, timeout=20, check=True
        ).stdout
        document = json.loads(output)
    except Exception as exc:
        print(f"[pytorch] {tool} per-process memory unavailable: {exc}", file=sys.stderr)
        return None
    return _find_pid_vram_bytes(document, os.getpid())


def _process_device_memory(dev: str):
    """Device memory attributed to this process.

    Device-wide free/total is deliberately not used: it counts every tenant
    on the GPU, so an unrelated workload would inflate it. The per-process
    figure is what Meganeura's Vulkan and Metal queries also report, which
    makes it the one memory number comparable across the two engines.
    """
    if dev == "mps":
        # Metal's driver-allocated size is already process-scoped.
        return int(torch.mps.driver_allocated_memory()), "metal-driver-allocated"
    if not dev.startswith("cuda"):
        return None, None
    # A ROCm build reports device strings as "cuda" but has no NVML; route it
    # to the AMD tools instead of failing two NVIDIA probes first.
    if getattr(torch.version, "hip", None):
        process_bytes = _amd_smi_process_bytes()
        if process_bytes is not None:
            return process_bytes, "amd-smi-per-process"
        process_bytes = _rocm_smi_process_bytes()
        if process_bytes is not None:
            return process_bytes, "rocm-smi-per-process"
        return None, None
    process_bytes = _nvml_process_bytes()
    if process_bytes is not None:
        return process_bytes, "nvml-per-process"
    process_bytes = _nvidia_smi_process_bytes()
    if process_bytes is not None:
        return process_bytes, "nvidia-smi-per-process"
    return None, None


def _configure_benchmark_precision(dev: str, strict: bool):
    if strict:
        torch.set_float32_matmul_precision("highest")
        if dev.startswith("cuda"):
            # `highest` is documented as equivalent to this assignment, but
            # set and later report both controls so the artifact proves the
            # effective contract instead of relying on a default.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        return {
            "comparison_class": "strict-f32",
            "tensor_storage": "f32",
            "matmul_inputs": "f32",
            "convolution_inputs": "f32",
            "accumulation": "f32",
            "output": "f32",
            "reduced_precision_allowed": False,
        }
    torch.set_float32_matmul_precision("high")
    if dev.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return {
        "comparison_class": "reduced-input-f32-accumulate",
        "tensor_storage": "f32",
        "matmul_inputs": "TF32 or backend high-precision algorithm",
        "convolution_inputs": "TF32 when supported",
        "accumulation": "f32",
        "output": "f32",
        "reduced_precision_allowed": True,
    }


def bench(model_name: str, spec: dict):
    """Matched benchmark: symmetric samples and full forward/loss/backward."""
    dev = detect_device()
    stream = torch.cuda.Stream(device=dev) if dev.startswith("cuda") and torch.version.cuda else None
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream(dev))
    # Compilation can retain AccumulateGrad nodes. Their stream must remain
    # valid for capture, so all CUDA conditions use one preparation/run stream.
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        _bench(model_name, spec, dev, stream)
    if stream is not None:
        torch.cuda.current_stream(dev).wait_stream(stream)


def _bench(model_name, spec, dev, stream):
    dev_name = device_name(dev)
    backend = backend_name(dev)
    model_type = spec["type"]
    training_requested = os.environ.get("INFERENA_INFERENCE_ONLY", "0") != "1"
    if not training_requested and model_type != "causal_lm":
        raise ValueError("inference-only currently supports SmolLM2 workloads")
    strict = os.environ.get("INFERENA_STRICT", "0") == "1"
    precision_mode = "strict-f32" if strict else "accelerated-f32"
    warmup_runs = int(os.environ.get("INFERENA_WARMUP_RUNS", "5"))
    measurement_runs = int(os.environ.get("INFERENA_MEASUREMENT_RUNS", "20"))
    if warmup_runs < 0 or measurement_runs < 1:
        raise ValueError("warmups must be >= 0 and measurement runs must be >= 1")
    precision = _configure_benchmark_precision(dev, strict)
    mode = os.environ.get("INFERENA_TORCH_MODE", "default")
    modes = torch._inductor.list_mode_options()
    if mode != "eager" and mode not in modes:
        raise ValueError(f"unknown INFERENA_TORCH_MODE: {mode}")
    native_cuda = dev.startswith("cuda") and torch.version.cuda is not None
    graph_setting = os.environ.get("INFERENA_CUDA_GRAPHS", "1" if native_cuda else "0")
    if graph_setting not in ("0", "1"):
        raise ValueError("INFERENA_CUDA_GRAPHS must be 0 or 1")
    use_graphs = graph_setting == "1"
    if use_graphs and not native_cuda:
        raise ValueError("this experiment's explicit graph path requires NVIDIA CUDA")
    execution = {
        "requested_mode": mode,
        "compiled": False,
        "stream_policy": "single dedicated CUDA preparation/run stream" if stream is not None else "backend default",
        "determinism": {
            "algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "cuda_graphs": {"requested": use_graphs, "phases": {}},
    }

    print(
        f"[pytorch] inferena-cuda-graphs-v2: {precision_mode}, {warmup_runs} warmups, "
        f"{measurement_runs} samples",
        file=sys.stderr,
    )
    print(
        f"[pytorch] device: {dev_name} ({dev}), backend: {backend}, "
        f"torch {torch.__version__}",
        file=sys.stderr,
    )
    print(f"[pytorch] determinism: {json.dumps(execution['determinism'])}", file=sys.stderr)

    load_start = time.perf_counter()
    eager_model = load_model(model_name, spec, dev)
    eager_model.to(dev)
    if model_type == "resnet":
        eager_model.eval()
    else:
        eager_model.train()
    synchronize(dev)
    load_s = time.perf_counter() - load_start

    execution["embedding_backward"] = (
        qualify_embedding_backward(eager_model, dev)
        if training_requested and model_type == "causal_lm"
        else {"status": "not-requested"}
    )

    model = eager_model
    compile_s = 0.0
    if mode != "eager" and dev != "mps":
        compile_cache = clear_compile_cache()
        compile_start = time.perf_counter()
        try:
            options = dict(modes[mode])
            # The explicit switch owns graph replay in this experiment. Do not
            # enable hidden partial graph trees in the no-graph control either.
            options["triton.cudagraphs"] = False
            execution["compiler_options"] = options
            candidate = torch.compile(eager_model, options=options)
            compile_inputs = prepare_inputs(model_type, candidate, dev)

            if training_requested:
                candidate.zero_grad(set_to_none=True)
                train_outputs = _benchmark_forward(model_type, candidate, compile_inputs)
                _benchmark_loss(model_type, train_outputs, compile_inputs).backward()
                synchronize(dev)
                candidate.zero_grad(set_to_none=True)
                del train_outputs

            with torch.no_grad():
                _benchmark_forward(model_type, candidate, compile_inputs)
                _benchmark_latency_call(
                    model_type, candidate, compile_inputs, dev
                )()
            synchronize(dev)
            compile_s = time.perf_counter() - compile_start
            model = candidate
            execution["compiled"] = True
            del compile_inputs
        except Exception as exc:
            raise RuntimeError(
                f"requested PyTorch mode {mode!r} failed; no eager timing substituted"
            ) from exc
    else:
        execution["compile_skipped"] = (
            "explicit eager mode" if mode == "eager" else "unsupported platform"
        )

    inputs = prepare_inputs(model_type, model, dev)

    def inference_call():
        with torch.no_grad():
            return _benchmark_logits(model_type, _benchmark_forward(model_type, model, inputs))

    def prepare_phase(name, fn, training_model=None):
        if use_graphs:
            fn, report = capture_phase(fn, training_model, stream=stream,
                                       reduced_precision=precision["reduced_precision_allowed"])
        else:
            report = {"status": "not-requested"}
        execution["cuda_graphs"]["phases"][name] = report
        return fn

    phase_memory = {}
    inference_call = prepare_phase("inference", inference_call)
    _reset_peak_memory(dev)
    inference_outputs, inference_samples = _measure_call(
        inference_call, warmup_runs, measurement_runs, dev, "inference"
    )
    phase_memory["inference"] = _phase_memory(dev)
    logits = _benchmark_logits(model_type, inference_outputs)
    loss = _benchmark_loss(model_type, inference_outputs, inputs)

    def train_call():
        outputs = _benchmark_forward(model_type, model, inputs)
        step_loss = _benchmark_loss(model_type, outputs, inputs)
        step_loss.backward()
        return _benchmark_logits(model_type, outputs), step_loss

    training_samples = None
    if training_requested:
        train_call = prepare_phase("training", train_call, model)
        _reset_peak_memory(dev)
        _, training_samples = _measure_call(
            train_call, warmup_runs, measurement_runs, dev, "training",
            before=None if use_graphs else lambda: model.zero_grad(set_to_none=True),
        )
        phase_memory["training"] = _phase_memory(dev)
    grad_norm_sq = 0.0
    gradient_norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad = parameter.grad.detach().float()
            parameter_norm_sq = float(torch.sum(grad * grad).item())
            grad_norm_sq += parameter_norm_sq
            gradient_norms[
                _gradient_parameter_name(model_type, name)
            ] = parameter_norm_sq ** 0.5
    # Meganeura's default optimizer concatenates each SwiGLU gate/up weight
    # pair into one physical parameter buffer. Compare the invariant norm of
    # that concatenation rather than treating storage layout as model wiring.
    for gate_name in [
        name
        for name in gradient_norms
        if name.endswith(".mlp.gate_proj.weight")
    ]:
        up_name = gate_name.removesuffix("gate_proj.weight") + "up_proj.weight"
        if up_name not in gradient_norms:
            continue
        fused_name = f"{gate_name}+{up_name}"
        gradient_norms[fused_name] = (
            gradient_norms.pop(gate_name) ** 2
            + gradient_norms.pop(up_name) ** 2
        ) ** 0.5
    grad_norm = grad_norm_sq ** 0.5 if training_requested else None

    latency_call = _benchmark_latency_call(model_type, model, inputs, dev)

    def no_grad_latency():
        with torch.no_grad():
            return _benchmark_logits(model_type, latency_call())

    no_grad_latency = prepare_phase("latency", no_grad_latency)
    _reset_peak_memory(dev)
    _, latency_samples = _measure_call(
        no_grad_latency, warmup_runs, measurement_runs, dev, "latency"
    )
    phase_memory["latency"] = _phase_memory(dev)

    inference_summary = _timing_summary(inference_samples)
    training_summary = _timing_summary(training_samples) if training_requested else None
    latency_summary = _timing_summary(latency_samples)
    precision["torch_float32_matmul_precision"] = (
        torch.get_float32_matmul_precision()
    )
    if dev.startswith("cuda"):
        precision["cuda_matmul_allow_tf32"] = bool(
            torch.backends.cuda.matmul.allow_tf32
        )
        precision["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        precision["nvidia_tf32_override"] = os.environ.get(
            "NVIDIA_TF32_OVERRIDE"
        )
        precision["torch_allow_tf32_cublas_override"] = os.environ.get(
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
        )

    process_bytes, process_source = _process_device_memory(dev)
    device_total_bytes = None
    if dev.startswith("cuda"):
        device_total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    elif dev.startswith("xpu"):
        device_total_bytes = int(torch.xpu.get_device_properties(dev).total_memory)
    memory_report = {
        "device": {
            "total_bytes": device_total_bytes,
            "process_bytes": process_bytes,
            "process_source": process_source,
            "sampled_at": "after all measured series",
        },
        "phases": {
            name: values
            for name, values in phase_memory.items()
            if values is not None
        },
        # Phases run in sequence on one process, so allocations made by an
        # earlier phase are still resident: the training peak includes the
        # inference activations' pool, and the latency peak includes the
        # gradients. Each figure bounds its phase from above, not exactly.
        "notes": (
            "Phases run in order inference, training, latency without "
            "releasing the allocator pool; each peak therefore includes "
            "residency established by earlier phases."
        ),
    }
    if not memory_report["phases"]:
        memory_report = None

    logits_hash = sha256_f32_tensor(logits)
    logits_flat = logits.detach().float().cpu().flatten()
    sample_count = min(256, logits_flat.numel())
    if sample_count == logits_flat.numel():
        logits_sample = logits_flat.tolist()
    else:
        logits_sample = [
            logits_flat[index * (logits_flat.numel() - 1) // (sample_count - 1)].item()
            for index in range(sample_count)
        ]
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_build_config": torch.__config__.show(),
        "inductor_compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        "cuda_version": torch.version.cuda,
        "hip_version": torch.version.hip,
        "xpu_version": getattr(torch.version, "xpu", None),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if dev.startswith("cuda"):
        properties = torch.cuda.get_device_properties(0)
        environment.update({
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "device_total_memory_bytes": properties.total_memory,
            "device_multiprocessor_count": properties.multi_processor_count,
            "device_uuid": str(properties.uuid),
        })
    elif dev.startswith("xpu"):
        properties = torch.xpu.get_device_properties(dev)
        environment.update({
            "device_total_memory_bytes": properties.total_memory,
            "device_properties": str(properties),
        })
    if execution["compiled"] and dev.startswith(("cuda", "xpu")):
        from torch._inductor.utils import is_big_gpu
        environment["inductor_is_big_gpu"] = is_big_gpu(torch.device(dev))
    profiles = {}
    if profile_dir := os.environ.get("INFERENA_PROFILE_DIR"):
        samples = int(os.environ.get("INFERENA_PROFILE_SAMPLES", "3"))
        if samples < 1:
            raise ValueError("profile samples must be positive")
        for name, fn in (
            ("inference", inference_call), ("training", train_call),
            ("latency", no_grad_latency),
        ):
            if name == "training" and not training_requested:
                continue
            before = (
                lambda: model.zero_grad(set_to_none=True)
            ) if name == "training" and not use_graphs else None
            path = os.path.join(profile_dir, f"{model_name}_pytorch_{name}.json")
            profiles[name] = profile_phase(fn, path, samples, before, device=dev)

    result = {
        "framework": "pytorch",
        "framework_rev": torch.__version__,
        "model": model_name,
        "device": dev_name,
        "gpu_name": dev_name,
        "torch_version": torch.__version__,
        "backend": backend,
        "environment": environment,
        "execution": execution,
        "profile_artifacts": profiles,
        "protocol": {
            "name": "inferena-cuda-graphs-v2",
            "warmup_runs": warmup_runs,
            "measurement_runs": measurement_runs,
            "statistic": "median",
            "training_requested": training_requested,
            "training_scope": "forward + loss + backward; no optimizer update" if training_requested else None,
            "diagnostic": "INFERENA_NSYS" in os.environ,
            "timing_scope": "synchronized host wall time; resident inputs; no readback",
            "gradient_reset": (
                "captured backward overwrites stable gradient buffers"
                if use_graphs else "set_to_none outside timed region"
            ),
            "capture_scope": "per-phase preparation and qualification, outside timing",
            "compile_scope": (
                "torch.compile plus first specializations of requested phases"
            ),
        },
        "precision": precision,
        "timings": {
            "compile_s": round(compile_s, 3),
            "inference_ms": round(inference_summary["median"], 3),
            "latency_ms": round(latency_summary["median"], 3),
            "training_ms": round(training_summary["median"], 3) if training_requested else None,
        },
        "timing_samples_ms": {
            "inference": inference_samples,
            "latency": latency_samples,
            "training": training_samples,
        },
        "timing_summary_ms": {
            "inference": inference_summary,
            "latency": latency_summary,
            "training": training_summary,
        },
        "memory": memory_report,
        "outputs": {
            "logits_hash": logits_hash,
            "output_shape": list(logits.shape),
            "logits_sample": [round(v, 6) for v in logits_sample],
            "loss": round(float(loss.item()), 6),
            "grad_norm": round(grad_norm, 6) if training_requested else None,
            "gradient_norms": {
                name: round(value, 9)
                for name, value in sorted(gradient_norms.items())
            },
        },
        "load_s": round(load_s, 3),
    }
    if model_type == "causal_lm":
        result["workload_metrics"] = {
            "prefill_ms": round(inference_summary["median"], 3),
            "prefill_tokens": int(inputs["input_ids"].shape[1]),
            "stateless_one_token_ms": round(latency_summary["median"], 3),
            "has_kv_cache": False,
            "decode_ms": None,
        }
    print(json.dumps(result))


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M"
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)
    if os.environ.get("INFERENA_DRY_RUN") == "1":
        print(f"[pytorch] dry-run OK: {model_name} ({spec['type']})", file=sys.stderr)
        sys.exit(0)
    if os.environ.get("INFERENA_LEGACY", "0") == "1":
        raise ValueError("legacy runner removed; reproduce it at paper-arxiv-1")
    bench(model_name, spec)
