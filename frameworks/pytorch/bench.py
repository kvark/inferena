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

import hashlib
import json
import os
import platform
import shutil
import struct
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _xpu_actually_works() -> bool:
    """XPU may report available but fail at kernel-launch time on older Intel
    iGPUs (Gen12 Raptor/Alder Lake UHD) — JIT compilation aborts with
    "program was built for 1 devices". Probe with a trivial matmul."""
    try:
        x = torch.ones(4, 4, device="xpu")
        _ = (x @ x.t()).cpu()
        return True
    except Exception as e:
        print(f"[pytorch] XPU present but compute probe failed ({e}); falling back", file=sys.stderr)
        return False


def detect_device() -> str:
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
    raw = struct.pack(f"<{flat.numel()}f", *flat.tolist())
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def clear_compile_cache():
    """Clear torch inductor cache so we measure real compilation time."""
    torch._dynamo.reset()
    for d in [
        os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "torch", "inductor",
        ),
    ]:
        if d and os.path.isdir(d):
            print(f"  clearing compile cache: {d}", file=sys.stderr)
            shutil.rmtree(d, ignore_errors=True)


def capture_cuda_graph(fn, warmup: int = 3):
    """Warm up fn on a side stream, then capture a no_grad replay graph.

    Used when torch.compile is unavailable (Windows/Triton missing) to reclaim
    the kernel-launch overhead that dominates small-model CUDA timings.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        fn()
    return g


def capture_cuda_graph_train(fwd_bwd_fn, model, warmup: int = 3):
    """Capture forward+backward as a CUDA graph for training.

    Prerequisites for capture:
    - All prior AccumulateGrad nodes must be destroyed before side-stream
      warmup, so new ones are created on the capture stream (otherwise
      stream mismatch invalidates the capture).
    - Parameters and inputs must live at stable addresses across replays.
    - No dynamic shapes, no Python control flow inside the captured region.

    fwd_bwd_fn should run one forward+backward pass using pre-set inputs.
    """
    # Fully detach gradients — `grad = None` destroys AccumulateGrad so the
    # next backward rebuilds it on whatever stream is active.
    for p in model.parameters():
        p.grad = None

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            for p in model.parameters():
                p.grad = None
            fwd_bwd_fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Now gradients exist and live at stable addresses. Zero them (preserving
    # the tensors) and capture: the captured graph zeros grads then adds to them.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        fwd_bwd_fn()
    return g


# --- Model registry ---

MODEL_REGISTRY = {
    "SmolLM2-135M": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "type": "causal_lm",
    },
    "SmolLM2-360M": {
        "hf_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
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
            print(f"[pytorch] local load failed ({e})", file=sys.stderr)

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
        return model(**kwargs)
    return model(**inputs)


def _benchmark_logits(model_type: str, outputs):
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
        return lambda: model(input_ids=token, attention_mask=mask)
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


def _measure_call(fn, warmup_runs: int, measurement_runs: int, before=None):
    last = None
    for _ in range(warmup_runs):
        if before is not None:
            before()
        last = fn()
        sync()

    samples = []
    for _ in range(measurement_runs):
        if before is not None:
            before()
        sync()
        start = time.perf_counter()
        last = fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000.0)
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


def bench_v2(model_name: str, spec: dict):
    """Matched benchmark: symmetric samples and full forward/loss/backward."""
    dev = detect_device()
    dev_name = device_name(dev)
    backend = backend_name(dev)
    model_type = spec["type"]
    strict = os.environ.get("INFERENA_STRICT", "0") == "1"
    precision_mode = "strict-f32" if strict else "accelerated-f32"
    warmup_runs = int(os.environ.get("INFERENA_WARMUP_RUNS", "5"))
    measurement_runs = int(os.environ.get("INFERENA_MEASUREMENT_RUNS", "20"))
    if warmup_runs < 0 or measurement_runs < 1:
        raise ValueError("warmups must be >= 0 and measurement runs must be >= 1")
    precision = _configure_benchmark_precision(dev, strict)

    print(
        f"[pytorch] inferena-paper-v1: {precision_mode}, {warmup_runs} warmups, "
        f"{measurement_runs} samples",
        file=sys.stderr,
    )
    print(
        f"[pytorch] device: {dev_name} ({dev}), backend: {backend}, "
        f"torch {torch.__version__}",
        file=sys.stderr,
    )

    load_start = time.perf_counter()
    eager_model = load_model(model_name, spec, dev)
    eager_model.to(dev)
    if model_type == "resnet":
        eager_model.eval()
    else:
        eager_model.train()
    sync()
    load_s = time.perf_counter() - load_start

    model = eager_model
    compile_s = 0.0
    if dev != "mps" and sys.platform != "win32":
        clear_compile_cache()
        compile_start = time.perf_counter()
        try:
            candidate = torch.compile(eager_model)
            compile_inputs = prepare_inputs(model_type, candidate, dev)

            candidate.zero_grad(set_to_none=True)
            train_outputs = _benchmark_forward(model_type, candidate, compile_inputs)
            _benchmark_loss(model_type, train_outputs, compile_inputs).backward()
            sync()
            candidate.zero_grad(set_to_none=True)

            with torch.no_grad():
                _benchmark_forward(model_type, candidate, compile_inputs)
                _benchmark_latency_call(
                    model_type, candidate, compile_inputs, dev
                )()
            sync()
            compile_s = time.perf_counter() - compile_start
            model = candidate
        except Exception as exc:
            message = str(exc).split("\n")[0][:200]
            print(
                f"[pytorch] torch.compile failed ({message}); using eager mode",
                file=sys.stderr,
            )
            torch._dynamo.reset()
            model = eager_model
            compile_s = 0.0

    inputs = prepare_inputs(model_type, model, dev)

    def inference_call():
        with torch.no_grad():
            return _benchmark_forward(model_type, model, inputs)

    phase_memory = {}
    _reset_peak_memory(dev)
    inference_outputs, inference_samples = _measure_call(
        inference_call, warmup_runs, measurement_runs
    )
    phase_memory["inference"] = _phase_memory(dev)
    logits = _benchmark_logits(model_type, inference_outputs)
    loss = _benchmark_loss(model_type, inference_outputs, inputs)

    def train_call():
        outputs = _benchmark_forward(model_type, model, inputs)
        step_loss = _benchmark_loss(model_type, outputs, inputs)
        step_loss.backward()
        return outputs, step_loss

    _reset_peak_memory(dev)
    _, training_samples = _measure_call(
        train_call,
        warmup_runs,
        measurement_runs,
        before=lambda: model.zero_grad(set_to_none=True),
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
    grad_norm = grad_norm_sq ** 0.5

    latency_call = _benchmark_latency_call(model_type, model, inputs, dev)

    def no_grad_latency():
        with torch.no_grad():
            return latency_call()

    _reset_peak_memory(dev)
    _, latency_samples = _measure_call(
        no_grad_latency, warmup_runs, measurement_runs
    )
    phase_memory["latency"] = _phase_memory(dev)

    inference_summary = _timing_summary(inference_samples)
    training_summary = _timing_summary(training_samples)
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
        "cuda_version": torch.version.cuda,
        "hip_version": torch.version.hip,
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
    result = {
        "framework": "pytorch",
        "framework_rev": torch.__version__,
        "model": model_name,
        "device": dev_name,
        "gpu_name": dev_name,
        "torch_version": torch.__version__,
        "backend": backend,
        "environment": environment,
        "protocol": {
            "name": "inferena-paper-v1",
            "warmup_runs": warmup_runs,
            "measurement_runs": measurement_runs,
            "statistic": "median",
            "training_scope": "forward + loss + backward; no optimizer update",
            "compile_scope": (
                "torch.compile plus first training, inference, and latency "
                "specializations"
            ),
        },
        "precision": precision,
        "timings": {
            "compile_s": round(compile_s, 3),
            "inference_ms": round(inference_summary["median"], 3),
            "latency_ms": round(latency_summary["median"], 3),
            "training_ms": round(training_summary["median"], 3),
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
            "grad_norm": round(grad_norm, 6),
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


def bench(model_name: str, spec: dict):
    dev = detect_device()
    dev_name = device_name(dev)
    backend = backend_name(dev)
    model_type = spec["type"]
    torch.set_float32_matmul_precision("high")

    print(f"[pytorch] device: {dev_name} ({dev}), backend: {backend}, torch {torch.__version__}", file=sys.stderr)

    # --- Load model ---
    print(f"[pytorch] loading {spec['hf_id']}...", file=sys.stderr)
    t0 = time.perf_counter()
    model = load_model(model_name, spec, dev)
    model.to(dev)
    if model_type == "resnet":
        model.eval()  # keep eval for fused-BN matching with meganeura
    else:
        model.train()
    sync()
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[pytorch] loaded in {load_ms:.0f}ms", file=sys.stderr)

    # --- torch.compile ---
    # Skip on MPS (poorly supported, adds overhead) and on Windows
    # (CPU path needs MSVC cl.exe; CUDA path needs Triton, which has no
    # official Windows wheels). Eager mode still runs so correctness
    # comparisons against other frameworks remain valid.
    if dev == "mps":
        compile_s = 0.0
        print("[pytorch] skipping torch.compile on MPS (not well supported)", file=sys.stderr)
    elif sys.platform == "win32":
        compile_s = 0.0
        print("[pytorch] skipping torch.compile on Windows (Triton unsupported)", file=sys.stderr)
    else:
        print("[pytorch] compiling with torch.compile()...", file=sys.stderr)
        clear_compile_cache()
        compile_t0 = time.perf_counter()
        try:
            compiled = torch.compile(model)

            # Force compilation with a dummy forward+backward pass.
            # Must run WITH gradients — compiling under no_grad() produces different
            # code, causing a costly recompilation on the first grad-enabled forward.
            dummy_kwargs = prepare_inputs(model_type, compiled, dev)
            if model_type == "sd_unet":
                dummy_out = _sd_forward(compiled, dummy_kwargs)
                F.mse_loss(dummy_out, dummy_kwargs["noise_target"]).backward()
            elif model_type == "smolvla":
                dummy_out = compiled(**dummy_kwargs)
                F.mse_loss(dummy_out, torch.zeros_like(dummy_out)).backward()
            elif model_type == "resnet":
                dummy_out = compiled(dummy_kwargs["images"])
                F.cross_entropy(dummy_out, dummy_kwargs["labels"]).backward()
            elif model_type == "whisper":
                dummy_out = compiled(dummy_kwargs["input_features"])
                dummy_out.last_hidden_state.sum().backward()
            else:
                # Drop labels — HF's built-in `loss` shifts logits/labels by
                # one position internally, assuming labels are input_ids-
                # aligned. Ours are pre-shifted (labels[i] = next token after
                # position i), so relying on outputs.loss double-shifts.
                dummy_kw = {k: v for k, v in dummy_kwargs.items() if k != "labels"}
                dummy_out = compiled(**dummy_kw)
                vocab_size = dummy_out.logits.shape[-1]
                F.cross_entropy(
                    dummy_out.logits.reshape(-1, vocab_size), dummy_kwargs["labels"].reshape(-1)
                ).backward()
            compiled.zero_grad()
            sync()
            model = compiled
            compile_s = time.perf_counter() - compile_t0
            print(f"[pytorch] compiled in {compile_s:.2f}s", file=sys.stderr)
        except Exception as e:
            # Inductor CPU backend needs a C++ toolchain + Python headers
            # (Python.h). On minimal Linux setups without python3-dev this
            # fails; XPU/CUDA kernel compilation can also fail on unsupported
            # hardware. Eager mode still runs — keep going with zero compile
            # time so the rest of the bench still produces valid results.
            msg = str(e).split("\n")[0][:200]
            print(f"[pytorch] torch.compile failed ({msg}); falling back to eager", file=sys.stderr)
            torch._dynamo.reset()
            compile_s = 0.0

    # --- Prepare deterministic input ---
    fwd_kwargs = prepare_inputs(model_type, model, dev)

    # --- Forward ---
    sync()
    t0 = time.perf_counter()
    if model_type == "sd_unet":
        target = fwd_kwargs["noise_target"]
        outputs = _sd_forward(model, fwd_kwargs)
    elif model_type == "resnet":
        outputs = model(fwd_kwargs["images"])
    elif model_type == "whisper":
        outputs = model(fwd_kwargs["input_features"])
    else:
        # Drop labels — see the loss-computation comment below for why we
        # never let HF compute `outputs.loss` for causal_lm internally.
        fwd_only_kwargs = {k: v for k, v in fwd_kwargs.items() if k != "labels"}
        outputs = model(**fwd_only_kwargs)
    sync()
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # --- Loss ---
    if model_type == "sd_unet":
        loss = F.mse_loss(outputs, target)
        logits = outputs
    elif model_type == "smolvla":
        target = torch.zeros_like(outputs)
        loss = F.mse_loss(outputs, target)
        logits = outputs
    elif model_type == "resnet":
        logits = outputs
        loss = F.cross_entropy(outputs, fwd_kwargs["labels"])
    elif model_type == "whisper":
        logits = outputs.last_hidden_state  # encoder hidden states
        loss = logits.pow(2).mean()  # MSE vs zero (matches meganeura)
    else:
        # Compute cross-entropy manually instead of using outputs.loss.
        # `prepare_inputs` already pre-shifts labels (labels[i] = the token
        # at position i+1), but HF's built-in `loss` shifts AGAIN internally
        # (comparing logits[i] against labels[i+1], assuming labels come in
        # input_ids-aligned) — that double shift silently inflates the loss
        # and was making PyTorch's own "ground truth" wrong in correctness
        # checks against other frameworks (which compute it manually, like
        # this, without relying on the model's internal loss).
        logits = outputs.logits
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), fwd_kwargs["labels"].reshape(-1))

    # --- Backward ---
    sync()
    t0 = time.perf_counter()
    loss.backward()
    sync()
    training_ms = (time.perf_counter() - t0) * 1000.0

    # --- Latency (minimal-input forward) ---
    # Measure single-sample / single-token / minimal-batch forward pass.
    # Warm-up pass first so torch.compile doesn't recompile during timing.
    model.zero_grad()
    if model_type == "causal_lm":
        lat_input = torch.tensor([[0]], device=dev, dtype=torch.long)
        lat_mask = torch.ones(1, 1, dtype=torch.long, device=dev)
        lat_fn = lambda: model(input_ids=lat_input, attention_mask=lat_mask)
    elif model_type == "resnet":
        lat_img = torch.zeros(1, 3, 224, 224, device=dev, dtype=torch.float32)
        lat_fn = lambda: model(lat_img)
    elif model_type == "sd_unet":
        # The conditioned workload is already batch 1.
        lat_fn = lambda: _sd_forward(model, fwd_kwargs)
    elif model_type == "smolvla":
        # Single action chunk (batch=1, chunk_size=1).
        lat_kw = {k: v[:, :1] if v.dim() >= 2 else v for k, v in fwd_kwargs.items()}
        lat_fn = lambda: model(**lat_kw)
    elif model_type == "whisper":
        lat_fn = lambda: model(fwd_kwargs["input_features"])
    else:
        lat_fn = None

    if lat_fn is not None:
        with torch.no_grad():
            lat_fn()
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            lat_fn()
        sync()
        latency_ms = (time.perf_counter() - t0) * 1000.0
    else:
        latency_ms = 0.0

    # --- CUDA graph replay (fallback when torch.compile is unavailable) ---
    # On Windows Triton is missing, so we lose the graph-capture pass
    # Inductor would do. Capturing a manual CUDA graph here closes most of
    # that gap on inference + latency (no_grad) AND training (forward+backward).
    if dev.startswith("cuda") and compile_s == 0.0:
        if model_type == "sd_unet":
            inf_fn = lambda: _sd_forward(model, fwd_kwargs)
        elif model_type == "resnet":
            inf_fn = lambda: model(fwd_kwargs["images"])
        elif model_type == "whisper":
            inf_fn = lambda: model(fwd_kwargs["input_features"])
        elif model_type == "causal_lm":
            # Drop labels so the model returns logits only (no internal loss).
            inf_kw = {k: v for k, v in fwd_kwargs.items() if k != "labels"}
            inf_fn = lambda: model(**inf_kw)
        else:  # smolvla
            inf_fn = lambda: model(**fwd_kwargs)

        try:
            print("[pytorch] capturing CUDA graph for inference...", file=sys.stderr)
            inf_graph = capture_cuda_graph(inf_fn)
            sync()
            t0 = time.perf_counter()
            inf_graph.replay()
            sync()
            inference_ms = (time.perf_counter() - t0) * 1000.0

            if lat_fn is not None:
                print("[pytorch] capturing CUDA graph for latency...", file=sys.stderr)
                lat_graph = capture_cuda_graph(lat_fn)
                sync()
                t0 = time.perf_counter()
                lat_graph.replay()
                sync()
                latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            print(f"[pytorch] CUDA graph capture failed ({e}); keeping eager timings", file=sys.stderr)

        # --- Training CUDA graph (forward + backward) ---
        # Free prior autograd graph refs — the earlier loss.backward() and
        # lingering logits/loss tensors keep AccumulateGrad nodes alive on
        # the default stream, which breaks capture on a side stream.
        # Save scalar values before dropping the tensors.
        loss_val = float(loss.item())
        logits_saved = logits.detach()
        del outputs, loss, logits
        try:
            del target, noisy
        except (NameError, UnboundLocalError):
            pass
        import gc
        gc.collect()
        torch.cuda.synchronize()

        # Suppress the stream-mismatch warning — we've done our best to clear
        # prior refs. If capture still fails we fall back to eager timing.
        try:
            torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        except AttributeError:
            pass

        # Define the forward+backward closure per model type.
        captured_logits = [None]
        def _train_step():
            if model_type == "sd_unet":
                out = _sd_forward(model, fwd_kwargs)
                captured_logits[0] = out
                F.mse_loss(out, fwd_kwargs["noise_target"]).backward()
            elif model_type == "smolvla":
                out = model(**fwd_kwargs)
                captured_logits[0] = out
                F.mse_loss(out, torch.zeros_like(out)).backward()
            elif model_type == "resnet":
                out = model(fwd_kwargs["images"])
                captured_logits[0] = out
                F.cross_entropy(out, fwd_kwargs["labels"]).backward()
            elif model_type == "whisper":
                out = model(fwd_kwargs["input_features"])
                captured_logits[0] = out.last_hidden_state
                out.last_hidden_state.pow(2).mean().backward()
            else:  # causal_lm
                # Drop labels so the model returns logits only (no internal
                # loss) — see the manual cross_entropy comment above.
                inf_kw = {k: v for k, v in fwd_kwargs.items() if k != "labels"}
                out = model(**inf_kw)
                captured_logits[0] = out.logits
                vocab_size = out.logits.shape[-1]
                F.cross_entropy(
                    out.logits.reshape(-1, vocab_size), fwd_kwargs["labels"].reshape(-1)
                ).backward()

        try:
            print("[pytorch] capturing CUDA graph for training...", file=sys.stderr)
            train_graph = capture_cuda_graph_train(_train_step, model)
            sync()
            t0 = time.perf_counter()
            train_graph.replay()
            sync()
            training_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[pytorch] training (graph): {training_ms:.2f}ms", file=sys.stderr)
        except Exception as e:
            print(f"[pytorch] training CUDA graph capture failed ({e}); keeping eager training timing", file=sys.stderr)

    # --- Collect outputs ---
    # Use saved tensor if training CUDA graph deleted the originals.
    logits_src = logits if 'logits' in dir() and isinstance(locals().get('logits', None), torch.Tensor) else logits_saved
    loss_out = loss.item() if 'loss' in dir() and hasattr(locals().get('loss', None), 'item') else loss_val
    logits_hash = sha256_f32_tensor(logits_src)
    logits_flat = logits_src.detach().float().cpu().flatten()
    logits_sample = logits_flat[:16].tolist()

    result = {
        "framework": "pytorch",
        "framework_rev": torch.__version__,
        "model": model_name,
        "device": dev_name,
        "gpu_name": dev_name,
        "torch_version": torch.__version__,
        "backend": backend,
        "timings": {
            "compile_s": round(compile_s, 2),
            "inference_ms": round(inference_ms, 3),
            "latency_ms": round(latency_ms, 3),
            "training_ms": round(training_ms, 3),
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": [round(v, 6) for v in logits_sample],
            "loss": round(loss_out, 6),
        },
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
        bench(model_name, spec)
    else:
        bench_v2(model_name, spec)
