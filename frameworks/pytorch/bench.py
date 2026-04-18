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


# --- SD U-Net (matches meganeura's sd_unet module: simplified, no cross-attn/timestep) ---

class ResBlock(nn.Module):
    """GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Conv3×3 + residual."""
    def __init__(self, in_c, out_c, num_groups=16, eps=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_c, eps=eps)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_c, eps=eps)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.res_conv = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.res_conv(x)


class SDUNet(nn.Module):
    """Simplified SD U-Net matching meganeura's sd_unet::SDUNetConfig::small().

    Architecture: Conv_in → [ResBlock + Downsample]×N → Middle ResBlock
                  → [Upsample + CatSkip + ResBlock]×N → GroupNorm → SiLU → Conv_out

    No cross-attention, no timestep embedding — pure convolutional U-Net.
    """
    def __init__(self, in_channels=4, base_channels=64, num_levels=3,
                 num_groups=16, eps=1e-5):
        super().__init__()
        self.num_levels = num_levels

        # Input conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False)

        # Encoder
        ch_mults = [1 << i for i in range(num_levels)]
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        prev_c = base_channels
        for level, mult in enumerate(ch_mults):
            out_c = base_channels * mult
            self.encoder_blocks.append(ResBlock(prev_c, out_c, num_groups, eps))
            if level < num_levels - 1:
                self.downsamples.append(
                    nn.Conv2d(out_c, out_c, 3, stride=2, padding=1, bias=False)
                )
            else:
                self.downsamples.append(nn.Identity())
            prev_c = out_c

        # Middle
        self.middle = ResBlock(prev_c, prev_c, num_groups, eps)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(num_levels)):
            out_c = base_channels * ch_mults[level]
            skip_c = out_c  # from encoder
            if level < num_levels - 1:
                self.upsamples.append(nn.Upsample(scale_factor=2, mode='nearest'))
            else:
                self.upsamples.append(nn.Identity())
            self.decoder_blocks.append(ResBlock(prev_c + skip_c, out_c, num_groups, eps))
            prev_c = out_c

        # Output
        self.norm_out = nn.GroupNorm(num_groups, base_channels, eps=eps)
        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)

        # Encoder
        skips = []
        for level in range(self.num_levels):
            x = self.encoder_blocks[level](x)
            skips.append(x)
            if level < self.num_levels - 1:
                x = self.downsamples[level](x)

        # Middle
        x = self.middle(x)

        # Decoder
        for i, level in enumerate(reversed(range(self.num_levels))):
            if level < self.num_levels - 1:
                x = self.upsamples[i](x)
            x = torch.cat([x, skips[level]], dim=1)
            x = self.decoder_blocks[i](x)

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
        self.gate = nn.Linear(dim, intermediate, bias=False)
        self.up = nn.Linear(dim, intermediate, bias=False)
        self.down = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GQAttention(nn.Module):
    def __init__(self, dim, num_heads=15, num_kv_heads=5, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.kv_repeat = num_heads // num_kv_heads

    def forward(self, q_input, kv_input=None):
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
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, sq, self.num_heads * self.head_dim)
        return self.o_proj(out)


class ExpertLayer(nn.Module):
    def __init__(self, dim, intermediate, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GQAttention(dim, num_heads, num_kv_heads, head_dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, intermediate)

    def forward(self, x, kv=None):
        x = x + self.attn(self.norm1(x), kv)
        x = x + self.mlp(self.norm2(x))
        return x


class ActionExpert(nn.Module):
    def __init__(self, action_dim=32, expert_hidden=720, intermediate=2048,
                 num_layers=16, num_heads=15, num_kv_heads=5, head_dim=64,
                 vlm_kv_dim=320, self_attn_every_n=2):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, expert_hidden, bias=False)
        self.time_proj = nn.Linear(expert_hidden * 2, expert_hidden, bias=False)
        self.kv_proj = nn.Linear(vlm_kv_dim, expert_hidden, bias=False)
        self.layers = nn.ModuleList([
            ExpertLayer(expert_hidden, intermediate, num_heads, num_kv_heads, head_dim)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(expert_hidden)
        self.head = nn.Linear(expert_hidden, action_dim, bias=False)
        self.self_attn_every_n = self_attn_every_n

    def forward(self, noisy_actions, timestep, vlm_kv):
        x = self.action_proj(noisy_actions) + self.time_proj(timestep)
        kv = self.kv_proj(vlm_kv)
        for i, layer in enumerate(self.layers):
            if i % self.self_attn_every_n == 0:
                x = layer(x)  # self-attention
            else:
                x = layer(x, kv)  # cross-attention
        return self.head(self.norm(x))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
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


def _name_seeded_init(p, name):
    """Fill parameter with sin(j*0.01 + name_seed) * 0.02."""
    seed = _name_seed(name)
    n = p.numel()
    p.copy_(torch.sin(torch.arange(n, dtype=torch.float32) * 0.01 + seed).view_as(p) * _INIT_SCALE)


def _transposed_init(p, name):
    """Init as [in, out] (meganeura layout), store as [out, in] (PyTorch layout).

    Ensures x @ W_torch.T == x @ W_mega for matching forward passes.
    """
    seed = _name_seed(name)
    out_f, in_f = p.shape
    w = torch.sin(torch.arange(in_f * out_f, dtype=torch.float32) * 0.01 + seed).view(in_f, out_f) * _INIT_SCALE
    p.copy_(w.T)


def _resnet_init(model):
    """Deterministic ResNet init matching meganeura's fused-BN approach.

    BN → identity (weight=1, bias=0); conv/fc → name-seeded sin values.
    FC weight uses transposed init to match meganeura's [in, out] matmul.
    Model set to eval mode so BN uses running stats (mean=0, var=1 → identity).

    Scale 0.01 (not 0.1) to prevent activation explosion through 50+ layers
    with identity BN — residual connections keep activations bounded.
    """
    scale = 0.01
    with torch.no_grad():
        for name, p in model.named_parameters():
            if '.bn' in name or '.downsample.1.' in name:
                if 'weight' in name:
                    p.fill_(1.0)
                else:
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
                _transposed_init(p, name)
            else:
                _name_seeded_init(p, name)
    return encoder


def _random_init(model_type: str, model_name: str):
    if model_type == "sd_unet":
        # Match meganeura's SDUNetConfig::small()
        model = SDUNet(
            in_channels=4, base_channels=64, num_levels=3,
            num_groups=16, eps=1e-5,
        ).to(torch.float32)
        # Name-seeded init: framework-independent, matches meganeura's init_params.
        with torch.no_grad():
            for name, p in model.named_parameters():
                _name_seeded_init(p, name)
        return model
    if model_type == "smolvla":
        model = ActionExpert().to(torch.float32)
        with torch.no_grad():
            for name, p in model.named_parameters():
                _name_seeded_init(p, name)
        return model
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
        # Match meganeura's SDUNetConfig::small(): batch=2, in_c=4, res=32.
        batch, in_c, res = 2, 4, 32
        in_size = batch * in_c * res * res
        noisy = torch.tensor(
            [(i * 0.01) for i in range(in_size)],
            dtype=torch.float32, device=dev,
        ).sin().reshape(batch, in_c, res, res)
        target = torch.tensor(
            [(i * 0.007) for i in range(in_size)],
            dtype=torch.float32, device=dev,
        ).cos().reshape(batch, in_c, res, res)
        return {"noisy_latent": noisy, "noise_target": target}

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
        model = torch.compile(model)

        # Force compilation with a dummy forward+backward pass.
        # Must run WITH gradients — compiling under no_grad() produces different
        # code, causing a costly recompilation on the first grad-enabled forward.
        dummy_kwargs = prepare_inputs(model_type, model, dev)
        if model_type == "sd_unet":
            dummy_out = model(dummy_kwargs["noisy_latent"])
            F.mse_loss(dummy_out, dummy_kwargs["noise_target"]).backward()
        elif model_type == "smolvla":
            dummy_out = model(**dummy_kwargs)
            F.mse_loss(dummy_out, torch.zeros_like(dummy_out)).backward()
        elif model_type == "resnet":
            dummy_out = model(dummy_kwargs["images"])
            F.cross_entropy(dummy_out, dummy_kwargs["labels"]).backward()
        elif model_type == "whisper":
            dummy_out = model(dummy_kwargs["input_features"])
            dummy_out.last_hidden_state.sum().backward()
        else:
            dummy_out = model(**dummy_kwargs)
            dummy_out.loss.backward()
        model.zero_grad()
        sync()
        compile_s = time.perf_counter() - compile_t0
        print(f"[pytorch] compiled in {compile_s:.2f}s", file=sys.stderr)

    # --- Prepare deterministic input ---
    fwd_kwargs = prepare_inputs(model_type, model, dev)

    # --- Forward ---
    sync()
    t0 = time.perf_counter()
    if model_type == "sd_unet":
        noisy = fwd_kwargs["noisy_latent"]
        target = fwd_kwargs["noise_target"]
        outputs = model(noisy)
    elif model_type == "resnet":
        outputs = model(fwd_kwargs["images"])
    elif model_type == "whisper":
        outputs = model(fwd_kwargs["input_features"])
    else:
        outputs = model(**fwd_kwargs)
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
        loss = outputs.loss
        logits = outputs.logits

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
        # Single-sample latency (batch=1).
        lat_noisy = fwd_kwargs["noisy_latent"][:1]
        lat_fn = lambda: model(lat_noisy)
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
            inf_fn = lambda: model(fwd_kwargs["noisy_latent"])
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
                out = model(fwd_kwargs["noisy_latent"])
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
                out = model(**fwd_kwargs)
                captured_logits[0] = out.logits
                out.loss.backward()

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
    bench(model_name, spec)
