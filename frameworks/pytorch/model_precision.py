"""Declared model-specific autocast boundaries, fixed before qualification."""

import torch


AUTOCAST_SELECTION = "model-policy-v2; no timing or numerical fallback"
SMOLLM2_GUARD = "final-decoder-linears-f32-v1"
# The pinned SmolLM2-135M architecture; declaration must not depend on whether
# the checkpoint has already been downloaded when the collector probes devices.
SMOLLM2_LAYERS = 30


def preferred_autocast(model_name):
    """A frozen recipe; other workloads keep the backend capability policy."""
    return "fp16" if model_name == "SmolLM2-135M" else None


def model_guard(model_name, requested, dtype):
    if model_name == "SmolLM2-135M" and requested == "auto" and dtype == "fp16":
        return SMOLLM2_GUARD
    return None


def guard_modules(layers):
    if type(layers) is not int or layers < 1:
        raise ValueError("invalid decoder layer count for the F32 boundary")
    prefix = f"model.layers.{layers - 1}."
    return [prefix + name for name in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]


class F32Linear(torch.nn.Module):
    """Keep a projection and its output in F32, sharing its master parameters."""

    def __init__(self, original):
        super().__init__()
        self.weight, self.bias = original.weight, original.bias

    def forward(self, x):
        with torch.autocast(self.weight.device.type, enabled=False):
            return torch.nn.functional.linear(x.float(), self.weight, self.bias)


def apply_model_guard(model, guard):
    if guard is None:
        return []
    if guard != SMOLLM2_GUARD:
        raise ValueError(f"unknown model precision boundary: {guard}")
    if model.config.num_hidden_layers != SMOLLM2_LAYERS:
        raise ValueError("checkpoint architecture differs from the declared F32 boundary")
    paths = guard_modules(SMOLLM2_LAYERS)
    # Validate the whole recipe before changing any module. A checkpoint or
    # architecture mismatch must not silently apply only part of the policy.
    originals = [model.get_submodule(path) for path in paths]
    if any(not isinstance(module, torch.nn.Linear) or module.weight.dtype != torch.float32
           or module.bias is not None and module.bias.dtype != torch.float32
           for module in originals):
        raise ValueError("the F32 boundary requires the declared linear modules and F32 master weights")
    for path, original in zip(paths, originals, strict=True):
        parent, _, name = path.rpartition(".")
        setattr(model.get_submodule(parent), name, F32Linear(original))
    return paths
