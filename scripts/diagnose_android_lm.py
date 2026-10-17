"""Isolate precision boundaries in the CPU oracle using an existing LM capture."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import types

import torch
from torch.nn import functional as F

from compare_android import aggregate, compare, inputs_for, read_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("original", "loss64", "norm64", "softmax64", "rope64", "all64"), required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frameworks/pytorch"))
    bench = importlib.import_module("bench")
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.models.llama import modeling_llama as llama

    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    config = LlamaConfig.from_dict(json.loads(args.config.read_text()))
    config.dtype = torch.float32
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).float().double().train()
    if args.variant in ("norm64", "all64"):
        def norm_forward(self, x):
            variance = x.square().mean(-1, keepdim=True)
            return self.weight * x * torch.rsqrt(variance + self.variance_epsilon)
        for module in model.modules():
            if isinstance(module, llama.LlamaRMSNorm):
                module.forward = types.MethodType(norm_forward, module)
    if args.variant in ("rope64", "all64"):
        @torch.no_grad()
        def rope_forward(self, x, position_ids):
            inv = self.inv_freq[None, :, None].to(x.dtype).expand(position_ids.shape[0], -1, 1)
            freqs = (inv @ position_ids[:, None, :].to(x.dtype)).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling
        model.model.rotary_emb.forward = types.MethodType(rope_forward, model.model.rotary_emb)
    if args.variant in ("softmax64", "all64"):
        def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            key = llama.repeat_kv(key, module.num_key_value_groups)
            value = llama.repeat_kv(value, module.num_key_value_groups)
            scores = (query @ key.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=dropout, training=module.training)
            return (weights @ value).transpose(1, 2).contiguous(), weights
        llama.eager_attention_forward = attention

    capture = json.loads((args.capture / "training.json").read_text())
    modules = dict(model.named_modules())
    parameters = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            canonical = bench._gradient_parameter_name("causal_lm", name)
            parent, _, leaf = name.rpartition(".")
            transpose = leaf == "weight" and isinstance(modules[parent], torch.nn.Linear)
            source = read_tensor(args.capture, capture["parameters"][canonical])
            shape = tuple(reversed(parameter.shape)) if transpose else parameter.shape
            source = source.reshape(shape)
            parameter.copy_(source.T if transpose else source)
            parameters[canonical] = (parameter, transpose)
    inputs = inputs_for("causal_lm", {k: read_tensor(args.capture, v) for k, v in capture["inputs"].items()})
    output = bench._benchmark_forward("causal_lm", model, inputs)
    if args.variant in ("loss64", "all64"):
        loss = F.cross_entropy(output.logits.reshape(-1, config.vocab_size), inputs["labels"].reshape(-1))
    else:
        loss = bench._benchmark_loss("causal_lm", output, inputs)
    loss.backward()
    checks = {"loss": compare(capture["loss"], loss.detach().numpy())}
    for name, (parameter, transpose) in parameters.items():
        grad = parameter.grad.T if transpose else parameter.grad
        checks["gradient:" + name] = compare(read_tensor(args.capture, capture["gradients"][name]), grad.detach().numpy())
    report = dict(variant=args.variant, checks=checks,
                  all_gradients=aggregate(v for k, v in checks.items() if k.startswith("gradient:")),
                  passed=all(v["passed"] for v in checks.values()))
    print(json.dumps({"variant": args.variant, "loss": loss.item(), "passed": report["passed"],
                      "failed": {k: v for k, v in checks.items() if not v["passed"]},
                      "all_gradients": report["all_gradients"]}, indent=2), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
