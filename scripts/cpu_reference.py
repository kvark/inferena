"""Loss arithmetic for offline CPU correctness checks, outside benchmark timing."""

LOSS_POLICY = "reference-dtype-v1"


def reference_loss(kind, logits, inputs):
    """Keep the oracle's chosen dtype instead of the timed AMP loss's f32 cast."""
    from torch.nn import functional as F

    if kind == "sd_unet":
        return F.mse_loss(logits, inputs["noise_target"])
    if kind == "smolvla":
        return logits.square().mean()
    if kind == "resnet":
        return F.cross_entropy(logits, inputs["labels"])
    if kind == "whisper":
        return logits.square().mean()
    if kind == "causal_lm":
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), inputs["labels"].reshape(-1))
    raise ValueError(f"unknown reference workload: {kind}")
