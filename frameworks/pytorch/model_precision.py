"""Declared phase arithmetic, fixed before compilation or qualification."""

AUTOCAST_SELECTION = "model-phase-policy-v4; no timing or numerical fallback"


def latency_autocast(model_name, requested, dtype):
    # The measured one-token workload is faster in F32 on both local GPUs.
    # Its prefill and training workloads retain capability-selected autocast.
    # Explicit dtype requests continue to exercise the requested arithmetic.
    if model_name == "SmolLM2-135M" and requested == "auto":
        return ""
    return dtype
