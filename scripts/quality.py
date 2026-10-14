"""Shared receipt contract for pretrained language-model forward quality."""
import math

LM_POLICY = "language-model-distribution-v1"
MEAN_KL_LIMIT = math.log(1.01)
TOKEN_KL_LIMIT = math.log(1.10)
LOSS_RTOL, LOSS_ATOL = 0.02, 1e-6


def language_model_report(kl, losses, reference_losses, vocabulary):
    """Recompute acceptance from per-token evidence; no fitted error envelope."""
    if (vocabulary < 2 or not kl or len(kl) != len(losses) or len(kl) != len(reference_losses)
            or any(not math.isfinite(x) or x < 0 for x in (*kl, *losses, *reference_losses))):
        raise ValueError("invalid language-model quality evidence")
    loss = math.fsum(losses) / len(losses)
    reference_loss = math.fsum(reference_losses) / len(reference_losses)
    mean_kl, max_kl = math.fsum(kl) / len(kl), max(kl)
    loss_error, loss_bound = abs(loss - reference_loss), LOSS_ATOL + LOSS_RTOL * abs(reference_loss)
    return {"policy": LM_POLICY, "status": "pass" if mean_kl <= MEAN_KL_LIMIT
            and max_kl <= TOKEN_KL_LIMIT and loss_error <= loss_bound else "failed",
            "tokens": len(kl), "vocabulary": vocabulary,
            "mean_kl_limit": MEAN_KL_LIMIT, "token_kl_limit": TOKEN_KL_LIMIT,
            "loss_rtol": LOSS_RTOL, "loss_atol": LOSS_ATOL,
            "mean_kl": mean_kl, "max_kl": max_kl,
            "loss": loss, "reference_loss": reference_loss,
            "loss_error": loss_error, "loss_bound": loss_bound,
            "kl_per_token": kl, "losses": losses, "reference_losses": reference_losses}


def check_language_model_report(report):
    expected = language_model_report(report["kl_per_token"], report["losses"],
                                     report["reference_losses"], report["vocabulary"])
    if set(report) != set(expected):
        raise ValueError("language-model quality receipt fields differ")
    for key, value in expected.items():
        actual = report[key]
        if isinstance(value, float):
            if not math.isfinite(actual) or not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError(f"language-model quality receipt disagrees: {key}")
        elif actual != value:
            raise ValueError(f"language-model quality receipt disagrees: {key}")
    if expected["status"] != "pass":
        raise ValueError("language-model quality exceeds fixed bounds")
    return expected
