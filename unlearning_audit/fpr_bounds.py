"""Pointwise upper bounds for an independently calibrated false-positive rate."""

from math import expm1, log, log1p

FPR_BOUND = "binomial-kl-95-v1"


def upper_fpr(events: int, total: int) -> float:
    """Invert the one-sided binomial Chernoff bound at 95% confidence."""
    if not 0 <= events <= total or total < 1:
        raise ValueError("expected 0 <= events <= total and total >= 1")
    limit = -log(0.05) / total
    if not events:
        return -expm1(-limit)
    if events == total:
        return 1.0

    observed = events / total
    lower, upper = observed, 1.0
    for _ in range(64):
        mid = (lower + upper) / 2
        if mid in (lower, upper):
            break
        gap = mid - observed
        divergence = -observed * log1p(gap / observed) - (1 - observed) * log1p(-gap / (1 - observed))
        if divergence >= limit:
            upper = mid
        else:
            lower = mid
    return upper


def calibrated_fpr_bound(point: dict, holdout: int) -> float | None:
    """Report the bound when calibration selected a numerical threshold."""
    if point["calibration_false_positives"] == 0:
        return None
    return upper_fpr(point["false_positives"], holdout)
