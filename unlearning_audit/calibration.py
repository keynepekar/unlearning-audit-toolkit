"""Select loss thresholds on a separate cohort of known non-members."""

from bisect import bisect_right
from itertools import groupby


def calibrated_points(
    forget: tuple[float, ...],
    holdout: tuple[float, ...],
    calibration: tuple[float, ...],
    targets: tuple[float, ...],
) -> list[dict]:
    """Select inclusive thresholds on calibration and measure on separate examples.

    A complete block of tied calibration losses must fit inside the budget.
    If no block fits, use the all-negative rule.
    """
    ordered_forget = sorted(forget)
    ordered_holdout = sorted(holdout)
    thresholds = []
    seen = 0
    for loss, ties in groupby(sorted(calibration)):
        seen += sum(1 for _ in ties)
        thresholds.append((seen, loss))
    calibration_counts = [count for count, _ in thresholds]
    results = []
    for target in targets:
        # Build the cutoff in empirical rate space to apply the same budget
        # comparison as the descriptive operating points, including equality.
        eligible = bisect_right(
            calibration_counts, target, key=lambda count: count / len(calibration)
        )
        if eligible:
            selected_count, threshold = thresholds[eligible - 1]
            true_positives = bisect_right(ordered_forget, threshold)
            false_positives = bisect_right(ordered_holdout, threshold)
        else:
            selected_count = true_positives = false_positives = 0
        results.append({
            "target_fpr": target,
            "calibration_false_positives": selected_count,
            "false_positives": false_positives,
            "true_positives": true_positives,
            "fpr": false_positives / len(holdout),
            "tpr": true_positives / len(forget),
        })
    return results
