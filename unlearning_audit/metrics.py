"""Descriptive ROC statistics for aligned, labeled loss exports."""

from dataclasses import dataclass
from itertools import groupby
from math import floor, isfinite
from random import Random
from statistics import mean

from .calibration import calibrated_points
from .fingerprints import fingerprint_inputs
from .manifest import ROLES, Manifest
from .validation import LossExport


@dataclass(frozen=True)
class RocPoint:
    false_positives: int
    true_positives: int


def roc_counts(forget: tuple[float, ...], holdout: tuple[float, ...]) -> list[RocPoint]:
    """Walk distinct loss thresholds. Equal losses enter the curve together."""
    if not forget or not holdout:
        raise ValueError("ROC requires both forget and holdout losses")
    ranked = sorted([(loss, True) for loss in forget] + [(loss, False) for loss in holdout])
    points = [RocPoint(0, 0)]
    false_positives = true_positives = 0
    for _, group in groupby(ranked, key=lambda item: item[0]):
        for _, is_forget in group:
            true_positives += is_forget
            false_positives += not is_forget
        points.append(RocPoint(false_positives, true_positives))
    return points


def area_under_roc(points: list[RocPoint]) -> float:
    """Integrate a ROC path with trapezoids in integer count space."""
    twice_area = sum(
        (right.false_positives - left.false_positives)
        * (left.true_positives + right.true_positives)
        for left, right in zip(points, points[1:])
    )
    end = points[-1]
    return twice_area / (2 * end.false_positives * end.true_positives)


def operating_points(points: list[RocPoint], targets: tuple[float, ...]) -> list[dict]:
    """Return the best observed TPR within each empirical FPR budget."""
    end = points[-1]
    results = []
    for target in targets:
        best = max(
            (point for point in points if point.false_positives / end.false_positives <= target),
            key=lambda point: (point.true_positives, -point.false_positives),
        )
        results.append({
            "target_fpr": target,
            "fpr": best.false_positives / end.false_positives,
            "tpr": best.true_positives / end.true_positives,
            "false_positives": best.false_positives,
            "true_positives": best.true_positives,
        })
    return results


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def _loss_ranks(forget: tuple[float, ...], holdout: tuple[float, ...]) -> tuple[list[int], list[int], int]:
    levels = {loss: rank for rank, loss in enumerate(sorted(set(forget) | set(holdout)))}
    return [levels[loss] for loss in forget], [levels[loss] for loss in holdout], len(levels)


def _resampled_auc(ranks: tuple[list[int], list[int], int], forget_idx: list[int], holdout_idx: list[int]) -> float:
    forget_ranks, holdout_ranks, n_levels = ranks
    forget_weights = [0] * n_levels
    holdout_weights = [0] * n_levels
    for index in forget_idx:
        forget_weights[forget_ranks[index]] += 1
    for index in holdout_idx:
        holdout_weights[holdout_ranks[index]] += 1

    # Each holdout counts earlier forget rows fully and tied rows at half weight.
    earlier_forget = twice_area = 0
    for forget, holdout in zip(forget_weights, holdout_weights):
        twice_area += holdout * (2 * earlier_forget + forget)
        earlier_forget += forget
    return twice_area / (2 * len(forget_idx) * len(holdout_idx))


def paired_auc_intervals(
    vectors: dict[str, dict[str, tuple[float, ...]]], repeats: int, seed: int
) -> dict[str, list[float] | None]:
    """Resample within each cohort using the same indices for all model roles."""
    comparisons = ("candidate_minus_before", "candidate_minus_reference")
    forget_n = len(vectors["candidate"]["forget"])
    holdout_n = len(vectors["candidate"]["holdout"])
    if not repeats or min(forget_n, holdout_n) < 2:
        return dict.fromkeys(comparisons)

    rng = Random(seed)
    differences: dict[str, list[float]] = {name: [] for name in comparisons}
    ranks = {role: _loss_ranks(values["forget"], values["holdout"]) for role, values in vectors.items()}
    for _ in range(repeats):
        forget_idx = rng.choices(range(forget_n), k=forget_n)
        holdout_idx = rng.choices(range(holdout_n), k=holdout_n)
        areas = {role: _resampled_auc(role_ranks, forget_idx, holdout_idx) for role, role_ranks in ranks.items()}
        differences["candidate_minus_before"].append(areas["candidate"] - areas["before"])
        differences["candidate_minus_reference"].append(areas["candidate"] - areas["reference"])
    return {
        key: [_percentile(ordered, 0.025), _percentile(ordered, 0.975)]
        for key, values in differences.items()
        if (ordered := sorted(values))
    }


def compare_losses(
    export: LossExport,
    manifest: Manifest,
    *,
    targets: tuple[float, ...] = (0.01, 0.10),
    repeats: int = 400,
    seed: int = 42,
    include_fingerprints: bool = False,
) -> dict:
    """Return an aggregate, JSON-compatible audit of three aligned model exports."""
    manifest.check_alignment(set(export.losses), set(export.sample_ids))
    if not targets or any(not isfinite(target) or not 0 < target < 1 for target in targets):
        raise ValueError("FPR targets must be strictly between 0 and 1")
    if len(set(targets)) != len(targets):
        raise ValueError("FPR targets must be distinct")
    if type(repeats) is not int or (repeats != 0 and not 200 <= repeats <= 10_000):
        raise ValueError("bootstrap repeats must be 0 or between 200 and 10000")
    if type(seed) is not int:
        raise ValueError("bootstrap seed must be an integer")

    counts = export.counts
    targets = tuple(sorted(targets))
    vectors = {role: export.losses[manifest.models[role]] for role in ROLES}
    models = {}
    for role, values in vectors.items():
        points = roc_counts(values["forget"], values["holdout"])
        models[role] = {
            "auc": area_under_roc(points),
            "roc": [[point.false_positives, point.true_positives] for point in points],
            "operating_points": operating_points(points, targets),
            "mean_utility_loss": mean(values["utility"]) if "utility" in values else None,
        }
        if "calibration" in values:
            models[role]["calibrated_points"] = calibrated_points(
                values["forget"], values["holdout"], values["calibration"], targets
            )

    intervals = paired_auc_intervals(vectors, repeats, seed)
    deltas = {
        f"candidate_minus_{role}": {
            "estimate": models["candidate"]["auc"] - models[role]["auc"],
            "interval": intervals[f"candidate_minus_{role}"],
        }
        for role in ("before", "reference")
    }

    warnings = [
        f"{target:.4g} FPR is below the one-event resolution with {counts['holdout']} holdout examples."
        for target in targets if target < 1 / counts["holdout"]
    ]
    warnings.extend(
        f"{target:.4g} FPR is below the one-event resolution with {counts['calibration']} calibration examples."
        for target in targets if "calibration" in counts and target < 1 / counts["calibration"]
    )
    if min(counts["forget"], counts["holdout"]) < 20:
        warnings.append("At least one audit cohort has fewer than 20 examples. Estimates can be unstable.")
    if repeats and min(counts["forget"], counts["holdout"]) < 2:
        warnings.append("Bootstrap intervals require at least two examples per audit cohort.")
    if not repeats:
        warnings.append("Bootstrap intervals were disabled for this run.")

    result = {
        "schema_version": 1,
        "manifest": manifest.as_dict(),
        "counts": counts,
        "models": models,
        "auc_deltas": deltas,
        "bootstrap": {"repeats": repeats, "seed": seed},
        "warnings": warnings,
    }
    if include_fingerprints:
        result["fingerprints"] = fingerprint_inputs(export, manifest)
    return result
