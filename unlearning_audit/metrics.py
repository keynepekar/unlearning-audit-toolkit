"""Descriptive ROC statistics for aligned, labeled loss exports."""

from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, groupby, pairwise
from math import floor, isfinite
from operator import add, itemgetter, mul
from random import Random
from statistics import mean
from typing import NamedTuple

from .calibration import calibrated_points
from .fingerprints import fingerprint_inputs
from .fpr_bounds import FPR_BOUND
from .groups import GroupIndices, align_groups, group_digest
from .manifest import ROLES, Manifest
from .validation import LossExport


class RocPoint(NamedTuple):
    false_positives: int
    true_positives: int


def _small_histogram(losses: tuple[float, ...]) -> Counter[float] | None:
    """Probe in bounded chunks so distinct scores do not build a large hash table."""
    counts = Counter()
    for start in range(0, len(losses), 16_384):
        counts.update(losses[start:start + 16_384])
        if len(counts) > 4096:
            return None
    return counts


def roc_counts(forget: tuple[float, ...], holdout: tuple[float, ...]) -> list[RocPoint]:
    """Walk distinct loss thresholds. Equal losses enter the curve together."""
    if not forget or not holdout:
        raise ValueError("ROC requires both forget and holdout losses")
    points = [RocPoint(0, 0)]
    false_positives = true_positives = 0
    if len(forget) + len(holdout) >= 4096 and (members := _small_histogram(forget)) is not None:
        nonmembers = _small_histogram(holdout)
        if nonmembers is not None:
            for loss in sorted(members.keys() | nonmembers.keys()):
                true_positives += members[loss]
                false_positives += nonmembers[loss]
                points.append(RocPoint(false_positives, true_positives))
            return points

    ranked = sorted([(loss, True) for loss in forget] + [(loss, False) for loss in holdout])
    for _, group in groupby(ranked, key=lambda item: item[0]):
        for _, is_forget in group:
            true_positives += is_forget
            false_positives += not is_forget
        points.append(RocPoint(false_positives, true_positives))
    return points


def area_under_roc(points: Sequence[Sequence[int]]) -> float:
    """Integrate a ROC path with trapezoids in integer count space."""
    twice_area = sum(
        (right_fp - left_fp) * (left_tp + right_tp)
        for (left_fp, left_tp), (right_fp, right_tp) in pairwise(points)
    )
    holdout, forget = points[-1]
    return twice_area / (2 * holdout * forget)


def operating_points(points: Sequence[Sequence[int]], targets: tuple[float, ...]) -> list[dict]:
    """Return the best observed TPR within each empirical FPR budget."""
    holdout, forget = points[-1]
    results = []
    for target in targets:
        stop = bisect_right(points, target, key=lambda point: point[0] / holdout)
        if not stop:
            raise ValueError("FPR target precedes the ROC curve")
        # The first point with the best TPR uses the fewest false positives.
        fp, tp = points[bisect_left(points, points[stop - 1][1], hi=stop, key=itemgetter(1))]
        results.append({
            "target_fpr": target,
            "fpr": fp / holdout,
            "tpr": tp / forget,
            "false_positives": fp,
            "true_positives": tp,
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


@dataclass(frozen=True)
class _PrefixRanks:
    forget_order: list[int]
    before: list[int]
    through: list[int]

    @classmethod
    def from_ranks(cls, ranks: tuple[list[int], list[int], int]):
        forget, holdout, _ = ranks
        order = sorted(range(len(forget)), key=forget.__getitem__)
        thresholds = [forget[i] for i in order]
        return cls(
            order,
            [bisect_left(thresholds, rank) for rank in holdout],
            [bisect_right(thresholds, rank) for rank in holdout],
        )

    def area(self, forget: list[int], holdout: list[int], denominator: int) -> float:
        cumulative = list(accumulate(map(forget.__getitem__, self.forget_order), initial=0))
        # A strictly earlier row counts twice. An equal row occurs only in through.
        before = map(cumulative.__getitem__, self.before)
        through = map(cumulative.__getitem__, self.through)
        return sum(map(mul, holdout, map(add, before, through))) / denominator


def _sample_weights(indices: list[int], size: int) -> list[int]:
    weights = [0] * size
    for index in indices:
        weights[index] += 1
    return weights


def _python_auc_draws(ranks, forget_n, holdout_n, repeats, seed, groups):
    rng = Random(seed)
    total = forget_n + holdout_n
    prefixes = {
        role: _PrefixRanks.from_ranks(values)
        for role, values in ranks.items() if total >= 4096 and 2 * values[2] >= total
    }
    histograms = {role: values for role, values in ranks.items() if role not in prefixes}
    for _ in range(repeats):
        if groups is None:
            forget_idx = rng.choices(range(forget_n), k=forget_n)
            holdout_idx = rng.choices(range(holdout_n), k=holdout_n)
        else:
            forget_idx, holdout_idx = (
                [index for group in rng.choices(groups[cohort], k=len(groups[cohort])) for index in group]
                for cohort in ("forget", "holdout")
            )
        areas = {
            role: _resampled_auc(role_ranks, forget_idx, holdout_idx)
            for role, role_ranks in histograms.items()
        }
        if prefixes:
            forget = _sample_weights(forget_idx, forget_n)
            holdout = _sample_weights(holdout_idx, holdout_n)
            denominator = 2 * len(forget_idx) * len(holdout_idx)
            areas.update({role: ranks.area(forget, holdout, denominator) for role, ranks in prefixes.items()})
        yield areas


def paired_auc_intervals(
    vectors: dict[str, dict[str, tuple[float, ...]]], repeats: int, seed: int,
    groups: GroupIndices | None = None, *, backend: str = "python",
) -> dict[str, list[float] | None]:
    """Resample within each cohort using the same indices for all model roles."""
    if backend not in ("python", "numpy"):
        raise ValueError("bootstrap backend must be python or numpy")
    comparisons = ("candidate_minus_before", "candidate_minus_reference")
    forget_n = len(vectors["candidate"]["forget"])
    holdout_n = len(vectors["candidate"]["holdout"])
    if not repeats or min(forget_n, holdout_n) < 2:
        return dict.fromkeys(comparisons)

    draw_aucs = _python_auc_draws
    if backend == "numpy":
        try:
            from ._bootstrap_numpy import auc_draws as draw_aucs
        except ImportError as error:
            raise ValueError("NumPy bootstrap requires the fast extra (pip install 'unlearning-audit-toolkit[fast]')") from error
    ranks = {role: _loss_ranks(values["forget"], values["holdout"]) for role, values in vectors.items()}
    differences: dict[str, list[float]] = {name: [] for name in comparisons}
    for areas in draw_aucs(ranks, forget_n, holdout_n, repeats, seed, groups):
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
    include_loss_gaps: bool = False,
    group_assignments: dict[str, str] | None = None,
    iid_holdout: bool = False,
    bootstrap_backend: str = "python",
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
    if group_assignments is not None and not repeats:
        raise ValueError("group bootstrap requires nonzero repeats")
    if iid_holdout and ("calibration" not in export.sample_ids or group_assignments is not None):
        raise ValueError("FPR bounds require a calibration cohort and an independent, ungrouped holdout")
    groups = align_groups(export, group_assignments) if group_assignments is not None else None

    counts = export.counts
    targets = tuple(sorted(targets))
    vectors = {role: export.losses[manifest.models[role]] for role in ROLES}
    models = {}
    for role, values in vectors.items():
        points = roc_counts(values["forget"], values["holdout"])
        models[role] = {
            "auc": area_under_roc(points),
            "roc": [list(point) for point in points],
            "operating_points": operating_points(points, targets),
            "mean_utility_loss": mean(values["utility"]) if "utility" in values else None,
        }
        if "calibration" in values:
            models[role]["calibrated_points"] = calibrated_points(
                values["forget"], values["holdout"], values["calibration"], targets
            )

    intervals = paired_auc_intervals(vectors, repeats, seed, groups, backend=bootstrap_backend)
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
    if groups is not None and min(map(len, groups.values())) < 20:
        warnings.append("At least one audit cohort has fewer than 20 groups. Interval coverage can be unstable.")

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
    if iid_holdout:
        result["holdout_fpr_bound"] = FPR_BOUND
    if groups is not None:
        result["grouping"] = {"counts": {cohort: len(units) for cohort, units in groups.items()}}
        if include_fingerprints:
            result["grouping"]["groups_sha256"] = group_digest(export, groups)
    if include_loss_gaps:
        result["loss_gaps"] = {
            role: {
                cohort: mean(abs(a - b) for a, b in zip(
                    vectors[role][cohort], vectors["reference"][cohort], strict=True,
                ))
                for cohort in counts
            }
            for role in ("before", "candidate")
        }
    return result
