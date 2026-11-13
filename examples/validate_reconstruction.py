"""Validate a reconstruction trigger on unseen sequential deletion paths."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from block_ridge import BlockRidge
from ridge_benchmark import ridge_svd
from ridge_control import RidgeStats
from sequential_benchmark import make_data, relative

TOLERANCE = 1e-6
THRESHOLDS = (1e4, 1e6, 1e8, 1e10, 1e12, 1e14)


def solve(stats, alpha):
    try:
        return stats.solve(alpha)
    except np.linalg.LinAlgError:
        return None


def errors(weights, reference, holdout):
    if weights is None or not np.isfinite(weights).all():
        return None
    return [relative(weights, reference), relative(holdout @ weights, holdout @ reference)]


def failed(error):
    return error is None or max(error) > TOLERANCE


def paths(seeds):
    return product(seeds, (1, 1e4, 1e8), (1, 1e6), (1e-12, 1.0), (10, 100), ("forward", "reverse", "shuffled"))


def replay(case, threshold=None):
    seed, spectrum, leverage, alpha, requests, order = case
    x, y, holdout = make_data(seed, spectrum, leverage)
    removed = np.arange(100)
    if order == "reverse":
        removed = removed[::-1]
    elif order == "shuffled":
        removed = np.random.default_rng(seed + 1000).permutation(removed)
    full = stats = RidgeStats.from_rows(x, y)
    qr = BlockRidge(x, y, alpha=alpha, block_size=64) if threshold is not None else None
    keep = np.ones(len(x), dtype=bool)
    penalty = alpha * np.eye(x.shape[1])
    steps = []
    for request in np.array_split(removed, requests):
        keep[request] = False
        stats = stats.subtract(RidgeStats.from_rows(x[request], y[request]))
        condition = float(np.linalg.cond(stats.gram + penalty))
        reference = ridge_svd(x[keep], y[keep], alpha)
        before = errors(solve(stats, alpha), reference, holdout)
        alert = threshold is not None and (not np.isfinite(condition) or condition > threshold)
        after = before
        if alert:
            stats = RidgeStats.from_rows(x[keep], y[keep])
            after = errors(solve(stats, alpha), reference, holdout)
        steps.append({
            "condition": condition if np.isfinite(condition) else None,
            "pre_failure": failed(before), "alert": alert, "post_failure": failed(after),
        })
        if qr is not None:
            qr_final, _ = qr.remove(request)

    result = {"case": case, "steps": steps}
    if qr is not None:
        single_qr, _ = BlockRidge(x, y, alpha=alpha, block_size=64).remove(np.sort(removed))
        single_gram = solve(full.subtract(RidgeStats.from_rows(x[~keep], y[~keep])), alpha)
        result["final"] = {
            "qr_sequential_vs_single": errors(qr_final, single_qr, holdout),
            "qr_sequential_vs_svd": errors(qr_final, reference, holdout),
            "gram_single_vs_svd": errors(single_gram, reference, holdout),
            "gram_policy_vs_svd": errors(solve(stats, alpha), reference, holdout),
        }
    return result


def confusion(steps, threshold=None):
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    for step in steps:
        condition = step["condition"]
        alert = step["alert"] if threshold is None else condition is None or condition > threshold
        key = ("t" if alert == step["pre_failure"] else "f") + ("p" if alert else "n")
        counts[key] += 1
    return counts


def balanced_accuracy(counts):
    return (counts["tp"] / (counts["tp"] + counts["fn"]) +
            counts["tn"] / (counts["tn"] + counts["fp"])) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    train, test = ((42,), (45,)) if args.quick else (range(42, 45), range(45, 53))
    with threadpool_limits(limits=1):
        training = [step for case in paths(train) for step in replay(case)["steps"]]
        candidates = [{"threshold": value, **confusion(training, value)} for value in THRESHOLDS]
        best = max(candidates, key=lambda row: (balanced_accuracy(row), -row["fn"]))
        results = [replay(case, best["threshold"]) for case in paths(test)]
    steps = [step for result in results for step in result["steps"]]
    summary = {
        "training_seeds": list(train), "validation_seeds": list(test), "tolerance": TOLERANCE,
        "training_candidates": candidates, "selected": best["threshold"],
        "validation_counts": confusion(steps), "validation_steps": len(steps),
        "post_failures": sum(step["post_failure"] for step in steps),
        "corrected_steps": sum(step["pre_failure"] and not step["post_failure"] for step in steps),
        "new_failure_steps": sum(not step["pre_failure"] and step["post_failure"] for step in steps),
        "sequences": results,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, allow_nan=False, separators=(",", ":"))
    print({key: value for key, value in summary.items() if key != "sequences"})


if __name__ == "__main__":
    main()
