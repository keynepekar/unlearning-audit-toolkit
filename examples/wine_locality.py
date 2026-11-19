"""Measure prediction discrepancies near a fixed deletion in UCI red wine data."""

import argparse
import csv
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from block_ridge import BlockRidge
from wine_experiment import ALPHA, EXPECTED_SHA256


def evaluate(features, targets, seed):
    train, test = train_test_split(np.arange(len(features)), test_size=0.3, random_state=seed)
    scale = np.std(features[train], axis=0)
    scale[scale == 0] = 1
    distance_to_anchor = np.linalg.norm((features[train] - features[train[0]]) / scale, axis=1)
    order = np.argsort(distance_to_anchor, kind="stable")
    n_forget = int(np.ceil(0.1 * len(train)))
    forget, retain = train[order[:n_forget]], train[order[n_forget:]]

    x = np.column_stack((features, np.ones(len(features))))
    regressor = Ridge(alpha=ALPHA, fit_intercept=False, solver="svd")
    original = regressor.fit(x[train], targets[train]).coef_.copy()
    reference = regressor.fit(x[retain], targets[retain]).coef_
    qr = BlockRidge(x[train], targets[train], alpha=ALPHA, block_size=128)
    exact, _ = qr.remove(order[:n_forget])

    # One Newton step with the retained Hessian replaced by its diagonal.
    retained_x = x[retain]
    gradient = retained_x.T @ (retained_x @ original - targets[retain]) + ALPHA * original
    diagonal = np.square(retained_x).sum(axis=0) + ALPHA
    approximate = original - gradient / diagonal

    scaled_test, scaled_forget = features[test] / scale, features[forget] / scale
    distances = np.linalg.norm(scaled_test[:, None] - scaled_forget, axis=2).min(axis=1)
    score_order = np.argsort(distances, kind="stable")
    bands = zip(("near", "middle", "far"), np.array_split(score_order, 3), strict=True)
    ref_predictions = x[test] @ reference
    records = []
    for band, indices in bands:
        test_rows = test[indices]
        for name, model in (("unchanged", original), ("diagonal", approximate), ("block_qr", exact)):
            predictions = x[test_rows] @ model
            gaps = np.abs(predictions - ref_predictions[indices])
            records.append({
                "split_seed": seed, "distance_band": band, "method": name,
                "removed_rows": n_forget, "test_rows": len(indices),
                "distance_min": float(distances[indices].min()),
                "distance_max": float(distances[indices].max()),
                "mean_absolute_prediction_gap": float(gaps.mean()),
                "p95_absolute_prediction_gap": float(np.quantile(gaps, 0.95)),
                "max_absolute_prediction_gap": float(gaps.max()),
                "test_mae": float(np.abs(predictions - targets[test_rows]).mean()),
                "reference_test_mae": float(np.abs(ref_predictions[indices] - targets[test_rows]).mean()),
            })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if sha256(args.dataset.read_bytes()).hexdigest() != EXPECTED_SHA256:
        parser.error("unexpected UCI red wine CSV hash")
    rows = np.unique(np.loadtxt(args.dataset, delimiter=";", skiprows=1), axis=0)
    with threadpool_limits(limits=1):
        records = [row for seed in range(42, 90) for row in evaluate(rows[:, :-1], rows[:, -1], seed)]
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} band comparisons to {args.output}")


if __name__ == "__main__":
    main()
