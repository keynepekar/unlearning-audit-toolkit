"""Compare ridge refits on fixed splits of the UCI red wine dataset."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from block_ridge import BlockRidge
from ridge_control import RidgeStats

EXPECTED_SHA256 = "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"
ALPHA = 4.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = args.dataset.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        parser.error("dataset SHA-256 differs from the expected UCI red wine CSV")
    rows = np.loadtxt(args.dataset, delimiter=";", skiprows=1)
    # Deduplicate entire rows before splitting to keep exact records together.
    rows = np.unique(rows, axis=0)
    x = np.column_stack((rows[:, :-1], np.ones(len(rows))))
    y = rows[:, -1]
    results = []
    with threadpool_limits(limits=1):
        for seed in (42, 43, 44, 45):
            train, holdout = train_test_split(np.arange(len(x)), test_size=0.3, random_state=seed)
            for fraction in (0.1, 0.5):
                random_retain, random_forget = train_test_split(train, test_size=fraction, random_state=seed + 100)
                # Scale only for selecting neighboring rows. Model features stay fixed.
                scale = np.std(x[train, :-1], axis=0)
                scale[scale == 0] = 1
                distance = np.linalg.norm((x[train, :-1] - x[train[0], :-1]) / scale, axis=1)
                order = np.argsort(distance, kind="stable")
                nearest_forget = train[order[:len(random_forget)]]
                nearest_retain = train[order[len(random_forget):]]
                for selection, retain, forget in (
                    ("random", random_retain, random_forget), ("nearest", nearest_retain, nearest_forget),
                ):
                    selected = np.concatenate((forget, retain))
                    full_x, full_y = x[selected], y[selected]
                    qr = BlockRidge(full_x, full_y, alpha=ALPHA, block_size=128)
                    candidate, info = qr.remove(np.arange(len(forget)))
                    stats = RidgeStats.from_rows(full_x, full_y).subtract(RidgeStats.from_rows(x[forget], y[forget]))
                    gram = stats.solve(ALPHA)
                    reference = Ridge(alpha=ALPHA, fit_intercept=False, solver="svd").fit(x[retain], y[retain]).coef_
                    for method, weights in (("block_qr", candidate), ("gram", gram)):
                        predictions = x[holdout] @ weights
                        truth = x[holdout] @ reference
                        results.append({
                            "split_seed": seed, "fraction": fraction, "selection": selection, "method": method,
                            "train_rows": len(train), "removed_rows": len(forget), "holdout_rows": len(holdout),
                            "coefficient_relative_l2": float(np.linalg.norm(weights - reference) / np.linalg.norm(reference)),
                            "prediction_relative_l2": float(np.linalg.norm(predictions - truth) / np.linalg.norm(truth)),
                            "reference_mse": float(np.mean((truth - y[holdout]) ** 2)),
                            "candidate_mse": float(np.mean((predictions - y[holdout]) ** 2)),
                            "removed_reference_mse": float(np.mean((x[forget] @ reference - y[forget]) ** 2)),
                            "touched_blocks": info["touched_blocks"],
                        })
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "wine.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    (args.output / "metadata.json").write_text(json.dumps({
        "source": "https://doi.org/10.24432/C56S3T", "license": "CC BY 4.0",
        "sha256": EXPECTED_SHA256, "unique_full_rows": len(rows), "alpha": ALPHA,
        "preprocessing": "Deduplicate full rows; raw features plus an explicit penalized intercept. Neighbors use train-set standard deviations for selection only.",
        "split_seeds": [42, 43, 44, 45],
    }, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} method comparisons to {args.output}")


if __name__ == "__main__":
    main()
