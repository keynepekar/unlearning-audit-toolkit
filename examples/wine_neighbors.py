"""Compare removal of one wine row with removal of its nearest group."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from wine_experiment import ALPHA, EXPECTED_SHA256


def anchor_group(train, features, size):
    anchor = train[0]
    scale = np.std(features[train], axis=0)
    scale[scale == 0] = 1
    others = train[train != anchor]
    distance = np.linalg.norm((features[others] - features[anchor]) / scale, axis=1)
    order = np.argsort(distance, kind="stable")
    return np.concatenate(([anchor], others[order[:size - 1]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != EXPECTED_SHA256:
        parser.error("unexpected UCI CSV hash")
    rows = np.unique(np.loadtxt(args.dataset, delimiter=";", skiprows=1), axis=0)
    x = np.column_stack((rows[:, :-1], np.ones(len(rows))))
    y = rows[:, -1]
    records = []
    with threadpool_limits(limits=1):
        for seed in range(42, 52):
            train, _ = train_test_split(np.arange(len(x)), test_size=0.3, random_state=seed)
            anchor = train[0]
            group = anchor_group(train, x[:, :-1], size=10)
            for mode, removed in (("single", group[:1]), ("group", group)):
                retain = train[~np.isin(train, removed)]
                model = Ridge(alpha=ALPHA, fit_intercept=False, solver="svd").fit(x[retain], y[retain])
                predictions = model.predict(x[group])
                records.append({
                    "split_seed": seed, "mode": mode, "removed_count": len(removed),
                    "anchor_target": y[anchor], "anchor_prediction": predictions[0],
                    "anchor_squared_error": (predictions[0] - y[anchor]) ** 2,
                    "group_prediction_mean": float(np.mean(predictions)),
                    "group_target_mean": float(np.mean(y[group])),
                    "neighbors_retained": int(np.isin(group[1:], retain).sum()),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
