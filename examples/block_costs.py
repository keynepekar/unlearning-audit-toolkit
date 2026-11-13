"""Measure block size and deletion locality for ridge QR refits."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from block_ridge import BlockRidge
from sequential_benchmark import make_data, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "seeds": [42, 43, 44, 45, 46, 47], "block_sizes": [32, 64, 128],
        "patterns": ["concentrated", "dispersed"], "removal_count": 100,
        "alpha": [1e-12, 1e-8, 1.0], "spectrum": [1, 10000, 100000000], "leverage": 1000000,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    results = []
    with threadpool_limits(limits=1):
        for seed in config["seeds"]:
            for spectrum in config["spectrum"]:
                x, y, holdout = make_data(seed, spectrum, config["leverage"])
                for pattern in config["patterns"]:
                    removed = np.arange(config["removal_count"])
                    if pattern == "dispersed":
                        removed = np.random.default_rng(seed + 1000).permutation(len(x))[:config["removal_count"]]
                    keep = np.ones(len(x), dtype=bool)
                    keep[removed] = False
                    for alpha in config["alpha"]:
                        started = perf_counter()
                        ref = Ridge(alpha=alpha, fit_intercept=False, solver="svd").fit(x[keep], y[keep]).coef_
                        refit_ms = (perf_counter() - started) * 1000
                        for size in config["block_sizes"]:
                            started = perf_counter()
                            model = BlockRidge(x, y, alpha=alpha, block_size=size)
                            build_ms = (perf_counter() - started) * 1000
                            started = perf_counter()
                            candidate, details = model.remove(removed)
                            update_ms = (perf_counter() - started) * 1000
                            results.append({
                                "seed": seed, "spectrum": spectrum, "pattern": pattern, "alpha": alpha,
                                "block_size": size, "build_ms": build_ms, "update_ms": update_ms, "svd_ms": refit_ms,
                                "coefficient_error": relative(candidate, ref),
                                "prediction_error": relative(holdout @ candidate, holdout @ ref), **details,
                            })
    with (args.output / "block-costs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} scenarios to {args.output}")


if __name__ == "__main__":
    main()
