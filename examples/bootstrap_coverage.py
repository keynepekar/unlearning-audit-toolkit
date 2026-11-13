"""Measure percentile interval coverage for paired loss differences."""

import argparse
import json
from pathlib import Path

import numpy as np


def simulate(rng, groups, correlation, trials, draws):
    size, truth = 8, 0.2
    observations = groups * size
    covered = dict(rows=0, groups=0)
    widths = dict(rows=0.0, groups=0.0)
    for _ in range(trials):
        common = rng.normal(size=(groups, 1))
        noise = rng.normal(size=(groups, size))
        differences = truth + np.sqrt(correlation) * common + np.sqrt(1 - correlation) * noise
        population = {"rows": differences.ravel(), "groups": differences.mean(axis=1)}
        for kind, values in population.items():
            indices = rng.integers(len(values), size=(draws, len(values)))
            samples = values[indices].mean(axis=1)
            lower, upper = np.quantile(samples, [0.025, 0.975])
            covered[kind] += lower <= truth <= upper
            widths[kind] += upper - lower
    return {
        "groups": groups, "rows": observations, "within_group_correlation": correlation,
        "trials": trials, "bootstrap_draws": draws,
        "coverage": {kind: float(value / trials) for kind, value in covered.items()},
        "mean_interval_width": {kind: float(value / trials) for kind, value in widths.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    trials, draws = (20, 99) if args.quick else (600, 799)
    rng = np.random.default_rng(20261113)
    results = [
        simulate(rng, groups, correlation, trials, draws)
        for groups in (12, 48) for correlation in (0.0, 0.3, 0.7)
    ]
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"seed": 20261113, "true_mean": 0.2, "results": results}, stream, indent=2)
    print(f"Wrote coverage for {len(results)} scenarios to {args.output}")


if __name__ == "__main__":
    main()
