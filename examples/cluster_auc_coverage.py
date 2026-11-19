"""Simulate coverage of paired AUC intervals when rows share sampling units."""

import argparse
import json
from math import erf
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

SEED = 20261120
SHIFT = 0.4


def draws(rng, pair_scores, sizes, repeats):
    weights = [
        rng.multinomial(len(counts), np.full(len(counts), 1 / len(counts)), size=repeats)
        for counts in sizes
    ]
    left, right = weights
    return ((left @ pair_scores) * right).sum(axis=1) / ((left @ sizes[0]) * (right @ sizes[1]))


def pair_wins(forget, holdout):
    return (forget[:, None] < holdout).astype(float) + 0.5 * (forget[:, None] == holdout)


def trial(rng, groups, correlation, unequal, tied, repeats):
    sizes = [rng.integers(2, 9, size=groups) if unequal else np.full(groups, 5) for _ in range(2)]

    def latent(counts):
        group = np.repeat(rng.normal(size=groups), counts)
        return np.sqrt(correlation) * group + np.sqrt(1 - correlation) * rng.normal(size=len(group))

    before = [latent(counts) for counts in sizes]
    if tied:
        # Equal marginal distributions across models with less than perfect pairing.
        after = [0.6 * score + 0.8 * latent(counts) for score, counts in zip(before, sizes, strict=True)]
        before = [np.round(score) for score in before]
        after = [np.round(score) for score in after]
    else:
        after = [before[0] - SHIFT, before[1]]
    pairs = pair_wins(*after) - pair_wins(*before)
    starts = [np.r_[0, counts.cumsum()[:-1]] for counts in sizes]
    grouped = np.add.reduceat(np.add.reduceat(pairs, starts[0], axis=0), starts[1], axis=1)
    row_sizes = [np.ones(counts.sum()) for counts in sizes]
    return {
        "rows": np.quantile(draws(rng, pairs, row_sizes, repeats), (0.025, 0.975)),
        "groups": np.quantile(draws(rng, grouped, sizes, repeats), (0.025, 0.975)),
    }


def scenario(rng, groups, correlation, unequal, tied, trials, repeats):
    truth = 0.0 if tied else erf(SHIFT / 2) / 2
    covered = {"rows": 0, "groups": 0}
    widths = {"rows": 0.0, "groups": 0.0}
    for _ in range(trials):
        for method, (lower, upper) in trial(rng, groups, correlation, unequal, tied, repeats).items():
            covered[method] += int(lower <= truth <= upper)
            widths[method] += upper - lower
    return {
        "groups_per_cohort": groups, "correlation": correlation,
        "unequal_sizes": unequal, "tied_null": tied, "true_auc_delta": truth,
        "coverage": {method: count / trials for method, count in covered.items()},
        "mean_width": {method: width / trials for method, width in widths.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    trials, repeats = (16, 199) if args.quick else (600, 799)
    rng = np.random.default_rng(SEED)
    with threadpool_limits(limits=1):
        results = [
            scenario(rng, groups, correlation, unequal, tied, trials, repeats)
            for groups in (12, 32)
            for correlation, unequal, tied in ((0.0, False, False), (0.7, False, False),
                                              (0.7, True, False), (0.7, True, True))
        ]
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"seed": SEED, "trials": trials, "draws": repeats, "results": results}, stream, indent=2)


if __name__ == "__main__":
    main()
