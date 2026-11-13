"""Compare reconstruction policies over sequential ridge deletions."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from block_ridge import BlockRidge
from ridge_control import RidgeStats
from sequential_benchmark import make_data, relative

CONFIG = {
    "seeds": [42, 43, 44, 45, 46, 47],
    "steps": [10, 100],
    "patterns": ["concentrated", "dispersed"],
    "spectrum": [1, 10000, 100000000],
    "alpha": [1e-12, 1e-8, 1.0],
    "leverage": 1e6,
    "period": 10,
    "condition_trigger": 1e10,
    "tolerance": 1e-6,
}


def run_sequence(seed, steps, pattern, spectrum, alpha):
    x, y, holdout = make_data(seed, spectrum, CONFIG["leverage"])
    rng = np.random.default_rng(seed + 1000)
    order = np.arange(100) if pattern == "concentrated" else rng.permutation(len(x))[:100]
    requests = np.array_split(order, steps)
    active = np.ones(len(x), dtype=bool)
    initial = RidgeStats.from_rows(x, y)
    states = {name: initial for name in ("continuous", "periodic", "condition")}
    qr = BlockRidge(x, y, alpha=alpha, block_size=64)
    records = []
    for step, removed in enumerate(requests, 1):
        active[removed] = False
        deleted = RidgeStats.from_rows(x[removed], y[removed])
        start = perf_counter()
        reference = Ridge(alpha=alpha, fit_intercept=False, solver="svd").fit(x[active], y[active]).coef_
        reference_ms = (perf_counter() - start) * 1000
        for method in ("continuous", "periodic", "condition", "block_qr"):
            start = perf_counter()
            rebuilt = False
            condition = None
            if method == "block_qr":
                weights, info = qr.remove(removed)
            else:
                current = states[method].subtract(deleted)
                if method == "condition":
                    condition = float(np.linalg.cond(current.gram + alpha * np.eye(x.shape[1])))
                rebuilt = (method == "periodic" and step % CONFIG["period"] == 0) or (
                    method == "condition" and (not np.isfinite(condition) or condition > CONFIG["condition_trigger"])
                )
                if rebuilt:
                    current = RidgeStats.from_rows(x[active], y[active])
                states[method] = current
                try:
                    weights = current.solve(alpha)
                except np.linalg.LinAlgError:
                    weights = None
                info = {"touched_blocks": None, "factor_bytes": None}
            elapsed = (perf_counter() - start) * 1000
            finite = weights is not None and np.isfinite(weights).all()
            coefficient_error = relative(weights, reference) if finite else None
            prediction_error = relative(holdout @ weights, holdout @ reference) if finite else None
            failure = not finite or max(coefficient_error, prediction_error) > CONFIG["tolerance"]
            records.append({
                "seed": seed, "steps": steps, "pattern": pattern, "spectrum": spectrum, "alpha": alpha,
                "step": step, "method": method, "failure": failure, "rebuilt": rebuilt,
                "coefficient_error": coefficient_error, "prediction_error": prediction_error,
                "condition_before_rebuild": condition if condition is not None and np.isfinite(condition) else None,
                "update_ms": elapsed, "svd_ms": reference_ms,
                "touched_blocks": info["touched_blocks"], "factor_bytes": info["factor_bytes"],
            })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    rows = []
    with threadpool_limits(limits=1):
        for seed in CONFIG["seeds"][:1] if args.quick else CONFIG["seeds"]:
            for steps in CONFIG["steps"][:1] if args.quick else CONFIG["steps"]:
                for pattern in CONFIG["patterns"]:
                    for spectrum in CONFIG["spectrum"]:
                        for alpha in CONFIG["alpha"]:
                            rows.extend(run_sequence(seed, steps, pattern, spectrum, alpha))
    with (args.output / "steps.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    keys = ("seed", "steps", "pattern", "spectrum", "alpha", "method")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summary = []
    for key, group in groups.items():
        summary.append(dict(zip(keys, key)) | {
            "any_failure": any(row["failure"] for row in group),
            "failure_steps": sum(row["failure"] for row in group),
            "rebuilds": sum(row["rebuilt"] for row in group),
            "median_ms": float(np.median([row["update_ms"] for row in group])),
            "p95_ms": float(np.quantile([row["update_ms"] for row in group], .95)),
            "total_ms": sum(row["update_ms"] for row in group),
            "svd_total_ms": sum(row["svd_ms"] for row in group),
        })
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {len(rows)} steps and {len(summary)} sequences to {args.output}")


if __name__ == "__main__":
    main()
