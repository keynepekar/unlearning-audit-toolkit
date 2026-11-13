"""Compare successive ridge removals with an independent SVD refit."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from block_ridge import BlockRidge
from ridge_control import RidgeStats

CONFIG = {
    "schema_version": 1,
    "seeds": [42, 43, 44, 45, 46, 47],
    "rows": 1000,
    "features": 16,
    "holdout": 200,
    "steps": [10, 100],
    "alpha": 1e-8,
    "block_size": 64,
    "tolerance": 1e-6,
    "regularization": [1e-12, 1e-8, 1.0],
    "spectrum_ratio": [1, 1e4, 1e8],
    "leverage": [1, 1e6],
}


def make_data(seed, spectrum, leverage):
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(CONFIG["features"], CONFIG["features"])))
    x = rng.normal(size=(CONFIG["rows"], CONFIG["features"]))
    x[:100, 0] *= leverage
    x = (x * np.geomspace(1, 1 / spectrum, CONFIG["features"])) @ rotation
    weights = rng.normal(size=CONFIG["features"])
    y = x @ weights + rng.normal(scale=0.01, size=len(x))
    holdout = (rng.normal(size=(CONFIG["holdout"], CONFIG["features"])) *
               np.geomspace(1, 1 / spectrum, CONFIG["features"])) @ rotation
    return x, y, holdout


def relative(actual, expected):
    scale = np.linalg.norm(expected)
    error = np.linalg.norm(actual - expected)
    return float(error / scale if scale else error)


def experiment(seed, steps, alpha, spectrum, leverage):
    x, y, holdout = make_data(seed, spectrum, leverage)
    initial = RidgeStats.from_rows(x, y)
    qr = BlockRidge(x, y, alpha=alpha, block_size=CONFIG["block_size"])
    stats = initial
    active = np.ones(len(x), dtype=bool)
    requests = np.array_split(np.arange(100), steps)
    results = []
    for step, removed in enumerate(requests, 1):
        active[removed] = False
        started = perf_counter()
        stats = stats.subtract(RidgeStats.from_rows(x[removed], y[removed]))
        try:
            gram = stats.solve(alpha)
        except np.linalg.LinAlgError:
            gram = None
        gram_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        block, info = qr.remove(removed)
        block_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        reference = Ridge(alpha=alpha, fit_intercept=False, solver="svd").fit(x[active], y[active]).coef_
        refit_ms = (perf_counter() - started) * 1000
        system = stats.gram + alpha * np.eye(x.shape[1])
        denominator = np.linalg.norm(system) * np.linalg.norm(gram) + np.linalg.norm(stats.cross) if gram is not None else 0
        residual = float(np.linalg.norm(system @ gram - stats.cross) / denominator) if denominator else None
        for method, weights, elapsed in (("gram", gram, gram_ms), ("block_qr", block, block_ms)):
            finite = weights is not None and np.isfinite(weights).all()
            results.append({
                "seed": seed, "steps": steps, "step": step, "alpha": alpha, "spectrum": spectrum,
                "leverage": leverage, "method": method, "removed": int((~active).sum()),
                "prediction_error": relative(holdout @ weights, holdout @ reference) if finite else None,
                "coefficient_error": relative(weights, reference) if finite else None,
                "gram_residual": residual if method == "gram" else None,
                "status": "ok" if finite else "solve_failed",
                "update_ms": elapsed, "svd_ms": refit_ms, "touched_blocks": info["touched_blocks"],
                "factor_bytes": info["factor_bytes"], "input_bytes": info["input_bytes"],
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    canonical = json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()
    metadata = {
        "config": CONFIG, "config_sha256": hashlib.sha256(canonical).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
        "profile": "quick" if args.quick else "full", "blas_threads": 1,
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    seeds = CONFIG["seeds"][:1] if args.quick else CONFIG["seeds"]
    steps = CONFIG["steps"][:1] if args.quick else CONFIG["steps"]
    results = []
    with threadpool_limits(limits=1):
        for seed in seeds:
            for count in steps:
                for alpha in CONFIG["regularization"]:
                    for spectrum in CONFIG["spectrum_ratio"]:
                        for leverage in CONFIG["leverage"]:
                            results.extend(experiment(seed, count, alpha, spectrum, leverage))
    with (args.output / "sequences.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} per-step observations to {args.output}")


if __name__ == "__main__":
    main()
