"""Benchmark ridge downdates against independent refits on CPU."""

import argparse
import csv
import json
import platform
from collections.abc import Callable
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_info, threadpool_limits

from ridge_control import RidgeStats
from unlearning_audit.metrics import area_under_roc, roc_counts
from unlearning_audit.output import write_artifact

COEFFICIENT_TOLERANCE = 1e-6
PREDICTION_TOLERANCE = 1e-6
FRACTIONS = (0.01, 0.10, 0.50, 0.90)


def ridge_svd(x, y, alpha):
    return Ridge(alpha=alpha, fit_intercept=False, solver="svd").fit(x, y).coef_


def relative_l2(candidate, reference):
    error = np.linalg.norm(candidate - reference)
    scale = np.linalg.norm(reference)
    return float(error / scale if scale else error)


def loss_auc(forget_loss, holdout_loss):
    return area_under_roc(roc_counts(tuple(forget_loss), tuple(holdout_loss)))


def direct_gram_error(retained, reference, alpha):
    try:
        error = relative_l2(retained.solve(alpha), reference)
    except np.linalg.LinAlgError:
        return None
    return error if np.isfinite(error) else None


def evaluate(train_x, train_y, forget_x, forget_y, holdout_x, holdout_y, alpha):
    full = RidgeStats.from_rows(
        np.concatenate((train_x, forget_x)), np.concatenate((train_y, forget_y)),
    )
    updated = full.subtract(RidgeStats.from_rows(forget_x, forget_y))
    reference = ridge_svd(train_x, train_y, alpha)
    original = ridge_svd(
        np.concatenate((train_x, forget_x)), np.concatenate((train_y, forget_y)), alpha,
    )
    retained = RidgeStats.from_rows(train_x, train_y)
    condition = float(np.linalg.cond(updated.gram + alpha * np.eye(train_x.shape[1])))
    row = {
        "retained_rows": len(train_x),
        "removed_rows": len(forget_x),
        "holdout_rows": len(holdout_x),
        "alpha": alpha,
        "updated_system_condition": condition if np.isfinite(condition) else None,
        "retained_gram_relative_l2": relative_l2(updated.gram, retained.gram),
        "retained_cross_relative_l2": relative_l2(updated.cross, retained.cross),
        "direct_gram_coefficient_relative_l2": direct_gram_error(retained, reference, alpha),
        "coefficient_relative_l2": None,
        "holdout_prediction_relative_l2": None,
        "holdout_paired_loss_gap": None,
        "original_auc": None,
        "candidate_auc": None,
        "reference_auc": None,
        "status": "solve_failed",
    }
    try:
        candidate = updated.solve(alpha)
    except np.linalg.LinAlgError:
        return row

    losses = []
    for weights in (original, candidate, reference):
        forget_loss = (forget_x @ weights - forget_y) ** 2
        holdout_loss = (holdout_x @ weights - holdout_y) ** 2
        if not np.isfinite(forget_loss).all() or not np.isfinite(holdout_loss).all():
            row["status"] = "nonfinite_loss"
            return row
        losses.append((forget_loss, holdout_loss))
    coefficients = relative_l2(candidate, reference)
    predictions = relative_l2(holdout_x @ candidate, holdout_x @ reference)
    row.update(
        coefficient_relative_l2=coefficients,
        holdout_prediction_relative_l2=predictions,
        holdout_paired_loss_gap=float(np.mean(np.abs(losses[1][1] - losses[2][1]))),
        original_auc=loss_auc(*losses[0]),
        candidate_auc=loss_auc(*losses[1]),
        reference_auc=loss_auc(*losses[2]),
        status="outside_tolerance" if (
            coefficients > COEFFICIENT_TOLERANCE or predictions > PREDICTION_TOLERANCE
        ) else "within_tolerance",
    )
    return row


def stress_matrix():
    for seed in (42, 43, 44):
        for dimension in (8, 32):
            rng = np.random.default_rng(seed)
            rotation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
            target = rng.normal(size=dimension)
            base_x = rng.normal(size=(1_200, dimension))
            base_holdout = rng.normal(size=(300, dimension))
            noise = rng.normal(scale=0.01, size=len(base_x))
            holdout_noise = rng.normal(scale=0.01, size=len(base_holdout))
            for spectrum in (1, 1e4, 1e8):
                scale = np.geomspace(1.0, 1.0 / spectrum, dimension)
                for leverage in (1, 1e6):
                    x = base_x.copy()
                    x[:180, 0] *= leverage
                    x = (x * scale) @ rotation
                    holdout = (base_holdout * scale) @ rotation
                    y = x @ target + noise
                    holdout_y = holdout @ target + holdout_noise
                    for fraction in (0.01, 0.15, 0.50):
                        count = round(fraction * len(x))
                        row = evaluate(
                            x[count:], y[count:], x[:count], y[:count], holdout, holdout_y, 1e-12,
                        )
                        yield {
                            "seed": seed, "features": dimension, "design_spectrum_ratio": spectrum,
                            "removed_direction_multiplier": leverage, "deletion_fraction": fraction, **row,
                        }


def digits_matrix():
    pixels, digits = load_digits(return_X_y=True)
    _, unique = np.unique(pixels, axis=0, return_index=True)
    pixels, digits = pixels[np.sort(unique)] / 16, digits[np.sort(unique)]
    y = (digits % 2).astype(float)
    rows = np.arange(len(digits))
    for split_seed in (42, 43, 44, 45):
        train, holdout = train_test_split(rows, test_size=0.30, stratify=digits, random_state=split_seed)
        for deletion_seed in (71, 72, 73):
            for fraction in (0.10, 0.50):
                retain, forget = train_test_split(
                    train, test_size=fraction, stratify=digits[train], random_state=deletion_seed,
                )
                yield {
                    "split_seed": split_seed, "deletion_seed": deletion_seed, "deletion_fraction": fraction,
                    "features": pixels.shape[1],
                    **evaluate(pixels[retain], y[retain], pixels[forget], y[forget], pixels[holdout], y[holdout], 4.0),
                }


def median_ms(operation: Callable, repeats: int = 7) -> float:
    operation()
    timings = []
    for _ in range(repeats):
        start = perf_counter_ns()
        operation()
        timings.append((perf_counter_ns() - start) / 1e6)
    return median(timings)


def timing_matrix(rows: int):
    for dimension in (16, 64, 128):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(rows, dimension))
        y = x @ rng.normal(size=dimension) + rng.normal(size=rows)
        alpha = 4.0
        full = RidgeStats.from_rows(x, y)
        precompute_ms = median_ms(lambda: RidgeStats.from_rows(x, y))
        for fraction in FRACTIONS:
            count = round(fraction * rows)
            x_f, y_f, x_r, y_r = x[:count], y[:count], x[count:], y[count:]

            def update():
                return full.subtract(RidgeStats.from_rows(x_f, y_f)).solve(alpha)

            def refit():
                return RidgeStats.from_rows(x_r, y_r).solve(alpha)

            candidate = update()
            reference = ridge_svd(x_r, y_r, alpha)
            update_ms = median_ms(update)
            refit_ms = median_ms(refit)
            yield {
                "training_rows": rows, "features": dimension, "deletion_fraction": fraction,
                "removed_rows": count, "retained_rows": rows - count,
                "precompute_ms": precompute_ms, "update_ms": update_ms, "refit_same_solver_ms": refit_ms,
                "refit_svd_ms": median_ms(lambda: ridge_svd(x_r, y_r, alpha)),
                "refit_over_update": refit_ms / update_ms,
                "coefficient_relative_l2": relative_l2(candidate, reference),
                "statistics_bytes": full.gram.nbytes + full.cross.nbytes,
                "retained_arrays_bytes": x_r.nbytes + y_r.nbytes,
            }


def write_results(output: Path, name: str, rows: list[dict]) -> None:
    with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timing-rows", type=int, default=12_000, help="training rows per timing workload")
    args = parser.parse_args()
    if args.timing_rows < 2_000:
        parser.error("--timing-rows must be at least 2000")
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        results = {
            "environment": {
                "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                "platform": platform.platform(), "processor": platform.processor(),
                "threadpools": [
                    {key: pool[key] for key in ("internal_api", "version", "num_threads")}
                    for pool in threadpool_info()
                ],
            },
            "tolerances": {"coefficient_relative_l2": COEFFICIENT_TOLERANCE,
                           "holdout_prediction_relative_l2": PREDICTION_TOLERANCE},
            "stress": list(stress_matrix()),
            "digits": list(digits_matrix()),
            "timing": list(timing_matrix(args.timing_rows)),
        }
    for name in ("stress", "digits", "timing"):
        write_results(args.output, name, results[name])
    write_artifact(args.output / "benchmark.json", json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"Wrote benchmark results to {args.output}")


if __name__ == "__main__":
    main()
