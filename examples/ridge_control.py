"""Compare exact ridge removal and an unchanged control with the loss audit."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from unlearning_audit.manifest import Manifest
from unlearning_audit.metrics import compare_losses
from unlearning_audit.output import write_artifact
from unlearning_audit.report import render_report
from unlearning_audit.validation import load_losses

ALPHA = 4.0
SEED = 42


@dataclass(frozen=True)
class RidgeStats:
    gram: np.ndarray
    cross: np.ndarray

    @classmethod
    def from_rows(cls, x: np.ndarray, y: np.ndarray):
        return cls(x.T @ x, x.T @ y)

    def subtract(self, removed: "RidgeStats") -> "RidgeStats":
        return RidgeStats(self.gram - removed.gram, self.cross - removed.cross)

    def solve(self, alpha: float) -> np.ndarray:
        return np.linalg.solve(self.gram + alpha * np.eye(len(self.cross)), self.cross)


def make_data(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)

    def cohort(size: int, *, rare: bool) -> tuple[np.ndarray, np.ndarray]:
        amplitude = rng.uniform(0.5, 1.5, size=size)
        x = np.zeros((size, 2))
        x[:, int(rare)] = amplitude
        return x, amplitude * (3.0 if rare else 1.5)

    return {
        "retain": cohort(400, rare=False),
        "forget": cohort(100, rare=True),
        "holdout": cohort(200, rare=True),
        "utility": cohort(200, rare=False),
    }


def fit_controls(data: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    retain_x, retain_y = data["retain"]
    forget_x, forget_y = data["forget"]
    # The downdate uses full-training aggregates and the rows to remove.
    all_stats = RidgeStats.from_rows(np.vstack((retain_x, forget_x)), np.concatenate((retain_y, forget_y)))
    removed = RidgeStats.from_rows(forget_x, forget_y)
    original = all_stats.solve(ALPHA)
    updated = all_stats.subtract(removed).solve(ALPHA)
    # Independently fit the reference using only retain rows and an SVD solver.
    reference = Ridge(alpha=ALPHA, fit_intercept=False, solver="svd").fit(retain_x, retain_y).coef_
    return {"original": original, "downdated": updated, "retrained": reference}


def audit_control(
    directory: Path,
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    models: dict[str, np.ndarray],
    *,
    exact: bool,
    seed: int,
    holdout_scale: float = 1.0,
) -> dict:
    name = "downdated" if exact else "unchanged"
    title = "Ridge deletion · exact control" if exact else "Ridge deletion · unchanged control"
    if holdout_scale != 1.0:
        title += f" · holdout scaled by {holdout_scale:g}"
    selected = {
        "original": models["original"],
        name: models["downdated"] if exact else models["original"],
        "retrained": models["retrained"],
    }
    directory.mkdir()
    path = directory / "losses.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        output = csv.writer(stream, lineterminator="\n")
        output.writerow(("sample_id", "cohort", "model", "loss"))
        for role, weights in selected.items():
            for cohort in ("forget", "holdout", "utility"):
                x, y = data[cohort]
                prefix = cohort
                if cohort == "holdout" and holdout_scale != 1.0:
                    prefix = f"holdout-{holdout_scale:g}"
                for index, loss in enumerate((x @ weights - y) ** 2):
                    output.writerow((f"{prefix}-{index:03d}", cohort, role, loss))

    metadata = {
        "schema_version": 1,
        "title": title,
        "loss_definition": "Squared prediction error for each synthetic observation.",
        "models": {"before": "original", "candidate": name, "reference": "retrained"},
        "cohorts": {
            "forget": "Original training rows along the second feature direction.",
            "holdout": (
                "Independent draws along the same second feature direction."
                if holdout_scale == 1.0 else
                f"Independent draws along the second feature direction, with features and targets scaled by {holdout_scale:g}."
            ),
            "utility": "Independent draws along the retained first feature direction.",
        },
        "provenance": {
            "Data": f"Synthetic continuous draws, NumPy {np.__version__}, seed {seed}.",
            "Objective": f"Ridge with alpha {ALPHA:g}, no intercept and fixed feature encoding.",
            "Reference": "SVD fit from scratch on retain rows with scikit-learn.",
            "Candidate": "Subtract removed rows' contributions and solve." if exact else "Original coefficients reused.",
        },
    }
    result = compare_losses(
        load_losses(path), Manifest.from_dict(metadata), seed=seed,
        include_fingerprints=True, include_loss_gaps=True,
    )
    reference = selected["retrained"]
    candidate = selected[name]
    comparison = {
        "candidate": name,
        "max_coefficient_gap": float(np.max(np.abs(candidate - reference))),
        "max_prediction_gap_holdout": float(np.max(np.abs(data["holdout"][0] @ (candidate - reference)))),
        "mean_holdout_loss": {
            role: float(np.mean((data["holdout"][0] @ weights - data["holdout"][1]) ** 2))
            for role, weights in selected.items()
        },
        "auc": {role: scores["auc"] for role, scores in result["models"].items()},
        "candidate_loss_gaps": result["loss_gaps"]["candidate"],
    }
    write_artifact(directory / "manifest.json", json.dumps(metadata, indent=2) + "\n")
    write_artifact(directory / "metrics.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_artifact(directory / "report.html", render_report(result))
    return comparison


def run(path: Path, seed: int) -> dict:
    path.mkdir(parents=True, exist_ok=False)
    data = make_data(seed)
    with threadpool_limits(limits=1):
        models = fit_controls(data)
        controls = [
            audit_control(path / name, data, models, exact=exact, seed=seed)
            for name, exact in (("exact", True), ("unchanged", False))
        ]
    summary = {"seed": seed, "alpha": ALPHA, "controls": controls}
    write_artifact(path / "controls.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for the two audits")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    try:
        summary = run(args.output, args.seed)
    except OSError as error:
        parser.exit(1, f"Ridge control failed: {error}\n")
    for control in summary["controls"]:
        print(
            f'{control["candidate"]:>10}  AUC {control["auc"]["candidate"]:.4f}'
            f'  max prediction gap vs retrained {control["max_prediction_gap_holdout"]:.6g}'
        )


if __name__ == "__main__":
    main()
