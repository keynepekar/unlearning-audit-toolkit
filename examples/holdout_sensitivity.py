"""Measure how changing only the holdout affects a known exact ridge deletion."""

import argparse
import csv
from io import StringIO
from pathlib import Path

from threadpoolctl import threadpool_limits

from ridge_control import audit_control, fit_controls, make_data
from unlearning_audit.metrics import area_under_roc, roc_counts
from unlearning_audit.output import write_artifact


HOLDOUTS = {"lower_amplitude": 0.65, "matched": 1.0, "higher_amplitude": 1.5}


def run(directory: Path, seeds: range) -> None:
    directory.mkdir()
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    roles = ("original", "downdated", "retrained")
    writer.writerow(("seed", "holdout", "holdout_scale", *(f"{role}_auc" for role in roles)))
    with threadpool_limits(limits=1):
        for seed in seeds:
            data = make_data(seed)
            models = fit_controls(data)
            forget_x, forget_y = data["forget"]
            holdout_x, holdout_y = data["holdout"]
            for name, scale in HOLDOUTS.items():
                shifted_x, shifted_y = scale * holdout_x, scale * holdout_y
                areas = [
                    area_under_roc(roc_counts(
                        tuple((forget_x @ models[role] - forget_y) ** 2),
                        tuple((shifted_x @ models[role] - shifted_y) ** 2),
                    ))
                    for role in roles
                ]
                writer.writerow((seed, name, scale, *areas))
                if seed == seeds.start:
                    audit_control(
                        directory / name, {**data, "holdout": (shifted_x, shifted_y)}, models,
                        exact=True, seed=seed, holdout_scale=scale,
                    )
    write_artifact(directory / "sensitivity.csv", buffer.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--quick", action="store_true", help="run the first seed only")
    args = parser.parse_args()
    try:
        run(args.output, range(42, 43 if args.quick else 90))
    except OSError as error:
        parser.exit(1, f"Holdout experiment failed: {error}\n")


if __name__ == "__main__":
    main()
