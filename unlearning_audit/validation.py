"""Validate a long-form CSV of aligned per-sample losses."""

import csv
import math
from pathlib import Path

COLUMNS = {"sample_id", "cohort", "model", "loss"}
COHORTS = {"forget", "holdout"}


def validate_losses(path: Path) -> dict[str, int]:
    """Return counts by cohort, raising ValueError for invalid input."""
    samples: dict[str, str] = {}
    models: dict[str, set[str]] = {}

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        header = reader.fieldnames or []
        if len(header) != len(COLUMNS) or set(header) != COLUMNS:
            raise ValueError("expected columns: sample_id, cohort, model, loss")

        for row in reader:
            line = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"line {line}: expected four fields")

            sample = row["sample_id"].strip()
            model = row["model"].strip()
            cohort = row["cohort"].strip()
            if not sample or not model:
                raise ValueError(f"line {line}: sample_id and model must be non-empty")
            if cohort not in COHORTS:
                raise ValueError(f"line {line}: cohort must be forget or holdout")

            try:
                loss = float(row["loss"])
            except ValueError as error:
                raise ValueError(f"line {line}: loss must be a number") from error
            if not math.isfinite(loss) or loss < 0:
                raise ValueError(f"line {line}: loss must be finite and non-negative")

            if samples.setdefault(sample, cohort) != cohort:
                raise ValueError(f"line {line}: inconsistent cohort for sample {sample!r}")
            seen = models.setdefault(model, set())
            if sample in seen:
                raise ValueError(f"line {line}: duplicate sample {sample!r} for model {model!r}")
            seen.add(sample)

    if len(models) < 2:
        raise ValueError("expected at least two models")
    for model, seen in models.items():
        missing_count = len(samples) - len(seen)
        if missing_count:
            raise ValueError(f"model {model!r} is missing {missing_count} sample(s)")

    counts = dict.fromkeys(sorted(COHORTS), 0)
    for cohort in samples.values():
        counts[cohort] += 1
    if not all(counts.values()):
        raise ValueError("both forget and holdout cohorts are required")
    return counts
