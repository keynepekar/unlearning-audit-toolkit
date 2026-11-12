"""Validate a long-form CSV of aligned per-sample losses."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

COLUMNS = {"sample_id", "cohort", "model", "loss"}
REQUIRED_COHORTS = {"forget", "holdout"}
COHORTS = REQUIRED_COHORTS | {"utility"}


@dataclass(frozen=True)
class LossExport:
    """Aligned loss vectors in canonical sample-ID order within each cohort."""

    sample_ids: dict[str, tuple[str, ...]]
    losses: dict[str, dict[str, tuple[float, ...]]]

    @property
    def counts(self) -> dict[str, int]:
        return {cohort: len(ids) for cohort, ids in self.sample_ids.items()}


def load_losses(path: Path) -> LossExport:
    """Read, validate and align the models in a long-form loss export."""
    samples: dict[str, str] = {}
    models: dict[str, dict[str, float]] = {}

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
                raise ValueError(f"line {line}: expected forget, holdout or utility cohort")

            try:
                loss = float(row["loss"])
            except ValueError as error:
                raise ValueError(f"line {line}: loss must be a number") from error
            if not math.isfinite(loss) or loss < 0:
                raise ValueError(f"line {line}: loss must be finite and non-negative")

            if samples.setdefault(sample, cohort) != cohort:
                raise ValueError(f"line {line}: inconsistent cohort for sample {sample!r}")
            seen = models.setdefault(model, {})
            if sample in seen:
                raise ValueError(f"line {line}: duplicate sample {sample!r} for model {model!r}")
            seen[sample] = loss

    if len(models) < 2:
        raise ValueError("expected at least two models")
    for model, seen in models.items():
        missing_count = len(samples) - len(seen)
        if missing_count:
            raise ValueError(f"model {model!r} is missing {missing_count} sample(s)")

    if not REQUIRED_COHORTS.issubset(samples.values()):
        raise ValueError("both forget and holdout cohorts are required")

    grouped: dict[str, list[str]] = {}
    for sample, cohort in sorted(samples.items()):
        grouped.setdefault(cohort, []).append(sample)
    ids = {cohort: tuple(grouped[cohort]) for cohort in sorted(grouped)}
    aligned = {
        model: {cohort: tuple(values[sample] for sample in group) for cohort, group in ids.items()}
        for model, values in sorted(models.items())
    }
    return LossExport(ids, aligned)


def validate_losses(path: Path) -> dict[str, int]:
    """Return validated sample counts by cohort."""
    return load_losses(path).counts
