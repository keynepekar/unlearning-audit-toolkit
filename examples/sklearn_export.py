"""Export per-example classification losses from scikit-learn estimators."""

import csv
from pathlib import Path

import numpy as np

from unlearning_audit.validation import COHORTS, REQUIRED_COHORTS


def export_losses(path: Path, models: dict, cohorts: dict):
    """Write aligned cross-entropy losses from fitted probability classifiers.

    cohorts maps names to (sample IDs, feature matrix, labels). The caller
    establishes data provenance and membership independently.
    """
    if len(models) < 2 or not REQUIRED_COHORTS.issubset(cohorts):
        raise ValueError("at least two models and forget/holdout cohorts are required")
    if not set(cohorts).issubset(COHORTS):
        raise ValueError("unknown cohort name")
    model_names = tuple(models)
    if any(not isinstance(name, str) or not name.strip() for name in model_names):
        raise ValueError("model names must be nonempty strings")
    normalized_names = tuple(name.strip() for name in model_names)
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("model names must be unique after trimming")
    prepared = []
    ids_seen = set()
    for name, (ids, x, y) in cohorts.items():
        ids = list(ids)
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 2 or y.ndim != 1 or len(ids) != len(x) or len(y) != len(x) or not ids:
            raise ValueError(f"incompatible cohort arrays: {name}")
        if any(not isinstance(item, str) or not item.strip() for item in ids):
            raise ValueError("sample IDs must be nonempty strings")
        ids = [item.strip() for item in ids]
        if len(set(ids)) != len(ids) or ids_seen.intersection(ids):
            raise ValueError("sample IDs must be unique across cohorts")
        ids_seen.update(ids)
        prepared.append((name, ids, x, y))
    output = []
    for model_name, model in zip(normalized_names, models.values(), strict=True):
        for cohort, ids, x, y in prepared:
            probabilities = np.asarray(model.predict_proba(x))
            classes = list(model.classes_)
            if probabilities.shape != (len(y), len(classes)) or not np.isfinite(probabilities).all():
                raise ValueError("invalid predicted probabilities")
            if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1):
                raise ValueError("predicted probabilities must sum to one")
            lookup = {label: index for index, label in enumerate(classes)}
            try:
                columns = [lookup[label] for label in y]
            except KeyError as error:
                raise ValueError("target class is absent from a model") from error
            loss = -np.log(np.maximum(probabilities[np.arange(len(y)), columns], 1e-15))
            output.extend((sample, cohort, model_name, value) for sample, value in zip(ids, loss, strict=True))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("sample_id", "cohort", "model", "loss"))
        writer.writerows(output)
