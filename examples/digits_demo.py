"""Train three small classifiers and audit their outputs with the installed package."""

import argparse
import csv
import json
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from unlearning_audit.manifest import load_manifest
from unlearning_audit.metrics import compare_losses
from unlearning_audit.output import write_artifact
from unlearning_audit.validation import load_losses

SEED = 42
TRAIN_EPOCHS = 30
UPDATE_EPOCHS = 10


def split_examples(images: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Keep one copy of each exact image before creating stratified partitions."""
    _, first_indices = np.unique(images, axis=0, return_index=True)
    unique_rows = np.sort(first_indices)
    train, evaluation = train_test_split(
        unique_rows, test_size=0.40, random_state=SEED, stratify=labels[unique_rows],
    )
    retain, forget = train_test_split(train, test_size=0.25, random_state=SEED, stratify=labels[train])
    holdout, reserve = train_test_split(
        evaluation, test_size=0.35, random_state=SEED, stratify=labels[evaluation],
    )
    calibration, utility = train_test_split(
        reserve, test_size=0.5, random_state=SEED, stratify=labels[reserve],
    )
    return {
        "retain": retain, "forget": forget, "holdout": holdout,
        "calibration": calibration, "utility": utility,
    }


def train(images: np.ndarray, labels: np.ndarray, indices: np.ndarray, training_seed: int) -> SGDClassifier:
    model = SGDClassifier(
        loss="log_loss", learning_rate="constant", eta0=0.03, random_state=training_seed, shuffle=False,
    )
    update(model, images, labels, indices, TRAIN_EPOCHS, training_seed)
    return model


def update(model, images, labels, indices, epochs, training_seed):
    rng = np.random.default_rng(training_seed)
    classes = np.unique(labels)
    for _ in range(epochs):
        rows = rng.permutation(indices)
        model.partial_fit(images[rows], labels[rows], classes=classes)


def export_losses(path, models, images, labels, groups):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("sample_id", "cohort", "model", "loss"))
        for name, model in models.items():
            for cohort in ("forget", "holdout", "calibration", "utility"):
                rows = groups[cohort]
                probabilities = model.predict_proba(images[rows])
                columns = np.searchsorted(model.classes_, labels[rows])
                true_probabilities = probabilities[np.arange(len(rows)), columns]
                losses = -np.log(np.maximum(true_probabilities, 1e-15))
                writer.writerows((f"digit-{index:04d}", cohort, name, loss) for index, loss in zip(rows, losses))


def run(output: Path, *, unchanged: bool, training_seed: int = SEED) -> None:
    output.mkdir(parents=True, exist_ok=False)
    pixels, labels = load_digits(return_X_y=True)
    groups = split_examples(pixels, labels)
    images = pixels / 16.0
    with threadpool_limits(limits=1):
        original = train(images, labels, np.concatenate((groups["retain"], groups["forget"])), training_seed)
        candidate = deepcopy(original)
        if not unchanged:
            update(candidate, images, labels, groups["retain"], UPDATE_EPOCHS, training_seed)
        reference = train(images, labels, groups["retain"], training_seed)
        export_losses(
            output / "losses.csv",
            {"original": original, "candidate": candidate, "retrained": reference},
            images, labels, groups,
        )

    candidate_method = (
        "Unchanged copy of the original classifier."
        if unchanged else f"Original classifier updated on retain only for {UPDATE_EPOCHS} additional epochs."
    )
    manifest = {
        "schema_version": 1,
        "title": "Handwritten digits · unchanged control" if unchanged else "Handwritten digits · retain-only update",
        "loss_definition": "Negative log of the true digit probability, with a probability floor of 1e-15.",
        "models": {"before": "original", "candidate": "candidate", "reference": "retrained"},
        "cohorts": {
            "forget": "Stratified 25% of the original training partition. Used in original training only.",
            "holdout": "Stratified 65% of the evaluation partition. Excluded from all training.",
            "calibration": "One half of the remaining evaluation partition, used to select thresholds only.",
            "utility": "Other half of the remaining evaluation partition, used to measure utility only.",
        },
        "provenance": {
            "Protocol": "digits-sgd-unchanged-v1" if unchanged else "digits-sgd-retain-update-v1",
            "Training seed": str(training_seed),
            "Dataset": "Optical Recognition of Handwritten Digits (Alpaydin and Kaynak, 1998), UCI DOI 10.24432/C50P49, CC BY 4.0, as bundled by scikit-learn.",
            "Split": f"Exact pixel duplicates removed before a stratified 60/40 train/evaluation split. Seed {SEED}.",
            "Training": f"SGD logistic classifier, {TRAIN_EPOCHS} epochs, constant learning rate 0.03, scaled pixels.",
            "Candidate": candidate_method,
            "Reference": "Same training procedure initialized from scratch on retain only.",
            "Environment": f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, NumPy {np.__version__}, SciPy {scipy.__version__}.",
        },
    }
    manifest_file = output / "manifest.json"
    write_artifact(manifest_file, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    result = compare_losses(
        load_losses(output / "losses.csv"), load_manifest(manifest_file), seed=SEED, include_fingerprints=True,
    )
    write_artifact(output / "metrics.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for the CSV, manifest and metrics")
    parser.add_argument("--unchanged", action="store_true", help="use an unchanged candidate as a control")
    parser.add_argument("--training-seed", type=int, default=SEED, help="training randomness; the split stays fixed")
    args = parser.parse_args()
    if not 0 <= args.training_seed < 2**32:
        parser.error("--training-seed must be between 0 and 4294967295")
    try:
        run(args.output, unchanged=args.unchanged, training_seed=args.training_seed)
    except OSError as error:
        parser.exit(1, f"Demo failed: {error}\n")
    print(f"Wrote demo artifacts to {args.output}")


if __name__ == "__main__":
    main()
