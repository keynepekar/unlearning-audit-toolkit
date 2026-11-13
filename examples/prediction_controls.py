"""Show how aggregate loss metrics respond to controlled prediction changes."""

import csv
import math
import sys

from unlearning_audit.metrics import area_under_roc, roc_counts


def diagnostic(reference, candidate, target):
    reference_loss = [(prediction - y) ** 2 for prediction, y in zip(reference, target, strict=True)]
    candidate_loss = [(prediction - y) ** 2 for prediction, y in zip(candidate, target, strict=True)]
    midpoint = len(target) // 2
    deviations = sorted(abs(a - b) for a, b in zip(candidate, reference, strict=True))
    return {
        "reference_auc": area_under_roc(roc_counts(tuple(reference_loss[:midpoint]), tuple(reference_loss[midpoint:]))),
        "candidate_auc": area_under_roc(roc_counts(tuple(candidate_loss[:midpoint]), tuple(candidate_loss[midpoint:]))),
        "mean_loss_gap": sum(abs(a - b) for a, b in zip(candidate_loss, reference_loss, strict=True)) / len(target),
        "mean_prediction_gap": sum(deviations) / len(target),
        "p95_prediction_gap": deviations[math.ceil(0.95 * len(deviations)) - 1],
        "max_prediction_gap": deviations[-1],
    }


def main():
    targets = [float(index) for index in range(100)]
    reference = [y + 0.5 for y in targets]
    cases = {
        "identical": reference,
        "equal_loss_opposite_prediction": [y - 0.5 for y in targets],
        "one_row_changed": [reference[0] + 10, *reference[1:]],
        "loss_rescaled": [y + 1 for y in targets],
    }
    writer = csv.DictWriter(sys.stdout, fieldnames=["case", *diagnostic(reference, reference, targets)])
    writer.writeheader()
    for name, prediction in cases.items():
        writer.writerow({"case": name, **diagnostic(reference, prediction, targets)})


if __name__ == "__main__":
    main()
