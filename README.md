# Unlearning Audit Toolkit

[![Tests](https://github.com/keynepekar/unlearning-audit-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/keynepekar/unlearning-audit-toolkit/actions/workflows/tests.yml)

Check whether loss exports from a machine unlearning experiment are ready to compare. The validator checks that each model was evaluated on the same examples, with consistent cohort labels.

## Validate a CSV

Run from the repository root with Python 3.12 or later.

```sh
python -m unlearning_audit validate examples/losses.csv
```

The example contains hand-written values to illustrate the file format. Each row contains a sample ID, its cohort (`forget` or `holdout`), a model name, and a finite, non-negative per-sample loss. Include at least two models. Each model must have exactly one row for every sample.

Use `forget` for examples selected for removal and `holdout` for examples excluded from every model's training data. The validator checks the consistency of these labels across exports. Confirm their provenance in the training pipeline.

## Tests

```sh
python -m unittest discover -s tests -v
```

Licensed under the [MIT License](LICENSE).
