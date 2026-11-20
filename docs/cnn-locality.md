# Prediction and representation differences in a small CNN

This experiment uses the public UCI Digits data bundled with scikit-learn. It asks whether a retain-only fine-tune approaches retraining across holdout examples at different distances from the forget set. The trained network, update and metrics are specific to this experiment. [Forgetting Has Neighbors](https://arxiv.org/abs/2605.31317) · [Do unlearning algorithms learn the right features?](https://arxiv.org/abs/2611.06681)

## Protocol

The split removes exact pixel duplicates and stays fixed across training seeds. The model has two convolutional layers and a 32-dimensional embedding before its classification head. In each run, fit the original for 30 epochs on retain and forget examples. Fine-tune a copy for eight more epochs on retain only. Fit two new models on retain only for 30 epochs, one with the original initialization seed and the other with an independent seed. Use AdamW throughout. Batch sizes are at most 64.

Compute the scaled-pixel Euclidean distance from each holdout example to its nearest forget example, then divide the ordered holdout into thirds. Within each band, measure the mean total variation distance between class probabilities and centered linear CKA between embeddings. Linear CKA compares relationships between samples and is invariant to orthogonal transformations and isotropic scaling of an embedding. The independent refit gives a baseline for variation between training seeds. [Kornblith et al.](https://proceedings.mlr.press/v97/kornblith19a.html)

Repeated seeds reuse the same split. The distance bands can have different digit mixtures. This is a descriptive experiment about one architecture, one update and one split.

## Observed results

The CUDA run used 24 training seeds, 42 through 65, with 467 holdout images and 270 forget images. Mean values across the seeds are shown below. Higher CKA indicates more similar relationships between the holdout embeddings.

| Model | Near TV | Middle TV | Far TV | Near CKA | Middle CKA | Far CKA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0.02145 | 0.03108 | 0.06335 | 0.96473 | 0.96176 | 0.95102 |
| Retain-only update | 0.02013 | 0.02660 | 0.05609 | 0.96547 | 0.96321 | 0.95293 |
| Independent refit | 0.02946 | 0.03813 | 0.07733 | 0.90183 | 0.90356 | 0.87996 |

The update lowered the mean prediction gap in all three bands. The far band still had the largest average gap in this experiment. Mean holdout accuracies were 97.29% for the original, 97.66% for the update and 96.42% for the independent refit. For seed 42, the fixed loss score assigned AUC 0.485 to the original, so this score showed weak membership separation. The probability and embedding comparisons measure agreement with the matched retrained model.

The RTX 3050 Laptop run took 291 seconds with PyTorch 2.15.0 and CUDA 13.2. Peak memory tracked by the PyTorch CUDA allocator was 33.8 MiB. The [release archive](https://github.com/keynepekar/unlearning-audit-toolkit/releases/download/v0.7.0/experiment-results-v070.zip) includes all 216 band comparisons and the run configuration.

## Reproduce

Install the demo dependencies in an isolated Python 3.12 environment.

```sh
python -m pip install . -r examples/requirements-demo.txt
```

Install PyTorch 2.15.0 for your platform. The CUDA run uses the official CUDA 13.2 wheel. Keep that choice separate from the core audit package dependencies. [PyTorch installation guide](https://pytorch.org/get-started/locally/)

```sh
python -m pip install torch==2.15.0+cu132 --index-url https://download.pytorch.org/whl/cu132
python examples/cnn_locality.py cnn-run --device cuda --seeds 24
unlearning-audit verify cnn-run/metrics.json cnn-run/losses.csv --manifest cnn-run/manifest.json
```

For a CPU run, use the PyTorch CPU wheel and pass `--device cpu --seeds 1`. Use a fresh output directory each time. `locality.csv` contains the per-seed rows and `run.json` records the environment, holdout accuracy, runtime and peak PyTorch CUDA allocation. The CSV, manifest and metrics for the first seed exercise the audit workflow. Training can vary across software and hardware stacks. [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
