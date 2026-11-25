# Fashion-MNIST membership control

The experiment tests whether a retain-only CNN fine-tune approaches retraining when the original model already gives lower losses to the forget cohort than to the official test cohort. The initial 60,000-image pilot had loss-based AUC near 0.50. This follow-up uses a smaller, stratified training set and longer training to make the initial membership comparison more informative. [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) · [Loss-based membership evaluation](https://arxiv.org/abs/2112.03570)

## Protocol

Draw 500 images per class from the official training data with a fixed split seed. Place 100 per class in the forget cohort and the other 400 in the retain cohort. Use all 10,000 official test images as holdout. For each training seed, fit the original on all 5,000 selected training images for 30 epochs. Fit the reference from the same initialization seed on the 4,000 retained images for 30 epochs. Continue training a copy of the original on the retained images for 10 epochs with a fresh AdamW optimizer. Score every model on the same forget and holdout rows using categorical cross-entropy. The paired bootstrap uses 400 draws within the two audit cohorts for each fixed model triplet.

## Observed results

| Model | Loss AUC across six seeds |
| --- | ---: |
| Original | 0.5273–0.5369 |
| Retain-only fine-tune | 0.5241–0.5334 |
| Retrained | 0.5047–0.5109 |

The candidate AUC fell relative to the original in five of six seeds. The paired interval for this difference excluded zero in one seed. The candidate AUC exceeded the retrained AUC in all six seeds, with each paired interval above zero. The comparison describes one fixed split and one loss score. Training repeats share the same data, and their spread describes variation across the fitted models. A low AUC alone gives no guarantee of erasure.

The six runs took 129 seconds, including scoring and audit calculation, on an NVIDIA RTX 3050 Laptop GPU with PyTorch 2.15.0 and CUDA 13.2. PyTorch tracked a peak allocation of 285.5 MiB. All six saved audits replayed with the NumPy backend.

## Reproduce

Download the four `*.gz` files from the dataset publisher into `fashion-data/`. The example checks each file against the publisher's checksum and validates its IDX dimensions. Install the toolkit with the optional `fast` extra and install PyTorch for your platform. [Data downloads](https://github.com/zalandoresearch/fashion-mnist#get-the-data) · [PyTorch installation guide](https://pytorch.org/get-started/locally/)

```sh
python -m pip install '.[fast]'
python examples/fashion_audit.py fashion-data fashion-run --device cuda --seeds 6
unlearning-audit verify fashion-run/seed-42/metrics.json fashion-run/seed-42/losses.csv --manifest fashion-run/seed-42/manifest.json --bootstrap-backend numpy
unlearning-audit study fashion-run/seed-*/metrics.json --format html --output fashion-study.html
```

For a CPU smoke run, pass `--device cpu --seeds 1`. Numeric results can vary across hardware and software stacks. The release archive contains aggregate audit results for all six seeds. [Reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
