# Prediction discrepancies near a fixed deletion

This experiment measures how a ridge model's prediction gap to retraining varies with distance to a fixed forget set. Dolgova and Stich study this question with deep classifiers and approximate updates. Here the data are tabular, and the ridge objective uses fixed features. [Forgetting Has Neighbors](https://arxiv.org/abs/2605.31317)

## Protocol

Use the red Wine Quality dataset. Deduplicate complete records and generate 48 deterministic 70/30 train/test splits using seeds 42 to 89. In each split, delete the 10% of training rows closest to the first training row, rounding up. Measure Euclidean distance on the 11 feature columns scaled by their full-training standard deviations. The training objective uses the raw features, a penalized intercept and a ridge penalty of 4.

Fit an independent retained-data SVD reference. Compare it with the unchanged original, one diagonal Newton step on the retained objective starting at the original weights, and a block QR update. Every model in a split uses the same deletion. Divide the held-out rows into near, middle and far thirds by distance to their nearest deleted row. Every model uses the same bands. The CSV reports each band's bounds, mean, 95th percentile and maximum absolute prediction gap from the reference, plus mean absolute error against the observed quality score.

## Results

Mean band-level absolute prediction gap across the 48 splits, in quality-score units.

| Method | Near | Middle | Far |
| --- | ---: | ---: | ---: |
| Unchanged control | 0.02562 | 0.01590 | 0.01856 |
| Diagonal Newton step | 0.05638 | 0.06923 | 0.08467 |
| Block QR | < 0.00001 | < 0.00001 | < 0.00001 |

The diagonal update had a larger mean gap in the far band in 46 of 48 splits and increased all three average gaps relative to the unchanged control. Block QR's largest pointwise gap was 1.20 × 10⁻¹³. The approximate update illustrates why inspecting several bands is useful even when a method's largest discrepancies are away from the forget set. The splits can share examples. These results describe the tested update and dataset. The [release archive](https://github.com/keynepekar/unlearning-audit-toolkit/releases/download/v0.6.0/experiment-results-v060.zip) contains the per-band CSV.

## Reproduce

Download `winequality-red.csv` from UCI. The script checks its SHA-256 before running.

```sh
python -m pip install . -r examples/requirements-demo.txt
python examples/wine_locality.py winequality-red.csv wine-locality.csv
```

The dataset is by Cortez et al. (2009) and is licensed under CC BY 4.0. [UCI Wine Quality](https://doi.org/10.24432/C56S3T) · [License](https://creativecommons.org/licenses/by/4.0/)
