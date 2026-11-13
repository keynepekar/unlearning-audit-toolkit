# Holdout sensitivity with an exact ridge deletion

Both the forget and holdout examples are unseen by a model retrained from scratch on the retained data. Its ROC AUC can still move away from 0.5 when the cohorts have different loss distributions. This experiment isolates that effect.

## Protocol

For each of 48 seeds, reuse the [ridge control](ridge-controls.md) to fit the original model, subtract the forget rows' sufficient statistics for the exact update, and refit independently using only the retained rows. The forget set contains 100 rows. For the matched holdout, draw 200 independent rows from the forget generator. The shifted conditions multiply both the features and targets of those same holdout rows by 0.65 or 1.50. This keeps the models, forget rows, holdout size and underlying draws fixed within each seed. The model has no intercept, so its holdout squared losses are multiplied by the square of the scale factor.

| Holdout | Mean retrained AUC | Observed range across seeds |
| --- | ---: | ---: |
| Lower amplitude (0.65×) | 0.170 | 0.117–0.255 |
| Matched (1.00×) | 0.496 | 0.409–0.613 |
| Higher amplitude (1.50×) | 0.808 | 0.738–0.869 |

The exact update and the retrained model had identical AUCs in all 144 comparisons. The range describes observed variation across seeds. The deliberately chosen shifts illustrate sensitivity on this synthetic generator. In other datasets a retrained-model AUC near 0.5 can coexist with cohort differences that the loss rank does not capture.

## Reproduce

```sh
python -m pip install . -r examples/requirements-demo.txt
python examples/holdout_sensitivity.py holdout-study
unlearning-audit verify holdout-study/higher_amplitude/metrics.json holdout-study/higher_amplitude/losses.csv --manifest holdout-study/higher_amplitude/manifest.json
```

`sensitivity.csv` gives per-seed AUCs for all three models. For seed 42, the script also writes the losses, manifest, metrics and standalone report for each holdout. Use `--quick` to run only seed 42.
