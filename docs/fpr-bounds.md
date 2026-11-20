# Uncertainty at calibrated false-positive rates

Set a threshold on the calibration cohort and measure false positives on a separate holdout. With `--iid-holdout`, the HTML and summary CSV also show a one-sided 95% upper bound for that threshold's holdout FPR. Use this option when holdout rows are independent, identically distributed draws from the target non-member population and the model and threshold were selected independently of them. The option requires a calibration cohort and rejects grouped comparisons. It applies pointwise to each model and threshold. Simultaneous claims across thresholds or models require an additional procedure.

```sh
python examples/digits_demo.py digits-demo
unlearning-audit compare digits-demo/losses.csv --manifest digits-demo/manifest.json --iid-holdout --fpr 0.001 --fpr 0.01 --fpr 0.1 --fingerprints --output digits-demo/bounds.json
unlearning-audit verify digits-demo/bounds.json digits-demo/losses.csv --manifest digits-demo/manifest.json
unlearning-audit report digits-demo/bounds.json --output digits-demo/bounds.html
```

With zero false positives in 467 independent holdout rows, the upper bound is approximately 0.639%. Reaching a bound of 0.1% with zero events requires at least 2,995 independent rows. When calibration has too few rows or encounters ties and selects the all-negative rule, the table identifies that rule and leaves the bound field blank. The empirical cap and the observed FPR are still reported.

## Computation

For an observed proportion `p = k / n`, the code finds the upper rate `q` satisfying `n × KL(Bernoulli(p) || Bernoulli(q)) = −log(0.05)`. This inverts the binomial Chernoff lower-tail inequality and gives a conservative upper bound. With zero events it reduces to `1 − 0.05^(1/n)`, also the exact one-sided binomial bound for that case. With all rows positive the upper bound is one. [Hoeffding, 1963](https://doi.org/10.1080/01621459.1963.10500830) · [NIST Technical Note 2119](https://doi.org/10.6028/NIST.TN.2119)

The JSON stores the method identifier `binomial-kl-95-v1`. Reports derive bounds from the validated counts. The optional summary column `fpr_upper_95` uses rates in the interval [0, 1]. Descriptive rows have a blank cell because their thresholds are selected from the same observations used to estimate FPR.
