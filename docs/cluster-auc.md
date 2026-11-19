# Coverage of paired AUC intervals with grouped data

The group bootstrap samples independent units within each ROC cohort and keeps observations within each sampled group together. All model AUCs in a draw use the same sampled rows.

The simulation gives each cohort 12 or 32 groups. Group sizes are five rows or independently drawn from two through eight. Gaussian scores share a group effect, with within-group correlation zero or 0.7 before rounding. One comparison shifts the candidate's forget scores by 0.4 and has a population AUC difference of approximately 0.11135. The tied scenario rounds the scores and compares two correlated models with the same marginal distributions, giving a population difference of zero. AUC gives half weight to a tie. The simulation uses pair-count matrices with the same row weighting as the toolkit.

## Observed coverage at nominal 95%

| Groups per cohort | Correlation | Group sizes | Scenario | Row bootstrap | Group bootstrap |
| ---: | ---: | --- | --- | ---: | ---: |
| 12 | 0.0 | Equal | Shift | 99.0% | 98.3% |
| 12 | 0.7 | Equal | Shift | 87.5% | 95.2% |
| 12 | 0.7 | Unequal | Shift | 85.3% | 96.5% |
| 12 | 0.7 | Unequal | Tied null | 65.8% | 93.7% |
| 32 | 0.0 | Equal | Shift | 95.7% | 95.2% |
| 32 | 0.7 | Equal | Shift | 84.8% | 96.5% |
| 32 | 0.7 | Unequal | Shift | 78.2% | 94.3% |
| 32 | 0.7 | Unequal | Tied null | 68.5% | 94.2% |

With unequal group sizes, ties and 12 groups per cohort, the mean interval width was 0.188 for row resampling and 0.343 for group resampling. Both intervals exceeded nominal coverage in the independent 12-group scenario. The [release archive](https://github.com/keynepekar/unlearning-audit-toolkit/releases/download/v0.6.0/experiment-results-v060.zip) contains the full simulation output.

## Reproduce

```sh
python -m pip install . -r examples/requirements-demo.txt
python examples/cluster_auc_coverage.py cluster-auc-coverage.json
```

The full run uses 600 simulated datasets per scenario and 799 bootstrap draws per dataset and method. The JSON records observed coverage and mean interval width. Use `--quick` to check the entry point.

The simulations draw independent groups in each cohort and make group size independent of scores. Coverage depends on the number and size of groups, the score distribution and the sampling design. [Paired ROC comparisons with clustered data](https://doi.org/10.1198/016214508000000364) · [Bootstrap intervals for correlated AUC data](https://doi.org/10.3389/fvets.2023.1274786)
