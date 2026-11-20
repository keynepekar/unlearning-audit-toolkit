# Comparing and rendering large audits

The bootstrap uses the same random draws for all three models. For mostly distinct scores, it now counts row multiplicities once and uses a cumulative sum in forget-score order. For each holdout row, it looks up the cumulative counts strictly before and through the corresponding score. The sum of these two counts gives full weight to an outranking forget row and half weight to a tie after normalization. For scores with many ties, a histogram over distinct ranks avoids a scan over the longer row order.

The offline HTML compacts dense ROC curves on the chart's display grid. It retains the first and last point in every occupied cell, plus the exact coordinates of reported descriptive and calibrated operating points. A straight segment between two retained points in one cell stays inside that cell. The JSON and ROC CSV retain every point.

## Measured comparison

We ran a grouped audit with 350,000 samples, 1,050,000 CSV rows, mostly distinct scores and 400 bootstrap repetitions. On one Windows Intel Core i7-12700H system, v0.6.0 took 104.36 seconds and the optimized run took 57.53 seconds. The generated metrics JSON matched byte for byte. The corresponding HTML went from 14,263,114 to 70,002 bytes in the measured run. These are wall-clock observations from single runs on the same machine.

```sh
python examples/generate_audit_load.py large --samples 350000 --distinct
unlearning-audit compare large/losses.csv --manifest large/manifest.json --groups large/groups.csv --bootstrap 400 --fingerprints --output large/metrics.json
unlearning-audit verify large/metrics.json large/losses.csv --manifest large/manifest.json --groups large/groups.csv
unlearning-audit report large/metrics.json --output large/report.html
unlearning-audit report large/metrics.json --format roc-csv --output large/roc.csv
```
