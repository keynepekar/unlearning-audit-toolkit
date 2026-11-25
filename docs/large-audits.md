# Comparing and rendering large audits

The bootstrap uses the same random draws for all three models. For mostly distinct scores, it now counts row multiplicities once and uses a cumulative sum in forget-score order. For each holdout row, it looks up the cumulative counts strictly before and through the corresponding score. The sum of these two counts gives full weight to an outranking forget row and half weight to a tie after normalization. For scores with many ties, a histogram over distinct ranks avoids a scan over the longer row order.

The offline HTML compacts dense ROC curves on the chart's display grid. It retains the first and last point in every occupied cell, plus the exact coordinates of reported descriptive and calibrated operating points. A straight segment between two retained points in one cell stays inside that cell. The JSON and ROC CSV retain every point.

For large inputs with few distinct scores, ROC construction counts occurrences by score and walks the sorted distinct thresholds. The histogram probe stops after a batch brings a cohort above 4,096 distinct scores. Inputs with a wider score range use the general sort. Both paths keep all tied rows together and retain every ROC point.

## Measured comparison

We ran a grouped audit with 350,000 samples, 1,050,000 CSV rows, mostly distinct scores and 400 bootstrap repetitions. On one Windows Intel Core i7-12700H system, v0.6.0 took 104.36 seconds and the optimized run took 57.53 seconds. The generated metrics JSON matched byte for byte. The corresponding HTML went from 14,263,114 to 70,002 bytes in the measured run. These are wall-clock observations from single runs on the same machine.

An interleaved comparison of the two bootstrap implementations used 100,000 observations, 35 draws and three runs per implementation. The median CPU time fell from 1.89 to 1.16 seconds for distinct scores. For scores with many ties it was 0.55 and 0.53 seconds. Every interval matched in both cases. The full v0.6.0 audit also replayed against inputs recreated by the load generator.

```sh
python examples/generate_audit_load.py large --samples 350000 --distinct
unlearning-audit compare large/losses.csv --manifest large/manifest.json --groups large/groups.csv --bootstrap 400 --fingerprints --output large/metrics.json
unlearning-audit verify large/metrics.json large/losses.csv --manifest large/manifest.json --groups large/groups.csv
unlearning-audit report large/metrics.json --output large/report.html
unlearning-audit report large/metrics.json --format roc-csv --output large/roc.csv
```
