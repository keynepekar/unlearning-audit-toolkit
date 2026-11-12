# Input contract and methodology

## Loss export

Provide a UTF-8 CSV with `sample_id,cohort,model,loss` as its header. For each model, include exactly one row for every sample. Use the same loss definition and evaluation protocol across models. The loss must be finite and non-negative. The reader trims IDs and model names and aligns rows by sample ID.

| Cohort | Meaning |
| --- | --- |
| `forget` | Examples in the original training data that were selected for removal. |
| `holdout` | Unseen examples used to estimate false positives. Match the forget cohort's relevant distribution as closely as the study allows. |
| `calibration` | Optional second set of non-members used to set thresholds before looking at the audit cohorts. |
| `utility` | Optional separate evaluation set for measuring mean loss. |

Use distinct IDs across cohorts. The exporter must establish training-set membership, split independence and the absence of duplicates with different IDs. The validator checks the declarations and alignment in the submitted files.

## Manifest

The manifest is a JSON object with `schema_version` set to `1`. Give the experiment a `title` and describe the metric in `loss_definition`. Assign the CSV model names to `before`, `candidate`, and `reference` in the `models` object. The reference represents the same training procedure run from scratch without the forget set. The `cohorts` object documents how each supplied cohort was formed. Use the `provenance` object for the data source, training procedure, environment and random seeds. All mapping values are non-empty strings. [Example manifest](../examples/manifest.json)

## Metrics

- **ROC AUC.** The audit ranks examples by negative loss. It builds the empirical ROC curve by moving through distinct loss values, with all ties entering together. Trapezoidal integration gives the probability that a forget example outranks a holdout example, with half credit for ties. AUC below 0.5 indicates separation in the opposite direction.
- **Descriptive operating points.** For each requested FPR cap, the toolkit reports the largest TPR on the empirical curve within the cap. Ties are handled as a group. The report shows the observed rates and their denominators. A cohort of size N can only resolve one observed false positive per N examples.
- **Calibrated operating points.** When a calibration cohort is present, choose the largest observed calibration loss whose inclusive threshold stays within the cap. If no tied block fits, use the all-negative rule. Apply the selected threshold to the forget and holdout cohorts. The measured holdout FPR can exceed the calibration cap.
- **Paired AUC differences.** The toolkit reports candidate minus original and candidate minus retrained. A stratified percentile bootstrap resamples each cohort with replacement and uses the same resampled indices for all three models. Its 95% interval describes variability across examples for these fixed models. Repeat training separately to study variation across model fits.
- **Utility.** If provided, the utility cohort contributes a separate mean loss for each model. Its interpretation depends on the declared loss and on how the cohort was formed.

Use `--fpr` once per requested target. Defaults are 0.01 and 0.10. The comparison defaults to 400 bootstrap repeats and seed 42. The output JSON retains counts, rates, curves and intervals. HTML and CSV consume that saved JSON independently of the original exports.

## Interpretation

Look at the original model's separation first. When its AUC is already close to 0.5, the chosen loss score may have little power to show a change after unlearning. Results describe one score, one comparison protocol and the supplied model fits. They do not establish deletion, formal equivalence to retraining or legal compliance. Dependencies between observations can make sample-level intervals optimistic.

## Demo data and references

- Alpaydin and Kaynak (1998), [Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49), UCI Machine Learning Repository, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The demo uses the subset bundled by scikit-learn, removes exact pixel duplicates, partitions it and scales pixel intensities. The dataset itself is supplied by scikit-learn.
- Carlini et al. (2022), [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570). Discusses why performance at low false-positive rates matters.
- Triantafillou et al. (2024), [Are we making progress in unlearning?](https://arxiv.org/abs/2406.09073). Describes the first NeurIPS unlearning challenge and comparison with retrained models.
- Thaker et al. (2025), [LLM Unlearning Benchmarks are Weak Measures of Progress](https://arxiv.org/abs/2410.02879). Examines how dataset construction and evaluation can affect the interpretation of unlearning results.
