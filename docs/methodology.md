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
- **Paired loss gaps.** With `compare --loss-gaps`, report the mean absolute difference between each model's loss and the reference loss on the same examples, separately by cohort. This keeps information about the magnitude of differences that ROC ranks discard. [Controlled ridge example](ridge-controls.md)

Use `--fpr` once per requested target. Defaults are 0.01 and 0.10. The comparison defaults to 400 bootstrap repeats and seed 42. The output JSON retains counts, rates, curves and intervals. HTML and CSV consume that saved JSON independently of the original exports.

Use `report --format roc-csv` to export every plotted ROC point with its false-positive and true-positive counts. The HTML report provides text and tables alongside the chart. Its chart lines differ by both color and stroke pattern.

## Verify a saved audit

Add `--fingerprints` to `compare` to record SHA-256 digests for cohort assignments, model losses and the manifest. Then run:

```sh
unlearning-audit verify metrics.json losses.csv --manifest manifest.json
```

Fingerprints are computed on parsed values. Row order, JSON key order, and equivalent numeric spellings give the same digests. A change to an assignment, a numeric loss or a normalized manifest field changes the relevant digest. Verification checks those digests and recomputes the results with the saved targets and bootstrap settings. The format is identified by `unlearning-audit-sha256-v1`.

Fingerprints establish consistency with a retained export. They do not authenticate the source of the files or establish that the declared cohorts are correct. A split digest can also link runs that used the same identifiers. Leave fingerprints disabled when sharing results that should not be linked.

## Compare training repeats

For a fixed-split study, create at least two comparisons with fingerprints. Put a distinct integer `Training seed` and a common `Protocol` in each manifest's `provenance`. Keep all other provenance fields, the loss definition, cohorts and FPR targets fixed. `study` rejects different split fingerprints, duplicate seeds and mismatched protocol metadata. The digits demo sets these fields automatically.

The study reports individual run estimates, mean, sample standard deviation and observed range. Keep this dispersion separate from the paired bootstrap interval of an individual run. Training-seed runs reuse the same examples, and their range describes these fitted models on this split. It is not a confidence interval over data splits.

In the six-seed digits example, the observed candidate-minus-original AUC range is −0.0080 to −0.0028. The candidate-minus-retrained range is +0.0089 to +0.0165. These are small changes in a comparison where the original model's AUC is already close to chance-level ranking. [Example study report](digits-study.png)

A longer run used the same evaluation split and training seeds 42 through 89. The first six seeds are included in the 48-run set.

| Candidate AUC minus | Six seeds, observed range | 48 seeds, observed range | 48 seeds, mean |
| --- | ---: | ---: | ---: |
| Original | −0.0080 to −0.0028 | −0.0117 to −0.0009 | −0.0063 |
| Retrained | +0.0089 to +0.0165 | +0.0073 to +0.0165 | +0.0122 |

Both differences kept their sign across the 48 runs, while the observed range widened. This measures sensitivity to training seeds on one split. To reproduce the extended run from a POSIX shell after installing the demo dependencies:

```sh
mkdir seed-study
for seed in $(seq 42 89); do
    python examples/digits_demo.py "seed-study/seed-$seed" --training-seed "$seed" || exit 1
done
unlearning-audit study seed-study/seed-*/metrics.json --format html --output seed-study/study.html
```

## Interpretation

Look at the original model's separation first. When its AUC is already close to 0.5, the chosen loss score may have little power to show a change after unlearning. Results describe one score, one comparison protocol and the supplied model fits. They do not establish deletion, formal equivalence to retraining or legal compliance. Dependencies between observations can make sample-level intervals optimistic.

## Demo data and references

The [implementation references](sources.md) map methods to their sources and code.

- Alpaydin and Kaynak (1998), [Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49), UCI Machine Learning Repository, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The demo uses the subset bundled by scikit-learn, removes exact pixel duplicates, partitions it and scales pixel intensities. The dataset itself is supplied by scikit-learn.
- Carlini et al. (2022), [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570). Discusses why performance at low false-positive rates matters.
- Triantafillou et al. (2024), [Are we making progress in unlearning?](https://arxiv.org/abs/2406.09073). Describes the first NeurIPS unlearning challenge and comparison with retrained models.
- Thaker et al. (2025), [LLM Unlearning Benchmarks are Weak Measures of Progress](https://arxiv.org/abs/2410.02879). Examines how dataset construction and evaluation can affect the interpretation of unlearning results.
- Xu et al. (2026), [Unlearning Isn't Deletion](https://proceedings.mlr.press/v306/xu26cd.html). Studies reversibility in language models and the limitations of task-level metrics in that setting.
- [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist). Discusses documenting the sources of variability in reported statistics.
