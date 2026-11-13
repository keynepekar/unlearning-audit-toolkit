# Sequential ridge removal experiment

The experimental block QR implementation retains a local copy of the training rows and caches a factor for each block. A deletion recomputes factors for touched blocks and solves using the combined factors and one ridge regularization term. The example is intended for numerical comparison. Input arrays and factors remain in memory and are included separately in its byte counts.

## Run the synthetic experiment

```sh
python -m pip install . -r examples/requirements-demo.txt
python examples/sequential_benchmark.py sequential-results
python examples/summarize_sequences.py sequential-results/sequences.csv sequential-results/summary.csv
```

The script writes a configuration with its SHA-256 and one CSV row per method and deletion step. The full profile crosses six seeds, 10 or 100 steps, three regularization strengths, three feature spectrum ratios and two leverage settings. Every sequence removes the first 100 rows and compares each step against a scikit-learn SVD refit on remaining rows. Requests arrive in fixed contiguous order. Timings include the update or refit for that step and exclude initial factorization and data generation. Use `--quick` for the CI profile.

On the recorded Windows run, 132 of 216 Gram-subtraction sequences and 26 of 216 QR sequences had at least one step where coefficient or holdout prediction relative L2 error exceeded 10⁻⁶. The largest holdout prediction errors were 5.98 for Gram and 7.72 × 10⁻⁷ for QR. These cases include deliberately ill-conditioned inputs. They are comparisons for the stated configuration and do not establish a numerical guarantee.

## Public-data example

Download the red wine CSV from [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), then run:

```sh
python examples/wine_experiment.py winequality-red.csv wine-results
```

The script verifies the expected SHA-256, removes duplicate full rows before splitting, uses the original quality score, and adds a penalized intercept. It reports separate results for four train/test splits and two removal fractions. The UCI dataset is by Cortez et al. (2009), licensed CC BY 4.0, DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T). [License](https://creativecommons.org/licenses/by/4.0/)

For the recorded run, the largest relative holdout prediction errors versus the SVD refit were 3.35 × 10⁻¹⁴ for Gram and 4.93 × 10⁻¹⁵ for block QR. Splits may share observations across runs. The example uses random deletion rather than a similarity-based deletion study.

## Scope

The CSV includes per-step timings and memory for stored inputs and factors. It does not measure peak process memory. The example does not automatically choose a reconstruction policy. The equal-squared-loss regression control in the test suite demonstrates that identical losses can mask different predictions.
