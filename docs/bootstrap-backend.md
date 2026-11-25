# Faster paired bootstrap with NumPy

The optional NumPy backend accelerates paired AUC intervals for individual rows and independent groups. The Python backend remains the default. Both implementations use the same draws across the original, candidate and reference models. Saved metrics from either backend can be verified by the other.

## Run an audit

```sh
python -m pip install '.[fast]'
unlearning-audit compare examples/comparison.csv --manifest examples/manifest.json --groups examples/groups.csv --fingerprints --bootstrap-backend numpy --output metrics.json
unlearning-audit verify metrics.json examples/comparison.csv --manifest examples/manifest.json --groups examples/groups.csv --bootstrap-backend numpy
```

The NumPy backend copies the CPython MT19937 state and checks that its sampled indices match CPython before running. It counts group draws with replacement, maps each group count to its member rows, and accumulates integer counts by loss rank. An earlier forget score contributes full weight and a tied score contributes half weight. The denominator uses the actual resampled cohort sizes, which vary when groups have unequal sizes. The backend checks the maximum possible count products against the signed 64-bit range. Use the Python backend if the range check fails. [NumPy legacy generator](https://numpy.org/doc/stable/reference/random/legacy.html) · [Grouping contract](methodology.md#grouped-observations)

## Measurements

On Windows with an Intel Core i7-12700H, CPython 3.12 and NumPy 2.5.4, 400 grouped draws on 100,000 synthetic observations with unequal group sizes took a median of 11.87 s in Python and 1.48 s in NumPy across three paired timings. All intervals matched. The benchmark times interval computation, including rank preparation. A separate end-to-end NumPy replay matched the saved Python metrics for the 350,000-observation grouped fixture with 1,050,000 CSV rows and 400 draws.

```sh
python examples/generate_audit_load.py grouped --samples 100000 --distinct
python examples/benchmark_bootstrap.py grouped/losses.csv grouped/manifest.json bootstrap.json --groups grouped/groups.csv
```
