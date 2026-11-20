# Reproduce the Digits loss export

The Digits example can record the inputs it used to produce its loss CSV. The scoring replay reconstructs the fitted classifiers from numerical JSON files and scores the examples again.

```sh
python -m pip install . -r examples/requirements-demo.txt
python examples/digits_demo.py digits-record --record-scoring
python examples/digits_demo.py digits-record --verify-scoring
unlearning-audit verify digits-record/metrics.json digits-record/losses.csv --manifest digits-record/manifest.json
```

`scoring.json` records SHA-256 digests of the bundled pixels, labels, ordered split indices, preprocessed features, scorer source, the three numeric checkpoints, loss CSV, manifest and metrics JSON. It also records the preprocessing divisor, probability floor and Python, NumPy, SciPy and scikit-learn versions. Array digests include shape and dtype, with little-endian values in C order. Source digests normalize line endings. Artifact digests use the exact bytes. The replay scores the original partitions with one BLAS thread and checks the regenerated CSV's digest.

The record binds scoring to the artifacts used in this example. Retain it with the run and keep the checkpoints local when they should stay private. Use the same source and package versions for replay. Differences in floating-point results across platforms can prevent byte-for-byte verification. The hashes support consistency checks against a trusted copy of the record. Training and data lineage still depend on the recorded workflow.

The separate `unlearning-audit verify` command starts from exported losses and recalculates the audit metrics. The scoring replay starts from the numeric classifier state. The provenance fields are informed by [TRAIL](https://github.com/sepehrkdi/trail).

For records produced by v0.6.0, run the scoring replay from the `v0.6.0` source checkout. That tag's manifest text says the holdout is 35% of the evaluation partition. The implementation uses 35% for the reserve split and the remaining 65% for the holdout, subject to rounding.
