"""Generate independently sampled synthetic calibration and holdout losses."""

import argparse
import csv
import json
from pathlib import Path
from random import Random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    rng = Random(42)
    cohorts = {"forget": 250, "holdout": 467, "calibration": 4000}
    models = {"before": "original", "candidate": "updated", "reference": "retrained"}
    with (args.output / "losses.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("sample_id", "cohort", "model", "loss"))
        for cohort, size in cohorts.items():
            for index in range(size):
                score = rng.random()
                for model, exponent in (("original", 2), ("updated", 1.5), ("retrained", 1)):
                    loss = score**exponent if cohort == "forget" else score
                    writer.writerow((f"{cohort}-{index:04d}", cohort, model, loss))
    manifest = {
        "schema_version": 1,
        "title": "IID calibration and holdout demonstration",
        "loss_definition": "Synthetic non-negative losses simulated from uniform draws.",
        "models": models,
        "cohorts": {cohort: f"{size} synthetic observations." for cohort, size in cohorts.items()},
        "provenance": {"Data": "Independent synthetic observations simulated with Python Random(42)."},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
