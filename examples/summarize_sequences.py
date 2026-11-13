"""Summarize per-step sequential benchmark observations by sequence."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


def summarize(rows, tolerance):
    sequences = defaultdict(list)
    keys = ("seed", "steps", "alpha", "spectrum", "leverage", "method")
    for row in rows:
        sequences[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, group in sorted(sequences.items()):
        group.sort(key=lambda row: int(row["step"]))
        expected = int(group[0]["steps"])
        if [int(row["step"]) for row in group] != list(range(1, expected + 1)):
            raise ValueError(f"incomplete or repeated steps for {key}")
        first_failure = None
        errors = []
        timings = []
        for row in group:
            timings.append(float(row["update_ms"]))
            error = float(row["prediction_error"]) if row["prediction_error"] else None
            coefficient = float(row["coefficient_error"]) if row["coefficient_error"] else None
            if error is not None:
                errors.append(error)
            if first_failure is None and (
                row["status"] != "ok" or error is None or coefficient is None
                or error > tolerance or coefficient > tolerance
            ):
                first_failure = int(row["step"])
        output.append(dict(zip(keys, key)) | {
            "first_failure_step": first_failure,
            "max_prediction_error": max(errors) if errors else None,
            "final_prediction_error": group[-1]["prediction_error"] or None,
            "total_update_ms": sum(timings),
            "median_update_ms": median(timings),
            "total_svd_ms": sum(float(row["svd_ms"]) for row in group),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if not 0 < args.tolerance < float("inf"):
        parser.error("--tolerance must be finite and positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("output must differ from input")
    with args.input.open(encoding="utf-8", newline="") as stream:
        result = summarize(csv.DictReader(stream), args.tolerance)
    if not result:
        parser.error("input has no observations")
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    print(f"Wrote {len(result)} sequence summaries to {args.output}")


if __name__ == "__main__":
    main()
