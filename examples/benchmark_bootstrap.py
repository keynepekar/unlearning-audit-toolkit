"""Time paired bootstrap backends on one aligned loss export."""

import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from unlearning_audit.groups import align_groups, load_groups
from unlearning_audit.manifest import load_manifest
from unlearning_audit.metrics import paired_auc_intervals
from unlearning_audit.validation import load_losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("losses", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--repeats", type=int, default=400)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if min(args.repeats, args.trials) < 1:
        parser.error("repeats and trials must be positive")
    export = load_losses(args.losses)
    manifest = load_manifest(args.manifest)
    manifest.check_alignment(set(export.losses), set(export.sample_ids))
    vectors = {role: export.losses[model] for role, model in manifest.models.items()}
    groups = align_groups(export, load_groups(args.groups)) if args.groups else None
    times = {"python": [], "numpy": []}
    for trial in range(args.trials):
        order = ("python", "numpy") if trial % 2 else ("numpy", "python")
        intervals = {}
        for backend in order:
            start = perf_counter()
            intervals[backend] = paired_auc_intervals(vectors, args.repeats, 42, groups, backend=backend)
            elapsed = perf_counter() - start
            times[backend].append(elapsed)
        if intervals["python"] != intervals["numpy"]:
            raise ValueError("bootstrap backends produced different intervals")
        print(f"Trial {trial + 1}: Python {times['python'][-1]:.3f} s, NumPy {times['numpy'][-1]:.3f} s", flush=True)
    summary = {"python": platform.python_version(), "numpy": np.__version__, "counts": export.counts,
               "group_counts": {cohort: len(parts) for cohort, parts in groups.items()} if groups else None,
               "repeats": args.repeats, "seconds": times,
               "median_seconds": {backend: median(values) for backend, values in times.items()},
               "intervals": intervals["python"]}
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
