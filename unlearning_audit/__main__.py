"""Command-line entry point for local audits."""

import argparse
import csv
from pathlib import Path

from .manifest import load_manifest
from .validation import load_losses


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m unlearning_audit")
    commands = parser.add_subparsers(required=True)
    validate = commands.add_parser("validate", help="check a CSV of per-sample losses")
    validate.add_argument("csv_file", type=Path)
    validate.add_argument("--manifest", type=Path, help="also check declared models and cohorts")
    args = parser.parse_args()

    try:
        export = load_losses(args.csv_file)
        if args.manifest:
            manifest = load_manifest(args.manifest)
            manifest.check_alignment(set(export.losses), set(export.sample_ids))
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Validation failed: {error}\n")

    counts = ", ".join(f"{count} {cohort}" for cohort, count in export.counts.items())
    print(f"Valid loss export: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
