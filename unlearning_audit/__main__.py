"""Command-line entry point for local audits."""

import argparse
import csv
from pathlib import Path

from .validation import validate_losses


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m unlearning_audit")
    commands = parser.add_subparsers(required=True)
    validate = commands.add_parser("validate", help="check a CSV of per-sample losses")
    validate.add_argument("csv_file", type=Path)
    args = parser.parse_args()

    try:
        counts = validate_losses(args.csv_file)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Validation failed: {error}\n")

    print(f"Valid loss export: {counts['forget']} forget, {counts['holdout']} holdout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
