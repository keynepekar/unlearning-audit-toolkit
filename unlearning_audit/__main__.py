"""Command-line entry point for local audits."""

import argparse
import csv
import json
from pathlib import Path

from .manifest import load_manifest
from .metrics import compare_losses
from .output import write_artifact
from .report import load_metrics, render_report, render_summary_csv
from .validation import load_losses


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m unlearning_audit")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="check a CSV of per-sample losses")
    validate.add_argument("csv_file", type=Path)
    validate.add_argument("--manifest", type=Path, help="also check declared models and cohorts")
    compare = commands.add_parser("compare", help="compare aligned losses and write aggregate metrics")
    compare.add_argument("csv_file", type=Path)
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--fpr", type=float, action="append", help="FPR target, repeatable")
    compare.add_argument("--bootstrap", type=int, default=400, help="bootstrap repeats, or 0 to disable")
    compare.add_argument("--seed", type=int, default=42)
    report = commands.add_parser("report", help="render aggregate metrics as HTML or summary CSV")
    report.add_argument("metrics_file", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--format", choices=("html", "csv"), default="html")
    args = parser.parse_args()

    try:
        if args.command == "report":
            render = render_report if args.format == "html" else render_summary_csv
            write_artifact(args.output, render(load_metrics(args.metrics_file)), inputs=(args.metrics_file,))
            print(f"Wrote report to {args.output}")
            return 0
        export = load_losses(args.csv_file)
        manifest = load_manifest(args.manifest) if args.manifest else None
        if args.command == "compare":
            result = compare_losses(
                export, manifest, targets=tuple(args.fpr or (0.01, 0.10)),
                repeats=args.bootstrap, seed=args.seed,
            )
            write_artifact(
                args.output, json.dumps(result, indent=2, allow_nan=False) + "\n",
                inputs=(args.csv_file, args.manifest),
            )
            print(f"Wrote metrics to {args.output}")
            return 0
        if manifest:
            manifest.check_alignment(set(export.losses), set(export.sample_ids))
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"{args.command.capitalize()} failed: {error}\n")

    counts = ", ".join(f"{count} {cohort}" for cohort, count in export.counts.items())
    print(f"Valid loss export: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
