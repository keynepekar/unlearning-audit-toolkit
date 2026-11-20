"""Write deterministic synthetic loss exports for a grouped audit load test."""

import argparse
import csv
import json
from pathlib import Path
from random import Random


def generate(directory, specimens, distinct):
    directory.mkdir()
    rng = Random(620 + specimens)
    with (directory / "losses.csv").open("x", newline="", encoding="utf-8") as losses, (
        directory / "groups.csv"
    ).open("x", newline="", encoding="utf-8") as groups:
        loss_writer, group_writer = csv.writer(losses, lineterminator="\n"), csv.writer(groups, lineterminator="\n")
        loss_writer.writerow(("sample_id", "cohort", "model", "loss"))
        group_writer.writerow(("sample_id", "group_id"))
        for cohort in ("forget", "holdout"):
            done = group_number = 0
            while done < specimens // 2:
                size = min(rng.randrange(2, 10), specimens // 2 - done)
                latent = rng.randrange(72)
                for index in range(done, done + size):
                    sample = f"{cohort}-{index:06}"
                    group_writer.writerow((sample, f"{cohort}-g{group_number}"))
                    reference = 30 + latent + (20 * rng.random() if distinct else rng.randrange(20))
                    candidate = reference - (5 if cohort == "forget" else 0) + rng.randrange(-2, 3)
                    original = reference - (8 if cohort == "forget" else 0)
                    for name, value in (("initial", original), ("updated", candidate), ("retrained", reference)):
                        loss_writer.writerow((sample, cohort, name, value))
                done += size
                group_number += 1
    template = Path(__file__).with_name("manifest.json")
    manifest = json.loads(template.read_text(encoding="utf-8"))
    manifest["cohorts"].pop("utility")
    manifest["title"] = "Synthetic grouped load benchmark"
    manifest["provenance"] = {"Data": f"Deterministic grouped synthetic losses, seed {620 + specimens}."}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--distinct", action="store_true", help="generate mostly distinct scores")
    args = parser.parse_args()
    if args.samples < 20 or args.samples % 2:
        parser.error("--samples must be an even integer of at least 20")
    generate(args.output, args.samples, args.distinct)


if __name__ == "__main__":
    main()
