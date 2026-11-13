"""Draw a descriptive count chart from the policy benchmark summary."""

import argparse
import csv
from collections import Counter
from html import escape
from pathlib import Path

from unlearning_audit.output import write_artifact

LABELS = {
    "continuous": "Continuous Gram subtraction",
    "periodic": "Gram with periodic reconstruction",
    "condition": "Gram with condition trigger",
    "block_qr": "Cached block QR",
}


def render(rows):
    totals, failures = Counter(), Counter()
    for row in rows:
        method = row["method"]
        if method not in LABELS or row["any_failure"] not in ("True", "False"):
            raise ValueError("unexpected method or failure value")
        totals[method] += 1
        failures[method] += row["any_failure"] == "True"
    if set(totals) != set(LABELS) or len(set(totals.values())) != 1:
        raise ValueError("expected the same number of sequences for each method")
    description = ". ".join(
        f"{label}: {failures[method]} of {totals[method]} sequences exceeded the tolerance"
        for method, label in LABELS.items()
    )
    width = 420
    marks = []
    for tick in (0, .25, .5, .75, 1):
        x = 295 + width * tick
        marks.append(f'<path d="M{x:g} 85V300" stroke="#d6dfe7"/>')
        marks.append(f'<text x="{x:g}" y="324" text-anchor="middle">{tick:.0%}</text>')
    bars = []
    for index, (method, label) in enumerate(LABELS.items()):
        y = 100 + index * 50
        fraction = failures[method] / totals[method]
        bars.append(f'<text x="280" y="{y + 15}" text-anchor="end">{escape(label)}</text>')
        bars.append(f'<rect x="295" y="{y}" width="{width * fraction:.2f}" height="19" rx="3" fill="#006b60"/>')
        bars.append(f'<text x="730" y="{y + 15}">{failures[method]}/{totals[method]}</text>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 370" role="img" aria-labelledby="title desc">
<title id="title">Sequences with at least one tolerance failure</title>
<desc id="desc">{escape(description)}. The CSV contains per-sequence results.</desc>
<rect width="820" height="370" rx="12" fill="#f7f9fb"/>
<g font-family="system-ui, sans-serif" fill="#192339" font-size="14">
<text x="30" y="32" font-size="21" font-weight="700">Sequences with at least one tolerance failure</text>
<text x="30" y="58">High-leverage synthetic data · coefficient or prediction relative error above 10⁻⁶</text>
{"".join(marks)}{"".join(bars)}
<text x="505" y="352" text-anchor="middle">Share of sequences</text>
</g></svg>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as stream:
        image = render(csv.DictReader(stream))
    write_artifact(args.output, image, inputs=(args.input,))


if __name__ == "__main__":
    main()
