"""Render benchmark timing as a standalone SVG with no plotting dependency."""

import argparse
import json
from html import escape
from math import ceil, floor, log10
from pathlib import Path

from unlearning_audit.output import write_artifact

COLORS = ("#263d73", "#006b60", "#a05a24")


def render_chart(results: dict) -> str:
    timings = results["timing"]
    dimensions = sorted({row["features"] for row in timings})
    fractions = sorted({row["deletion_fraction"] for row in timings})
    left, right, top, bottom = 87, 765, 82, 345
    values = [row["refit_over_update"] for row in timings]
    low = min(-1, floor(log10(min(values))))
    high = max(1, ceil(log10(max(values))))

    def x(fraction):
        return left + fraction * (right - left)

    def y(value):
        return bottom - (log10(value) - low) * (bottom - top) / (high - low)

    grid = []
    for power in range(low, high + 1):
        value = 10**power
        line = "#8594a4" if power == 0 else "#dce4eb"
        grid.append(f'<path stroke="{line}" d="M{left} {y(value):.2f}H{right}"/>')
        grid.append(f'<text x="{left - 12}" y="{y(value) + 4:.2f}" text-anchor="end">{value:g}×</text>')
    for fraction in fractions:
        position = x(fraction)
        grid.append(f'<path stroke="#8594a4" d="M{position:.2f} {bottom}v6"/>')
        grid.append(f'<text x="{position:.2f}" y="373" text-anchor="middle">{fraction:.0%}</text>')

    series = []
    for index, (dimension, color, pattern) in enumerate(zip(dimensions, COLORS, ("none", "10 5", "3 4"), strict=True)):
        rows = sorted((row for row in timings if row["features"] == dimension), key=lambda row: row["deletion_fraction"])
        coords = [(x(row["deletion_fraction"]), y(row["refit_over_update"])) for row in rows]
        points = " ".join(f"{a:.2f},{b:.2f}" for a, b in coords)
        series.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{pattern}"/>')
        for a, b in coords:
            series.append(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="3.2" fill="{color}"/>')
        legend_x = 105 + 165 * index
        series.append(f'<path d="M{legend_x} 428h32" stroke="{color}" stroke-width="3" stroke-dasharray="{pattern}"/>')
        series.append(f'<text x="{legend_x + 41}" y="433">{dimension} features</text>')

    n = escape(f'{timings[0]["training_rows"]:,}')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 460" role="img" aria-labelledby="title desc">
<title id="title">Ridge update versus refit at four deletion fractions</title>
<desc id="desc">Above one means updating was faster. Below one means refitting was faster. See timing.csv for exact values.</desc>
<rect width="820" height="460" rx="12" fill="#f7f9fb"/>
<g font-family="system-ui, sans-serif" fill="#192339" font-size="14">
<text x="32" y="33" font-size="22" font-weight="700">When is the ridge update faster?</text>
<text x="32" y="56">Refit time / update time · log scale · {n} training rows · one BLAS thread</text>
{"".join(grid)}
{"".join(series)}
<text x="425" y="404" text-anchor="middle">Fraction of training rows removed</text>
</g></svg>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.benchmark.open(encoding="utf-8") as stream:
        results = json.load(stream)
    write_artifact(args.output, render_chart(results), inputs=(args.benchmark,))


if __name__ == "__main__":
    main()
