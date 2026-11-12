"""Summarize training repeats on one fixed, fingerprinted evaluation split."""

import re
from csv import writer
from html import escape
from io import StringIO
from statistics import mean, stdev
from typing import Iterable

from .manifest import ROLES
from .report import DELTA_LABELS, render_document, validate_metrics

SEED_FIELD = "Training seed"
PROTOCOL_FIELD = "Protocol"
COMPARISONS = tuple(DELTA_LABELS)


def summarize_study(documents: Iterable[object]) -> dict:
    """Summarize run estimates and their spread, without pooling example-level CIs."""
    runs = {}
    baseline = None
    for index, document in enumerate(documents, start=1):
        result, manifest = validate_metrics(document)
        if "fingerprints" not in result:
            raise ValueError(f"run {index} has no input fingerprints")
        metadata = dict(manifest.provenance)
        seed_text = metadata.pop(SEED_FIELD, "")
        if not re.fullmatch(r"0|[1-9][0-9]{0,9}", seed_text):
            raise ValueError(f"run {index} must declare a non-negative integer {SEED_FIELD!r}")
        seed = int(seed_text)
        if seed >= 2**32:
            raise ValueError(f"run {index} has a training seed outside the uint32 range")
        if seed in runs:
            raise ValueError(f"duplicate training seed: {seed}")
        if PROTOCOL_FIELD not in metadata:
            raise ValueError(f"run {index} must declare {PROTOCOL_FIELD!r}")

        contract = {
            "split": result["fingerprints"]["split_sha256"],
            "cohort counts": result["counts"],
            "cohort descriptions": manifest.cohorts,
            "loss definition": manifest.loss_definition,
            "protocol metadata": metadata,
            "FPR targets": [point["target_fpr"] for point in result["models"]["before"]["operating_points"]],
        }
        if baseline is None:
            baseline = contract
        else:
            different = [field for field in contract if baseline[field] != contract[field]]
            if different:
                raise ValueError(f"run {index} differs in " + ", ".join(different))
        runs[seed] = {
            "training_seed": seed,
            "auc": {role: result["models"][role]["auc"] for role in ROLES},
            "auc_deltas": {key: result["auc_deltas"][key]["estimate"] for key in COMPARISONS},
        }
    if len(runs) < 2:
        raise ValueError("a study needs at least two distinct training seeds")

    ordered = [runs[seed] for seed in sorted(runs)]
    return {
        "schema_version": 1,
        "protocol": baseline["protocol metadata"][PROTOCOL_FIELD],
        "counts": baseline["cohort counts"],
        "runs": ordered,
        "auc_delta_summary": {
            key: _spread([run["auc_deltas"][key] for run in ordered]) for key in COMPARISONS
        },
    }


def _spread(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "sample_sd": stdev(values), "min": min(values), "max": max(values)}


def study_csv(study: dict) -> str:
    stream = StringIO(newline="")
    output = writer(stream, lineterminator="\n")
    output.writerow(("training_seed", *(f"{role}_auc" for role in ROLES), *COMPARISONS))
    for run in study["runs"]:
        output.writerow((
            run["training_seed"], *(run["auc"][role] for role in ROLES),
            *(run["auc_deltas"][key] for key in COMPARISONS),
        ))
    return stream.getvalue()


def _range_chart(study: dict) -> str:
    summary = study["auc_delta_summary"]
    extent = max(0.01, max(abs(summary[key][bound]) for key in COMPARISONS for bound in ("min", "max")))
    left, right = 48, 552
    center = (left + right) / 2

    def x(value: float) -> float:
        return center + value / extent * (right - center)

    rows = []
    for y, key in zip((45, 112), COMPARISONS):
        group = summary[key]
        rows.append(
            f'<text x="{left}" y="{y - 14}">{DELTA_LABELS[key]}</text>'
            f'<line class="study-range" x1="{x(group["min"]):.2f}" x2="{x(group["max"]):.2f}" y1="{y}" y2="{y}"/>'
            f'<circle class="study-mean" cx="{x(group["mean"]):.2f}" cy="{y}" r="5"/>'
        )
    return (
        '<svg viewBox="0 0 600 166" role="img" aria-labelledby="spread-title spread-desc">'
        '<title id="spread-title">Observed ranges of paired AUC differences</title>'
        '<desc id="spread-desc">Horizontal lines span the minimum and maximum across runs. '
        'Dots mark means. The summary table lists exact values.</desc>'
        f'<path class="chance" d="M{center:g} 35V139"/>'
        + "".join(rows)
        + f'<text x="{left}" y="155" text-anchor="middle">−{extent:.3f}</text>'
        + f'<text x="{center:g}" y="155" text-anchor="middle">0</text>'
        + f'<text x="{right}" y="155" text-anchor="middle">+{extent:.3f}</text></svg>'
    )


def study_html(study: dict) -> str:
    summary_rows = []
    for key, values in study["auc_delta_summary"].items():
        cells = "".join(f'<td>{values[field]:.4f}</td>' for field in ("mean", "sample_sd", "min", "max"))
        summary_rows.append(f'<tr><th scope="row">{DELTA_LABELS[key]}</th>{cells}</tr>')
    run_rows = []
    for run in study["runs"]:
        values = [*run["auc"].values(), *run["auc_deltas"].values()]
        cells = "".join(f'<td>{value:.4f}</td>' for value in values)
        run_rows.append(f'<tr><th scope="row">{run["training_seed"]}</th>{cells}</tr>')
    return render_document("Training repeats", f"""  <header class="hero">
    <p class="project">Unlearning Audit Toolkit</p>
    <h1>Training repeats</h1>
    <p class="cohorts">{len(study["runs"])} training seeds on one fixed evaluation split</p>
  </header>
  <section class="panel">
    <h2>Observed spread in AUC differences</h2>
    <p>The line shows the observed range and the dot marks the mean. These are descriptive statistics across runs.</p>
    <div class="study-chart">{_range_chart(study)}</div>
    <p class="table-hint">Scroll the tables sideways for more columns.</p>
    <div class="table-wrap" tabindex="0" role="region" aria-label="AUC difference summary">
      <table><thead><tr><th scope="col">Comparison</th><th scope="col">Mean</th><th scope="col">Sample SD</th><th scope="col">Min</th><th scope="col">Max</th></tr></thead>
      <tbody>{''.join(summary_rows)}</tbody></table>
    </div>
  </section>
  <section class="panel">
    <h2>Results by training seed</h2>
    <p>Protocol {escape(study["protocol"])}. The observed spread reflects these runs on the shared cohort.</p>
    <div class="table-wrap" tabindex="0" role="region" aria-label="Per-seed AUC results">
      <table><thead><tr><th scope="col">Seed</th><th scope="col">Original AUC</th><th scope="col">Candidate AUC</th>
      <th scope="col">Retrained AUC</th><th scope="col">Δ vs original</th><th scope="col">Δ vs retrained</th></tr></thead>
      <tbody>{''.join(run_rows)}</tbody></table>
    </div>
  </section>
""")
