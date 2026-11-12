"""Render a saved audit as a self-contained, offline HTML report."""

import json
from csv import writer
from html import escape
from importlib.resources import files
from io import StringIO
from pathlib import Path
from sys import float_info

from .manifest import ROLES, Manifest, _unique_fields
from .metrics import RocPoint, area_under_roc, operating_points

LABELS = {"before": "Original", "candidate": "Candidate", "reference": "Retrained"}
COLORS = {"before": "#263d73", "candidate": "#006b60", "reference": "#a05a24"}
DELTA_LABELS = {
    "candidate_minus_before": "Candidate − original",
    "candidate_minus_reference": "Candidate − retrained",
}


def _mapping(value: object, fields: set[str], label: str) -> dict:
    if not isinstance(value, dict) or value.keys() != fields:
        raise ValueError(f"invalid {label} fields")
    return value


def _number(value: object, low: float, high: float, label: str) -> float:
    if type(value) not in (int, float) or not low <= value <= high:
        raise ValueError(f"invalid {label}")
    return float(value)


def _count(value: object, low: int, high: int, label: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid {label}")
    return value


def _check_curve(points: object, holdout: int, forget: int) -> None:
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("expected a ROC curve with at least two points")
    if points[0] != [0, 0] or points[-1] != [holdout, forget]:
        raise ValueError("ROC curve endpoints do not match cohort counts")
    last_fp, last_tp = -1, -1
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("invalid ROC point")
        fp = _count(point[0], 0, holdout, "ROC false positives")
        tp = _count(point[1], 0, forget, "ROC true positives")
        if fp < last_fp or tp < last_tp or (fp, tp) == (last_fp, last_tp):
            raise ValueError("ROC points must increase monotonically")
        last_fp, last_tp = fp, tp


def _check_operating_points(
    rows: object, holdout: int, forget: int, *, calibration: int | None = None
) -> tuple[float, ...]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing operating points")
    targets = []
    fields = {"target_fpr", "fpr", "tpr", "false_positives", "true_positives"}
    if calibration is not None:
        fields.add("calibration_false_positives")
    for row in rows:
        _mapping(row, fields, "operating point")
        target = _number(row["target_fpr"], 0, 1, "target FPR")
        if not 0 < target < 1 or (targets and target <= targets[-1]):
            raise ValueError("FPR targets must increase strictly between 0 and 1")
        fp = _count(row["false_positives"], 0, holdout, "false positives")
        tp = _count(row["true_positives"], 0, forget, "true positives")
        maximum = target
        if calibration is not None:
            calibrated_fp = _count(row["calibration_false_positives"], 0, calibration, "calibration count")
            if calibrated_fp / calibration > target:
                raise ValueError("calibration count exceeds its FPR target")
            maximum = 1
        fpr = _number(row["fpr"], 0, maximum, "observed FPR")
        tpr = _number(row["tpr"], 0, 1, "observed TPR")
        if abs(fpr - fp / holdout) > 1e-12 or abs(tpr - tp / forget) > 1e-12:
            raise ValueError("operating point rates do not match counts")
        targets.append(target)
    return tuple(targets)


def _validate_result(value: object) -> tuple[dict, Manifest]:
    result = _mapping(
        value, {"schema_version", "manifest", "counts", "models", "auc_deltas", "bootstrap", "warnings"},
        "metrics",
    )
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("unsupported metrics schema_version (expected 1)")
    manifest = Manifest.from_dict(result["manifest"])
    counts = _mapping(result["counts"], set(manifest.cohorts), "cohort counts")
    for cohort, size in counts.items():
        _count(size, 1, 2**53 - 1, f"{cohort} count")
    models = _mapping(result["models"], set(ROLES), "models")
    targets = None
    model_fields = {"auc", "roc", "operating_points", "mean_utility_loss"}
    if "calibration" in counts:
        model_fields.add("calibrated_points")
    for metrics in models.values():
        _mapping(metrics, model_fields, "model metrics")
        auc = _number(metrics["auc"], 0, 1, "AUC")
        _check_curve(metrics["roc"], counts["holdout"], counts["forget"])
        roc = [RocPoint(*point) for point in metrics["roc"]]
        if abs(auc - area_under_roc(roc)) > 1e-12:
            raise ValueError("AUC does not match the ROC curve")
        observed = _check_operating_points(metrics["operating_points"], counts["holdout"], counts["forget"])
        if metrics["operating_points"] != operating_points(roc, observed):
            raise ValueError("descriptive operating points do not match the ROC curve")
        if targets is not None and observed != targets:
            raise ValueError("models must use the same FPR targets")
        targets = observed
        if "calibration" in counts:
            calibrated = _check_operating_points(
                metrics["calibrated_points"], counts["holdout"], counts["forget"],
                calibration=counts["calibration"],
            )
            if calibrated != targets:
                raise ValueError("calibrated and descriptive FPR targets must match")
        utility = metrics["mean_utility_loss"]
        if "utility" in counts:
            _number(utility, 0, float_info.max, "mean utility loss")
        elif utility is not None:
            raise ValueError("utility loss requires a utility cohort")

    deltas = _mapping(result["auc_deltas"], set(DELTA_LABELS), "AUC differences")
    for role in ("before", "reference"):
        delta = deltas[f"candidate_minus_{role}"]
        _mapping(delta, {"estimate", "interval"}, "AUC difference")
        estimate = _number(delta["estimate"], -1, 1, "AUC difference")
        if abs(estimate - (models["candidate"]["auc"] - models[role]["auc"])) > 1e-12:
            raise ValueError("AUC difference does not match model AUCs")
        interval = delta["interval"]
        if interval is not None:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("invalid AUC interval")
            lower = _number(interval[0], -1, 1, "lower AUC interval")
            _number(interval[1], lower, 1, "upper AUC interval")
    bootstrap = _mapping(result["bootstrap"], {"repeats", "seed"}, "bootstrap")
    _count(bootstrap["repeats"], 0, 10_000, "bootstrap repeats")
    if type(bootstrap["seed"]) is not int:
        raise ValueError("invalid bootstrap seed")
    if not isinstance(result["warnings"], list) or any(not isinstance(item, str) for item in result["warnings"]):
        raise ValueError("warnings must be a list of strings")
    return result, manifest


def _chart(result: dict) -> str:
    left, top, width, height = 65, 20, 462, 270
    bottom = top + height
    counts = result["counts"]
    ticks = []
    for step in range(6):
        fraction = step / 5
        x = left + width * fraction
        y = bottom - height * fraction
        ticks.append(
            f'<path class="grid" d="M{x:g} {top}V{bottom} M{left} {y:g}H{left + width}"/>'
            f'<text x="{x:g}" y="{bottom + 23}" text-anchor="middle">{fraction:.0%}</text>'
            f'<text x="{left - 9}" y="{y + 4:g}" text-anchor="end">{fraction:.0%}</text>'
        )
    paths = []
    for role in ROLES:
        points = " ".join(
            f"{left + width * fp / counts['holdout']:.2f},{bottom - height * tp / counts['forget']:.2f}"
            for fp, tp in result["models"][role]["roc"]
        )
        dash = {"before": "", "candidate": ' stroke-dasharray="8 3"', "reference": ' stroke-dasharray="2 3"'}[role]
        paths.append(f'<polyline points="{points}" stroke="{COLORS[role]}"{dash}/>')
    return (
        '<svg viewBox="0 0 550 356" role="img" aria-label="ROC curves for the three models">'
        + "".join(ticks)
        + f'<path class="chance" d="M{left} {bottom}L{left + width} {top}"/>'
        + "".join(paths)
        + f'<text x="{left + width / 2:g}" y="342" text-anchor="middle">False-positive rate</text>'
        + '<text x="15" y="155" text-anchor="middle" transform="rotate(-90 15 155)">True-positive rate</text>'
        + '</svg>'
    )


def _cards(result: dict, manifest: Manifest) -> str:
    return "".join(
        f'<article class="card {role}"><span class="model-role">{LABELS[role]}</span>'
        f'<strong>{result["models"][role]["auc"]:.3f}</strong>'
        f'<span class="model-name">{escape(manifest.models[role])}</span></article>'
        for role in ROLES
    )


def _operating_table(result: dict, *, calibrated: bool = False) -> str:
    counts = result["counts"]
    rows = []
    field = "calibrated_points" if calibrated else "operating_points"
    calibration_header = '<th scope="col">Calibration FPR</th>' if calibrated else ""
    for role in ROLES:
        for point in result["models"][role][field]:
            calibration_cell = ""
            if calibrated:
                events = point["calibration_false_positives"]
                total = counts["calibration"]
                calibration_cell = f"<td>{events / total:.2%} <small>({events}/{total})</small></td>"
            rows.append(
                f'<tr><th scope="row">{LABELS[role]}</th><td>{point["target_fpr"]:.3g}</td>{calibration_cell}'
                f'<td>{point["fpr"]:.2%} <small>({point["false_positives"]}/{counts["holdout"]})</small></td>'
                f'<td>{point["tpr"]:.2%} <small>({point["true_positives"]}/{counts["forget"]})</small></td></tr>'
            )
    return (
        '<div class="table-wrap"><table><thead><tr><th scope="col">Model</th><th scope="col">FPR cap (rate)</th>'
        + calibration_header
        + '<th scope="col">Observed FPR</th><th scope="col">Observed TPR</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>"
    )


def _calibration(result: dict) -> str:
    if "calibration" not in result["counts"]:
        return ""
    return (
        '<section class="panel"><h2>Calibrated operating points</h2>'
        '<p>Each threshold is selected using the separate calibration cohort. '
        'The observed holdout FPR can exceed the calibration cap.</p>'
        + _operating_table(result, calibrated=True)
        + '</section>'
    )


def _differences(result: dict) -> str:
    entries = []
    for key, label in DELTA_LABELS.items():
        difference = result["auc_deltas"][key]
        interval = difference["interval"]
        bounds = f"{interval[0]:+.3f} to {interval[1]:+.3f}" if interval is not None else "Unavailable"
        entries.append(
            f'<tr><th scope="row">{label}</th><td>{difference["estimate"]:+.3f}</td><td>{bounds}</td></tr>'
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th scope="col">Comparison</th><th scope="col">Δ AUC</th>'
        '<th scope="col">95% interval</th></tr></thead><tbody>' + "".join(entries) + "</tbody></table></div>"
    )


def _utility(result: dict) -> str:
    if "utility" not in result["counts"]:
        return ""
    losses = "".join(
        f'<li>{LABELS[role]} <strong>{result["models"][role]["mean_utility_loss"]:.4g}</strong></li>'
        for role in ROLES
    )
    return (
        '<section class="panel"><h2>Utility cohort</h2><p>Mean loss on '
        f'{result["counts"]["utility"]:,} separate evaluation examples. Lower is better for this loss.</p>'
        f'<ul class="utility">{losses}</ul></section>'
    )


def _definitions(values: dict[str, str]) -> str:
    return "".join(f"<dt>{escape(key)}</dt><dd>{escape(text)}</dd>" for key, text in values.items())


def render_report(document: object) -> str:
    """Validate an aggregate audit document and render it without external assets."""
    result, manifest = _validate_result(document)
    css = files(__package__).joinpath("report.css").read_text(encoding="utf-8")
    title = escape(manifest.title)
    counts = result["counts"]
    warning_items = "".join(f"<li>{escape(message)}</li>" for message in result["warnings"])
    warnings = (
        f'<aside class="warnings" aria-label="Audit warnings"><h2>Audit warnings</h2><ul>{warning_items}</ul></aside>'
        if warning_items else ""
    )
    bootstrap = result["bootstrap"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>{title} | Unlearning Audit Toolkit</title>
<style>{css}</style>
</head>
<body>
<main>
  <header class="hero">
    <p class="project">Unlearning Audit Toolkit</p>
    <h1>{title}</h1>
    <p class="cohorts">{counts['forget']:,} forget examples <span aria-hidden="true">/</span> {counts['holdout']:,} held-out non-members</p>
  </header>
  <section aria-labelledby="auc-heading">
    <h2 id="auc-heading">ROC AUC</h2>
    <div class="cards">{_cards(result, manifest)}</div>
  </section>
  {warnings}
  <div class="grid-panels">
    <section class="panel">
      <h2>Receiver operating characteristic</h2>
      <p>Lower loss ranks an example higher. The diagonal marks chance-level ranking.</p>
      {_chart(result)}
      <ul class="legend"><li class="before">Original</li><li class="candidate">Candidate</li><li class="reference">Retrained</li></ul>
    </section>
    <section class="panel">
      <h2>Change in AUC</h2>
      <p>Differences use the candidate minus each comparator.</p>
      {_differences(result)}
      <p>Paired bootstrap over the same examples within each cohort. Repeats: {bootstrap['repeats']:,}. Seed: {bootstrap['seed']}.</p>
      <p>Intervals describe sampling variation for these fitted models. AUC near 0.5 shows weak ranking for this score alone.</p>
    </section>
  </div>
  <section class="panel">
    <h2>Operating points</h2>
    <p>Maximum observed true-positive rate within each false-positive cap on these same audit cohorts.</p>
    {_operating_table(result)}
  </section>
  {_calibration(result)}
  {_utility(result)}
  <section class="panel">
    <h2>Experiment provenance</h2>
    <p>{escape(manifest.loss_definition)}</p>
    <dl>{_definitions(manifest.cohorts)}{_definitions(manifest.provenance)}</dl>
  </section>
</main>
</body>
</html>
"""


def load_metrics(path: Path) -> object:
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream, object_pairs_hook=_unique_fields)


def render_summary_csv(document: object) -> str:
    """Export operating points and model-level metrics for downstream analysis."""
    result, _ = _validate_result(document)
    stream = StringIO(newline="")
    output = writer(stream, lineterminator="\n")
    output.writerow((
        "role", "method", "auc", "mean_utility_loss", "target_fpr", "calibration_false_positives",
        "calibration_count", "false_positives", "holdout_count", "fpr", "true_positives", "forget_count", "tpr",
    ))
    counts = result["counts"]
    for role in ROLES:
        model = result["models"][role]
        for method, field in (("descriptive", "operating_points"), ("calibrated", "calibrated_points")):
            for point in model.get(field, []):
                output.writerow((
                    role, method, model["auc"], model["mean_utility_loss"], point["target_fpr"],
                    point.get("calibration_false_positives"), counts["calibration"] if method == "calibrated" else None,
                    point["false_positives"], counts["holdout"], point["fpr"],
                    point["true_positives"], counts["forget"], point["tpr"],
                ))
    return stream.getvalue()
