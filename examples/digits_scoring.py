"""Record and replay the Digits scorer using numeric classifier state."""

import json
import platform
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import SGDClassifier

from sklearn_export import PROBABILITY_FLOOR
from unlearning_audit.manifest import _unique_fields

MODELS = ("original", "candidate", "retrained")
ARTIFACTS = ("losses.csv", "manifest.json", "metrics.json", *(f"{name}.json" for name in MODELS))


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _array_digest(values, dtype):
    array = np.asarray(values, dtype=dtype, order="C")
    digest = sha256(json.dumps([array.dtype.str, array.shape], separators=(",", ":")).encode() + b"\n")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _record(directory, pixels, images, labels, groups):
    source = Path(__file__).resolve().parent
    return {
        "schema_version": 1,
        "data": {
            "pixels_sha256": _array_digest(pixels, "<f8"),
            "labels_sha256": _array_digest(labels, "<i8"),
        },
        "split_sha256": {
            cohort: _array_digest(indices, "<i8") for cohort, indices in sorted(groups.items())
        },
        "preprocessing": {"divisor": 16, "features_sha256": _array_digest(images, "<f8")},
        "scoring": {"class": "SGDClassifier", "loss": "log_loss", "probability_floor": PROBABILITY_FLOOR},
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "source_sha256": {
            filename: sha256((source / filename).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for filename in ("digits_demo.py", "digits_scoring.py", "sklearn_export.py")
        },
        "artifact_sha256": {filename: _digest(directory / filename) for filename in ARTIFACTS},
    }


def _write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")


def record_scoring(directory, pixels, images, labels, groups, models):
    for name in MODELS:
        model = models[name]
        _write_json(directory / f"{name}.json", {
            "classes": model.classes_.tolist(),
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
        })
    _write_json(directory / "scoring.json", _record(directory, pixels, images, labels, groups))


def _read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_fields)


def _classifier(path, expected_classes, features):
    state = _read_json(path)
    if not isinstance(state, dict) or state.keys() != {"classes", "coef", "intercept"}:
        raise ValueError(f"invalid checkpoint fields in {path.name}")
    coefficients = np.asarray(state["coef"], dtype=float)
    intercept = np.asarray(state["intercept"], dtype=float)
    outputs = 1 if len(expected_classes) == 2 else len(expected_classes)
    if (
        state["classes"] != expected_classes.tolist()
        or coefficients.shape != (outputs, features)
        or intercept.shape != (outputs,)
        or not np.isfinite(coefficients).all() or not np.isfinite(intercept).all()
    ):
        raise ValueError(f"invalid checkpoint arrays in {path.name}")
    model = SGDClassifier(loss="log_loss")
    model.classes_ = expected_classes
    model.coef_, model.intercept_ = coefficients, intercept
    model.n_features_in_ = features
    return model


def verify_scoring(directory, pixels, images, labels, groups, export):
    saved = _read_json(directory / "scoring.json")
    expected = _record(directory, pixels, images, labels, groups)
    if saved != expected:
        fields = [key for key, value in expected.items() if not isinstance(saved, dict) or saved.get(key) != value]
        raise ValueError("scoring provenance differs: " + (", ".join(fields) or "unexpected fields"))
    classes = np.unique(labels)
    models = {name: _classifier(directory / f"{name}.json", classes, images.shape[1]) for name in MODELS}
    with TemporaryDirectory() as temporary:
        replay = Path(temporary) / "losses.csv"
        export(replay, models, images, labels, groups)
        if _digest(replay) != expected["artifact_sha256"]["losses.csv"]:
            raise ValueError("replayed scoring did not reproduce the loss CSV")
