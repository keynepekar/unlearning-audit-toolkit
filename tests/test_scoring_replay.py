import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from sklearn.linear_model import SGDClassifier
except ImportError:
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
if np is not None:
    from digits_scoring import MODELS, record_scoring, verify_scoring
    from sklearn_export import export_losses


@unittest.skipIf(np is None, "demo dependencies are not installed")
class ScoringReplayTests(unittest.TestCase):
    def test_numeric_state_replays_and_changes_are_detected(self):
        pixels = np.array([[0.0, 16.0], [16.0, 0.0], [1.0, 15.0], [15.0, 1.0]])
        labels = np.array([0, 1, 0, 1])
        features = pixels / 16
        groups = {"forget": np.array([0, 1]), "holdout": np.array([2, 3])}
        classifier = SGDClassifier(loss="log_loss", random_state=42).fit(features, labels)
        models = dict.fromkeys(MODELS, classifier)

        def export(path, estimators, x, y, partitions):
            export_losses(path, estimators, {
                cohort: ([str(i) for i in rows], x[rows], y[rows])
                for cohort, rows in partitions.items()
            })

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            export(path / "losses.csv", models, features, labels, groups)
            for name in ("manifest.json", "metrics.json"):
                (path / name).write_text("{}", encoding="utf-8")
            record_scoring(path, pixels, features, labels, groups, models)
            verify_scoring(path, pixels, features, labels, groups, export)
            with self.assertRaisesRegex(ValueError, "split_sha256"):
                verify_scoring(path, pixels, features, labels, {**groups, "forget": groups["forget"][::-1]}, export)
            with (path / "candidate.json").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "artifact_sha256"):
                verify_scoring(path, pixels, features, labels, groups, export)
