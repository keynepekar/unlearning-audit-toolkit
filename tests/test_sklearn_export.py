import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
except ImportError:
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
if np is not None:
    from sklearn_export import export_losses

from unlearning_audit.validation import load_losses


@unittest.skipIf(np is None, "demo dependencies are not installed")
class SklearnExportTests(unittest.TestCase):
    def test_export_and_duplicate_rejection(self):
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        classifier = LogisticRegression().fit(x, y)
        models = {"first": classifier, "second": classifier}
        cohorts = {"forget": (["a", "b"], x[:2], y[:2]), "holdout": (["c", "d"], x[2:], y[2:])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "losses.csv"
            export_losses(path, models, cohorts)
            self.assertEqual(load_losses(path).counts, {"forget": 2, "holdout": 2})
            cohorts["holdout"] = (["a", "d"], x[2:], y[2:])
            with self.assertRaisesRegex(ValueError, "unique"):
                export_losses(Path(directory) / "invalid.csv", models, cohorts)

    def test_metadata_validation_precedes_prediction_and_writing(self):
        models = {"first": None, "second": None}
        cohorts = {"forget": None, "holdout": None}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            with self.assertRaisesRegex(ValueError, "unknown cohort"):
                export_losses(path, models, cohorts | {"test": None})
            with self.assertRaisesRegex(ValueError, "unique after trimming"):
                export_losses(path, {"first": None, " first ": None}, cohorts)
            self.assertFalse(path.exists())
