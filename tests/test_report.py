import csv
import unittest
from copy import deepcopy
from dataclasses import replace
from io import StringIO
from pathlib import Path

from unlearning_audit.manifest import load_manifest
from unlearning_audit.metrics import compare_losses
from unlearning_audit.report import render_report, render_roc_csv, render_summary_csv
from unlearning_audit.validation import load_losses

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.metrics = compare_losses(
            load_losses(EXAMPLES / "comparison.csv"),
            load_manifest(EXAMPLES / "manifest.json"),
            repeats=0,
        )

    def test_escapes_title_and_provenance(self):
        self.metrics["manifest"]["title"] = '<script>alert("title")</script>'
        self.metrics["manifest"]["provenance"] = {"<b>data</b>": "<img src=x>"}
        html = render_report(self.metrics)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img src=x>", html)
        self.assertIn("&lt;b&gt;data&lt;/b&gt;", html)
        self.assertIn("ROC curves for the three models", html)
        self.assertIn("1.000", html)

    def test_rejects_inconsistent_result(self):
        self.metrics["models"]["before"]["operating_points"][0]["fpr"] = 0.01
        with self.assertRaisesRegex(ValueError, "rates do not match counts"):
            render_report(self.metrics)

    def test_rejects_invalid_curve(self):
        self.metrics["models"]["before"]["roc"][-1] = [2, 3]
        with self.assertRaisesRegex(ValueError, "curve endpoints"):
            render_report(self.metrics)

    def test_calibrated_points_follow_the_saved_roc_curve(self):
        export = load_losses(EXAMPLES / "comparison.csv")
        manifest = load_manifest(EXAMPLES / "manifest.json")
        metrics = compare_losses(
            replace(
                export, sample_ids={**export.sample_ids, "calibration": ("c1", "c2", "c3", "c4")},
                losses={
                    model: {**values, "calibration": (0.15, 0.25, 1.0, 2.0)}
                    for model, values in export.losses.items()
                },
            ),
            replace(manifest, cohorts={**manifest.cohorts, "calibration": "Independent non-members"}),
            targets=(0.1, 0.25, 0.4, 0.5), repeats=0,
        )
        for index, changes, message in (
            (0, {"true_positives": 1, "tpr": 0.5}, "equal calibration counts"),
            (1, {"false_positives": 1, "fpr": 0.5}, "does not lie on the ROC"),
            (2, {"true_positives": 2, "tpr": 1.0}, "equal calibration counts"),
            (3, {"true_positives": 0, "tpr": 0.0}, "nondecreasing"),
        ):
            with self.subTest(index=index):
                altered = deepcopy(metrics)
                altered["models"]["before"]["calibrated_points"][index].update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    render_report(altered)
        self.assertTrue(render_report(metrics))

    def test_summary_csv_has_fixed_roles_and_numeric_values(self):
        self.metrics["manifest"]["title"] = "=1+2"
        rows = list(csv.DictReader(StringIO(render_summary_csv(self.metrics))))
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["role"], "before")
        self.assertEqual(rows[0]["auc"], "1.0")
        self.assertEqual(rows[0]["calibration_count"], "")

    def test_roc_csv_contains_all_curve_points(self):
        rows = list(csv.DictReader(StringIO(render_roc_csv(self.metrics))))
        self.assertEqual(len(rows), sum(len(model["roc"]) for model in self.metrics["models"].values()))
        self.assertEqual((rows[0]["fpr"], rows[0]["tpr"]), ("0.0", "0.0"))
        self.assertEqual((rows[-1]["fpr"], rows[-1]["tpr"]), ("1.0", "1.0"))
