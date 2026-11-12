import csv
import unittest
from io import StringIO
from pathlib import Path

from unlearning_audit.manifest import load_manifest
from unlearning_audit.metrics import compare_losses
from unlearning_audit.report import render_report, render_summary_csv
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

    def test_summary_csv_has_fixed_roles_and_numeric_values(self):
        self.metrics["manifest"]["title"] = "=1+2"
        rows = list(csv.DictReader(StringIO(render_summary_csv(self.metrics))))
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["role"], "before")
        self.assertEqual(rows[0]["auc"], "1.0")
        self.assertEqual(rows[0]["calibration_count"], "")
