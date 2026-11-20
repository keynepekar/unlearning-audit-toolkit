import csv
import unittest
from dataclasses import replace
from io import StringIO
from math import log
from pathlib import Path

from unlearning_audit.fpr_bounds import FPR_BOUND, upper_fpr
from unlearning_audit.manifest import load_manifest
from unlearning_audit.metrics import compare_losses
from unlearning_audit.report import render_report, render_summary_csv
from unlearning_audit.validation import load_losses
from unlearning_audit.verification import verify_comparison

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class FprBoundTests(unittest.TestCase):
    def test_zero_events_and_bound_monotonicity(self):
        self.assertGreater(upper_fpr(0, 2994), 0.001)
        self.assertLessEqual(upper_fpr(0, 2995), 0.001)
        for total in (1, 5, 100, 1000):
            values = [upper_fpr(count, total) for count in range(total + 1)]
            self.assertEqual(values, sorted(values))
            self.assertTrue(all(low / total <= high <= 1 for low, high in enumerate(values)))
        candidate = upper_fpr(1, 10)
        self.assertAlmostEqual(0.1 * log(0.1 / candidate) + 0.9 * log(0.9 / (1 - candidate)), -log(0.05) / 10)

    def test_bounds_render_only_for_numeric_calibrated_thresholds(self):
        export = load_losses(EXAMPLES / "comparison.csv")
        manifest = load_manifest(EXAMPLES / "manifest.json")
        with self.assertRaisesRegex(ValueError, "require a calibration cohort"):
            compare_losses(export, manifest, iid_holdout=True)
        export = replace(
            export,
            sample_ids={**export.sample_ids, "calibration": ("c1", "c2", "c3", "c4")},
            losses={role: {**losses, "calibration": (0.15, 0.25, 1.0, 2.0)} for role, losses in export.losses.items()},
        )
        manifest = replace(manifest, cohorts={**manifest.cohorts, "calibration": "Independent non-members"})
        with self.assertRaisesRegex(ValueError, "ungrouped holdout"):
            compare_losses(export, manifest, iid_holdout=True, group_assignments={})
        metrics = compare_losses(
            export, manifest, targets=(0.1, 0.25), iid_holdout=True, include_fingerprints=True, repeats=0,
        )
        self.assertEqual(metrics["holdout_fpr_bound"], FPR_BOUND)
        verify_comparison(metrics, export, manifest)
        rows = list(csv.DictReader(StringIO(render_summary_csv(metrics))))
        self.assertEqual(rows[0]["fpr_upper_95"], "")
        self.assertEqual(rows[2]["fpr_upper_95"], "")
        self.assertEqual(float(rows[3]["fpr_upper_95"]), upper_fpr(0, 2))
        html = render_report(metrics)
        self.assertIn("95% upper FPR bound", html)
        self.assertIn("All-negative rule", html)
