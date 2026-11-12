import unittest
from pathlib import Path

from unlearning_audit.manifest import load_manifest
from unlearning_audit.metrics import area_under_roc, compare_losses, operating_points, roc_counts
from unlearning_audit.validation import load_losses

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class MetricsTests(unittest.TestCase):
    def test_roc_polarity_and_ties(self):
        self.assertEqual(area_under_roc(roc_counts((0.1,), (0.8,))), 1)
        self.assertEqual(area_under_roc(roc_counts((0.8,), (0.1,))), 0)
        self.assertEqual(area_under_roc(roc_counts((0.1,), (0.1,))), 0.5)

    def test_tied_threshold_cannot_exceed_budget(self):
        points = roc_counts((0.1, 0.2), (0.2, 0.2, 0.8))
        result = operating_points(points, (0.1, 0.5, 0.7))
        self.assertEqual([row["tpr"] for row in result], [0.5, 0.5, 1])
        self.assertEqual([row["false_positives"] for row in result], [0, 0, 2])

    def test_paired_bootstrap_and_utility(self):
        export = load_losses(EXAMPLES / "comparison.csv")
        manifest = load_manifest(EXAMPLES / "manifest.json")
        result = compare_losses(export, manifest, repeats=200, seed=7)
        self.assertEqual(result, compare_losses(export, manifest, repeats=200, seed=7))
        self.assertEqual(result["models"]["before"]["auc"], 1)
        self.assertEqual(result["models"]["candidate"]["auc"], 0.5)
        self.assertEqual(result["auc_deltas"]["candidate_minus_reference"]["interval"], [0, 0])
        self.assertAlmostEqual(result["models"]["candidate"]["mean_utility_loss"], 0.4)
        self.assertTrue(any("one-event resolution" in warning for warning in result["warnings"]))

    def test_invalid_configuration(self):
        export = load_losses(EXAMPLES / "comparison.csv")
        manifest = load_manifest(EXAMPLES / "manifest.json")
        for options in ({"targets": (float("nan"),)}, {"targets": (0.1, 0.1)}, {"repeats": 10}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                compare_losses(export, manifest, **options)
