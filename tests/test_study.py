import unittest
from dataclasses import replace
from pathlib import Path

from unlearning_audit.manifest import load_manifest
from unlearning_audit.metrics import compare_losses
from unlearning_audit.study import study_csv, study_html, summarize_study
from unlearning_audit.validation import load_losses

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class StudyTests(unittest.TestCase):
    def run_metrics(self, seed, protocol="test", fingerprints=True):
        export = load_losses(EXAMPLES / "comparison.csv")
        manifest = replace(
            load_manifest(EXAMPLES / "manifest.json"),
            provenance={"Protocol": protocol, "Training seed": str(seed)},
        )
        return compare_losses(export, manifest, repeats=0, include_fingerprints=fingerprints)

    def test_run_order_and_observed_spread(self):
        study = summarize_study([self.run_metrics(5, "<protocol>"), self.run_metrics(2, "<protocol>")])
        self.assertEqual([run["training_seed"] for run in study["runs"]], [2, 5])
        self.assertEqual(study["auc_delta_summary"]["candidate_minus_before"], {
            "mean": -0.5, "sample_sd": 0.0, "min": -0.5, "max": -0.5,
        })
        self.assertIn("2,1.0,0.5,0.5,-0.5,0.0", study_csv(study))
        self.assertIn("&lt;protocol&gt;", study_html(study))
        self.assertNotIn("<protocol>", study_html(study))

    def test_rejects_missing_fingerprints_and_duplicate_seeds(self):
        with self.assertRaisesRegex(ValueError, "no input fingerprints"):
            summarize_study([self.run_metrics(2, fingerprints=False), self.run_metrics(5)])
        with self.assertRaisesRegex(ValueError, "duplicate training seed"):
            summarize_study([self.run_metrics(2), self.run_metrics(2)])

    def test_chart_shows_a_signed_scale_with_units(self):
        study = summarize_study([self.run_metrics(2), self.run_metrics(5)])
        report = study_html(study)
        self.assertIn("AUC difference (points). One point is 0.01 AUC.", report)
        for tick in ("−100", "−50", "0", "+50", "+100"):
            self.assertIn(f'text-anchor="middle">{tick}</text>', report)

    def test_rejects_mixed_protocols_and_splits(self):
        first = self.run_metrics(2)
        with self.assertRaisesRegex(ValueError, "protocol metadata"):
            summarize_study([first, self.run_metrics(5, "other")])
        second = self.run_metrics(5)
        second["fingerprints"]["split_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs in split"):
            summarize_study([first, second])
