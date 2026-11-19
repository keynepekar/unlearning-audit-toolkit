import unittest
from dataclasses import replace
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from unlearning_audit.groups import load_groups
from unlearning_audit.manifest import load_manifest
from unlearning_audit.metrics import compare_losses, roc_counts, area_under_roc
from unlearning_audit.report import render_report
from unlearning_audit.study import summarize_study
from unlearning_audit.validation import LossExport
from unlearning_audit.verification import verify_comparison

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class GroupTests(unittest.TestCase):
    def setUp(self):
        self.export = LossExport(
            {"forget": ("f1", "f2", "f3"), "holdout": ("h1", "h2", "h3", "h4")},
            {
                "initial": {"forget": (2, 3, 4), "holdout": (1, 1, 3, 4)},
                "updated": {"forget": (1, 1, 4), "holdout": (1, 2, 3, 3)},
                "retrained": {"forget": (3, 3, 3), "holdout": (1, 3, 3, 4)},
            },
        )
        manifest = load_manifest(EXAMPLES / "manifest.json")
        self.manifest = replace(manifest, cohorts={k: manifest.cohorts[k] for k in self.export.counts})
        self.groups = {"f1": "a", "f2": "a", "f3": "b", "h1": "c", "h2": "d", "h3": "d", "h4": "d"}

    def compare(self, groups):
        return compare_losses(
            self.export, self.manifest, group_assignments=groups, repeats=200, include_fingerprints=True,
        )

    def test_uneven_groups_match_explicit_duplicate_rows(self):
        actual = self.compare(self.groups)
        rng = Random(42)
        draws = []
        for _ in range(200):
            forget = sum(rng.choices(((0, 1), (2,)), k=2), ())
            holdout = sum(rng.choices(((0,), (1, 2, 3)), k=2), ())
            areas = {
                role: area_under_roc(roc_counts(
                    tuple(values["forget"][i] for i in forget),
                    tuple(values["holdout"][i] for i in holdout),
                ))
                for role, values in self.export.losses.items()
            }
            draws.append(areas["updated"] - areas["retrained"])
        ordered = sorted(draws)
        expected = [
            ordered[index] + fraction * (ordered[index + 1] - ordered[index])
            for index, fraction in ((4, 0.975), (194, 0.025))
        ]
        for got, want in zip(actual["auc_deltas"]["candidate_minus_reference"]["interval"], expected, strict=True):
            self.assertAlmostEqual(got, want)
        self.assertEqual(actual["models"], self.compare(None)["models"])
        self.assertIn("Paired bootstrap of whole groups", render_report(actual))

    def test_replay_rejects_changed_partition_and_accepts_relabeling(self):
        metrics = self.compare(self.groups)
        with self.assertRaisesRegex(ValueError, "provide group assignments"):
            verify_comparison(metrics, self.export, self.manifest)
        relabeled = {sample: f"new-{label}" for sample, label in reversed(tuple(self.groups.items()))}
        verify_comparison(metrics, self.export, self.manifest, group_assignments=relabeled)
        with self.assertRaisesRegex(ValueError, "groups_sha256"):
            verify_comparison(metrics, self.export, self.manifest, group_assignments={**self.groups, "f2": "b"})

    def test_rejects_incomplete_shared_and_single_groups(self):
        for groups, message in (
            ({"f1": "a"}, "assign every"),
            ({**self.groups, "f3": "c"}, "must not span"),
            ({**self.groups, "f3": "a"}, "at least two groups"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.compare(groups)

    def test_study_keeps_group_partitions_fixed(self):
        self.manifest = replace(self.manifest, provenance={"Protocol": "test", "Training seed": "1"})
        first = self.compare(self.groups)
        self.manifest = replace(self.manifest, provenance={"Protocol": "test", "Training seed": "2"})
        second = self.compare(self.groups)
        self.assertEqual(len(summarize_study([first, second])["runs"]), 2)
        changed = self.compare({**self.groups, "f2": "b"})
        with self.assertRaisesRegex(ValueError, "differs in grouping"):
            summarize_study([first, changed])

    def test_csv_rejects_duplicate_sample(self):
        self.assertEqual(len(load_groups(EXAMPLES / "groups.csv")), 4)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "groups.csv"
            path.write_text("sample_id,group_id\nf1,a\nf1,b\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_groups(path)
