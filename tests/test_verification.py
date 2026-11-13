import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from unlearning_audit.manifest import load_manifest
from unlearning_audit.metrics import compare_losses
from unlearning_audit.validation import load_losses
from unlearning_audit.verification import verify_comparison

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.source = EXAMPLES / "comparison.csv"
        self.export = load_losses(self.source)
        self.manifest = load_manifest(EXAMPLES / "manifest.json")
        self.metrics = compare_losses(self.export, self.manifest, include_fingerprints=True, repeats=200)

    def test_replay_accepts_reordered_rows_and_equivalent_numbers(self):
        header, *rows = self.source.read_text(encoding="utf-8").splitlines()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reordered.csv"
            path.write_text("\n".join([header, *reversed(rows)]).replace(",0.1\n", ",1e-1\n"), encoding="utf-8")
            verify_comparison(self.metrics, load_losses(path), self.manifest)

    def test_replay_detects_loss_and_split_changes(self):
        content = self.source.read_text(encoding="utf-8")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "changed.csv"
            path.write_text(content.replace(",0.1\n", ",0.11\n"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "losses_sha256"):
                verify_comparison(self.metrics, load_losses(path), self.manifest)
            path.write_text(content.replace("f01,", "f03,"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split_sha256"):
                verify_comparison(self.metrics, load_losses(path), self.manifest)

    def test_replay_detects_manifest_and_result_changes(self):
        renamed = replace(self.manifest, title="Changed title")
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            verify_comparison(self.metrics, self.export, renamed)
        self.metrics["warnings"].append("Unrecorded annotation")
        with self.assertRaisesRegex(ValueError, "replayed metrics"):
            verify_comparison(self.metrics, self.export, self.manifest)

    def test_replay_requires_opt_in(self):
        del self.metrics["fingerprints"]
        with self.assertRaisesRegex(ValueError, "no fingerprints"):
            verify_comparison(self.metrics, self.export, self.manifest)

    def test_replay_checks_paired_loss_gaps(self):
        result = compare_losses(
            self.export, self.manifest, include_fingerprints=True, include_loss_gaps=True, repeats=0,
        )
        self.assertAlmostEqual(result["loss_gaps"]["candidate"]["holdout"], 0.1)
        verify_comparison(result, self.export, self.manifest)
        result["loss_gaps"]["candidate"]["holdout"] = 0.0
        with self.assertRaisesRegex(ValueError, "replayed metrics"):
            verify_comparison(result, self.export, self.manifest)
