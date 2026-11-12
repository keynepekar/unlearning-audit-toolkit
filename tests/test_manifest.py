import json
import tempfile
import unittest
from pathlib import Path

from unlearning_audit.manifest import Manifest, load_manifest


def manifest_data():
    return {
        "schema_version": 1,
        "title": "Example",
        "loss_definition": "Cross-entropy",
        "models": {"before": "m1", "candidate": "m2", "reference": "m3"},
        "cohorts": {"forget": "Original training rows", "holdout": "Disjoint rows"},
        "provenance": {"seed": "42"},
    }


class ManifestTests(unittest.TestCase):
    def test_checks_export_alignment(self):
        manifest = Manifest.from_dict(manifest_data())
        manifest.check_alignment({"m1", "m2", "m3"}, {"forget", "holdout"})
        with self.assertRaisesRegex(ValueError, "model names"):
            manifest.check_alignment({"m1", "m2"}, {"forget", "holdout"})
        with self.assertRaisesRegex(ValueError, "cohort names"):
            manifest.check_alignment({"m1", "m2", "m3"}, {"forget", "holdout", "utility"})

    def test_rejects_duplicate_roles_and_invalid_version(self):
        data = manifest_data()
        data["models"]["reference"] = "m2"
        with self.assertRaisesRegex(ValueError, "different name"):
            Manifest.from_dict(data)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            Manifest.from_dict(manifest_data() | {"schema_version": True})

    def test_rejects_duplicate_json_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"schema_version": 1, ' + json.dumps(manifest_data())[1:], encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                load_manifest(path)
