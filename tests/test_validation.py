import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from unlearning_audit.validation import load_losses, validate_losses

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "losses.csv"
VALID = EXAMPLE.read_text(encoding="utf-8")


class ValidationTests(unittest.TestCase):
    def test_valid_export(self):
        self.assertEqual(validate_losses(EXAMPLE), {"forget": 1, "holdout": 1})

    def test_row_order_does_not_affect_alignment(self):
        header, *rows = VALID.splitlines()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reversed.csv"
            path.write_text("\n".join([header, *reversed(rows)]), encoding="utf-8")
            self.assertEqual(load_losses(path), load_losses(EXAMPLE))

    def test_rejects_invalid_exports(self):
        cases = {
            "duplicate sample": VALID + "f01,forget,before,0.12\n",
            "inconsistent cohort": VALID.replace("f01,forget,after", "f01,holdout,after"),
            "missing 1 sample": VALID.replace("h01,holdout,after,0.78\n", ""),
            "finite and non-negative": VALID.replace("0.84", "NaN"),
            "expected four fields": VALID.replace("0.84", "0.84,extra"),
            "expected columns": VALID.replace("sample_id", "sample"),
            "at least two models": VALID.replace("f01,forget,after,0.67\nh01,holdout,after,0.78\n", ""),
            "both forget and holdout": "\n".join(line for line in VALID.splitlines() if "holdout" not in line),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "losses.csv"
            for message, content in cases.items():
                with self.subTest(message=message):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        validate_losses(path)

    def test_cli_exit_codes(self):
        for path, expected in ((EXAMPLE, 0), (ROOT / "missing.csv", 1)):
            with self.subTest(path=path):
                result = subprocess.run(
                    [sys.executable, "-m", "unlearning_audit", "validate", str(path)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, expected, result.stderr)
