import unittest

from unlearning_audit.calibration import calibrated_points


class CalibrationTests(unittest.TestCase):
    def test_calibration_and_measured_rates_are_distinct(self):
        points = calibrated_points(
            forget=(0.05, 0.21),
            holdout=(0.08, 0.09, 0.10, 0.11),
            calibration=(0.1, 0.2, 0.2, 0.3),
            targets=(0.1, 0.25, 0.5, 0.75),
        )
        self.assertEqual([point["calibration_false_positives"] for point in points], [0, 1, 1, 3])
        self.assertEqual([point["fpr"] for point in points], [0, 0.75, 0.75, 1.0])
        self.assertEqual([point["tpr"] for point in points], [0, 0.5, 0.5, 0.5])
