import unittest

from unlearning_audit.metrics import area_under_roc, roc_counts


class DiagnosticControls(unittest.TestCase):
    def test_equal_squared_loss_can_hide_prediction_disagreement(self):
        targets = (1.0, 2.0, 3.0, 4.0)
        delta = 0.5
        first = tuple(y + delta for y in targets)
        second = tuple(y - delta for y in targets)
        losses_a = tuple((prediction - y) ** 2 for prediction, y in zip(first, targets))
        losses_b = tuple((prediction - y) ** 2 for prediction, y in zip(second, targets))
        self.assertEqual(losses_a, losses_b)
        self.assertEqual(area_under_roc(roc_counts(losses_a[:2], losses_a[2:])), 0.5)
        self.assertEqual(max(abs(a - b) for a, b in zip(first, second)), 1.0)
