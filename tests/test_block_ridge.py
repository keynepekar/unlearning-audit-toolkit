import sys
import unittest
from pathlib import Path

try:
    import numpy as np
    from sklearn.linear_model import Ridge
except ImportError:
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
if np is not None:
    from block_ridge import BlockRidge, block_refit


@unittest.skipIf(np is None, "demo dependencies are not installed")
class BlockRidgeTests(unittest.TestCase):
    def test_matches_independent_svd_for_short_and_rank_deficient_blocks(self):
        rng = np.random.default_rng(14)
        x = rng.normal(size=(23, 5))
        x[:, 4] = x[:, 0]
        y = rng.normal(size=len(x))
        removed = np.array([0, 1, 2, 3, 7, 22])
        keep = np.ones(len(x), dtype=bool)
        keep[removed] = False
        expected = Ridge(alpha=0.1, fit_intercept=False, solver="svd").fit(x[keep], y[keep]).coef_
        for size in (1, 4, 100):
            with self.subTest(block_size=size):
                actual, details = block_refit(x, y, removed, alpha=0.1, block_size=size)
                np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
                self.assertGreater(details["touched_blocks"], 0)

    def test_empty_and_invalid_deletion(self):
        x = np.eye(2)
        y = np.array([1., 2.])
        actual, _ = block_refit(x, y, [], alpha=1, block_size=1)
        np.testing.assert_allclose(actual, y / 2)
        for removed in ([0, 0], [-1], [2], [0.5]):
            with self.subTest(removed=removed), self.assertRaises(ValueError):
                block_refit(x, y, removed, alpha=1, block_size=1)

    def test_sequential_removals_are_atomic_and_match_refit(self):
        rng = np.random.default_rng(24)
        x, y = rng.normal(size=(30, 4)), rng.normal(size=30)
        model = BlockRidge(x, y, alpha=0.1, block_size=7)
        for group in ([1, 9], [2, 14], [7, 8]):
            weights, _ = model.remove(group)
            expected = Ridge(alpha=0.1, fit_intercept=False, solver="svd").fit(x[model.keep], y[model.keep]).coef_
            np.testing.assert_allclose(weights, expected, atol=1e-11, rtol=1e-11)
        before = model.keep.copy()
        with self.assertRaises(ValueError):
            model.remove([1, 5])
        np.testing.assert_array_equal(model.keep, before)

    def test_order_and_complete_removal(self):
        rng = np.random.default_rng(6)
        x, y = rng.normal(size=(12, 3)), rng.normal(size=12)
        first = BlockRidge(x, y, alpha=1, block_size=4)
        second = BlockRidge(x, y, alpha=1, block_size=4)
        first.remove([0, 1])
        a, _ = first.remove([5, 6])
        second.remove([5, 6])
        b, _ = second.remove([0, 1])
        np.testing.assert_allclose(a, b, atol=1e-12)
        result, details = first.remove(np.flatnonzero(first.keep))
        np.testing.assert_allclose(result, np.zeros(3), atol=1e-12)
        self.assertEqual(details["active_rows"], 0)
