"""Experimental block QR refit for a fixed-feature ridge objective."""

import numpy as np


def factor_block(x, y):
    """Compress a block of the augmented least-squares matrix."""
    augmented = np.column_stack((x, y))
    return np.linalg.qr(augmented, mode="r")


class BlockRidge:
    """Keep row-level data and reusable QR factors for sequential removals."""

    def __init__(self, x, y, *, alpha, block_size):
        self.x = np.array(x, dtype=float, copy=True)
        self.y = np.array(y, dtype=float, copy=True)
        if self.x.ndim != 2 or self.y.shape != (len(self.x),):
            raise ValueError("expected a feature matrix and one target per row")
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError("inputs must be finite")
        if not np.isfinite(alpha) or alpha <= 0 or type(block_size) is not int or block_size < 1:
            raise ValueError("alpha and block_size must be positive")
        self.alpha = alpha
        self.block_size = block_size
        self.keep = np.ones(len(self.x), dtype=bool)
        self.factors = [
            factor_block(self.x[start:start + block_size], self.y[start:start + block_size])
            for start in range(0, len(self.x), block_size)
        ]

    def remove(self, removed):
        indices = np.asarray(removed)
        if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
            raise ValueError("removed indices must be a one-dimensional integer array")
        indices = indices.astype(int)
        if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(self.x)):
            raise ValueError("removed indices must be distinct and in range")
        if not self.keep[indices].all():
            raise ValueError("removed indices must still be active")
        keep = self.keep.copy()
        keep[indices] = False
        replacements = {}
        for block in np.unique(indices // self.block_size):
            start = block * self.block_size
            stop = min(start + self.block_size, len(self.x))
            mask = keep[start:stop]
            replacements[block] = factor_block(self.x[start:stop][mask], self.y[start:stop][mask]) if mask.any() else None
        factors = self.factors.copy()
        for block, factor in replacements.items():
            factors[block] = factor
        regularization = np.column_stack((np.sqrt(self.alpha) * np.eye(self.x.shape[1]), np.zeros(self.x.shape[1])))
        compressed = np.vstack([*(factor for factor in factors if factor is not None), regularization])
        coefficients = np.linalg.lstsq(compressed[:, :-1], compressed[:, -1], rcond=None)[0]
        self.keep, self.factors = keep, factors
        return coefficients, {
            "touched_blocks": len(replacements),
            "total_blocks": len(factors),
            "factor_bytes": sum(factor.nbytes for factor in factors if factor is not None),
            "input_bytes": self.x.nbytes + self.y.nbytes,
            "active_rows": int(keep.sum()),
        }


def block_refit(x, y, removed, *, alpha, block_size):
    """Refit from block factors after removing the specified row indices.

    Unchanged factors are cached for the duration of this call. The returned
    counts describe array payloads and affected blocks.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("expected a feature matrix and one target per row")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("inputs must be finite")
    if not np.isfinite(alpha) or alpha <= 0 or block_size < 1:
        raise ValueError("alpha and block_size must be positive")
    indices = np.asarray(removed)
    if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
        raise ValueError("removed indices must be a one-dimensional integer array")
    indices = indices.astype(int)
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(x)):
        raise ValueError("removed indices must be distinct and in range")

    keep = np.ones(len(x), dtype=bool)
    keep[indices] = False
    factors = []
    cached_bytes = 0
    touched = 0
    for start in range(0, len(x), block_size):
        stop = min(start + block_size, len(x))
        factor = factor_block(x[start:stop], y[start:stop])
        cached_bytes += factor.nbytes
        mask = keep[start:stop]
        if not mask.all():
            touched += 1
            if not mask.any():
                continue
            factor = factor_block(x[start:stop][mask], y[start:stop][mask])
        factors.append(factor)

    regularization = np.column_stack((np.sqrt(alpha) * np.eye(x.shape[1]), np.zeros(x.shape[1])))
    compressed = np.vstack([*factors, regularization])
    # lstsq handles short blocks and rank-deficient inputs. The ridge penalty
    # ensures full column rank for the coefficient part.
    coefficients = np.linalg.lstsq(compressed[:, :-1], compressed[:, -1], rcond=None)[0]
    return coefficients, {
        "touched_blocks": touched,
        "total_blocks": (len(x) + block_size - 1) // block_size,
        "factor_bytes": cached_bytes,
        "input_bytes": x.nbytes + y.nbytes,
    }
