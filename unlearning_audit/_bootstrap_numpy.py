"""Exact integer AUC resampling with an optional NumPy backend."""

import sys
from random import Random

import numpy as np


def _rng(seed):
    python = Random(seed)
    version, state, gaussian = python.getstate()
    if sys.implementation.name != "cpython" or version != 3 or len(state) != 625 or gaussian is not None:
        raise ValueError("NumPy bootstrap requires CPython's version 3 Random state")
    generator = np.random.RandomState(0)
    initial = ("MT19937", np.asarray(state[:-1], dtype=np.uint32), state[-1], 0, 0.0)
    generator.set_state(initial)
    # Keep the public replay contract even if a runtime changes its random draws.
    for size in (7, 1024, 1025):
        if not np.array_equal(_choices(generator, size), python.choices(range(size), k=size)):
            raise ValueError("NumPy bootstrap draws differ from CPython")
    generator.set_state(initial)
    return generator


def _choices(generator, size):
    return (generator.random_sample(size) * size).astype(np.intp)


def _row_units(size, groups):
    units = np.empty(size, dtype=np.intp)
    for unit, indices in enumerate(groups):
        units[list(indices)] = unit
    return units


def auc_draws(ranks, forget_n, holdout_n, repeats, seed, groups):
    """Yield AUCs with the same paired rows or clusters as the Python backend."""
    generator = _rng(seed)
    converted = {role: (np.asarray(forget), np.asarray(holdout), levels)
                 for role, (forget, holdout, levels) in ranks.items()}
    if groups is None:
        max_forget, max_holdout = forget_n, holdout_n
    else:
        forget_groups, holdout_groups = groups["forget"], groups["holdout"]
        max_forget = len(forget_groups) * max(map(len, forget_groups))
        max_holdout = len(holdout_groups) * max(map(len, holdout_groups))
        forget_units = _row_units(forget_n, forget_groups)
        holdout_units = _row_units(holdout_n, holdout_groups)
    if max(2 * max_forget, max_holdout, 2 * max_forget * max_holdout) > np.iinfo(np.int64).max:
        raise ValueError("bootstrap counts exceed the NumPy int64 range; use the Python backend")

    for _ in range(repeats):
        if groups is None:
            forget_idx = _choices(generator, forget_n)
            holdout_idx = _choices(generator, holdout_n)
            denominator = 2 * forget_n * holdout_n
        else:
            forget_counts = np.bincount(_choices(generator, len(forget_groups)), minlength=len(forget_groups))
            holdout_counts = np.bincount(_choices(generator, len(holdout_groups)), minlength=len(holdout_groups))
            forget_weights = forget_counts[forget_units]
            holdout_weights = holdout_counts[holdout_units]
            denominator = 2 * int(forget_weights.sum(dtype=np.int64)) * int(holdout_weights.sum(dtype=np.int64))
        areas = {}
        for role, (forget, holdout, levels) in converted.items():
            if groups is None:
                members = np.bincount(forget[forget_idx], minlength=levels).astype(np.int64, copy=False)
                nonmembers = np.bincount(holdout[holdout_idx], minlength=levels).astype(np.int64, copy=False)
            else:
                members = np.zeros(levels, dtype=np.int64)
                nonmembers = np.zeros(levels, dtype=np.int64)
                np.add.at(members, forget, forget_weights)
                np.add.at(nonmembers, holdout, holdout_weights)
            twice_area = int(nonmembers @ (2 * members.cumsum(dtype=np.int64) - members))
            areas[role] = twice_area / denominator
        yield areas
