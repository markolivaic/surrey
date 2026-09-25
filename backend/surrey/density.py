"""Draw a block's crop at a chosen point density, and say what the crop actually ended up with.

A crop is always POINTS_PER_CROP long, whatever the block holds: a block with more points is sampled
without replacement, a block with fewer is sampled with repeats, which is how training saw the sparser
Swiss tiles. So thinning the pool a crop is drawn from changes nothing at all while the pool stays
above the crop size, and everything once it falls below: thinning the pool would change the input over
small areas and leave the larger ones untouched.

This makes the thing being varied explicit: how many distinct points of the block the crop is allowed
to carry. Ask for a density in points per square metre, get that many distinct points over the block's
area (or all of them, if the block has fewer), padded with repeats to the crop length.
"""

from __future__ import annotations

import numpy as np


def distinct_for_density(density_per_m2: float, block_m: float, available: int) -> int:
    """How many distinct points a block of this size holds at this density, capped by what exists."""
    wanted = int(round(density_per_m2 * block_m * block_m))
    return max(1, min(wanted, available))


def crop_at_density(
    idx: np.ndarray, n_points: int, rng: np.random.Generator, distinct: int | None = None
) -> tuple[np.ndarray, dict]:
    """`n_points` indices drawn from `idx`, carrying at most `distinct` different points.

    Returns the indices and what the crop is: how many different points it holds and whether repeats
    were needed, so a measurement can report the input it actually fed the model rather than the input
    it meant to.
    """
    if distinct is None or distinct >= idx.size:
        pool = idx
    else:
        pool = rng.choice(idx, distinct, replace=False)
    repeats_needed = pool.size < n_points
    pick = rng.choice(pool, n_points, replace=repeats_needed)
    return pick, {
        "block_points": int(idx.size),
        "distinct_in_pool": int(pool.size),
        "distinct_in_crop": int(np.unique(pick).size),
        "repeats_needed": bool(repeats_needed),
    }
