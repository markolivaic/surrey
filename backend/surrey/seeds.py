"""How a claim is read across training seeds. Written before M0 seed C was trained.

Two seeds of M0 disagreed by more than any single-model reading could survive:
building lift in Financial 0.2384 and 0.3307, easy-error AUROC in Golden Gate Park
0.9462 and 0.8284, water AUROC in Embarcadero 0.725 and 0.6018. So from seed C on,
for every recipe (m0, m0prime, m1):

  1. Every claim is read on each seed separately, with the rule it was written with.
     That covers both confounds and the split of easy errors into transfer and home
     weakness.
  2. A claim stands only if it holds on every seed. Otherwise it is written as holding
     "on k of n seeds", and that phrase is the claim.
  3. Every figure is given as the median over seeds with the min and max beside it.
  4. The published model is seed A of each recipe, as decided before any result. A seed
     is never chosen by its result.

The relief rule was written for one model. Reading it per seed is a faithful extension,
not a new threshold: the margin stays 0.05 and the metric stays delivery ground called
building. No numeric definition of "at the edge" is added here, because any one written
now would be written after seeing seed A at 0.0516. Seed A's relief reading is marked at
the edge by name (EDGE_READINGS), the decision recorded before seed C, and every other
reading is given as its number.

Side effects between recipes (m1 against m0prime, m0prime against m0), per metric and
area: a drop is reported when every seed of the new recipe is worse than every seed of
the old one, max(new) < min(old) for a figure where higher is better and
min(new) > max(old) where lower is better. Anything short of that is not a drop.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

SEED_LETTERS = ("A", "B", "C")
SEED_NUMBERS = {"A": 0, "B": 1, "C": 2}
PUBLISHED_SEED = "A"
# Readings the text keeps at the edge, by decision and not by a threshold.
EDGE_READINGS = {
    ("m0", "A", "relief"): "0.0516 against a 0.05 bar; decided before seed C, stays at the edge"
}


def variant(recipe: str, seed: str) -> str:
    return f"{recipe}_seed{seed}"


def spread(values: Mapping[str, float | None]) -> dict:
    """Per-seed values, and median, min and max over the seeds that have one."""
    have = {k: v for k, v in values.items() if v is not None}
    out: dict = {"per_seed": dict(values), "n": len(have)}
    if not have:
        return out | {"median": None, "min": None, "max": None}
    arr = np.asarray(list(have.values()), dtype=float)
    return out | {
        "median": round(float(np.median(arr)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def holds(readings: Mapping[str, str | None], expected: str) -> dict:
    """Whether a reading holds on every seed, and the phrase the text must use."""
    have = {k: v for k, v in readings.items() if v is not None}
    k = sum(1 for v in have.values() if v == expected)
    n = len(have)
    if n == 0:
        phrase = "not measured"
    elif k == n:
        phrase = f"on all {n} seeds"
    else:
        phrase = f"on {k} of {n} seeds"
    return {
        "per_seed": dict(readings),
        "expected": expected,
        "k": k,
        "n": n,
        "stands": n > 0 and k == n,
        "phrase": phrase,
    }


def drop(
    new: Iterable[float | None], old: Iterable[float | None], higher_is_better: bool
) -> bool | None:
    """Every seed of the new recipe worse than every seed of the old one."""
    a = [v for v in new if v is not None]
    b = [v for v in old if v is not None]
    if not a or not b:
        return None
    return max(a) < min(b) if higher_is_better else min(a) > max(b)


def gain(
    new: Iterable[float | None], old: Iterable[float | None], higher_is_better: bool
) -> bool | None:
    """The mirror of drop, reported with the same strictness."""
    return drop(old, new, higher_is_better)


def chance_of_a_move(n_new: int, n_old: int) -> float:
    """How often chance alone puts every seed of one side beyond every seed of the other.

    With exchangeable runs every ordering of the n_new + n_old values is equally likely, and
    exactly one of the comb(n_new + n_old, n_new) ways to choose the new side's ranks puts all of
    them on top, and one puts all of them at the bottom. Two-sided, since a drop and a gain are
    both read: 2 / comb(n_new + n_old, n_new), 1 in 10 for three seeds against three.
    """
    from math import comb

    return 2 / comb(n_new + n_old, n_new)


def expected_by_chance(comparisons: int, n_new: int, n_old: int) -> float:
    """The number of moves chance alone would give over a set of comparisons."""
    return round(comparisons * chance_of_a_move(n_new, n_old), 1)
