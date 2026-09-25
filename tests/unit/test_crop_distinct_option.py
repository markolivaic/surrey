"""The crop density option, and the thing that would quietly ruin it: changing the draw when it is off.

Training with the crop's density drawn at random is the diagnosis for water. It is only readable if a
run without the option draws exactly what it drew before the option existed, generator order included.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile, write_tile  # noqa: E402


def tile(tmp_path: Path, n: int = 40_000) -> CachedTile:
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 60, (n, 2))
    z = rng.uniform(0, 5, n)
    arrays = {
        "xy_cm": (xy * 100).astype(np.uint16),
        "z_cm": (z * 100).astype(np.uint32),
        "hag_cm": (z * 100).astype(np.int16),
        "ret_q": np.zeros(n, np.uint8),
        "label": rng.integers(0, 15, n).astype(np.int8),
        "asprs": np.ones(n, np.uint8),
    }
    write_tile(
        tmp_path / "t",
        arrays,
        {"name": "t", "split": "train", "origin": [0.0, 0.0, 0.0], "epsg": 32610},
    )
    return CachedTile(tmp_path / "t", "train")


def test_a_crop_carries_the_distinct_count_it_was_given(tmp_path):
    t = tile(tmp_path)
    feat, _label, _rep = t.random_crop(np.random.default_rng(1), 40.0, 8192, 3, distinct=2000)
    rows = {tuple(r) for r in feat}
    assert len(rows) <= 2000


def test_self_test_without_the_option_the_draw_is_what_it_always_was(tmp_path):
    t = tile(tmp_path)
    a = t.random_crop(np.random.default_rng(7), 40.0, 8192, 3)
    b = t.random_crop(np.random.default_rng(7), 40.0, 8192, 3, distinct=None)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    # And the generator is left in the same state, which is what keeps a whole run reproducible.
    ga, gb = np.random.default_rng(7), np.random.default_rng(7)
    t.random_crop(ga, 40.0, 8192, 3)
    t.random_crop(gb, 40.0, 8192, 3, distinct=None)
    assert ga.integers(0, 10**9) == gb.integers(0, 10**9)


def test_a_distinct_count_larger_than_the_block_changes_nothing(tmp_path):
    t = tile(tmp_path)
    a = t.random_crop(np.random.default_rng(3), 40.0, 8192, 3)
    b = t.random_crop(np.random.default_rng(3), 40.0, 8192, 3, distinct=10**9)
    assert np.array_equal(a[0], b[0])
