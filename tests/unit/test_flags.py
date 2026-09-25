"""The LAS flag filter, through the real preparation of a small file."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))

from prep_tiles import prep_one  # noqa: E402
from surrey.cache import CachedTile  # noqa: E402
from surrey.flags import (  # noqa: E402
    DROP,
    KEY_POINT,
    OVERLAP,
    SYNTHETIC,
    WITHHELD,
    flag_bits,
    kept,
)


def small_las(path: Path, n: int = 4000, seed: int = 0):
    import laspy

    rng = np.random.default_rng(seed)
    las = laspy.LasData(laspy.LasHeader(version="1.4", point_format=6))
    las.header.scales = np.array([0.001, 0.001, 0.001])
    las.x = rng.uniform(0, 60, n)
    las.y = rng.uniform(0, 60, n)
    las.z = rng.uniform(0, 10, n)
    las.classification = rng.integers(0, 15, n).astype(np.uint8)
    las.return_number = np.ones(n, np.uint8)
    las.number_of_returns = np.ones(n, np.uint8)
    # Heights set apart for each flag, so a flagged point is recognisable after the thinning.
    marked = {"synthetic": 101.0, "withheld": 102.0, "key_point": 103.0, "overlap": 104.0}
    z = np.asarray(las.z)
    for i, (flag, height) in enumerate(marked.items()):
        rows = np.arange(i * 25, i * 25 + 25)
        z[rows] = height
        mask = np.zeros(n, bool)
        mask[rows] = True
        setattr(las, flag, mask)
    las.z = z
    las.write(path)
    return marked


def test_flag_bits_read_each_flag_and_kept_drops_only_synthetic_and_withheld():
    bits = np.array([0, SYNTHETIC, KEY_POINT, WITHHELD, OVERLAP, SYNTHETIC | OVERLAP], np.uint8)
    assert kept(bits).tolist() == [True, False, True, False, True, False]
    assert DROP == SYNTHETIC | WITHHELD


def test_preparation_leaves_out_synthetic_and_withheld_points_and_keeps_the_rest(tmp_path):
    import laspy

    src = tmp_path / "tile.laz"
    marked = small_las(src)
    bits = flag_bits(laspy.read(src))
    assert {int(b) for b in np.unique(bits)} == {0, SYNTHETIC, WITHHELD, KEY_POINT, OVERLAP}

    rec = prep_one(src, "train", tmp_path / "prep", labelled=True, voxel_m=0.15, seed=0)
    tile = CachedTile(tmp_path / "prep" / "tile")
    z = np.asarray(tile.z_cm, np.float64) / 100.0 + tile.origin[2]
    heights = {flag: int(np.isclose(z, h, atol=0.01).sum()) for flag, h in marked.items()}
    assert heights["synthetic"] == 0 and heights["withheld"] == 0
    assert heights["key_point"] > 0 and heights["overlap"] > 0
    assert ((np.asarray(tile.flags) & DROP) == 0).all()
    assert rec["points_dropped_by_flag"]["synthetic"] and rec["points_dropped_by_flag"]["withheld"]
