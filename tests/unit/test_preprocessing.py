"""Preprocessing behaviour, against cases whose answers can be worked out by hand."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.data import height_above, min_z_grid, voxel_pick  # noqa: E402


def test_voxel_pick_keeps_one_point_per_occupied_voxel():
    # Nine points, but only three distinct 1 m voxels.
    x = np.array([0.1, 0.2, 0.3, 5.1, 5.2, 5.3, 9.9, 9.8, 9.7])
    y = np.zeros(9)
    z = np.zeros(9)
    keep = voxel_pick(x, y, z, voxel_m=1.0, rng=np.random.default_rng(0))
    assert keep.size == 3
    assert np.unique(np.floor(x[keep])).size == 3


def test_voxel_pick_separates_points_stacked_vertically():
    # One footprint, three heights. A 2D grid would collapse these to one point
    # and a canopy would vanish; the voxel is three dimensional on purpose.
    x = np.zeros(3)
    y = np.zeros(3)
    z = np.array([0.0, 5.0, 10.0])
    keep = voxel_pick(x, y, z, voxel_m=1.0, rng=np.random.default_rng(0))
    assert keep.size == 3


def test_voxel_pick_returns_sorted_indices_within_range():
    rng = np.random.default_rng(3)
    x, y, z = (rng.random(500) * 20 for _ in range(3))
    keep = voxel_pick(x, y, z, voxel_m=0.5, rng=rng)
    assert np.all(np.diff(keep) > 0)
    assert keep.min() >= 0 and keep.max() < 500


def test_voxel_pick_is_reproducible_for_a_given_seed():
    rng_args = {"voxel_m": 0.4}
    x, y, z = (np.random.default_rng(7).random(300) * 10 for _ in range(3))
    a = voxel_pick(x, y, z, rng=np.random.default_rng(11), **rng_args)
    b = voxel_pick(x, y, z, rng=np.random.default_rng(11), **rng_args)
    assert np.array_equal(a, b)


def test_voxel_pick_refuses_a_key_that_would_overflow():
    x = np.array([0.0, 1e12])
    y = np.array([0.0, 1e12])
    z = np.array([0.0, 1e12])
    with pytest.raises(ValueError, match="overflow"):
        voxel_pick(x, y, z, voxel_m=0.001)


def test_height_above_ground_on_a_flat_plane_is_the_offset():
    rng = np.random.default_rng(0)
    x = rng.random(400) * 20
    y = rng.random(400) * 20
    z = np.full(400, 3.0)
    grid, x0, y0, cell = min_z_grid(x, y, z, cell_m=2.0)
    hag = height_above(x, y, z, grid, x0, y0, cell)
    assert np.allclose(hag, 0.0)


def test_height_above_ground_follows_a_slope_rather_than_absolute_z():
    # A 10 percent ramp with a 5 m mast standing on it. Height above ground must
    # report 5 m for the mast wherever it stands, which is the whole reason the
    # feature exists: San Francisco has relief and Surrey mostly does not.
    gx = np.repeat(np.arange(0, 40, 1.0), 4)
    gy = np.tile(np.arange(0, 4, 1.0), 40)
    gz = 0.1 * gx
    mast_x = np.full(6, 30.5)
    mast_y = np.full(6, 1.5)
    mast_z = 0.1 * 30.5 + np.linspace(0, 5, 6)

    x = np.concatenate([gx, mast_x])
    y = np.concatenate([gy, mast_y])
    z = np.concatenate([gz, mast_z])
    grid, x0, y0, cell = min_z_grid(x, y, z, cell_m=1.0)
    hag = height_above(x, y, z, grid, x0, y0, cell)

    assert hag[-1] == pytest.approx(5.0, abs=0.2)
    assert hag[: gx.size].max() < 0.2


def test_min_z_grid_fills_cells_that_hold_no_points():
    # A hole in the middle must still get a ground value, or height above ground
    # is undefined exactly where a wide roof sits.
    x = np.array([0.0, 0.0, 10.0, 10.0])
    y = np.array([0.0, 10.0, 0.0, 10.0])
    z = np.array([1.0, 1.0, 1.0, 1.0])
    grid, _, _, _ = min_z_grid(x, y, z, cell_m=1.0)
    assert np.isfinite(grid).all()
    assert grid.min() == pytest.approx(1.0)


def test_min_z_grid_rejects_an_empty_cloud():
    with pytest.raises(ValueError):
        min_z_grid(np.array([]), np.array([]), np.array([]))
