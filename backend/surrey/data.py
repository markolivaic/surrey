"""Turning tiles into training crops, identically on both sides of the transfer.

Every preprocessing choice here is made twice: once for Surrey, once for San
Francisco. Where the two could diverge, this module makes them agree, because a
difference in preprocessing looks exactly like a difference between cities and
there is no way to tell them apart afterwards.

Three choices worth arguing about, all of them made to protect the transfer:

No height above ground is fed to the network. That was the first design and
scripts/investigations/ground_cell_choice.py killed it: with a rule fixed in advance, no single
label-free terrain surface both kept Surrey's buildings above a 2 m gate and kept
San Francisco's hillside ground below it. A coarse cell finds real ground under a
wide roof and cannot follow a hill, a fine cell follows the hill and sits on the
roof, and a two stage version still put a quarter of a synthetic slope above the
gate. Rather than keep bolting onto a ground filter, the network is given a 40 m
block and three set abstraction levels and learns where ground is, which is what
the models on the DALES benchmark do.

Z reaches the network relative to its own block's first percentile, so a metre is
a metre and no terrain model is involved. The quantile grid survives for one
narrower job: the 2 m gate used when scoring against footprints, where its error
rates are measured and small enough to state.

Density is equalised by voxel downsampling. DALES runs 26 to 70 points per square
metre; the San Francisco areas run 76 to 147. A model trained at one density and
run at three times that sees every neighbourhood shrink. A fixed voxel size
thins both to the same ceiling.

Intensity is not used. Both deliveries carry it and neither is calibrated to the
other, so a model leaning on intensity would fail on the target for reasons that
have nothing to do with the city. Geometry only, and the cost of that shows up
in the in-domain score where it can be seen.
"""

from __future__ import annotations

import numpy as np

# DALES-2 class ids, from the dataset card.
CLASS_NAMES = [
    "ground",
    "vegetation",
    "car",
    "powerline",
    "fence",
    "tree",
    "pickup",
    "van-truck",
    "heavy-duty",
    "utility-pole",
    "light-pole",
    "traffic-pole",
    "habitat",
    "complex",
    "annex",
]
N_CLASSES = len(CLASS_NAMES)

# Coarse grouping used when scoring against a target that cannot tell a pickup
# from a van. Values index CLASS_NAMES.
COARSE = {
    "ground": ["ground"],
    "vegetation": ["vegetation", "tree"],
    "building": ["habitat", "complex", "annex"],
    "vehicle": ["car", "pickup", "van-truck", "heavy-duty"],
    "pole": ["utility-pole", "light-pole", "traffic-pole", "powerline"],
    "fence": ["fence"],
}

BLOCK_M = 40.0
VOXEL_M = 0.15
GROUND_CELL_M = 10.0
GROUND_QUANTILE = 0.05
POINTS_PER_CROP = 32768


def min_z_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray, cell_m: float = GROUND_CELL_M):
    """Minimum z per cell over ALL points, empty cells filled from the nearest.

    Label free on purpose. This is the same surface on Surrey and on San
    Francisco, which is the only reason a height feature can cross between them.
    """
    x0, y0 = float(x.min()), float(y.min())
    nx = max(1, int(np.ceil((x.max() - x0) / cell_m)) + 1)
    ny = max(1, int(np.ceil((y.max() - y0) / cell_m)) + 1)
    ix = np.clip(((x - x0) / cell_m).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - y0) / cell_m).astype(np.int64), 0, ny - 1)
    grid = np.full(nx * ny, np.inf)
    np.minimum.at(grid, iy * nx + ix, z)
    grid = grid.reshape(ny, nx)
    empty = ~np.isfinite(grid)
    if empty.any():
        from scipy.ndimage import distance_transform_edt

        _, (sy, sx) = distance_transform_edt(empty, return_distances=True, return_indices=True)
        grid = grid[sy, sx]
    return grid, x0, y0, cell_m


def quantile_z_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell_m: float = GROUND_CELL_M,
    q: float = GROUND_QUANTILE,
):
    """A low quantile of z per cell, rather than the minimum. Label free.

    The minimum is not robust. San Francisco's financial district carries about
    half a percent of points classified low noise, some of them 70 m below the
    street, and one such point drags a whole cell's minimum down with it. At a
    25 m cell that made 90 percent of the delivery's own ground points read as
    more than 2 m above ground, in an area with 1.3 m of relief. The same failure
    in the opposite direction is recorded in nadir, where high noise put cells
    200 m above a city that tops out at 217 m.

    A quantile ignores a sparse tail at either end without needing a noise label,
    which matters because the target has one and the training set does not, and
    using it on one side only would be a difference in preparation dressed up as
    a difference between cities.
    """
    x0, y0 = float(x.min()), float(y.min())
    nx = max(1, int(np.ceil((x.max() - x0) / cell_m)) + 1)
    ny = max(1, int(np.ceil((y.max() - y0) / cell_m)) + 1)
    ix = np.clip(((x - x0) / cell_m).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - y0) / cell_m).astype(np.int64), 0, ny - 1)
    key = iy * nx + ix

    order = np.lexsort((z, key))
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    counts = np.diff(np.r_[starts, ks.size])
    # Index q of the way into each cell's sorted heights, at least the first.
    pick = starts + np.minimum((counts * q).astype(np.int64), counts - 1)

    grid = np.full(nx * ny, np.inf)
    grid[ks[starts]] = z[order][pick]
    grid = grid.reshape(ny, nx)

    empty = ~np.isfinite(grid)
    if empty.any():
        if empty.all():
            raise ValueError("ground grid is empty everywhere")
        from scipy.ndimage import distance_transform_edt

        _, (sy, sx) = distance_transform_edt(empty, return_distances=True, return_indices=True)
        grid = grid[sy, sx]
    return grid, x0, y0, cell_m


def height_above(x, y, z, grid, x0, y0, cell_m) -> np.ndarray:
    ny, nx = grid.shape
    ix = np.clip(((x - x0) / cell_m).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - y0) / cell_m).astype(np.int64), 0, ny - 1)
    return z - grid[iy, ix]


def voxel_pick(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    voxel_m: float = VOXEL_M,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Indices of one point per occupied voxel, chosen at random within it.

    Random within the voxel rather than first-seen, so the choice does not
    correlate with file order, which on a lidar tile correlates with flight line.

    Voxel indices are folded into one int64 key rather than kept as three
    columns. np.unique over an (N, 3) array sorts 288 MB for a 12 million point
    tile and took minutes per tile; over one column it takes seconds. The key is
    checked for overflow rather than assumed safe.
    """
    rng = rng or np.random.default_rng(0)
    ix = np.floor(x / voxel_m).astype(np.int64)
    iy = np.floor(y / voxel_m).astype(np.int64)
    iz = np.floor(z / voxel_m).astype(np.int64)
    ix -= ix.min()
    iy -= iy.min()
    iz -= iz.min()
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    nz = int(iz.max()) + 1
    if nx * ny * nz > np.iinfo(np.int64).max // 2:
        raise ValueError(f"voxel grid {nx}x{ny}x{nz} overflows an int64 key")

    order = rng.permutation(x.size)
    key = ix * (ny * nz) + iy * nz + iz
    _, first = np.unique(key[order], return_index=True)
    return np.sort(order[first])
