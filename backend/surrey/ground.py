"""Height above ground from a point cloud's own ground points.

3DEP deliveries classify ground (ASPRS class 2) even when they classify nothing
else, so a ground surface is available on the target without any model. That is
what makes a height gate possible on unlabelled data.

This is a minimum-z grid with nearest-neighbour fill, not a triangulated surface.
It is deliberately blunt: it exists to answer "is this point near the ground or
well above it", not to publish a terrain model.
"""

from __future__ import annotations

import numpy as np

GROUND_CLASS = 2


def ground_grid(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, cell_m: float = 2.0
) -> tuple[np.ndarray, float, float, float]:
    """Minimum z per cell over the given points, with empty cells filled.

    Returns (grid, x0, y0, cell_m). Cells with no ground point take the value of
    the nearest cell that has one, so the surface is defined everywhere.
    """
    if x.size == 0:
        raise ValueError("no ground points, cannot build a ground surface")
    x0, y0 = float(x.min()), float(y.min())
    nx = max(1, int(np.ceil((x.max() - x0) / cell_m)) + 1)
    ny = max(1, int(np.ceil((y.max() - y0) / cell_m)) + 1)

    ix = np.clip(((x - x0) / cell_m).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - y0) / cell_m).astype(np.int64), 0, ny - 1)
    flat = iy * nx + ix

    grid = np.full(nx * ny, np.inf, dtype=np.float64)
    np.minimum.at(grid, flat, z)
    grid = grid.reshape(ny, nx)

    empty = ~np.isfinite(grid)
    if empty.any():
        if empty.all():
            raise ValueError("ground grid is empty everywhere")
        from scipy.ndimage import distance_transform_edt

        # Indices of the nearest filled cell for every empty cell.
        _, (iy_src, ix_src) = distance_transform_edt(
            empty, return_distances=True, return_indices=True
        )
        grid = grid[iy_src, ix_src]
    return grid, x0, y0, cell_m


def height_above_ground(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid: np.ndarray,
    x0: float,
    y0: float,
    cell_m: float,
) -> np.ndarray:
    ny, nx = grid.shape
    ix = np.clip(((x - x0) / cell_m).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - y0) / cell_m).astype(np.int64), 0, ny - 1)
    return z - grid[iy, ix]
