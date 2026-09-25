"""The target's own ground class, and how far it can be trusted as a reference.

The San Francisco delivery labels ground (ASPRS class 2) and leaves most of the
rest as class 1, unassigned. scripts/investigations/ground_reference_check.py measured what
unassigned actually is, into data/artifacts/ground_reference_check.json, and the
answer decides what a ground score means. A rule fixed before the numbers were
read allowed at most a tenth of unassigned points to sit at ground level. Every
area failed it.

So unassigned is not a clean negative. The vendor classified some ground and left
other ground unlabelled, which means a ground point the model finds correctly can
be counted as a false positive.

The contamination runs one way only, and that is what makes it survivable. Class 2
is ground, certainly. Unassigned is a mix of ground and not-ground. So:

    recall over class 2 is clean       every point in the reference really is ground
    precision has a floor and a ceiling  strict, and generous, and the truth is between

The floor counts only class 2 as correct, which undercounts. The ceiling also
accepts a predicted ground point sitting on the vendor's own ground surface, which
overcounts, because a kerb, a parked car's underside and low vegetation all sit
there too. Reporting the band is honest where reporting either end alone is not.
"""

from __future__ import annotations

import numpy as np

ASPRS_GROUND = 2
ASPRS_UNASSIGNED = 1
ASPRS_NOISE = (7, 18)
NEAR_GROUND_M = 0.5
SURFACE_CELL_M = 5.0
# How far a ground surface may be carried into cells the vendor left blank.
SURFACE_REACH_M = 25.0

# Figures where the reference itself fails, with the measurement that shows it. A model
# figure against one of these is never reported as the model's error; it stands beside a
# note that the reference does not hold there. The figures behind each reason are in the
# named artefact and reach prose only through the registry.
REFERENCE_NOT_VALID = {
    ("ggpark", "not_vegetation_core_called_vegetation"): (
        "the NAIP not-vegetation core in Golden Gate Park has a multiple-return share typical "
        "of canopy on both masks (naip_mask_choice.json), so NDVI probably misses canopy there, "
        "by shadow or relief displacement, and vegetation called inside it is likely right"
    ),
}


def vendor_ground_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    is_ground: np.ndarray,
    cell_m: float = SURFACE_CELL_M,
    reach_m: float = SURFACE_REACH_M,
):
    """Median z of the delivery's ground class per cell, carried a bounded distance.

    Cells with no ground class take the nearest labelled cell's value, but only
    within `reach_m`. Beyond that the surface stays undefined, because the middle
    of the bay has no ground to interpolate from and inventing one there would
    manufacture a height the delivery never claimed. Callers check coverage and
    report how much of the area they could speak for.
    """
    x0, y0 = float(x.min()), float(y.min())
    nx = max(1, int(np.ceil((x.max() - x0) / cell_m)) + 1)
    ny = max(1, int(np.ceil((y.max() - y0) / cell_m)) + 1)
    ix = np.clip(((x - x0) / cell_m).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - y0) / cell_m).astype(np.int64), 0, ny - 1)
    key = iy * nx + ix

    grid = np.full(nx * ny, np.nan)
    gk, gz = key[is_ground], z[is_ground]
    if gk.size:
        order = np.lexsort((gz, gk))
        ks, zs = gk[order], gz[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        counts = np.diff(np.r_[starts, ks.size])
        grid[ks[starts]] = zs[starts + counts // 2]

    grid = grid.reshape(ny, nx)
    empty = ~np.isfinite(grid)
    if empty.any() and (~empty).any() and reach_m > 0:
        # Fill from the nearest labelled cell, but only within reach. A roof in the
        # middle of a labelled street sits a few metres from real ground and is
        # plainly not ground itself, so declining to judge it throws away an easy
        # call. The middle of the bay is a different matter, and beyond reach the
        # surface stays undefined and the points there are declared unjudgeable.
        from scipy.ndimage import distance_transform_edt

        dist, (sy, sx) = distance_transform_edt(empty, return_distances=True, return_indices=True)
        within = dist <= (reach_m / cell_m)
        filled = grid[sy, sx]
        grid = np.where(empty & within, filled, grid)
    return grid.ravel(), key


def score_ground(
    asprs: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    predicted_ground: np.ndarray,
    cell_m: float = SURFACE_CELL_M,
    reach_m: float = SURFACE_REACH_M,
) -> dict:
    """Ground against the delivery, as a clean recall and a bounded precision."""
    clean = ~np.isin(asprs, ASPRS_NOISE)
    ref = (asprs == ASPRS_GROUND) & clean
    unassigned = (asprs == ASPRS_UNASSIGNED) & clean
    pred = predicted_ground & clean
    if not ref.any():
        return {
            "reference_points": 0,
            "note": "this area carries no ground class, nothing to score against",
        }

    grid, key = vendor_ground_surface(x, y, z, ref, cell_m, reach_m)
    covered = np.isfinite(grid[key])
    height = np.full(asprs.size, np.nan)
    height[covered] = z[covered] - grid[key[covered]]

    # Generous: a predicted ground point standing on the vendor's own ground
    # surface is plausibly ground even where the vendor left it unlabelled.
    plausible = ref | (unassigned & covered & (np.abs(height) <= NEAR_GROUND_M))

    # Precision is only asked where the delivery gave something to ask against. In
    # a cell holding no class 2 at all there is no surface, so a predicted ground
    # point there is neither confirmed nor refuted. Counting it wrong would charge
    # the model for the delivery's silence; counting it right would be worse. It
    # is excluded and declared instead.
    judgeable = covered | ref
    pred_j = pred & judgeable

    hits_strict = int((pred_j & ref).sum())
    hits_generous = int((pred_j & plausible).sum())
    n_pred = int(pred.sum())
    n_judgeable = int(pred_j.sum())

    return {
        "reference": "delivery ASPRS class 2, automated vendor output, not hand labels",
        "reference_points": int(ref.sum()),
        "predicted_points": n_pred,
        "predicted_points_judgeable": n_judgeable,
        "predicted_points_unjudgeable_fraction": round(1.0 - n_judgeable / n_pred, 4)
        if n_pred
        else None,
        "recall_clean": round(hits_strict / int(ref.sum()), 4),
        "precision_floor": round(hits_strict / n_judgeable, 4) if n_judgeable else None,
        "precision_ceiling": round(hits_generous / n_judgeable, 4) if n_judgeable else None,
        "iou_floor": round(hits_strict / max(int((pred | ref).sum()), 1), 4),
        "null_iou_predict_all_ground": round(int(ref.sum()) / max(int(clean.sum()), 1), 4),
        "surface_cell_m": cell_m,
        "surface_reach_m": reach_m,
        "unassigned_at_ground_level": round(
            float((np.abs(height[unassigned & covered]) <= NEAR_GROUND_M).mean()), 4
        )
        if (unassigned & covered).any()
        else None,
        "area_with_no_ground_class": round(float((~covered).mean()), 4),
        "reading": (
            "recall is the clean number: every point in the reference really is "
            "ground. Precision is a band because the vendor left some ground "
            "unlabelled, so the floor undercounts and the ceiling overcounts"
        ),
    }
