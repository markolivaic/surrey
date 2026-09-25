"""Scoring a building prediction against Microsoft Building Footprints.

The target delivery has no building labels, so footprints are the only reference
available for that class. They are a weak one, and the weakness is measurable:
a footprint is a 2D polygon, so its interior holds the ground inside a courtyard,
a car in a driveway and a tree overhanging a roof. Half the points in a dense
block fall inside one regardless of what they are.

So every number this module produces comes with its null: the score a model gets
by calling every point a building. A precision of 0.8 against a null of 0.7 is
almost nothing. Against a null of 0.001 it is the whole result.

Microsoft Building Footprints, ODbL-1.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_footprints(path: Path, epsg: int) -> list:
    """Footprint polygons reprojected from WGS84 into the given CRS."""
    import shapely
    from pyproj import Transformer
    from shapely.geometry import shape

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    geoms = [shape(f["geometry"]) for f in doc["features"] if f.get("geometry")]
    tf = Transformer.from_crs(4326, epsg, always_xy=True)

    def to_target(arr: np.ndarray) -> np.ndarray:
        x, y = tf.transform(arr[:, 0], arr[:, 1])
        return np.column_stack([x, y])

    return [shapely.transform(g, to_target) for g in geoms]


def clip_to_box(polys: list, x: np.ndarray, y: np.ndarray) -> tuple[list, float]:
    """Footprints clipped to the bounding box of the given points."""
    import shapely

    box = shapely.box(x.min(), y.min(), x.max(), y.max())
    local = [p for p in polys if p.intersects(box)]
    return [shapely.intersection(p, box) for p in local], float(box.area)


def inside_mask(clipped: list, x: np.ndarray, y: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """Boolean per point: does it fall inside any footprint."""
    import shapely

    if not clipped:
        return np.zeros(x.size, dtype=bool)
    union = shapely.union_all(clipped)
    shapely.prepare(union)
    out = np.zeros(x.size, dtype=bool)
    for i in range(0, x.size, chunk):
        j = min(i + chunk, x.size)
        out[i:j] = shapely.contains_xy(union, x[i:j], y[i:j])
    return out


def footprints_hit(
    clipped: list, x: np.ndarray, y: np.ndarray, selected: np.ndarray, chunk: int = 400_000
) -> np.ndarray:
    """Boolean per footprint: does any selected point fall within it."""
    import shapely

    hit = np.zeros(len(clipped), dtype=bool)
    if not clipped:
        return hit
    tree = shapely.STRtree(clipped)
    for i in range(0, x.size, chunk):
        j = min(i + chunk, x.size)
        pts = shapely.points(x[i:j], y[i:j])
        idx_pt, idx_poly = tree.query(pts, predicate="within")
        if idx_poly.size:
            keep = selected[i:j][idx_pt]
            if keep.any():
                hit[np.unique(idx_poly[keep])] = True
    return hit


def score_building(
    clipped: list,
    x: np.ndarray,
    y: np.ndarray,
    predicted_building: np.ndarray,
    eligible: np.ndarray | None = None,
) -> dict:
    """Precision and recall for a building prediction, each beside its null.

    `eligible` restricts the population, for example to points above 2 m. The
    null is computed over the same population, so the pair stays comparable.
    """
    if eligible is None:
        eligible = np.ones(x.size, dtype=bool)

    inside = inside_mask(clipped, x, y)
    pred = predicted_building & eligible

    null_precision = float(inside[eligible].mean()) if eligible.any() else float("nan")
    precision = float(inside[pred].mean()) if pred.any() else float("nan")

    hit_pred = footprints_hit(clipped, x, y, pred)
    hit_all = footprints_hit(clipped, x, y, eligible)
    recall = float(hit_pred.mean()) if len(clipped) else float("nan")
    null_recall = float(hit_all.mean()) if len(clipped) else float("nan")

    # How much of the room above the null the model actually took. Negative means
    # it did worse than calling everything a building.
    room = 1.0 - null_precision
    lift = (
        (precision - null_precision) / room
        if room > 1e-9 and precision == precision
        else float("nan")
    )

    return {
        "footprints": len(clipped),
        "eligible_points": int(eligible.sum()),
        "predicted_building_points": int(pred.sum()),
        "predicted_building_fraction": round(float(pred.sum() / max(eligible.sum(), 1)), 4),
        "precision": None if precision != precision else round(precision, 4),
        "null_precision": None if null_precision != null_precision else round(null_precision, 4),
        "precision_lift_over_null": None if lift != lift else round(lift, 4),
        "recall": None if recall != recall else round(recall, 4),
        "null_recall": None if null_recall != null_recall else round(null_recall, 4),
    }


def footprint_coverage(
    clipped: list,
    x: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    cell_m: float = 1.0,
    min_cells: int = 4,
) -> dict:
    """How much of each footprint's area actually receives selected points.

    Counting a footprint as recognised because one point landed inside it gives
    every building full marks, since every building has something above two
    metres on it. So this measures area instead: the footprint is divided into
    cells, and coverage is the fraction of its cells that contain at least one
    selected point. That separates touching a building from finding one.

    Coverage has a high null by construction, because lidar covers the whole
    footprint whatever the labels say. It is therefore a one-sided instrument: it
    catches a model that under-covers, and precision catches one that
    over-predicts. The null is returned alongside rather than described, so the
    asymmetry is visible instead of assumed.

    Footprints smaller than `min_cells` cells are reported separately. Coverage
    over three cells is a ratio of small integers and its median would be noise.
    """
    import shapely

    out: dict[str, Any] = {
        "cell_m": cell_m,
        "min_cells": min_cells,
        "footprints": len(clipped),
        "footprints_scored": 0,
        "footprints_too_small": 0,
        "coverage_median": None,
        "coverage_mean": None,
        "coverage_p10": None,
        "coverage_p90": None,
        "per_footprint": [],
    }
    if not clipped or x.size == 0:
        return out

    # The grid has to cover the footprints, not just the points. Deriving it from
    # the point bounding box alone clipped any footprint reaching past the points
    # down to a single cell, and a lone point inside it then scored full coverage.
    bounds = np.array([p.bounds for p in clipped if not p.is_empty], dtype=float)
    if bounds.size == 0:
        return out
    x0 = min(float(x.min()), float(bounds[:, 0].min()))
    y0 = min(float(y.min()), float(bounds[:, 1].min()))
    x1 = max(float(x.max()), float(bounds[:, 2].max()))
    y1 = max(float(y.max()), float(bounds[:, 3].max()))
    nx = int(np.ceil((x1 - x0) / cell_m)) + 1
    ny = int(np.ceil((y1 - y0) / cell_m)) + 1

    sel = np.flatnonzero(selected)
    marked = np.zeros(nx * ny, dtype=bool)
    if sel.size:
        ix = np.clip(((x[sel] - x0) / cell_m).astype(np.int64), 0, nx - 1)
        iy = np.clip(((y[sel] - y0) / cell_m).astype(np.int64), 0, ny - 1)
        marked[iy * nx + ix] = True

    scores = []
    for poly in clipped:
        if poly.is_empty:
            continue
        bx0, by0, bx1, by1 = poly.bounds
        i0 = max(int((bx0 - x0) / cell_m), 0)
        i1 = min(int((bx1 - x0) / cell_m) + 1, nx - 1)
        j0 = max(int((by0 - y0) / cell_m), 0)
        j1 = min(int((by1 - y0) / cell_m) + 1, ny - 1)
        if i1 < i0 or j1 < j0:
            continue

        gi, gj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
        gi, gj = gi.ravel(), gj.ravel()
        cx = x0 + (gi + 0.5) * cell_m
        cy = y0 + (gj + 0.5) * cell_m
        shapely.prepare(poly)
        inside = shapely.contains_xy(poly, cx, cy)
        total = int(inside.sum())
        if total < min_cells:
            out["footprints_too_small"] += 1
            continue
        hit = int(marked[(gj[inside]) * nx + (gi[inside])].sum())
        scores.append(hit / total)

    if scores:
        arr = np.asarray(scores)
        out["footprints_scored"] = int(arr.size)
        out["coverage_median"] = round(float(np.median(arr)), 4)
        out["coverage_mean"] = round(float(arr.mean()), 4)
        out["coverage_p10"] = round(float(np.percentile(arr, 10)), 4)
        out["coverage_p90"] = round(float(np.percentile(arr, 90)), 4)
        out["per_footprint"] = [round(float(v), 4) for v in arr]
    return out


def coverage_with_null(
    clipped: list,
    x: np.ndarray,
    y: np.ndarray,
    predicted: np.ndarray,
    eligible: np.ndarray,
    cell_m: float = 1.0,
    min_cells: int = 4,
) -> dict:
    """Coverage and its null in one value, so neither can be reported alone.

    Coverage saturates: selecting every eligible point reaches above 0.95, because
    lidar covers a footprint whatever the labels say. A coverage of 0.97 quoted on
    its own is therefore not a result, it is noise with a decimal point. The only
    honest form is the pair, so this returns the pair and the gap between them and
    there is no call that hands back one half.
    """
    got = footprint_coverage(
        clipped, x, y, predicted & eligible, cell_m=cell_m, min_cells=min_cells
    )
    null = footprint_coverage(clipped, x, y, eligible, cell_m=cell_m, min_cells=min_cells)
    got.pop("per_footprint", None)
    null.pop("per_footprint", None)

    med, nmed = got["coverage_median"], null["coverage_median"]
    shortfall = None
    if med is not None and nmed is not None:
        # How much of the reachable area the model failed to reach. Zero means it
        # covered everything the lidar did, one means it covered none of it.
        shortfall = round(1.0 - (med / nmed), 4) if nmed > 1e-9 else None

    return {
        "cell_m": cell_m,
        "min_cells": min_cells,
        "footprints_scored": got["footprints_scored"],
        "footprints_too_small": got["footprints_too_small"],
        "coverage_median": med,
        "coverage_p10": got["coverage_p10"],
        "coverage_p90": got["coverage_p90"],
        "null_coverage_median": nmed,
        "null_coverage_p10": null["coverage_p10"],
        "shortfall_against_null": shortfall,
        "reading": (
            "null_coverage is what selecting every eligible point reaches. "
            "coverage_median alone means nothing; the gap is the measurement"
        ),
    }
