"""Do the points labelled water sit on a regular grid? Swiss training tiles, Zug, and San Francisco.

The water inset showed the Zug water points in rows and columns and the Embarcadero ones scattered. That
is a picture of one window. This measures it over every labelled water point, with ground as the control,
because a scanner's own pattern could put any class in rows.

Two readings per set of points, both on the points as prepared (the voxel step included):

- nearest neighbour distance: median, and its coefficient of variation. A lattice has one distance and
  a variation near zero; real returns have a spread.
- on the lattice: the grid step is taken as the most common nearest neighbour distance, rounded to the
  centimetre; each point's x and y are reduced modulo that step, and the share of points within 1 cm
  of the most common remainder in both axes is reported. A lattice puts nearly every point there;
  scattered points put about the chance share, printed beside it.

Every tile is read on its own and whole. Thinning a set to a fixed size by taking every n-th point was
the first version, and it breaks the neighbourhoods a lattice is seen in: it read the Swiss training
water as scattered. A set's figure is the point-weighted pool of its tiles.

Nothing here says why a model does anything. It says what the points are.

Run:
    python scripts/investigations/water_lattice.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.sources import M1_CLASS_NAMES, WATER  # noqa: E402

OUT = ROOT / "data" / "artifacts" / "water_lattice.json"
PREP = ROOT / "data" / "prep"
GROUND_M1 = M1_CLASS_NAMES.index("ground")
ASPRS = {"water": 9, "ground": 2}
TOL_CM = 1.0


def xy_cm(tile: CachedTile, idx: np.ndarray) -> np.ndarray:
    return np.asarray(tile.xy_cm[idx], np.float64)


def reading(xy: np.ndarray) -> dict:
    if len(xy) < 100:
        return {"points": int(len(xy)), "note": "too few points to read"}
    d, _ = cKDTree(xy).query(xy, k=2)
    nn = d[:, 1]
    nn = nn[nn > 0]
    step = int(np.bincount(np.round(nn).astype(int)).argmax())
    out = {
        "points": int(len(xy)),
        "nn_median_cm": round(float(np.median(nn)), 2),
        "nn_cv": round(float(nn.std() / nn.mean()), 4),
        "step_cm": step,
    }
    if step >= 2:
        on = np.ones(len(xy), bool)
        for axis in (0, 1):
            r = np.mod(xy[:, axis], step)
            mode = np.bincount(np.round(r).astype(int) % step).argmax()
            gap = np.minimum(np.abs(r - mode), step - np.abs(r - mode))
            on &= gap <= TOL_CM
        out["share_on_the_lattice"] = round(float(on.mean()), 4)
        out["chance_share"] = round(float(min(1.0, (2 * TOL_CM + 1) / step) ** 2), 4)
    return out


def pooled(tiles: list[CachedTile], pick) -> dict:
    """Each tile read whole and on its own, then pooled, weighted by points."""
    per_tile = {}
    for t in tiles:
        idx = np.nonzero(pick(t))[0]
        per_tile[t.name] = reading(xy_cm(t, idx))
    read = [r for r in per_tile.values() if "nn_cv" in r]
    n = sum(r["points"] for r in read)
    if not n:
        return {"points": 0, "per_tile": per_tile}
    out = {
        "points": n,
        "tiles_read": len(read),
        "nn_median_cm_weighted": round(sum(r["nn_median_cm"] * r["points"] for r in read) / n, 2),
        "nn_cv_weighted": round(sum(r["nn_cv"] * r["points"] for r in read) / n, 4),
    }
    on = [r for r in read if "share_on_the_lattice" in r]
    if on:
        m = sum(r["points"] for r in on)
        out["share_on_the_lattice"] = round(
            sum(r["share_on_the_lattice"] * r["points"] for r in on) / m, 4
        )
        out["chance_share"] = round(sum(r["chance_share"] * r["points"] for r in on) / m, 4)
    out["per_tile"] = per_tile
    return out


def main() -> int:
    swiss = [CachedTile(p, "train") for p in sorted((PREP / "swiss").iterdir()) if p.is_dir()]
    zug = [CachedTile(p, "train") for p in sorted((PREP / "swiss_holdout").iterdir()) if p.is_dir()]
    sf = {a: CachedTile(PREP / "target" / a, "sf") for a in ("embarcadero", "baybridge")}
    rows = {}
    for name, tiles in (("swiss_training", swiss), ("zug_held_out", zug)):
        for cls, cid in (("water", WATER), ("ground", GROUND_M1)):
            rows[f"{name}.{cls}"] = pooled(tiles, lambda t, c=cid: np.asarray(t.label[:]) == c)
            print(
                name,
                cls,
                {k: v for k, v in rows[f"{name}.{cls}"].items() if k != "per_tile"},
                flush=True,
            )
    for area, tile in sf.items():
        asprs = np.load(PREP / "target" / area / "asprs.npy", mmap_mode="r")
        for cls, code in ASPRS.items():
            rows[f"{area}.{cls}"] = pooled([tile], lambda t, c=code, a=asprs: np.asarray(a) == c)
            print(
                area,
                cls,
                {k: v for k, v in rows[f"{area}.{cls}"].items() if k != "per_tile"},
                flush=True,
            )
    doc = {
        "what": "whether the points labelled water, and ground as the control, sit on a regular grid",
        "labels": "swisstopo's classes for the Swiss tiles, the delivery's ASPRS codes for San Francisco",
        "tolerance_cm": TOL_CM,
        "sets": rows,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
