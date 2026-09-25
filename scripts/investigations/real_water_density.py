"""How densely real water is measured: the Swiss tiles without their synthetic points, against the
water San Francisco's delivery labels, per area. No model is involved.

swisstopo fills lakes with synthetic points on a regular grid (data/artifacts/las_flags_audit.json).
With those dropped, what remains is the water the scanner actually returned. This counts it the same
way on both sides, on the source files and on the prepared tiles the model reads:

    a 40 m block, on the grid of the tile's own origin, is a water block when at least half of its
    points are water (class 9, and on the Swiss side not flagged synthetic); its density is its
    water points over 1600 square metres

Per area: the number of water blocks, the water points in them, and the 10th, 50th and 90th
percentile of their density. The Swiss areas are the four training tiles and the held-out Zug tile;
the San Francisco areas are the two whose delivery labels water, Embarcadero and Bay Bridge.

Writes data/artifacts/real_water_density.json.

Run:
    python scripts/investigations/real_water_density.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.flags import flag_bits, kept  # noqa: E402
from surrey.sources import WATER  # noqa: E402

BLOCK_M = 40.0
SWISS = {
    "zurich 2683-1246": "2683-1246",
    "zurich 2683-1243": "2683-1243",
    "geneva 2500-1118": "2500-1118",
    "geneva 2500-1119": "2500-1119",
    "zug 2680-1225, held out": "2680-1225",
}
SF = ("embarcadero", "baybridge")
OUT = ROOT / "data" / "artifacts" / "real_water_density.json"


def water_blocks(x: np.ndarray, y: np.ndarray, water: np.ndarray) -> dict:
    bx = np.floor((x - x.min()) / BLOCK_M).astype(np.int64)
    by = np.floor((y - y.min()) / BLOCK_M).astype(np.int64)
    key = bx * 100_000 + by
    _, inv, n_all = np.unique(key, return_inverse=True, return_counts=True)
    n_water = np.bincount(inv, weights=water.astype(np.float64), minlength=n_all.size)
    majority = n_water / n_all >= 0.5
    dens = n_water[majority] / (BLOCK_M * BLOCK_M)
    return {
        "water_blocks": int(majority.sum()),
        "water_points_in_them": int(n_water[majority].sum()),
        "density_p10_p50_p90": [round(float(v), 2) for v in np.percentile(dens, [10, 50, 90])]
        if dens.size
        else None,
    }


def swiss_source(tile_id: str) -> dict:
    import laspy

    path = next((ROOT / "data" / "cache" / "swiss").glob(f"*{tile_id}*.laz"))
    las = laspy.read(path)
    use = kept(flag_bits(las))
    x, y = np.asarray(las.x)[use], np.asarray(las.y)[use]
    return water_blocks(x, y, np.asarray(las.classification)[use] == 9)


def swiss_prepared(tile_id: str) -> dict:
    xs, ys, ws = [], [], []
    for prep in ("swiss", "swiss_holdout"):
        for d in sorted((ROOT / "data" / "prep" / prep).glob(f"*{tile_id}_*")):
            t = CachedTile(d)
            ux, uy = t.utm_xy(np.arange(t.n))
            xs.append(ux)
            ys.append(uy)
            ws.append(np.asarray(t.label) == WATER)
    return water_blocks(np.concatenate(xs), np.concatenate(ys), np.concatenate(ws))


def sf_source(area: str) -> dict:
    import laspy

    las = laspy.read(ROOT / "data" / "target" / "sf" / f"{area}.laz")
    use = kept(flag_bits(las))
    x, y = np.asarray(las.x)[use], np.asarray(las.y)[use]
    return water_blocks(x, y, np.asarray(las.classification)[use] == 9)


def sf_prepared(area: str) -> dict:
    t = CachedTile(ROOT / "data" / "prep" / "target" / area)
    ux, uy = t.utm_xy(np.arange(t.n))
    return water_blocks(ux, uy, np.asarray(t.asprs) == 9)


def main() -> int:
    doc = {
        "what": "real water, measured returns only, per 40 m block where water is the majority",
        "block_m": BLOCK_M,
        "swiss": {},
        "san_francisco": {},
    }
    for name, tile_id in SWISS.items():
        doc["swiss"][name] = {"source": swiss_source(tile_id), "prepared": swiss_prepared(tile_id)}
        print(name, doc["swiss"][name], flush=True)
    for area in SF:
        doc["san_francisco"][area] = {"source": sf_source(area), "prepared": sf_prepared(area)}
        print(area, doc["san_francisco"][area], flush=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
