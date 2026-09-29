"""Pull an area of interest out of a USGS 3DEP EPT octree and write true-metre LAZ.

Resumable: every octree node is cached under data/cache/ept/, so a re-run after
an interruption re-downloads nothing.

The output is reprojected from the delivery's EPSG:3857 to a UTM zone, because
Web Mercator's horizontal unit is not a metre. See backend/surrey/ept.py for why
that decides whether the transfer result means anything.

Run:
    python scripts/pipeline/fetch_sf_aoi.py --aoi soma
    python scripts/pipeline/fetch_sf_aoi.py --aoi all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.ept import EptSource, NodeKey, overlaps_xy, true_metre_epsg  # noqa: E402
from surrey.flags import BITS  # noqa: E402

DELIVERY = "CA_SanFrancisco_1_B23"

# Five morphologies, so a transfer number is never read off one patch of city.
# Chosen before any model existed, to keep the area of interest from being
# picked for a result.
AOIS = {
    "financial": (37.7935, -122.3985, "high rise towers, deep street canyons"),
    "soma": (37.7785, -122.4056, "mixed mid rise, wide streets, parking"),
    "mission": (37.7599, -122.4148, "dense low rise rowhouses on a grid"),
    "sunset": (37.7500, -122.4900, "uniform two storey grid, small gardens"),
    "ggpark": (37.7694, -122.4862, "tree canopy, few buildings"),
    # Added after the first five, to test a taxonomy mismatch rather than a
    # domain gap. The delivery labels water and bridge deck. DALES-2 has neither
    # class, so the model cannot name either one and has to call them something.
    # These are candidates: which ones actually carry those labels is measured,
    # not assumed, and the ones that do not are dropped.
    "channel": (37.7740, -122.3920, "Mission Creek channel and its bridges"),
    "embarcadero": (37.7905, -122.3875, "bay shoreline, piers and open water"),
    "baybridge": (37.7955, -122.3810, "Bay Bridge deck over water"),
    # Added last, to answer a question the first seven could not. The measurement
    # of slope leaking into the vertical origin came back saying San Francisco was
    # no worse than Surrey, but none of the first seven areas sat on the steep
    # streets. This is Filbert between Hyde and Leavenworth, about a 31 percent
    # grade, which is among the steepest in the city.
    "steep": (37.7998, -122.4190, "Russian Hill, one of the steepest grades in the city"),
}

# Measured and dropped. `channel` came back 0.04 percent water and 0.71 percent
# bridge deck, which is too little of either to say anything about what the model
# does over them. It stays in the list above so the measurement that rejected it
# can be repeated, and it is excluded from scoring rather than quietly deleted.
EXCLUDED = {
    "channel": "0.04% water and 0.71% bridge deck, too little of either to measure",
}
DEFAULT_SIZE_M = 300.0


def fetch_aoi(name: str, size_m: float, cache: Path, out_dir: Path, workers: int = 8) -> dict:
    import laspy
    from pyproj import Transformer

    lat, lon, blurb = AOIS[name]
    src = EptSource.open(DELIVERY)
    cube = src.cube

    to_wm = Transformer.from_crs(4326, 3857, always_xy=True)
    cx, cy = to_wm.transform(lon, lat)

    # One Web Mercator unit is cos(latitude) ground metres.
    unit_m = math.cos(math.radians(lat))
    half_wm = (size_m / 2.0) / unit_m
    aoi = (cx - half_wm, cy - half_wm, cx + half_wm, cy + half_wm)

    t0 = time.time()
    nodes = src.walk(keep=lambda k: overlaps_xy(k.bounds(cube), aoi))
    walk_s = time.time() - t0

    cache_dir = cache / DELIVERY
    cache_dir.mkdir(parents=True, exist_ok=True)

    def load(key: NodeKey) -> np.ndarray:
        blob = cache_dir / f"{key}.npy"
        arr = np.load(blob) if blob.exists() else None
        # A node cached before the reader kept the LAS flags is read again.
        if arr is None or "flags" not in arr.dtype.names:
            arr = src.node_points(key)
            np.save(blob, arr)
        m = (arr["x"] >= aoi[0]) & (arr["x"] < aoi[2]) & (arr["y"] >= aoi[1]) & (arr["y"] < aoi[3])
        return arr[m]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(load, nodes.keys()))
    fetch_s = time.time() - t0

    pts = np.concatenate([p for p in parts if len(p)]) if parts else np.empty(0)
    if len(pts) == 0:
        raise SystemExit(f"{name}: no points inside the area of interest")

    epsg_out = true_metre_epsg(lon, lat)
    to_utm = Transformer.from_crs(3857, epsg_out, always_xy=True)
    ux, uy = to_utm.transform(pts["x"], pts["y"])

    area_m2 = size_m * size_m
    classes, counts = np.unique(pts["classification"], return_counts=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_laz = out_dir / f"{name}.laz"

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.offsets = np.array([ux.min(), uy.min(), pts["z"].min()])
    hdr.scales = np.array([0.001, 0.001, 0.001])
    # Write the CRS into the file. DALES ships without one and that cost real
    # time; there is no reason to hand the next step the same puzzle.
    from pyproj import CRS as PyCRS

    hdr.add_crs(PyCRS.from_epsg(epsg_out))
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = ux, uy, pts["z"]
    las.intensity = pts["intensity"]
    las.classification = pts["classification"]
    las.return_number = pts["return_number"]
    las.number_of_returns = pts["number_of_returns"]
    for flag, bit in BITS.items():
        setattr(las, flag, (pts["flags"] & bit) != 0)
    las.write(out_laz)

    rec = {
        "aoi": name,
        "description": blurb,
        "delivery": DELIVERY,
        "centre_wgs84": {"lat": lat, "lon": lon},
        "size_m": size_m,
        "source_epsg": src.epsg,
        "output_epsg": epsg_out,
        "web_mercator_unit_in_metres": round(unit_m, 5),
        "nodes_walked": len(nodes),
        "node_points_available": int(sum(nodes.values())),
        "points_in_aoi": int(len(pts)),
        "density_pts_per_m2": round(len(pts) / area_m2, 1),
        "z_min_m": round(float(pts["z"].min()), 2),
        "z_max_m": round(float(pts["z"].max()), 2),
        "classification_histogram": {
            str(int(c)): int(n) for c, n in zip(classes, counts, strict=True)
        },
        "classification_fraction": {
            str(int(c)): round(float(n / len(pts)), 5) for c, n in zip(classes, counts, strict=True)
        },
        "walk_seconds": round(walk_s, 1),
        "fetch_seconds": round(fetch_s, 1),
        "output": str(out_laz.relative_to(ROOT)).replace("\\", "/"),
        "output_bytes": out_laz.stat().st_size,
    }
    print(
        f"{name:10s} {len(pts):>9,} pts  {rec['density_pts_per_m2']:>6.1f} pts/m2  "
        f"z {rec['z_min_m']:>7.2f}..{rec['z_max_m']:<7.2f}  "
        f"{len(nodes):>3} nodes  {rec['output_bytes'] / 1e6:.1f} MB"
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aoi", default="all", help=f"one of {sorted(AOIS)} or 'all'")
    ap.add_argument("--size-m", type=float, default=DEFAULT_SIZE_M)
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "cache" / "ept")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "target" / "sf")
    ap.add_argument("--artifact", type=Path, default=ROOT / "data" / "artifacts" / "sf_aois.json")
    args = ap.parse_args()

    names = sorted(AOIS) if args.aoi == "all" else [args.aoi]
    for n in names:
        if n not in AOIS:
            raise SystemExit(f"unknown aoi {n!r}, have {sorted(AOIS)}")

    records = [fetch_aoi(n, args.size_m, args.cache, args.out) for n in names]

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if args.artifact.exists():
        prior = {r["aoi"]: r for r in json.loads(args.artifact.read_text(encoding="utf-8"))["aois"]}
    for r in records:
        prior[r["aoi"]] = r
    args.artifact.write_text(
        json.dumps({"delivery": DELIVERY, "aois": [prior[k] for k in sorted(prior)]}, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
