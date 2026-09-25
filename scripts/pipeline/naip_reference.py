"""A vegetation reference for San Francisco, built without looking at the model.

The delivery labels ground and nothing else, and the building footprints say where
roofs are, not where trees are. So "the model calls the park's trees buildings" was an
inference from class shares, not a measurement. This builds the missing reference
from NAIP, the US Department of Agriculture's four-band aerial imagery, which is
public domain and carries a near infrared band.

Vegetation is a point whose NDVI clears a threshold AND that stands more than 2 m
above ground, so lawns and green paint are not trees.

The threshold is fixed from the NDVI histogram alone, by Otsu's method over the
pixels of every area, and written to the artefact before any prediction exists in
this process. This script never opens a prediction file.
tests/integrity/test_blind_reference.py checks that by reading its source, so the
threshold cannot be quietly tuned against the result it is used to score.

Two corrections the imagery needs, both measured elsewhere first:

The imagery and the lidar are not the same day. NAIP here was flown 2022-05-18. The
lidar's own GPS time is read from its points, because the STAC dates for 3DEP
deliveries are wrong and this delivery's time-encoding flag contradicts the values it
carries. The gap is written into the artefact.

A tree in an orthophoto leans away from the camera. nadir measured that displacement
at 0.083 to 0.096 m sideways per metre of height on NAIP. So the vegetation raster is
eroded by the displacement of the tallest canopy scored, and points above that height
are left out, leaving only crown cores where imagery and lidar still agree. The rate
is borrowed from nadir's Austin imagery and not re-measured on San Francisco.

Registration. scripts/pipeline/crs_check.py measured the lidar against NAIP per area, and the
offset is not systematic: it changes within one quarter quad and changes sign between
them. With --registered, each area's imagery is sampled at every point's position
minus that area's measured offset (surrey.registration.naip_offsets), and the reference
is written apart, to data/reference/registered and naip_reference_registered.json. An
area with no usable offset is left unregistered and says so. The offsets come from
intensity and red-band texture at ground level, not from any prediction.

Run:
    python scripts/pipeline/naip_reference.py
    python scripts/pipeline/naip_reference.py --registered
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.registration import naip_offsets, snap_bounds  # noqa: E402

TARGET_PREP = ROOT / "data" / "prep" / "target"
CACHE = ROOT / "data" / "cache" / "naip"
OUT_DIR = ROOT / "data" / "reference"
ARTIFACT = ROOT / "data" / "artifacts" / "naip_reference.json"
CRS_ARTIFACT = ROOT / "data" / "artifacts" / "crs_check.json"

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"
DELIVERY = "CA_SanFrancisco_1_B23"
NAIP_YEAR = "2022"

HAG_MIN_M = 2.0
HEIGHT_CAP_M = 25.0
DISPLACEMENT_M_PER_M = 0.096  # nadir's upper measured rate
ASPRS_NOISE = (7, 18)
GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.UTC)


def retry(fn, what: str, attempts: int = 6):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # network errors, 429, 5xx
            last = e
            time.sleep(2.0 * 2**i)
    raise RuntimeError(f"{what} failed after {attempts} attempts: {last}")


def naip_items(bbox_wgs84) -> list[dict]:
    body = {"collections": ["naip"], "bbox": list(bbox_wgs84), "limit": 100}
    r = retry(lambda: requests.post(STAC, json=body, timeout=60), "NAIP search")
    r.raise_for_status()
    return [f for f in r.json()["features"] if f["properties"].get("naip:year") == NAIP_YEAR]


def cached_image(item: dict) -> Path:
    """Download a whole quarter quad once. Remote windowed reads hang on this machine."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{item['id']}.tif"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    href = item["assets"]["image"]["href"]

    def fetch() -> None:
        url = href
        head = requests.head(url, timeout=60)
        if head.status_code in (401, 403, 409):
            token = requests.get(SAS, timeout=60).json()["token"]
            url = f"{href}?{token}"
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            tmp = path.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.replace(path)

    retry(fetch, f"download {item['id']}")
    return path


def otsu(
    values: np.ndarray, bins: int = 400, lo: float = -1.0, hi: float = 1.0
) -> tuple[float, list]:
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(np.float64)
    total = w.sum()
    cum_w = np.cumsum(w)
    cum_m = np.cumsum(w * centres)
    w0 = cum_w / total
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = cum_m / cum_w
        m1 = (cum_m[-1] - cum_m) / (total - cum_w)
        between = w0 * w1 * (m0 - m1) ** 2
    k = int(np.nanargmax(between[:-1]))
    return float(edges[k + 1]), hist.tolist()


def flight_dates(tile: CachedTile) -> dict:
    """Read the lidar's own GPS time from deep octree nodes over this area."""
    import laspy
    from pyproj import Transformer
    from surrey.ept import EptSource, _get, overlaps_xy

    src = EptSource.open(DELIVERY)
    cube = src.cube
    # The area's full extent, from its origin and size. Two arbitrary points would
    # give a box that need not overlap the area at all.
    ex, ey = tile.extent_m()
    to_wm = Transformer.from_crs(tile.epsg, 3857, always_xy=True)
    x0, y0 = to_wm.transform(float(tile.origin[0]), float(tile.origin[1]))
    x1, y1 = to_wm.transform(float(tile.origin[0] + ex), float(tile.origin[1] + ey))
    aoi = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    nodes = src.walk(keep=lambda k: overlaps_xy(k.bounds(cube), aoi), max_depth=7)
    deep = sorted(nodes, key=lambda k: -k.d)[:3]
    times = []
    flags = set()
    for k in deep:
        las = laspy.read(io.BytesIO(_get(f"{src.base}/ept-data/{k}.laz")))
        times.append(np.asarray(las.gps_time))
        flags.add(int(las.header.global_encoding.gps_time_type))
    t = np.concatenate(times)
    lo, hi = float(np.percentile(t, 1)), float(np.percentile(t, 99))
    week_time = hi <= 604800
    out = {
        "header_time_flag": sorted(flags),
        "magnitude_says": "week" if week_time else "adjusted standard",
        "points_read": int(t.size),
    }
    if not week_time:
        out["first"] = (GPS_EPOCH + dt.timedelta(seconds=lo + 1e9)).strftime("%Y-%m-%d")
        out["last"] = (GPS_EPOCH + dt.timedelta(seconds=hi + 1e9)).strftime("%Y-%m-%d")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--registered",
        action="store_true",
        help="sample NAIP at each point minus its area's measured offset",
    )
    args = ap.parse_args()
    out_dir = OUT_DIR / "registered" if args.registered else OUT_DIR
    if args.out is None:
        args.out = (
            ARTIFACT.with_name("naip_reference_registered.json") if args.registered else ARTIFACT
        )
    offsets = (
        naip_offsets(json.loads(CRS_ARTIFACT.read_text(encoding="utf-8")))
        if args.registered
        else {}
    )

    import rasterio
    from pyproj import Transformer
    from rasterio.merge import merge
    from rasterio.windows import from_bounds
    from scipy.ndimage import binary_erosion

    tiles = [
        CachedTile(p, "sf")
        for p in sorted(TARGET_PREP.iterdir())
        if p.is_dir() and (p / "meta.json").exists()
    ]
    t0 = time.time()

    erode_m = HEIGHT_CAP_M * DISPLACEMENT_M_PER_M
    areas = {}
    ndvi_pool = []

    # ---- pass 1: imagery only. Build every NDVI window, then fix the threshold.
    for tile in tiles:
        idx = np.arange(tile.n)
        ux, uy = tile.utm_xy(idx)
        to_ll = Transformer.from_crs(tile.epsg, 4326, always_xy=True)
        lon0, lat0 = to_ll.transform(float(ux.min()), float(uy.min()))
        lon1, lat1 = to_ll.transform(float(ux.max()), float(uy.max()))
        bbox = (
            min(lon0, lon1) - 0.001,
            min(lat0, lat1) - 0.001,
            max(lon0, lon1) + 0.001,
            max(lat0, lat1) + 0.001,
        )
        items = naip_items(bbox)
        if not items:
            areas[tile.name] = {"skipped": "no 2022 NAIP covers this area"}
            continue
        paths = [cached_image(it) for it in items]
        dss = [rasterio.open(p) for p in paths]
        crs = dss[0].crs
        to_img = Transformer.from_crs(tile.epsg, crs.to_epsg(), always_xy=True)
        # The lidar sits displaced by the offset relative to NAIP, so the imagery that
        # belongs to a point is found at the point minus the offset.
        off = offsets.get(tile.name, {"east_m": 0.0, "north_m": 0.0, "method": "none"})
        ix, iy = to_img.transform(ux - off["east_m"], uy - off["north_m"])
        margin = 20.0
        bounds = snap_bounds(
            (
                float(ix.min()) - margin,
                float(iy.min()) - margin,
                float(ix.max()) + margin,
                float(iy.max()) + margin,
            ),
            dss[0].transform,
        )
        if len(dss) == 1:
            ds = dss[0]
            # Snapped to whole pixels before reading. A fractional window is read from
            # offsets rounded to the nearest pixel while window_transform keeps the
            # fraction, so pixels and coordinates disagreed by up to half a pixel,
            # 0.3 m, found by the planted-shift check in scripts/pipeline/crs_check.py.
            win = from_bounds(*bounds, transform=ds.transform)
            win = win.round_offsets(op="floor").round_lengths(op="ceil")
            arr = ds.read(window=win, boundless=True, fill_value=0).astype(np.float32)
            transform = ds.window_transform(win)
            res = ds.res[0]
        else:
            arr, transform = merge(dss, bounds=bounds, nodata=0)
            arr = arr.astype(np.float32)
            res = dss[0].res[0]
        for d in dss:
            d.close()

        red, nir = arr[0], arr[3]
        valid = arr.sum(axis=0) > 0  # the black collar is not imagery
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.where(valid, (nir - red) / np.maximum(nir + red, 1.0), np.nan)
        ndvi_pool.append(ndvi[valid].ravel())

        cols = np.floor((ix - transform.c) / transform.a).astype(np.int64)
        rows = np.floor((iy - transform.f) / transform.e).astype(np.int64)
        inside = (rows >= 0) & (rows < ndvi.shape[0]) & (cols >= 0) & (cols < ndvi.shape[1])
        areas[tile.name] = {
            "tile": tile,
            "ndvi": ndvi,
            "valid": valid,
            "rows": rows,
            "cols": cols,
            "inside": inside,
            "res_m": float(res),
            "offset": off,
            "items": [
                {"id": it["id"], "acquired": it["properties"]["datetime"][:10]} for it in items
            ],
        }
        print(f"  imagery {tile.name:12s} {len(items)} item(s), window {ndvi.shape}", flush=True)

    threshold, hist = otsu(np.concatenate(ndvi_pool))
    print(f"\nNDVI threshold by Otsu over every area's pixels: {threshold:.4f}", flush=True)

    # ---- pass 2: apply the fixed threshold to the points. Still no predictions.
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    lidar_dates = None
    for name, a in areas.items():
        if "skipped" in a:
            rows_out.append({"aoi": name, **a})
            continue
        tile = a["tile"]
        if lidar_dates is None:
            lidar_dates = flight_dates(tile)
        erode_px = math.ceil(erode_m / a["res_m"])
        veg = np.nan_to_num(a["ndvi"], nan=-1.0) > threshold
        core = binary_erosion(veg, iterations=erode_px) & a["valid"]
        not_veg_core = binary_erosion(~veg & a["valid"], iterations=erode_px)

        r, c, inside = a["rows"], a["cols"], a["inside"]
        in_core = np.zeros(tile.n, dtype=bool)
        in_not = np.zeros(tile.n, dtype=bool)
        ndvi_pt = np.full(tile.n, np.nan, dtype=np.float32)
        in_core[inside] = core[r[inside], c[inside]]
        in_not[inside] = not_veg_core[r[inside], c[inside]]
        ndvi_pt[inside] = a["ndvi"][r[inside], c[inside]]

        asprs = np.asarray(tile.asprs[:], dtype=np.int16)
        hag = np.asarray(tile.hag_cm[:], dtype=np.float32) / 100.0
        clean = ~np.isin(asprs, ASPRS_NOISE)
        eligible = clean & inside & (hag > HAG_MIN_M) & (hag <= HEIGHT_CAP_M)
        vegetation = eligible & in_core
        not_vegetation = eligible & in_not

        np.savez(
            out_dir / f"naip_{name}.npz",
            vegetation=vegetation,
            not_vegetation=not_vegetation,
            eligible=eligible,
            ndvi=ndvi_pt.astype(np.float16),
        )
        rows_out.append(
            {
                "aoi": name,
                "naip_items": a["items"],
                "lidar_minus_naip_offset_applied": a["offset"],
                "image_resolution_m": a["res_m"],
                "erosion_px": erode_px,
                "points": tile.n,
                "points_eligible": int(eligible.sum()),
                "points_vegetation_core": int(vegetation.sum()),
                "points_not_vegetation_core": int(not_vegetation.sum()),
                "vegetation_share_of_eligible": round(
                    float(vegetation.sum() / max(eligible.sum(), 1)), 4
                ),
                "points_outside_imagery": int((~inside).sum()),
            }
        )
        print(
            f"  mask    {name:12s} eligible {eligible.sum():>9,}  "
            f"vegetation core {vegetation.sum():>9,}  not-vegetation core {not_vegetation.sum():>9,}",
            flush=True,
        )

    naip_date = (
        dt.date.fromisoformat(rows_out[0]["naip_items"][0]["acquired"])
        if rows_out and "naip_items" in rows_out[0]
        else None
    )
    gap = None
    if naip_date and lidar_dates and "first" in lidar_dates:
        gap = (dt.date.fromisoformat(lidar_dates["first"]) - naip_date).days

    result = {
        "question": "where is vegetation in San Francisco, independent of the model",
        "reference": "NAIP 2022, four bands, NDVI above a fixed threshold AND more than 2 m above ground",
        "blind": "this script never opens a prediction; see tests/integrity/test_blind_reference.py",
        "registered": bool(args.registered),
        "offsets_from": "data/artifacts/crs_check.json" if args.registered else None,
        "naip_year": NAIP_YEAR,
        "lidar_delivery": DELIVERY,
        "lidar_flight_dates_from_gps_time": lidar_dates,
        "days_from_imagery_to_lidar": gap,
        "threshold_method": "Otsu over the NDVI of every valid pixel in every area, pooled",
        "ndvi_threshold": round(threshold, 4),
        "ndvi_histogram": {"range": [-1.0, 1.0], "bins": len(hist), "counts": hist},
        "height_gate_m": [HAG_MIN_M, HEIGHT_CAP_M],
        "displacement_rate_m_per_m": DISPLACEMENT_M_PER_M,
        "displacement_rate_source": "nadir, measured on Austin NAIP; not re-measured on San Francisco",
        "erosion_m": round(erode_m, 2),
        "height_used": "the label-free quantile floor stored as hag_cm, the same gate as the footprint score",
        "areas": rows_out,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nimagery to lidar: {gap} days   lidar {lidar_dates}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
