"""Build the physical water null for Embarcadero and Bay Bridge. No model involved.

The rule, the margin and its reasons are in backend/surrey/water_null.py and were
written before this ran. This fetches what the rule needs and applies it:

    1. every EPT point over each area, with its GpsTime, matched to the cached point by
       position (a cached point is a voxel pick of an EPT point, so a match is exact to
       the centimetre the cache stores)
    2. NOAA CO-OPS 9414290 verified six minute water level over the flight, in NAVD88
    3. a check that the EPT heights are the source heights: each source file's z range
       from its own header against its bounds in the EPT source list
    4. per point: time, gauge level at that time, and the null at the margin and at the
       two sensitivity margins

Writes data/reference/water_null/<area>.npz and data/artifacts/water_null.json. The
artefact holds the null's own precision, recall and IoU against delivery water; no
model is scored here.

Run:
    python scripts/pipeline/water_null.py
"""

from __future__ import annotations

import datetime as dt
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from surrey.cache import CachedTile  # noqa: E402
from surrey.ept import EptSource, _get, overlaps_xy  # noqa: E402
from surrey.water_null import (  # noqa: E402
    AREAS,
    ASPRS_NOISE,
    ASPRS_WATER,
    DATUM,
    MARGIN_M,
    PRODUCT,
    SENSITIVITY_MARGINS_M,
    STATION_ID,
    STATION_NAME,
    as_null,
    gps_adjusted_to_utc,
    level_at,
    null_water,
    water_scores,
)

DELIVERY = "CA_SanFrancisco_1_B23"
TARGET_PREP = ROOT / "data" / "prep" / "target"
GPS_CACHE = ROOT / "data" / "cache" / "ept_gps" / DELIVERY
NOAA_CACHE = ROOT / "data" / "cache" / "noaa"
OUT = ROOT / "data" / "reference" / "water_null"
ARTIFACT = ROOT / "data" / "artifacts" / "water_null.json"
MATCH_TOLERANCE_M = 0.03
API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def node_with_time(src: EptSource, key) -> np.ndarray:
    blob = GPS_CACHE / f"{key}.npy"
    if blob.exists():
        return np.load(blob)
    import laspy

    las = laspy.read(io.BytesIO(_get(f"{src.base}/ept-data/{key}.laz")))
    out = np.empty(len(las.points), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("t", "f8")])
    out["x"], out["y"], out["z"], out["t"] = las.x, las.y, las.z, las.gps_time
    GPS_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(blob, out)
    return out


def gauge(begin: dt.datetime, end: dt.datetime) -> tuple[np.ndarray, np.ndarray, dict]:
    params = {
        "product": PRODUCT,
        "application": "surrey_research",
        "station": STATION_ID,
        "begin_date": begin.strftime("%Y%m%d %H:%M"),
        "end_date": end.strftime("%Y%m%d %H:%M"),
        "datum": DATUM,
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
    }
    NOAA_CACHE.mkdir(parents=True, exist_ok=True)
    cache = NOAA_CACHE / f"{STATION_ID}_{PRODUCT}_{DATUM}_{begin:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.json"
    if cache.exists():
        body = json.loads(cache.read_text(encoding="utf-8"))
    else:
        url = f"{API}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "surrey-research"})
        body = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
        if "error" in body:
            raise SystemExit(f"NOAA: {body['error']}")
        cache.write_text(json.dumps(body) + "\n", encoding="utf-8", newline="\n")
    rows = [r for r in body["data"] if r.get("v") not in ("", None)]
    t = np.array(
        [
            dt.datetime.strptime(r["t"], "%Y-%m-%d %H:%M").replace(tzinfo=dt.UTC).timestamp()
            for r in rows
        ]
    )
    v = np.array([float(r["v"]) for r in rows])
    quality = sorted({r.get("q") for r in rows})
    return (
        t,
        v,
        {
            "request": {k: v for k, v in params.items() if k != "application"},
            "readings": len(rows),
            "quality_flags": quality,
            "metadata": body.get("metadata"),
        },
    )


def vertical_check(limit: int = 12) -> dict:
    """Source header z range against the file's bounds in the EPT source record, for the
    same evenly spread files scripts/pipeline/datum_check.py reads."""
    import requests
    from datum_check import BUCKET, list_sources

    rows = []
    for key in list_sources(limit):
        rec = requests.get(f"{BUCKET}/{key}", timeout=60).json()
        las = rec["metadata"]["readers.las"]
        b = rec["bounds"]
        rows.append(
            {
                "file": key.rsplit("/", 1)[-1],
                "vertical_crs": las["srs"]["vertical"].split('"')[1],
                "header_minz_minus_ept": round(float(las["minz"]) - float(b[2]), 3),
                "header_maxz_minus_ept": round(float(las["maxz"]) - float(b[5]), 3),
            }
        )
        time.sleep(0.2)
    worst = max(max(abs(r["header_minz_minus_ept"]), abs(r["header_maxz_minus_ept"])) for r in rows)
    return {
        "files": rows,
        "vertical_crs": sorted({r["vertical_crs"] for r in rows}),
        "worst_abs_z_difference_m": worst,
        "reading": "EPT heights are the source NAVD88 heights"
        if worst <= 0.01
        else "EPT heights differ from the source heights",
    }


def _quantiles(v: np.ndarray) -> dict:
    return {f"p{q}": round(float(np.percentile(v, q)), 3) for q in (1, 50, 99)} if v.size else {}


def _by_class(mask: np.ndarray, asprs: np.ndarray, rel: np.ndarray) -> dict:
    """What the null calls water that the delivery does not, by delivery class, with the
    height of those points relative to the gauge level."""
    out = {}
    for c in np.unique(asprs[mask]).tolist():
        m = mask & (asprs == c)
        out[str(c)] = {"points": int(m.sum()), "height_minus_level_m": _quantiles(rel[m])}
    return out


def main() -> int:
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    t0 = time.time()
    src = EptSource.open(DELIVERY)
    cube = src.cube
    result = {
        "rule": "backend/surrey/water_null.py, written before this ran",
        "station": {
            "id": STATION_ID,
            "name": STATION_NAME,
            "product": PRODUCT,
            "datum": DATUM,
            "licence": "public domain (NOAA NOS); attribution requested",
        },
        "margin_m": MARGIN_M,
        "sensitivity_margins_m": list(SENSITIVITY_MARGINS_M),
        "vertical_check": vertical_check(),
        "areas": [],
    }
    print(
        f"vertical check: {result['vertical_check']['reading']}, worst {result['vertical_check']['worst_abs_z_difference_m']}"
    )

    per_area = {}
    for aoi in AREAS:
        tile = CachedTile(TARGET_PREP / aoi, "sf")
        ex, ey = tile.extent_m()
        to_wm = Transformer.from_crs(tile.epsg, 3857, always_xy=True)
        xs = [float(tile.origin[0]) - 5, float(tile.origin[0]) + ex + 5]
        ys = [float(tile.origin[1]) - 5, float(tile.origin[1]) + ey + 5]
        cx, cy = to_wm.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
        box = (min(cx), min(cy), max(cx), max(cy))
        nodes = src.walk(keep=lambda k, b=box: overlaps_xy(k.bounds(cube), b))
        with ThreadPoolExecutor(max_workers=6) as pool:
            parts = list(pool.map(lambda k: node_with_time(src, k), nodes))
        pts = np.concatenate([p for p in parts if len(p)])
        m = (pts["x"] >= box[0]) & (pts["x"] < box[2]) & (pts["y"] >= box[1]) & (pts["y"] < box[3])
        pts = pts[m]
        to_utm = Transformer.from_crs(3857, tile.epsg, always_xy=True)
        ux, uy = to_utm.transform(pts["x"], pts["y"])
        tree = cKDTree(
            np.column_stack([ux - tile.origin[0], uy - tile.origin[1], pts["z"] - tile.origin[2]])
        )
        idx = np.arange(tile.n)
        q = np.column_stack(
            [
                np.asarray(tile.xy_cm[:, 0], np.float64) / 100.0,
                np.asarray(tile.xy_cm[:, 1], np.float64) / 100.0,
                np.asarray(tile.z_cm[:], np.float64) / 100.0,
            ]
        )
        dist, nn = tree.query(q, k=1, workers=2)
        matched = dist <= MATCH_TOLERANCE_M
        times = gps_adjusted_to_utc(pts["t"][nn])
        per_area[aoi] = (tile, idx, times, matched, dist)
        print(
            f"  {aoi}: {len(nodes)} nodes, {len(pts):,} EPT points, {matched.mean():.5f} of cached points "
            f"matched within {MATCH_TOLERANCE_M} m, {time.time() - t0:.0f} s",
            flush=True,
        )

    all_t = np.concatenate([v[2][v[3]] for v in per_area.values()])
    begin = dt.datetime.fromtimestamp(all_t.min(), dt.UTC).replace(
        second=0, microsecond=0
    ) - dt.timedelta(minutes=30)
    end = dt.datetime.fromtimestamp(all_t.max(), dt.UTC).replace(
        second=0, microsecond=0
    ) + dt.timedelta(minutes=30)
    gt, gl, gmeta = gauge(begin, end)
    result["gauge"] = gmeta | {"first_utc": begin.isoformat(), "last_utc": end.isoformat()}

    OUT.mkdir(parents=True, exist_ok=True)
    for aoi, (tile, _idx, times, matched, dist) in per_area.items():
        unmatched = int((~matched).sum())
        if unmatched > 0.001 * tile.n:
            raise SystemExit(
                f"{aoi}: {unmatched} cached points have no EPT point within {MATCH_TOLERANCE_M} m"
            )
        level = level_at(times, gt, gl)
        z = tile.z_abs(np.arange(tile.n))
        asprs = np.asarray(tile.asprs[:], np.int16)
        clean = ~np.isin(asprs, ASPRS_NOISE)
        truth = asprs == ASPRS_WATER
        nulls = {
            f"null_{str(mm).replace('.', '_')}m": null_water(z, level, mm)
            for mm in (MARGIN_M, *SENSITIVITY_MARGINS_M)
        }
        np.savez_compressed(
            OUT / f"{aoi}.npz",
            time_utc=times,
            level_navd88_m=level.astype(np.float32),
            matched=matched,
            **nulls,
        )
        tt = times[matched]
        row = {
            "aoi": aoi,
            "points": tile.n,
            "unmatched_points": unmatched,
            "match_distance_p99_m": round(float(np.percentile(dist, 99)), 4),
            "flight_utc_first": dt.datetime.fromtimestamp(tt.min(), dt.UTC).isoformat(
                timespec="seconds"
            ),
            "flight_utc_last": dt.datetime.fromtimestamp(tt.max(), dt.UTC).isoformat(
                timespec="seconds"
            ),
            "flight_span_minutes": round(float((tt.max() - tt.min()) / 60), 1),
            "gauge_level_navd88_m": {
                "min": round(float(level.min()), 3),
                "max": round(float(level.max()), 3),
                "median": round(float(np.median(level)), 3),
            },
            "null": {k: as_null(water_scores(v, truth, clean)) for k, v in nulls.items()},
            "headline_null_false_positives_by_delivery_class": _by_class(
                nulls[f"null_{str(MARGIN_M).replace('.', '_')}m"] & clean & ~truth, asprs, z - level
            ),
            "delivery_water_height_minus_level_m": _quantiles((z - level)[truth & clean]),
            "headline_null": f"null_{str(MARGIN_M).replace('.', '_')}m",
        }
        result["areas"].append(row)
        print(
            f"  {aoi}: flown {row['flight_utc_first']} to {row['flight_utc_last']}, level "
            f"{row['gauge_level_navd88_m']}, null {row['null'][row['headline_null']]}",
            flush=True,
        )
    result["seconds"] = round(time.time() - t0, 1)
    ARTIFACT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
