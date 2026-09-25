"""An independent water polygon for the second upper edge of the water band. No height involved.

The first upper edge counts class 1 points below the gauge level plus a margin as water, which is
the null's own rule, so it cannot judge the null. This edge counts class 1 points as water when
they fall inside a water polygon drawn by someone else, from imagery and maps, with no lidar
height (backend/surrey/m1_criteria.py).

Source: the USGS National Hydrography Dataset through The National Map's service
(hydro.nationalmap.gov, nhd MapServer), layers "Area - Large Scale" and "Waterbody - Large Scale",
queried by each area's extent. USGS-produced data are in the U.S. public domain. The service states
its own refresh date, recorded with each feature's attributes. NHD was retired in October 2023 and
is no longer maintained, so the polygons are the last maintained version, and the artefact says so.

Per point of Embarcadero and Bay Bridge: whether it lies inside any returned polygon. Writes
data/reference/water_polygon/<area>.npz and data/artifacts/water_polygon.json.

Run:
    python scripts/pipeline/water_polygon.py
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402

BASE = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"
LAYERS = {9: "Area - Large Scale", 12: "Waterbody - Large Scale"}
AREAS = ("embarcadero", "baybridge")
OUT = ROOT / "data" / "reference" / "water_polygon"
ARTIFACT = ROOT / "data" / "artifacts" / "water_polygon.json"
CACHE = ROOT / "data" / "cache" / "nhd"


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "surrey-research"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def main() -> int:
    import shapely
    from pyproj import Transformer
    from shapely.geometry import shape

    service = get(f"{BASE}?f=json")
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "source": BASE,
        "layers": LAYERS,
        "service_copyright_text": service.get("copyrightText"),
        "licence": "USGS-produced data, U.S. public domain",
        "maintenance": "NHD retired October 2023 and no longer maintained; the service's own refresh "
        "note is recorded above",
        "height_used": False,
        "areas": [],
    }
    for aoi in AREAS:
        tile = CachedTile(ROOT / "data" / "prep" / "target" / aoi, "sf")
        ex, ey = tile.extent_m()
        to_ll = Transformer.from_crs(tile.epsg, 4326, always_xy=True)
        xs = [tile.origin[0], tile.origin[0] + ex]
        ys = [tile.origin[1], tile.origin[1] + ey]
        lon, lat = to_ll.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
        env = f"{min(lon)},{min(lat)},{max(lon)},{max(lat)}"
        polys, features = [], []
        for layer, lname in LAYERS.items():
            params = {
                "geometry": env,
                "geometryType": "esriGeometryEnvelope",
                "inSR": 4326,
                "outSR": tile.epsg,
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "f": "geojson",
            }
            cache = CACHE / f"{aoi}_{layer}.geojson"
            if cache.exists():
                gj = json.loads(cache.read_text(encoding="utf-8"))
            else:
                gj = get(f"{BASE}/{layer}/query?{urllib.parse.urlencode(params)}")
                cache.write_text(json.dumps(gj), encoding="utf-8", newline="\n")
            for f in gj.get("features", []):
                geom = shape(f["geometry"])
                if geom.is_empty:
                    continue
                polys.append(shapely.make_valid(geom))
                props = {k.lower(): v for k, v in (f.get("properties") or {}).items()}
                features.append(
                    {
                        "layer": lname,
                        "ftype": props.get("ftype"),
                        "fcode": props.get("fcode"),
                        "gnis_name": props.get("gnis_name"),
                        "loaddate": props.get("loaddate"),
                    }
                )
        ux, uy = tile.utm_xy(np.arange(tile.n))
        inside = np.zeros(tile.n, bool)
        if polys:
            union = shapely.union_all(polys)
            inside = shapely.contains_xy(union, ux, uy)
        asprs = np.asarray(tile.asprs[:], np.int16)
        np.savez_compressed(OUT / f"{aoi}.npz", inside=inside)
        by_class = {
            str(c): {
                "points": int((asprs == c).sum()),
                "inside_polygon": int(((asprs == c) & inside).sum()),
            }
            for c in (1, 2, 9, 17)
        }
        row = {
            "aoi": aoi,
            "features": features,
            "points": tile.n,
            "points_inside": int(inside.sum()),
            "by_delivery_class": by_class,
        }
        result["areas"].append(row)
        print(
            aoi, len(features), "features", int(inside.sum()), "points inside", by_class, flush=True
        )
    ARTIFACT.write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
