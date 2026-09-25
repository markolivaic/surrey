"""Many tiles per national set, from metadata and file headers, before anything is fixed.

One tile gave IGN 17.3 points per m² against a specification of at least 10 pulses,
and one tile can mislead. So density, acquisition dates and how the labels were made
are read across many tiles, without downloading point data:

    IGN LiDAR HD      the per-tile metadata layer (WFS), which carries the point count,
                      the acquisition dates and the classification process of each
                      1 km² tile
    swissSURFACE3D    the COPC header of each tile, fetched by an HTTP range request:
                      point count and extent, and the GPS time range of the root node
                      for the flight date
    AHN5 (GeoTiles)   the LAS header of each sub-tile, by range request: point count
                      and extent; the sub-tiles overlap their neighbours, so the extent
                      is the header's

Every request identifies itself and stays far below the published limit of the
Géoplateforme download API, ten requests a second from one address.

Run:
    python scripts/investigations/national_survey.py
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "artifacts" / "national_survey.json"
HEADERS = {"User-Agent": "surrey-research/0.1 (github.com/markolivaic)"}
PAUSE_S = 0.3

WFS = "https://data.geopf.fr/wfs/ows"
IGN_REGIONS = {
    # name: (lat_min, lon_min, lat_max, lon_max), city centres of about 10 km
    "paris": (48.80, 2.25, 48.90, 2.42),
    "lyon": (45.72, 4.78, 45.80, 4.90),
    "marseille": (43.25, 5.33, 43.35, 5.45),
    "nice": (43.66, 7.20, 43.74, 7.30),
    "toulon": (43.09, 5.88, 43.15, 5.98),
    "montpellier": (43.57, 3.83, 43.64, 3.92),
    "toulouse": (43.56, 1.39, 43.64, 1.49),
    "bordeaux": (44.80, -0.62, 44.88, -0.52),
    "nantes": (47.18, -1.60, 47.25, -1.50),
    "lille": (50.60, 3.00, 50.66, 3.10),
    "strasbourg": (48.55, 7.70, 48.62, 7.80),
    "ajaccio": (41.90, 8.70, 41.96, 8.78),
}
IGN_FIELDS = (
    "coordonnees_nw,nombre_points,date_debut_acquisition,date_fin_acquisition,"
    "procede_classement,moe_classement_automatique,moe_classement_manuel,code_mission"
)

STAC_CH = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swisssurface3d/items"
CH_REGIONS = {
    "zurich": (8.50, 47.35, 8.58, 47.40),
    "st_gallen": (9.34, 47.41, 9.42, 47.44),
    "chur": (9.50, 46.84, 9.56, 46.87),
    "lugano": (8.92, 45.99, 8.97, 46.03),
    "bern": (7.40, 46.93, 7.47, 46.97),
    "geneva": (6.12, 46.19, 6.18, 46.22),
    "basel": (7.56, 47.54, 7.62, 47.58),
    "lucerne": (8.28, 47.03, 8.33, 47.07),
}

AHN_BLADEN = {
    # kaartblad: city; sub-tiles 1 to 25 each
    "25GN1": "amsterdam",
    "37EN2": "rotterdam",
    "30GZ2": "den_haag_leiden",
    "31HZ2": "utrecht",
}


def get(url, **kw):
    for attempt in range(5):
        try:
            r = requests.get(url, headers={**HEADERS, **kw.pop("headers", {})}, timeout=90, **kw)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    raise SystemExit(f"gave up on {url}")


def summary(values) -> dict:
    v = np.asarray([x for x in values if x is not None], float)
    if not v.size:
        return {"n": 0}
    return {
        "n": int(v.size),
        "p10": round(float(np.percentile(v, 10)), 1),
        "median": round(float(np.median(v)), 1),
        "p90": round(float(np.percentile(v, 90)), 1),
    }


def ign() -> dict:
    out = {}
    for region, (la0, lo0, la1, lo1) in IGN_REGIONS.items():
        r = get(
            WFS,
            params={
                "SERVICE": "WFS",
                "VERSION": "2.0.0",
                "REQUEST": "GetFeature",
                "TYPENAMES": "IGNF_LIDAR-HD_METADONNEE:metadata",
                "OUTPUTFORMAT": "application/json",
                "COUNT": 500,
                "PROPERTYNAME": IGN_FIELDS,
                "BBOX": f"{la0},{lo0},{la1},{lo1},urn:ogc:def:crs:EPSG::4326",
            },
        )
        feats = [f["properties"] for f in r.json().get("features", [])]
        time.sleep(PAUSE_S)
        # A 1 km² tile holds its points over 1e6 m², less any sea or no-data part, so
        # the median is the figure to read and the low tail is coastline.
        dens = [f["nombre_points"] / 1e6 for f in feats if f.get("nombre_points")]
        months = Counter(
            f["date_debut_acquisition"][:7] for f in feats if f.get("date_debut_acquisition")
        )
        out[region] = {
            "tiles": len(feats),
            "points_per_m2": summary(dens),
            "acquisition_months": dict(sorted(months.items())),
            "procede_classement": dict(Counter(f.get("procede_classement") for f in feats)),
            "moe_classement_automatique": dict(
                Counter(f.get("moe_classement_automatique") for f in feats)
            ),
            "moe_classement_manuel": dict(Counter(f.get("moe_classement_manuel") for f in feats)),
            "missions": sorted({f.get("code_mission") for f in feats if f.get("code_mission")}),
        }
        print(
            f"  ign {region:12s} tiles {len(feats):4d}  density {out[region]['points_per_m2']}  "
            f"months {list(months)[:4]}  process {out[region]['procede_classement']}",
            flush=True,
        )
    return out


def copc_header(url: str) -> dict:
    """Point count, extent, and the root node's GPS time, from a COPC file by ranges."""
    head = get(url, headers={"Range": "bytes=0-1023"}).content
    if head[:4] != b"LASF":
        return {"error": "not a LAS file"}
    # LAS 1.4 public header offsets
    point_count = struct.unpack_from("<Q", head, 247)[0]
    maxx, minx, maxy, miny = struct.unpack_from("<4d", head, 179)
    header_size = struct.unpack_from("<H", head, 94)[0]
    # COPC info VLR directly follows the header: 54 byte VLR header, then the info.
    info = head[header_size + 54 : header_size + 54 + 160]
    cx, cy, cz, halfsize, spacing, root_off, root_size, gmin, gmax = struct.unpack_from(
        "<5d2Q2d", info, 0
    )
    return {
        "points": int(point_count),
        "extent_m": [round(maxx - minx, 1), round(maxy - miny, 1)],
        "gpstime_min": gmin,
        "gpstime_max": gmax,
    }


def gps_date(t: float) -> str | None:
    import datetime as dt

    if not t or t < 1e6:
        return None
    return (dt.datetime(1980, 1, 6) + dt.timedelta(seconds=t + 1e9)).strftime("%Y-%m-%d")


def swiss() -> dict:
    out = {}
    for region, bbox in CH_REGIONS.items():
        r = get(STAC_CH, params={"bbox": ",".join(map(str, bbox)), "limit": 100})
        items = r.json().get("features", [])
        time.sleep(PAUSE_S)
        rows = []
        for it in items:
            href = next(
                (a["href"] for a in it["assets"].values() if a["href"].endswith(".copc.laz")), None
            )
            if href is None:
                continue
            h = copc_header(href)
            time.sleep(PAUSE_S)
            if "error" in h:
                continue
            area = h["extent_m"][0] * h["extent_m"][1]
            rows.append(
                {
                    "id": it["id"],
                    "points_per_m2": round(h["points"] / area, 1) if area else None,
                    "first_day": gps_date(h["gpstime_min"]),
                    "last_day": gps_date(h["gpstime_max"]),
                }
            )
        out[region] = {
            "copc_tiles": len(rows),
            "tiles_listed": len(items),
            "points_per_m2": summary(r_["points_per_m2"] for r_ in rows),
            "flight_days": dict(Counter(r_["first_day"] for r_ in rows)),
            "note": "tiles without a .copc.laz asset are the pre-2024 LAS 1.2 delivery",
        }
        print(
            f"  ch  {region:12s} copc {len(rows):3d}/{len(items):3d}  density {out[region]['points_per_m2']}  "
            f"days {list(out[region]['flight_days'])[:3]}",
            flush=True,
        )
    return out


def las_header(url: str) -> dict:
    head = get(url, headers={"Range": "bytes=0-374"}).content
    if head[:4] != b"LASF":
        return {"error": "not a LAS file"}
    version = (head[24], head[25])
    legacy = struct.unpack_from("<I", head, 107)[0]
    count = struct.unpack_from("<Q", head, 247)[0] if version >= (1, 4) else legacy
    maxx, minx, maxy, miny = struct.unpack_from("<4d", head, 179)
    return {"points": int(count or legacy), "extent_m": [maxx - minx, maxy - miny]}


def ahn() -> dict:
    out = {}
    for blad, city in AHN_BLADEN.items():
        rows = []
        for k in range(1, 26):
            url = f"https://geotiles.citg.tudelft.nl/AHN5_T/{blad}_{k:02d}.LAZ"
            h = las_header(url)
            time.sleep(PAUSE_S)
            if "error" in h:
                continue
            area = h["extent_m"][0] * h["extent_m"][1]
            rows.append(h["points"] / area if area else None)
        out[f"{blad}_{city}"] = {
            "subtiles": len(rows),
            "points_per_m2": summary(rows),
            "note": "extent from the header, which includes the overlap margin",
        }
        print(
            f"  ahn {blad} {city:16s} subtiles {len(rows)}  density {out[f'{blad}_{city}']['points_per_m2']}",
            flush=True,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    result = {
        "question": "density, dates and label process across many tiles per national set",
        "ign_lidar_hd": ign(),
        "swisssurface3d": swiss(),
        "ahn5_geotiles": ahn(),
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
