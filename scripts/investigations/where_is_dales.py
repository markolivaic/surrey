"""Where was DALES actually flown? Answer it from the tiles, not from the name.

DALES stands for Dayton Annotated Laser Earth Scan. Dayton is the University of
Dayton, where the annotation was done. It is not where the lidar was flown. The
paper says the collection spanned 330 km2 over the City of Surrey, British
Columbia, Canada.

This script does not take the paper's word for it.

The tiles carry no CRS and their points are normalised to a local 0..500 m box,
so the only georeference left in the distribution is the filename. A tile called
5080_54400 sits at easting 508000, northing 5440000. Two things make that more
than a guess:

  1. Every tile origin decoded that way is a multiple of 500 m, and the tiles
     tile. A wrong scale factor would not produce a grid.
  2. Placing that grid in the CRS the paper states puts the data on Surrey and
     over 6000 km from Surrey.

If the naming convention were wrong, check 1 would fail before check 2 was ever
reached.

Run:
    python scripts/investigations/where_is_dales.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TILES = ROOT / "data" / "dales2"
DEFAULT_OUT = ROOT / "data" / "artifacts" / "where_is_dales.json"

# The paper states: UTM zone 10N, horizontal datum NAD83. That is EPSG:26910.
STATED_CRS = "EPSG:26910"
STATED_CRS_SOURCE = "Varney et al. 2020, DALES, section 3.1"

# Tile names are the tile origin in units of 100 m: 5080_54400 is 508000 E,
# 5440000 N. Verified by the grid check below, not assumed.
NAME_UNIT_M = 100.0
TILE_NOMINAL_M = 500.0
TILE_NAME = re.compile(r"^(\d{4})_(\d{5})$")

PLACES = {
    "Dayton, Ohio, USA": (39.7589, -84.1916),
    "Surrey, British Columbia, Canada": (49.1913, -122.8490),
}


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles", type=Path, default=DEFAULT_TILES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    import laspy
    from pyproj import CRS, Transformer

    paths = sorted(args.tiles.rglob("*.laz"))
    if not paths:
        raise SystemExit(f"no tiles under {args.tiles}, run scripts/pipeline/fetch_dales2.py first")

    per_tile = []
    for p in paths:
        m = TILE_NAME.match(p.stem)
        if not m:
            raise SystemExit(f"{p.name} does not match the tile naming convention")
        with laspy.open(p) as fh:
            hdr = fh.header
            try:
                declared = hdr.parse_crs()
            except Exception:
                declared = None
            per_tile.append(
                {
                    "split": p.parent.name,
                    "name": p.stem,
                    "points": int(hdr.point_count),
                    "point_format": int(hdr.point_format.id),
                    "local_min": [round(float(v), 2) for v in hdr.mins],
                    "local_max": [round(float(v), 2) for v in hdr.maxs],
                    "origin_e": int(m.group(1)) * NAME_UNIT_M,
                    "origin_n": int(m.group(2)) * NAME_UNIT_M,
                    "crs_declared": declared.to_string() if declared is not None else None,
                }
            )

    # Check 1: the decoded origins form a grid on a 500 m pitch, and every tile
    # holds a box of about the nominal tile size starting at its own zero.
    origins_on_pitch = sum(
        1
        for t in per_tile
        if t["origin_e"] % TILE_NOMINAL_M == 0 and t["origin_n"] % TILE_NOMINAL_M == 0
    )
    boxes_nominal = sum(
        1
        for t in per_tile
        if abs(t["local_max"][0] - TILE_NOMINAL_M) < 1.0
        and abs(t["local_max"][1] - TILE_NOMINAL_M) < 1.0
        and t["local_min"][0] == 0.0
        and t["local_min"][1] == 0.0
    )
    grid_holds = origins_on_pitch == len(per_tile) and boxes_nominal == len(per_tile)

    declared_any = [t["crs_declared"] for t in per_tile if t["crs_declared"]]
    crs_used = declared_any[0] if declared_any else STATED_CRS
    crs_origin = (
        "declared in the LAS header"
        if declared_any
        else f"absent from every LAS header, taken from {STATED_CRS_SOURCE}"
    )

    # Check 2: place the grid and see what city it lands on.
    es = [t["origin_e"] for t in per_tile] + [t["origin_e"] + TILE_NOMINAL_M for t in per_tile]
    ns = [t["origin_n"] for t in per_tile] + [t["origin_n"] + TILE_NOMINAL_M for t in per_tile]
    bbox = [min(es), min(ns), max(es), max(ns)]
    centre = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    tf = Transformer.from_crs(CRS.from_user_input(crs_used), CRS.from_epsg(4326), always_xy=True)
    lon, lat = tf.transform(*centre)
    corners = [
        tf.transform(x, y)
        for x, y in ((bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3]))
    ]

    distances = {k: round(haversine_km((lat, lon), v), 1) for k, v in PLACES.items()}
    nearest = min(distances, key=distances.get)

    tiled_area_km2 = len(per_tile) * (TILE_NOMINAL_M**2) / 1e6
    z_maxes = [t["local_max"][2] for t in per_tile]
    formats: dict[str, int] = {}
    for t in per_tile:
        k = str(t["point_format"])
        formats[k] = formats.get(k, 0) + 1

    result = {
        "question": "where was the lidar behind DALES actually flown",
        "tiles": len(per_tile),
        "points_total": sum(t["points"] for t in per_tile),
        "georeference": "tile filename only",
        "crs_used": crs_used,
        "crs_origin": crs_origin,
        "grid_check": {
            "origins_on_500m_pitch": f"{origins_on_pitch}/{len(per_tile)}",
            "local_boxes_are_500m_from_zero": f"{boxes_nominal}/{len(per_tile)}",
            "holds": grid_holds,
        },
        "bbox_utm": bbox,
        "extent_km": [round((bbox[2] - bbox[0]) / 1000, 2), round((bbox[3] - bbox[1]) / 1000, 2)],
        "tiled_area_km2": round(tiled_area_km2, 2),
        "centroid_wgs84": {"lat": round(lat, 5), "lon": round(lon, 5)},
        "corners_wgs84": [[round(a, 5), round(b, 5)] for a, b in corners],
        "distance_km": distances,
        "nearest_named_place": nearest,
        "point_formats": formats,
        "z_normalised_per_tile": all(t["local_min"][2] == 0.0 for t in per_tile),
        "z_max_range_m": [round(min(z_maxes), 2), round(max(z_maxes), 2)],
        "per_tile": per_tile,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"tiles            {len(per_tile)}   points {result['points_total']:,}")
    print(f"georeference     {result['georeference']}")
    print(f"crs              {crs_used}  ({crs_origin})")
    print(
        f"grid check       origins on 500 m pitch {origins_on_pitch}/{len(per_tile)}, "
        f"local boxes nominal {boxes_nominal}/{len(per_tile)}  -> "
        f"{'holds' if grid_holds else 'FAILS'}"
    )
    print(
        f"extent           {result['extent_km'][0]} x {result['extent_km'][1]} km, "
        f"{tiled_area_km2:.1f} km2 tiled"
    )
    print(f"centroid WGS84   {lat:.5f}, {lon:.5f}")
    for name, km in sorted(distances.items(), key=lambda kv: kv[1]):
        print(f"distance to      {name:36s} {km:>8.1f} km")
    print(f"point formats    {formats}")
    print(
        f"z normalised     {result['z_normalised_per_tile']}  "
        f"(z max ranges {result['z_max_range_m'][0]} to {result['z_max_range_m'][1]} m)"
    )
    print(f"wrote            {args.out}")

    if not grid_holds:
        print("\ngrid check failed, the filename convention is not what this assumes")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
