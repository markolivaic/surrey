"""Where a vegetation source could come from: evergreen canopy, flown in leaf.

San Francisco was flown on 2023-04-20 and much of its canopy is evergreen. Paris
was flown in March, when its plane trees are bare, and swissSURFACE3D is flown on
purpose while vegetation is low. A model taught vegetation from bare branches is
taught the wrong thing. So candidate tiles are chosen on two sourced facts, not on
what a city is said to be planted with:

    canopy type     IGN BD Forêt V2 (WFS LANDCOVER.FORESTINVENTORY.V2:formation_vegetale),
                    the essence of every forest polygon, grouped into evergreen,
                    deciduous and not stated
    flight date     the LiDAR HD tile metadata, date_debut_acquisition

Evergreen here means an essence the inventory names as coniferous other than larch,
or as evergreen oak. "Feuillus" alone says broadleaf, not deciduous, and in the
Mediterranean it can hide holm oak, so it is kept apart as not stated rather than
counted as deciduous.

Forest polygons in the inventory overlap one another, so every area is a union,
evergreen taking precedence where two groups claim the same ground, and that
contested share is reported beside the rest.

No point data is downloaded. Candidates are reported with their dates and the
classification process of each tile, for a decision to be made on them.

Run:
    python scripts/investigations/evergreen_candidates.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
import shapely
from shapely.geometry import box, shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "artifacts" / "evergreen_candidates.json"
HEADERS = {"User-Agent": "surrey-research/0.1 (github.com/markolivaic)"}
WFS = "https://data.geopf.fr/wfs/ows"
REGIONS = {
    "marseille": (43.20, 5.30, 43.38, 5.55),
    "nice": (43.65, 7.15, 43.78, 7.32),
    "toulon": (43.08, 5.85, 43.17, 6.02),
    "montpellier": (43.55, 3.78, 43.68, 3.95),
    "ajaccio": (41.88, 8.68, 41.98, 8.82),
    "paris": (48.80, 2.20, 48.90, 2.45),
}
LEAF_ON_MONTHS = {5, 6, 7, 8, 9}
# Overlays on a 1 cm grid: exact floating overlays of these polygons fail inside GEOS.
GRID_M = 0.01
EVERGREEN_WORDS = (
    "conifère",
    "pin ",
    "pins",
    "pin d",
    "sapin",
    "épicéa",
    "douglas",
    "cèdre",
    "sempervirent",
)
DECIDUOUS_WORDS = ("décidu", "hêtre", "châtaignier", "robinier", "peuplier")


def wfs(typename: str, bbox, sort_by: str, extra=None) -> tuple[list[dict], int | None]:
    """Every feature in the box, paged in a stable order and checked against the count.

    Pages are sorted so that paging cannot skip or repeat features, repeats are
    dropped by feature id regardless, and the unique count is returned beside the
    server's own total. (An over-full tile first looked like repeated features; it
    was overlapping polygons, handled in main.)
    """
    feats, start, matched = {}, 0, None
    while True:
        params = {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": typename,
            "OUTPUTFORMAT": "application/json",
            "SORTBY": sort_by,
            "COUNT": 1000,
            "STARTINDEX": start,
            "SRSNAME": "EPSG:2154",
            "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},urn:ogc:def:crs:EPSG::4326",
            **(extra or {}),
        }
        for attempt in range(5):
            try:
                r = requests.get(WFS, params=params, headers=HEADERS, timeout=120)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(3 * (attempt + 1))
        body = r.json()
        matched = body.get("numberMatched", matched)
        page = body.get("features", [])
        for f in page:
            feats[f.get("id")] = f
        time.sleep(0.3)
        if len(page) < 1000:
            return list(feats.values()), matched
        start += 1000


def polygonal(g):
    """Only the areal part. Clipping a polygon to a tile edge can leave lines and points
    in a collection, and GEOS refuses to overlay mixed dimensions."""
    parts = [q for q in shapely.get_parts(g) if q.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(parts, grid_size=GRID_M) if parts else shapely.Polygon()


def group(essence: str | None) -> str:
    e = (essence or "").lower()
    if "mélèze" in e:
        return "deciduous"
    if any(w in e for w in EVERGREEN_WORDS):
        return "evergreen"
    if any(w in e for w in DECIDUOUS_WORDS):
        return "deciduous"
    return "not_stated"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    result = {
        "question": "tiles with evergreen canopy flown in leaf",
        "canopy_source": "IGN BD Forêt V2, formation_vegetale, attribute essence",
        "date_source": "LiDAR HD tile metadata, date_debut_acquisition",
        "evergreen_rule": "essence naming a conifer other than larch, or evergreen oak",
        "leaf_on_months": sorted(LEAF_ON_MONTHS),
        "regions": {},
    }
    for region, bbox in REGIONS.items():
        forest, forest_matched = wfs("LANDCOVER.FORESTINVENTORY.V2:formation_vegetale", bbox, "id")
        polys = [shapely.make_valid(shape(f["geometry"])) for f in forest if f.get("geometry")]
        groups = [group(f["properties"].get("essence")) for f in forest if f.get("geometry")]
        essences = {}
        for f in forest:
            essences[f["properties"].get("essence")] = (
                essences.get(f["properties"].get("essence"), 0) + 1
            )
        tree = STRtree(polys)
        tiles, tiles_matched = wfs(
            "IGNF_LIDAR-HD_METADONNEE:metadata",
            bbox,
            "coordonnees_nw",
            {
                "PROPERTYNAME": "coordonnees_nw,nombre_points,date_debut_acquisition,procede_classement,url_npl"
            },
        )
        rows = []
        for t in tiles:
            p = t["properties"]
            nw = p.get("coordonnees_nw")
            if not nw:
                continue
            e_km, n_km = (int(v) for v in nw.split("-"))
            sq = box(e_km * 1000, (n_km - 1) * 1000, (e_km + 1) * 1000, n_km * 1000)
            # Inventory polygons overlap each other: on one Paris tile they summed to
            # 1,017,985 m² over a union of 835,164 m². Areas are therefore unions,
            # evergreen first, and ground claimed by two groups is reported apart.
            pieces = {"evergreen": [], "deciduous": [], "not_stated": []}
            for i in tree.query(sq):
                inter = polygonal(shapely.intersection(polys[i], sq, grid_size=GRID_M))
                if not inter.is_empty:
                    pieces[groups[i]].append(inter)
            ev = shapely.union_all(pieces["evergreen"], grid_size=GRID_M)
            de = shapely.union_all(pieces["deciduous"], grid_size=GRID_M)
            ns = shapely.union_all(pieces["not_stated"], grid_size=GRID_M)
            claimed_twice = sum(
                polygonal(shapely.intersection(a, b, grid_size=GRID_M)).area
                for a, b in ((ev, de), (ev, ns), (de, ns))
            )
            area = {
                "evergreen": ev.area,
                "deciduous": shapely.difference(de, ev, grid_size=GRID_M).area,
                "not_stated": shapely.difference(
                    ns, shapely.union(ev, de, grid_size=GRID_M), grid_size=GRID_M
                ).area,
            }
            covered = sum(area.values()) / 1e6
            if covered > 1.0 + 1e-6:
                raise SystemExit(f"{region} {nw}: forest covers {covered:.4f} of a 1 km² tile")
            month = (
                int(p["date_debut_acquisition"][5:7]) if p.get("date_debut_acquisition") else None
            )
            rows.append(
                {
                    "tile": nw,
                    "acquired": (p.get("date_debut_acquisition") or "")[:10],
                    "leaf_on": month in LEAF_ON_MONTHS,
                    "procede_classement": p.get("procede_classement"),
                    "points_per_m2": round(p["nombre_points"] / 1e6, 1)
                    if p.get("nombre_points")
                    else None,
                    "evergreen_forest_share": round(area["evergreen"] / 1e6, 3),
                    "deciduous_forest_share": round(area["deciduous"] / 1e6, 3),
                    "broadleaf_not_stated_share": round(area["not_stated"] / 1e6, 3),
                    "claimed_by_two_groups_share": round(claimed_twice / 1e6, 3),
                    "url": p.get("url_npl"),
                }
            )
        rows.sort(key=lambda r: (-int(r["leaf_on"]), -r["evergreen_forest_share"]))
        result["regions"][region] = {
            "forest_polygons": len(polys),
            "forest_polygons_server_count": forest_matched,
            "tiles_server_count": tiles_matched,
            "essences_by_polygon_count": dict(sorted(essences.items(), key=lambda kv: -kv[1])[:12]),
            "tiles": len(rows),
            "tiles_leaf_on": sum(r["leaf_on"] for r in rows),
            "top": rows[: args.top],
        }
        best = rows[0] if rows else {}
        print(
            f"  {region:12s} tiles {len(rows):4d} leaf-on {result['regions'][region]['tiles_leaf_on']:4d}  "
            f"best {best.get('tile')} {best.get('acquired')} evergreen {best.get('evergreen_forest_share')} "
            f"{best.get('procede_classement')}",
            flush=True,
        )
    result["seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
