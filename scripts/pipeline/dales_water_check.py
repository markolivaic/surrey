"""Does any DALES tile cross water, and which of its points must M1 ignore?

DALES-2 has no water class, so whatever water lies on its tiles is labelled something
else, most likely ground. M1 adds water from swissSURFACE3D. If the same flat,
low-return surface were called ground by one source and water by the other, the model
would get contradictory supervision. So water on the DALES tiles is found from an
independent map and those points are set to ignore for M1, never relabelled.

Map: the BC Freshwater Atlas, Open Government Licence - British Columbia, over the
openmaps WFS: lakes, river polygons, man-made waterbodies, and bays and channels for
the sea. Tiles are placed from their names, the origin in hundreds of metres in UTM
zone 10N NAD83 (EPSG:26910), as scripts/investigations/where_is_dales.py established.

Writes data/prep/dales_water_ignore/<tile>.npy (boolean per cached point) for every
tile with any water, and data/artifacts/dales_water_check.json with the water area per
tile and what DALES labels the points inside it.

Run:
    python scripts/pipeline/dales_water_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import CLASS_NAMES  # noqa: E402

DALES_PREP = ROOT / "data" / "prep" / "dales"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
OUT_MASKS = ROOT / "data" / "prep" / "dales_water_ignore"
OUT = ROOT / "data" / "artifacts" / "dales_water_check.json"
WFS = "https://openmaps.gov.bc.ca/geo/pub/wfs"
LAYERS = [
    "WHSE_BASEMAPPING.FWA_LAKES_POLY",
    "WHSE_BASEMAPPING.FWA_RIVERS_POLY",
    "WHSE_BASEMAPPING.FWA_MANMADE_WATERBODIES_POLY",
    "WHSE_BASEMAPPING.FWA_BAYS_AND_CHANNELS_POLY",
]
LICENCE = "Open Government Licence - British Columbia"
TILE_M = 500.0
NAME_UNIT_M = 100.0
HEADERS = {"User-Agent": "surrey-research/0.1 (github.com/markolivaic)"}


def polygons(layer: str, bbox) -> list:
    from shapely.geometry import shape

    r = requests.get(
        WFS,
        params={
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": layer,
            "outputFormat": "json",
            "srsName": "EPSG:26910",
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},urn:ogc:def:crs:EPSG::26910",
        },
        headers=HEADERS,
        timeout=120,
    )
    r.raise_for_status()
    return [shape(f["geometry"]) for f in r.json().get("features", []) if f.get("geometry")]


def main() -> int:
    import shapely
    from shapely.geometry import box

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    tiles = json.loads(PREP_ARTIFACT.read_text(encoding="utf-8"))["tiles"]
    e0 = min(int(t["name"].split("_")[0]) for t in tiles) * NAME_UNIT_M
    n0 = min(int(t["name"].split("_")[1]) for t in tiles) * NAME_UNIT_M
    e1 = max(int(t["name"].split("_")[0]) for t in tiles) * NAME_UNIT_M + TILE_M
    n1 = max(int(t["name"].split("_")[1]) for t in tiles) * NAME_UNIT_M + TILE_M
    water = {layer: polygons(layer, (e0, n0, e1, n1)) for layer in LAYERS}
    print({k.split(".")[-1]: len(v) for k, v in water.items()}, flush=True)
    everything = [(layer, g) for layer, gs in water.items() for g in gs]
    OUT_MASKS.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in sorted(tiles, key=lambda t: t["name"]):
        oe = int(t["name"].split("_")[0]) * NAME_UNIT_M
        on = int(t["name"].split("_")[1]) * NAME_UNIT_M
        sq = box(oe, on, oe + TILE_M, on + TILE_M)
        hits = [(layer, g.intersection(sq)) for layer, g in everything if g.intersects(sq)]
        hits = [(layer, g) for layer, g in hits if not g.is_empty and g.area > 0]
        row = {
            "tile": t["name"],
            "split": t["split"],
            "water_area_share": round(sum(g.area for _, g in hits) / sq.area, 4),
            "layers": sorted({layer.split(".")[-1] for layer, _ in hits}),
        }
        if hits:
            tile = CachedTile(DALES_PREP / t["name"], t["split"])
            x = np.asarray(tile.xy_cm[:, 0], np.float64) / 100.0 + oe
            y = np.asarray(tile.xy_cm[:, 1], np.float64) / 100.0 + on
            union = shapely.union_all([g for _, g in hits])
            shapely.prepare(union)
            inside = shapely.contains_xy(union, x, y)
            np.save(OUT_MASKS / f"{t['name']}.npy", inside)
            label = np.asarray(tile.label[:], np.int16)
            codes, counts = np.unique(label[inside], return_counts=True)
            row.update(
                {
                    "points_in_water": int(inside.sum()),
                    "points": tile.n,
                    "dales_labels_in_water": {
                        (CLASS_NAMES[c] if c >= 0 else "unlabelled"): round(
                            float(k / max(inside.sum(), 1)), 4
                        )
                        for c, k in zip(codes, counts, strict=True)
                    },
                    "ignore_mask": f"data/prep/dales_water_ignore/{t['name']}.npy",
                }
            )
        rows.append(row)
        print(
            f"  {t['name']} {t['split']:5s} water {row['water_area_share']:.4f} "
            f"points {row.get('points_in_water', 0):,} {row.get('dales_labels_in_water', '')}",
            flush=True,
        )

    result = {
        "question": "which DALES tiles cross water, by an independent map",
        "map": {
            "source": "BC Freshwater Atlas via openmaps.gov.bc.ca WFS",
            "layers": LAYERS,
            "licence": LICENCE,
            "crs": "EPSG:26910",
        },
        "tiles_with_water": sum(1 for r in rows if r["water_area_share"] > 0),
        "tiles": rows,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"tiles with water: {result['tiles_with_water']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
