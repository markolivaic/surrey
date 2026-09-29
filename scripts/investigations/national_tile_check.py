"""One urban tile from each national lidar set, measured before any of them is used.

The research datasets were rejected for DALES-style reasons: licence, registration,
photogrammetry. The national sets were then rejected too broadly, as if they carried
the same contamination as the San Francisco delivery. They may not. A national
classification is automatic, but automatic at a different quality and with a
different class list, so each is measured on its own tile.

For each tile:

    class distribution     every code in the file, counted, named from the source's
                           own class table where that table was found
    density                all points, and first returns as a pulse proxy, over the
                           1 m cells that hold any point
    contamination proxy    the share of points more than 2 m above ground that the
                           delivery left unclassified. Above 2 m a point is a roof, a
                           tree, a wire or a vehicle roof; an unclassified one is a
                           label the delivery did not commit to.

Ground comes from the delivery's own class 2, per 5 m cell, carried at most 25 m,
through surrey.reference.vendor_ground_surface, the same surface every San Francisco
ground figure uses. The shares are computed on the 0.15 m voxel-thinned cloud, the
population the San Francisco cache holds, and the same proxy is computed on every
San Francisco area so the national figures have something to be read against.

Tiles are downloaded anonymously into data/cache/national and not committed.

Run:
    python scripts/investigations/national_tile_check.py --download-only
    python scripts/investigations/national_tile_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import VOXEL_M, voxel_pick  # noqa: E402
from surrey.reference import SURFACE_CELL_M, SURFACE_REACH_M, vendor_ground_surface  # noqa: E402

CACHE = ROOT / "data" / "cache" / "national"
TARGET_PREP = ROOT / "data" / "prep" / "target"
OUT = ROOT / "data" / "artifacts" / "national_tile_check.json"
ABOVE_M = 2.0
# The Geoplateforme answers Python's default user agent with 403 and curl's with 200.
HEADERS = {"User-Agent": "surrey-research/0.1 (github.com/markolivaic)"}

TILES = {
    "ign_lidarhd_paris": {
        "dataset": "IGN LiDAR HD",
        "place": "Paris, Lambert-93 tile 0653_6863",
        "url": (
            "https://data.geopf.fr/telechargement/download/LiDARHD-NUALID/"
            "NUALHD_1-0__LAZ_LAMB93_KE_2025-06-06/LHD_FXX_0653_6863_PTS_LAMB93_IGN69.copc.laz"
        ),
        "found_via": (
            "WFS data.geopf.fr/wfs/ows, typename IGNF_LIDAR-HD_METADONNEE:metadata, "
            "property url_npl; feature says procede_classement IGN_AUTO_V5, "
            "acquisition 2023-03-03"
        ),
        "unclassified": [1],
        "vegetation": [3, 4, 5],
        "ground": 2,
        "not_measured": {66: "virtual points added under bridges, not a laser return"},
        # 65 is not in the current class table. The change log SE_LiDAR_HD.pdf records
        # it as the artefact class, removed from the specification on 2024-09-09; this
        # tile, edited 2025-06-06, still carries it.
        "noise": [65],
        "class_table_source": (
            "https://data.geopf.fr/annexes/ressources/documentation/"
            "DC_LiDAR_HD_1-0.pdf, section 2.2.8; code 65 from "
            "https://data.geopf.fr/annexes/ressources/documentation/"
            "SE_LiDAR_HD.pdf"
        ),
        "classes": {
            1: "non classé",
            2: "sol",
            3: "végétation basse 0-50 cm",
            4: "végétation moyenne 50 cm-1,50 m",
            5: "végétation haute +1,50 m",
            6: "bâtiment",
            9: "eau",
            17: "tablier de pont",
            64: "sursol pérenne",
            65: "artefact (removed from the specification 2024-09-09)",
            66: "points virtuels",
            67: "divers - bâtis",
        },
    },
    "swisssurface3d_zurich": {
        "dataset": "swissSURFACE3D",
        "place": "Zurich, LV95 tile 2683-1247, 2024 acquisition",
        "url": (
            "https://data.geo.admin.ch/ch.swisstopo.swisssurface3d/"
            "swisssurface3d_2024_2683-1247/swisssurface3d_2024_2683-1247_2056_5728.copc.laz"
        ),
        "found_via": (
            "STAC data.geo.admin.ch/api/stac/v0.9, collection "
            "ch.swisstopo.swisssurface3d, bbox over central Zurich"
        ),
        "unclassified": [1],
        "vegetation": [3],
        "ground": 2,
        "not_measured": {},
        "noise": [7, 18],
        "class_table_source": (
            "https://www.swisstopo.admin.ch/en/height-model-swisssurface3d, "
            "table 'Classification from 2024 onwards'. The table for data "
            "until 2024 has six classes and no facade, mast or wire class"
        ),
        "classes": {
            1: "unclassified and temporary objects",
            2: "ground",
            3: "vegetation (low - medium - high)",
            6: "building roofs",
            9: "water",
            14: "high-voltage / power line",
            15: "masts with a load-bearing function",
            17: "bridge-, overpass-, viaduct- and footbridge deck",
            26: "building facades",
            27: "bridge piers and viaduct stay cables",
        },
    },
    "ahn5_amsterdam": {
        "dataset": "AHN5",
        "place": "Amsterdam, GeoTiles sub-tile 25GN1_14",
        "url": "https://geotiles.citg.tudelft.nl/AHN5_T/25GN1_14.LAZ",
        "found_via": (
            "GeoTiles, TU Delft re-tiling of the AHN kaartbladen; the official "
            "download is the 5 x 6.25 km kaartblad"
        ),
        "unclassified": [1],
        "vegetation": [],
        "ground": 2,
        "not_measured": {},
        "noise": [7, 18],
        "class_table_source": (
            "https://www.ahn.nl/dataroom and https://www.ahn.nl/4-classificatie "
            "name the classes maaiveld, gebouwen, kunstwerken, water, overig, "
            "and from AHN4 high-voltage lines; neither page gives the numeric "
            "codes, so the code to name pairing below is not from a source"
        ),
        "classes": {},
    },
}


def download(name: str, spec: dict) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / f"{name}.laz"
    recorded = target.with_suffix(".size")
    # A finished download records the size the server announced, so a rerun can
    # verify the cached file without the network.
    if (
        target.exists()
        and recorded.exists()
        and int(recorded.read_text(encoding="utf-8")) == target.stat().st_size
    ):
        return target
    head = urllib.request.Request(spec["url"], method="HEAD", headers=HEADERS)
    with urllib.request.urlopen(head, timeout=60) as r:
        size = int(r.headers["Content-Length"])
    if target.exists() and target.stat().st_size == size:
        recorded.write_text(str(size), newline="\n")
        return target
    part = target.with_suffix(".part")
    t0 = time.time()
    get = urllib.request.Request(spec["url"], headers=HEADERS)
    with urllib.request.urlopen(get, timeout=120) as r, part.open("wb") as fh:
        while block := r.read(1 << 22):
            fh.write(block)
    if part.stat().st_size != size:
        raise SystemExit(f"{name}: got {part.stat().st_size} bytes, header said {size}")
    part.replace(target)
    recorded.write_text(str(size), newline="\n")
    print(f"  downloaded {name} {size / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
    return target


def read_tile(path: Path):
    """x, y, z relative to the header minimum as float32, plus class and return number."""
    with laspy.open(path) as fh:
        h = fh.header
        n = h.point_count
        x = np.empty(n, np.float32)
        y = np.empty(n, np.float32)
        z = np.empty(n, np.float32)
        cls = np.empty(n, np.uint8)
        ret = np.empty(n, np.uint8)
        i = 0
        for chunk in fh.chunk_iterator(5_000_000):
            k = len(chunk)
            x[i : i + k] = np.asarray(chunk.x) - h.mins[0]
            y[i : i + k] = np.asarray(chunk.y) - h.mins[1]
            z[i : i + k] = np.asarray(chunk.z) - h.mins[2]
            cls[i : i + k] = np.asarray(chunk.classification)
            ret[i : i + k] = np.asarray(chunk.return_number)
            i += k
        meta = {
            "points_in_header": int(n),
            "las_version": str(h.version),
            "point_format": int(h.point_format.id),
            "crs": str(h.parse_crs().to_epsg()) if h.parse_crs() else None,
            "extent_m": [
                round(float(h.maxs[0] - h.mins[0]), 1),
                round(float(h.maxs[1] - h.mins[1]), 1),
            ],
            "origin": [round(float(h.mins[0]), 1), round(float(h.mins[1]), 1)],
        }
    if i != n:
        raise SystemExit(f"{path.name}: read {i} points, header says {n}")
    return x, y, z, cls, ret, meta


def contamination(x, y, z, cls, ground: int, unclassified, vegetation, keep) -> dict:
    grid, key = vendor_ground_surface(
        x.astype(np.float64),
        y.astype(np.float64),
        z.astype(np.float64),
        (cls == ground) & keep,
        SURFACE_CELL_M,
        SURFACE_REACH_M,
    )
    surf = grid[key]
    covered = np.isfinite(surf) & keep
    above = covered & ((z - surf) > ABOVE_M)
    n_above = int(above.sum())
    unc = above & np.isin(cls, unclassified)
    veg = above & np.isin(cls, vegetation) if vegetation else np.zeros_like(above)
    return {
        "points": int(keep.sum()),
        "points_with_a_ground_surface": int(covered.sum()),
        "points_over_2m": n_above,
        "unclassified_share_of_points_over_2m": round(float(unc.sum() / n_above), 4)
        if n_above
        else None,
        "unclassified_over_2m_share_of_all_points": round(float(unc.sum() / keep.sum()), 4),
        "vegetation_share_of_points_over_2m": round(float(veg.sum() / n_above), 4)
        if n_above and vegetation
        else None,
        # Where the unclassified points stand: vehicle roofs sit near 2 m, roofs far above.
        "unclassified_over_2m_height_m_p10_p50_p90": [
            round(float(v), 1) for v in np.percentile((z - surf)[unc], [10, 50, 90])
        ]
        if unc.any()
        else None,
    }


def measure(name: str, spec: dict, path: Path) -> dict:
    t0 = time.time()
    x, y, z, cls, ret, meta = read_tile(path)
    codes, counts = np.unique(cls, return_counts=True)
    total = int(counts.sum())
    distribution = {
        int(c): {
            "points": int(k),
            "share": round(float(k / total), 4),
            "name": spec["classes"].get(int(c), "not named by a source found"),
        }
        for c, k in zip(codes, counts, strict=True)
    }

    occupied = np.unique((np.floor(y).astype(np.int64) << 21) | np.floor(x).astype(np.int64)).size
    density = {
        "occupied_1m_cells_m2": int(occupied),
        "points_per_m2": round(total / occupied, 1),
        "first_returns_per_m2": round(int((ret == 1).sum()) / occupied, 1),
    }

    keep_codes = ~np.isin(cls, list(spec["noise"]) + list(spec["not_measured"]))
    pick = voxel_pick(x.astype(np.float64), y.astype(np.float64), z.astype(np.float64), VOXEL_M)
    x, y, z, cls, keep_codes = x[pick], y[pick], z[pick], cls[pick], keep_codes[pick]
    proxy = contamination(
        x, y, z, cls, spec["ground"], spec["unclassified"], spec["vegetation"], keep_codes
    )
    proxy["population"] = f"{VOXEL_M} m voxel-thinned, noise and non-laser codes removed"

    row = {
        "dataset": spec["dataset"],
        "place": spec["place"],
        "url": spec["url"],
        "found_via": spec["found_via"],
        "file_bytes": path.stat().st_size,
        "header": meta,
        "class_table_source": spec["class_table_source"],
        "excluded_codes": {
            **{str(c): "noise" for c in spec["noise"]},
            **{str(c): r for c, r in spec["not_measured"].items()},
        },
        "class_distribution_raw": distribution,
        "density_raw": density,
        "contamination_proxy": proxy,
        "seconds": round(time.time() - t0, 1),
    }
    print(
        f"  {name:24s} {total:>11,} pts  {density['points_per_m2']} pts/m2  "
        f"unclassified over 2 m {proxy['unclassified_share_of_points_over_2m']}  "
        f"vegetation over 2 m {proxy['vegetation_share_of_points_over_2m']}",
        flush=True,
    )
    return row


def san_francisco_yardstick() -> list[dict]:
    rows = []
    for p in sorted(TARGET_PREP.iterdir()):
        if not (p.is_dir() and (p / "meta.json").exists()):
            continue
        tile = CachedTile(p, "sf")
        asprs = np.asarray(tile.asprs[:], dtype=np.int16)
        if not (asprs == 2).any():
            rows.append({"aoi": p.name, "skipped": "no ground class"})
            continue
        x = np.asarray(tile.xy_cm[:, 0], np.float32) / 100.0
        y = np.asarray(tile.xy_cm[:, 1], np.float32) / 100.0
        z = np.asarray(tile.z_cm[:], np.float32) / 100.0
        r = contamination(x, y, z, asprs, 2, [1], [], ~np.isin(asprs, [7, 18]))
        rows.append({"aoi": p.name, **r})
        print(
            f"  sf {p.name:12s} unclassified over 2 m {r['unclassified_share_of_points_over_2m']}",
            flush=True,
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    paths = {name: download(name, spec) for name, spec in TILES.items()}
    if args.download_only:
        return 0
    rows = {name: measure(name, spec, paths[name]) for name, spec in TILES.items()}
    result = {
        "question": "how contaminated is each national classification, measured on one tile",
        "above_ground_m": ABOVE_M,
        "ground_surface": {
            "from": "the delivery's own class 2",
            "cell_m": SURFACE_CELL_M,
            "reach_m": SURFACE_REACH_M,
        },
        "tiles": rows,
        "san_francisco_same_proxy": san_francisco_yardstick(),
        "caveat": "one tile per dataset; a tile is not a country",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
