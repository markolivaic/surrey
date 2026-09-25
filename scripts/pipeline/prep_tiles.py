"""Preprocess tiles once into compact caches, identically for both cities.

Decoding 40 LAZ tiles takes minutes and training needs them many times over, so
this runs the label-free preprocessing once and writes arrays that memory-map.

Everything is quantised to 1 cm, which is exactly the scale the source LAS files
already use, so nothing is lost. Storing float64 coordinates would cost four
times the disk for precision that was never in the file.

The preprocessing is the same code on Surrey and on San Francisco. That is the
point: a difference in preparation is indistinguishable from a difference between
cities once the numbers are out.

Run:
    python scripts/pipeline/prep_tiles.py --source dales
    python scripts/pipeline/prep_tiles.py --source target
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CELL_M, write_tile  # noqa: E402
from surrey.data import (  # noqa: E402
    GROUND_CELL_M,
    N_CLASSES,
    VOXEL_M,
    height_above,
    quantile_z_grid,
    voxel_pick,
)
from surrey.flags import dropped_counts, flag_bits, kept  # noqa: E402

SOURCES = {
    "dales": {
        "glob": [
            ("train", ROOT / "data" / "dales2" / "train"),
            ("test", ROOT / "data" / "dales2" / "test"),
        ],
        "out": ROOT / "data" / "prep" / "dales",
        "labelled": True,
    },
    "target": {
        "glob": [("sf", ROOT / "data" / "target" / "sf")],
        "out": ROOT / "data" / "prep" / "target",
        "labelled": False,
    },
}
XY_MAX_CM = 65535
HAG_MIN_CM, HAG_MAX_CM = -32768, 32767


def prep_one(
    path: Path, split: str, out_dir: Path, labelled: bool, voxel_m: float, seed: int
) -> dict:
    import laspy

    rng = np.random.default_rng(seed)
    las = laspy.read(path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    n_raw = x.size

    raw_area = float((x.max() - x.min()) * (y.max() - y.min()))
    # Points with a dropped LAS flag go before anything else, so the thinning never picks one. `keep`
    # indexes the file's own point order, as every read below expects.
    bits = flag_bits(las)
    usable = np.flatnonzero(kept(bits))
    dropped = dropped_counts(bits, np.asarray(las.classification))
    keep = usable[voxel_pick(x[usable], y[usable], z[usable], voxel_m, rng)]
    x, y, z = x[keep], y[keep], z[keep]

    # Height above ground is NOT a model feature, see backend/surrey/data.py.
    # It is stored only for the 2 m gate used when scoring against footprints, and
    # it uses the robust quantile floor rather than a minimum.
    grid, gx, gy, cell = quantile_z_grid(x, y, z, cell_m=GROUND_CELL_M)
    hag = height_above(x, y, z, grid, gx, gy, cell)

    nret = np.asarray(las.number_of_returns, dtype=np.float32)[keep]
    rnum = np.asarray(las.return_number, dtype=np.float32)[keep]
    ret = np.divide(rnum, np.maximum(nret, 1.0))

    x0, y0 = float(x.min()), float(y.min())
    xy_cm_f = np.stack([(x - x0) * 100.0, (y - y0) * 100.0], axis=1)
    xy_over = int((xy_cm_f > XY_MAX_CM).sum())
    xy_cm = np.clip(xy_cm_f, 0, XY_MAX_CM).astype(np.uint16)

    hag_cm_f = hag * 100.0
    hag_clipped = int(((hag_cm_f < HAG_MIN_CM) | (hag_cm_f > HAG_MAX_CM)).sum())
    hag_cm = np.clip(hag_cm_f, HAG_MIN_CM, HAG_MAX_CM).astype(np.int16)

    asprs = np.asarray(las.classification, dtype=np.uint8)[keep]
    if labelled:
        label = asprs.astype(np.int8)
        label[asprs >= N_CLASSES] = -1
    else:
        label = np.full(asprs.shape, -1, dtype=np.int8)

    # The absolute origin has to travel with the cache. Without it a prediction
    # cannot be put back on the map, and the footprint score needs the points in
    # the CRS the footprints are in.
    try:
        crs = las.header.parse_crs()
        epsg = crs.to_epsg() if crs is not None else None
    except Exception:
        epsg = None

    kept_area = float((x.max() - x.min()) * (y.max() - y.min()))
    written = write_tile(
        out_dir / path.stem,
        {
            "xy_cm": xy_cm,
            "z_cm": np.clip((z - z.min()) * 100.0, 0, 4_294_967_295),
            "hag_cm": hag_cm,
            "ret_q": np.clip(ret * 255.0, 0, 255),
            "label": label,
            "asprs": asprs,
            "flags": bits[keep],
        },
        {
            "name": path.stem,
            "split": split,
            "source": path.resolve().relative_to(ROOT).as_posix()
            if path.resolve().is_relative_to(ROOT)
            else path.name,
            "point_format": int(las.header.point_format.id),
            "origin": [x0, y0, float(z.min())],
            "epsg": epsg,
            "voxel_m": voxel_m,
            "points_raw": int(n_raw),
            "density_raw": round(n_raw / raw_area, 1) if raw_area > 0 else None,
            "density_kept": round(x.size / kept_area, 1) if kept_area > 0 else None,
        },
        cell_m=CELL_M,
    )

    kept_area = float((x.max() - x.min()) * (y.max() - y.min()))
    rec = {
        "name": path.stem,
        "split": split,
        "source": written["source"],
        "point_format": written["point_format"],
        "points_raw": int(n_raw),
        "points_dropped_by_flag": dropped,
        "points_kept": int(x.size),
        "kept_fraction": round(float(x.size / n_raw), 4),
        "density_raw": written["density_raw"],
        "density_kept": written["density_kept"],
        "hag_min_m": round(float(hag.min()), 2),
        "hag_max_m": round(float(hag.max()), 2),
        "xy_clipped": xy_over,
        "hag_clipped": hag_clipped,
        "origin": [round(x0, 3), round(y0, 3), round(float(z.min()), 3)],
        "epsg": epsg,
        "cells": written["cells"],
        "cache": f"data/prep/{out_dir.name}/{path.stem}",
        "cache_bytes": written["bytes"],
    }
    if xy_over or hag_clipped:
        print(f"    WARNING {path.stem}: clipped xy={xy_over} hag={hag_clipped}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--voxel-m", type=float, default=VOXEL_M)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="redo tiles already cached")
    args = ap.parse_args()

    spec = SOURCES[args.source]
    out_dir = spec["out"]
    artifact = ROOT / "data" / "artifacts" / f"prep_{args.source}.json"

    jobs = []
    for split, d in spec["glob"]:
        if not d.exists():
            continue
        jobs += [(p, split) for p in sorted(d.glob("*.laz"))]
    # An area measured and dropped (fetch_sf_aoi.EXCLUDED) is never prepared. Each tile's thinning is
    # seeded by its place in this list, so one extra file would change every tile after it.
    if args.source == "target":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fetch_sf_aoi import EXCLUDED

        jobs = [(p, split) for p, split in jobs if p.stem not in EXCLUDED]
    if not jobs:
        raise SystemExit(f"nothing to preprocess for source {args.source!r}")

    done = {}
    if artifact.exists() and not args.force:
        done = {r["name"]: r for r in json.loads(artifact.read_text())["tiles"]}

    t0 = time.time()
    records = []
    for i, (p, split) in enumerate(jobs, 1):
        cached = done.get(p.stem)
        if cached and (ROOT / cached["cache"] / "meta.json").exists():
            records.append(cached)
            print(f"  [{i:>2}/{len(jobs)}] {p.stem:12s} cached", flush=True)
            continue
        r = prep_one(p, split, out_dir, spec["labelled"], args.voxel_m, args.seed + i)
        records.append(r)
        print(
            f"  [{i:>2}/{len(jobs)}] {p.stem:12s} {r['points_raw']:>10,} -> "
            f"{r['points_kept']:>9,}  ({r['kept_fraction']:.2f})  "
            f"{r['density_raw']:>6.1f} -> {r['density_kept']:>5.1f} pts/m2",
            flush=True,
        )

    total_kept = sum(r["points_kept"] for r in records)
    total_raw = sum(r["points_raw"] for r in records)
    dens = [r["density_kept"] for r in records if r["density_kept"]]
    result = {
        "source": args.source,
        "voxel_m": args.voxel_m,
        "tiles": records,
        "points_raw_total": total_raw,
        "points_kept_total": total_kept,
        "kept_fraction_overall": round(total_kept / total_raw, 4),
        "density_kept": {
            "min": round(min(dens), 1),
            "median": round(float(np.median(dens)), 1),
            "max": round(max(dens), 1),
        },
        "cache_bytes_total": sum(r["cache_bytes"] for r in records),
        "seconds": round(time.time() - t0, 1),
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(
        f"\n{total_raw:,} -> {total_kept:,} points "
        f"({result['kept_fraction_overall']:.2f} kept at {args.voxel_m} m voxels)"
    )
    print(
        f"density after voxel: min {result['density_kept']['min']}  "
        f"median {result['density_kept']['median']}  max {result['density_kept']['max']} pts/m2"
    )
    print(f"cache {result['cache_bytes_total'] / 1e9:.2f} GB")
    print(f"wrote {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
