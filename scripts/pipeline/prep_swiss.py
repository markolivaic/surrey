"""Prepare the swissSURFACE3D tiles M1 trains on, the way DALES was prepared.

Tiles come from data/artifacts/swiss_tile_selection.json, chosen by a rule fixed before
it ran. Each 1 km tile is split into four 500 m sub-tiles: the cache stores horizontal
position as uint16 centimetres, which holds 655 m, and a 500 m tile is also the size of
a DALES tile, so drawing a tile at random draws a square metre at random in both sources.

Same voxel (0.15 m, one random point per voxel), same label-free height gate, same
cache layout as scripts/pipeline/prep_tiles.py. Labels go through surrey.sources.map_swiss: the
decision and reason for every code are there. The return ratio column is written as 0,
as the three channel model never reads it.

Writes data/prep/swiss/<tile>_<col><row>/ and data/artifacts/prep_swiss.json with
point counts, density before and after thinning, and the label mix, per sub-tile.

Run:
    python scripts/pipeline/prep_swiss.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import write_tile  # noqa: E402
from surrey.data import (  # noqa: E402
    GROUND_CELL_M,
    VOXEL_M,
    height_above,
    quantile_z_grid,
    voxel_pick,
)
from surrey.flags import dropped_counts, flag_bits, kept  # noqa: E402
from surrey.sources import M1_CLASS_NAMES, SUPERCLASSES, SWISS_2024, map_swiss  # noqa: E402

SELECTION = ROOT / "data" / "artifacts" / "swiss_tile_selection.json"
CACHE = ROOT / "data" / "cache" / "swiss"
OUT = ROOT / "data" / "prep" / "swiss"
ARTIFACT = ROOT / "data" / "artifacts" / "prep_swiss.json"
HEADERS = {"User-Agent": "surrey-research/0.1 (github.com/markolivaic)"}
SUB_M = 500.0


def download(url: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / url.rsplit("/", 1)[-1]
    head = urllib.request.urlopen(
        urllib.request.Request(url, method="HEAD", headers=HEADERS), timeout=60
    )
    size = int(head.headers["Content-Length"])
    if target.exists() and target.stat().st_size == size:
        return target
    part = target.with_suffix(".part")
    with (
        urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r,
        part.open("wb") as fh,
    ):
        while block := r.read(1 << 22):
            fh.write(block)
    if part.stat().st_size != size:
        raise SystemExit(f"{url}: got {part.stat().st_size} bytes, expected {size}")
    part.replace(target)
    return target


def label_mix(labels: np.ndarray) -> dict:
    names = dict(enumerate(M1_CLASS_NAMES)) | dict.fromkeys(SUPERCLASSES, "building (superclass)")
    names[-1] = "ignore"
    codes, counts = np.unique(labels, return_counts=True)
    return {
        names[int(c)]: round(float(k / labels.size), 4) for c, k in zip(codes, counts, strict=True)
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    # The held-out tile is prepared the same way into its own directory and artefact, so nothing that
    # reads the training list can reach it by accident.
    ap.add_argument("--selection", type=Path, default=SELECTION)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    args = ap.parse_args()
    t0 = time.time()
    selected = json.loads(args.selection.read_text(encoding="utf-8"))["selected"]
    records, unknown_total, dropped_by_tile = [], {}, {}
    for k, tile in enumerate(selected):
        path = download(tile["url"])
        las = laspy.read(path)
        # Points with a dropped LAS flag go first (surrey.flags): in these tiles that is the synthetic
        # water and the little synthetic ground, which swisstopo placed rather than measured.
        bits = flag_bits(las)
        usable = kept(bits)
        tile_dropped = dropped_counts(bits, np.asarray(las.classification))
        x = np.asarray(las.x, np.float64)[usable]
        y = np.asarray(las.y, np.float64)[usable]
        z = np.asarray(las.z, np.float64)[usable]
        codes = np.asarray(las.classification, np.int16)[usable]
        flags = bits[usable]
        ox, oy = np.floor(x.min() / 1000.0) * 1000.0, np.floor(y.min() / 1000.0) * 1000.0
        dropped_by_tile[tile["id"]] = {
            "points_in_file": int(bits.size),
            "dropped_by_flag": tile_dropped,
        }
        for i in range(2):
            for j in range(2):
                m = (
                    (x >= ox + i * SUB_M)
                    & (x < ox + (i + 1) * SUB_M)
                    & (y >= oy + j * SUB_M)
                    & (y < oy + (j + 1) * SUB_M)
                )
                if m.sum() < 100_000:
                    continue
                name = f"{tile['id'].replace('swisssurface3d_', 'ch')}_{i}{j}"
                rng = np.random.default_rng(args.seed + 10 * k + 2 * i + j)
                xs, ys, zs, cs, fs = x[m], y[m], z[m], codes[m], flags[m]
                keep = voxel_pick(xs, ys, zs, VOXEL_M, rng)
                xs, ys, zs, cs, fs = xs[keep], ys[keep], zs[keep], cs[keep], fs[keep]
                grid, gx, gy, cell = quantile_z_grid(xs, ys, zs, cell_m=GROUND_CELL_M)
                hag = height_above(xs, ys, zs, grid, gx, gy, cell)
                labels, unknown = map_swiss(cs)
                for c, n in unknown.items():
                    unknown_total[c] = unknown_total.get(c, 0) + n
                x0, y0 = ox + i * SUB_M, oy + j * SUB_M
                write_tile(
                    args.out_dir / name,
                    {
                        "xy_cm": np.clip(
                            np.stack([(xs - x0) * 100.0, (ys - y0) * 100.0], 1), 0, 65535
                        ),
                        "z_cm": np.clip((zs - zs.min()) * 100.0, 0, 4_294_967_295),
                        "hag_cm": np.clip(hag * 100.0, -32768, 32767),
                        "ret_q": np.zeros(xs.size),
                        "label": labels,
                        "asprs": np.clip(cs, 0, 255),
                        "flags": fs,
                    },
                    {
                        "name": name,
                        "split": "train",
                        "source": tile["url"],
                        "epsg": 2056,
                        "origin": [x0, y0, float(zs.min())],
                        "voxel_m": VOXEL_M,
                        "city": tile["city"],
                        "flight_day": tile["flight_day"],
                    },
                )
                raw_n, kept_n = int(m.sum()), int(keep.size)
                records.append(
                    {
                        "name": name,
                        "city": tile["city"],
                        "tile": tile["id"],
                        "points_raw": raw_n,
                        "points_kept": kept_n,
                        "water_points_kept": int((cs == 9).sum()),
                        "density_raw": round(raw_n / (SUB_M * SUB_M), 1),
                        "density_kept": round(kept_n / (SUB_M * SUB_M), 1),
                        "label_mix": label_mix(labels),
                        "cache": f"{args.out_dir.resolve().relative_to(ROOT).as_posix()}/{name}",
                    }
                )
                print(
                    f"  {name:28s} {raw_n:>10,} -> {kept_n:>9,}  {records[-1]['density_raw']} -> "
                    f"{records[-1]['density_kept']} pts/m2  {records[-1]['label_mix']}",
                    flush=True,
                )

    # A sub-tile left from an earlier run and not written by this one (the filter can take one below
    # the minimum) would otherwise still be read by training.
    written = {r["name"] for r in records}
    for stale in sorted(d for d in args.out_dir.iterdir() if d.is_dir() and d.name not in written):
        shutil.rmtree(stale)
        print(f"  removed {stale.name}, not written by this run", flush=True)

    result = {
        "source": "swissSURFACE3D 2024 classification, tiles from swiss_tile_selection.json",
        "licence": "swisstopo free geodata, attribution ©swisstopo",
        "voxel_m": VOXEL_M,
        "sub_tile_m": SUB_M,
        "class_table": {
            str(c): {"name": n, "m1_label": lab, "reason": why}
            for c, (n, lab, why) in SWISS_2024.items()
        },
        "codes_not_in_table_ignored": unknown_total,
        "flags_dropped": "synthetic and withheld (surrey.flags); raw points each removed, per class",
        "dropped_by_tile": dropped_by_tile,
        "tiles": records,
        "points_kept_total": sum(r["points_kept"] for r in records),
        "seconds": round(time.time() - t0, 1),
    }
    args.artifact.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
