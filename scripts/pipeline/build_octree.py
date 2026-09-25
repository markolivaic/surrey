"""Build the viewer's octree for an area: lidar, NAIP colour, and every variant's class and
confidence as point attributes, so switching variant never reloads geometry.

Built locally from public sources and saved predictions, never downloaded from the host,
so a clone can serve its own copy with surrey.static_server.

Per point, in the LAS handed to PotreeConverter 2.1.3 (BSD-2-Clause) with Brotli:

    position        UTM zone 10N, centimetre scale, as the cache stores it
    rgb             NAIP at the point, sampled at the point minus the area's measured
                    lidar-to-NAIP offset (surrey.registration.naip_offsets), raw values;
                    any display gamma is the shader's business
    slots           each variant's predicted fine class id (uint8) and confidence, the
                    top class probability as round(255 * p) (uint8), in standard LAS
                    fields: slot 0 in classification and user data, slots 1 to 3 in
                    intensity, point source id and scan angle, class in the low byte and
                    confidence in the high byte. Four variants at most.

Why not LAS extra bytes, the obvious place: PotreeConverter 2.1.3 returns them wrong, the
first holding another attribute's value and the rest zeros, in both encodings, while its
metadata still reports the right minimum and maximum. Reproduced on synthetic points in
docs/potreeconverter_extra_bytes.md; standard fields come back exact. The slot map is in each area's manifest.json.

PotreeConverter 2.1.3 crashed, a segmentation fault while chunking, on the 9.0 million
point Golden Gate Park file whenever extra attributes were requested, three times out of
three, and not on the same points split into two files. A three-way split failed
differently ("point to node lookup failed"); two parts worked. So the LAS goes to the converter
in parts of at most PART_POINTS points, split along x at quantiles; the converter merges its inputs.

Surrey, for card 2 (--surrey): the held-out tile's points from the mirror mbendjilali/DALES-2,
which carry the City of Surrey's Open Government Licence, and this project's predictions. The
DALES annotations carry different terms ("Creative Commons 3.0" with a non-commercial
sentence), so a hosted Surrey octree must not contain them: this script never reads a tile's
label array, the LAS classification field holds the first variant's prediction, and the manifest
says so. There is no NAIP in Canada, so colour is left at zero and the divider is not offered
on that card.

Three conversions per area measure what each part costs: position alone, position and
colour, and everything. Bytes per point and per area go to
data/artifacts/octree_budget.json, and every output file gets a SHA-256 in the area's
manifest.json.

Run:
    python scripts/pipeline/build_octree.py --areas ggpark embarcadero --variants returns_ood ret0 m0_seedA
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.registration import naip_offsets  # noqa: E402

TARGET_PREP = ROOT / "data" / "prep" / "target"
DALES_PREP = ROOT / "data" / "prep" / "dales"
PRED = ROOT / "data" / "predictions"
CONVERTER = (
    ROOT
    / "data"
    / "tools"
    / "PotreeConverter"
    / "PotreeConverter_2.1.3_x64_windows"
    / "PotreeConverter.exe"
)
OUT = ROOT / "data" / "viewer"
WORK = ROOT / "data" / "cache" / "octree_las"
ARTIFACT = ROOT / "data" / "artifacts" / "octree_budget.json"
CRS = ROOT / "data" / "artifacts" / "crs_check.json"
VARIANT_DIRS = {"returns_ood": PRED, "ret0": PRED / "ret0"}
PART_POINTS = 5_000_000
SLOT_FIELDS = [
    ("classification", "user data"),
    ("intensity",),
    ("point source id",),
    ("scan angle",),
]
# The four variants the viewer carries, fixed in docs/design/directions.md. The fields above hold
# four and no more; m0prime is never drawn.
VIEWER_VARIANTS = ("returns_ood", "ret0", "m0_seedA", "m1_seedA")


def variant_dir(v: str, city: str = "sf") -> Path:
    # On Surrey the return ratio is already 0 on every point, so ret0's Surrey predictions are the
    # as-delivered ones (scripts/pipeline/score_variant.py records the same).
    if city == "surrey" and v == "ret0":
        return PRED
    return VARIANT_DIRS.get(v, PRED / v)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def naip_at(tile: CachedTile, offset: dict | None) -> np.ndarray:
    """uint8 RGB per point from the cached NAIP tiles, zero where none covers it."""
    import rasterio
    from pyproj import Transformer

    ux, uy = tile.utm_xy(np.arange(tile.n))
    if offset:
        ux, uy = ux - offset["east_m"], uy - offset["north_m"]
    out = np.zeros((tile.n, 3), np.uint8)
    got = np.zeros(tile.n, bool)
    for path in sorted((ROOT / "data" / "cache" / "naip").glob("*.tif")):
        with rasterio.open(path) as ds:
            t = Transformer.from_crs(tile.epsg, ds.crs.to_epsg(), always_xy=True)
            ix, iy = t.transform(ux, uy)
            inv = ~ds.transform
            cols = np.floor(inv.a * ix + inv.b * iy + inv.c).astype(np.int64)
            rows = np.floor(inv.d * ix + inv.e * iy + inv.f).astype(np.int64)
            ok = (~got) & (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
            if not ok.any():
                continue
            arr = ds.read([1, 2, 3])
            out[ok] = arr[:, rows[ok], cols[ok]].T
            got |= ok
    if not got.all():
        print(f"  {int((~got).sum()):,} points outside NAIP, left black")
    return out


def write_las(
    path: Path,
    tile: CachedTile,
    rgb: np.ndarray | None,
    extras: list[tuple[np.ndarray, np.ndarray]],
) -> list[Path]:

    parts = max(1, int(np.ceil(tile.n / PART_POINTS)))
    ux_all, _ = tile.utm_xy(np.arange(tile.n))
    edges = np.quantile(ux_all, np.linspace(0, 1, parts + 1))
    band = np.clip(np.searchsorted(edges, ux_all, side="right") - 1, 0, parts - 1)
    paths = []
    for k in range(parts):
        paths.append(path.with_name(f"{path.stem}_part{k}.las"))
        _write_part(paths[-1], tile, np.flatnonzero(band == k), rgb, extras)
    return paths


def _write_part(
    path: Path,
    tile: CachedTile,
    idx: np.ndarray,
    rgb: np.ndarray | None,
    extras: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    import laspy

    ux, uy = tile.utm_xy(idx)
    z = tile.z_abs(idx)
    hdr = laspy.LasHeader(version="1.4", point_format=7 if rgb is not None else 6)
    hdr.offsets = np.array([float(tile.origin[0]), float(tile.origin[1]), float(tile.origin[2])])
    hdr.scales = np.array([0.01, 0.01, 0.01])
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = ux, uy, z
    if rgb is not None:
        # LAS colour is 16 bit; the byte value is stored in the high byte, as LAS readers expect.
        las.red, las.green, las.blue = (rgb[idx, i].astype(np.uint16) * 257 for i in range(3))
    for slot, (cls, conf) in enumerate(extras):
        c, q = cls[idx], conf[idx]
        if slot == 0:
            las.classification, las.user_data = c, q
        else:
            packed = c.astype(np.uint16) | (q.astype(np.uint16) << 8)
            if slot == 1:
                las.intensity = packed
            elif slot == 2:
                las.point_source_id = packed
            else:
                las.scan_angle = packed.view(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)


def convert(srcs: list[Path], out_dir: Path, attributes: list[str]) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        str(CONVERTER),
        *map(str, srcs),
        "-o",
        str(out_dir),
        "--encoding",
        "BROTLI",
        "--attributes",
        *attributes,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"PotreeConverter failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    # The converter's own log is not part of what a page serves.
    (out_dir / "log.txt").unlink(missing_ok=True)
    files = {p.name: p.stat().st_size for p in sorted(out_dir.iterdir()) if p.is_file()}
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    written = [a["name"] for a in meta["attributes"]]
    if len(written) != len(set(written)):
        raise SystemExit(f"the converter wrote an attribute twice: {written}")
    return {
        "attributes_requested": attributes,
        "attributes_written": written,
        "files": files,
        "bytes": sum(files.values()),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--areas", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--skip-costing", action="store_true", help="only the full conversion")
    ap.add_argument(
        "--surrey",
        action="store_true",
        help="the areas are held-out Surrey tiles; points and predictions only, never DALES labels",
    )
    args = ap.parse_args()
    offsets = naip_offsets(json.loads(CRS.read_text(encoding="utf-8")))
    budget = (
        json.loads(ARTIFACT.read_text(encoding="utf-8")) if ARTIFACT.exists() else {"areas": {}}
    )
    budget["converter"] = "PotreeConverter 2.1.3, BSD-2-Clause, --encoding BROTLI"
    budget["costing_note"] = (
        "position is measured with the constant LAS gps time field, the smallest "
        "attribute list the converter accepts; each marginal figure is the difference "
        "between two whole conversions"
    )

    for area in args.areas:
        city = "surrey" if args.surrey else "sf"
        tile = CachedTile(
            (DALES_PREP if args.surrey else TARGET_PREP) / area, "test" if args.surrey else "sf"
        )
        print(f"{area}: {tile.n:,} points", flush=True)
        off = offsets.get(area)
        rgb = np.zeros((tile.n, 3), np.uint8) if args.surrey else naip_at(tile, off)
        if len(args.variants) > len(SLOT_FIELDS):
            raise SystemExit(f"at most {len(SLOT_FIELDS)} variants fit the LAS fields")
        if list(args.variants) != list(VIEWER_VARIANTS[: len(args.variants)]):
            raise SystemExit(f"variants must be a prefix of {VIEWER_VARIANTS}, in that order")
        extras, present = [], {}
        for v in args.variants:
            z = np.load(variant_dir(v, city) / f"{city}_{area}.npz")
            if z["pred"].shape[0] != tile.n:
                raise SystemExit(f"{v} {area}: prediction count does not match the cache")
            cls_ = z["pred"].astype(np.uint8)
            extras.append(
                (
                    cls_,
                    np.clip(np.round(z["conf"].astype(np.float32) * 255), 0, 255).astype(np.uint8),
                )
            )
            cls, n = np.unique(cls_, return_counts=True)
            present[v] = {int(c): int(k) for c, k in zip(cls, n, strict=True)}
        slot_attrs = [f for fields in SLOT_FIELDS[: len(args.variants)] for f in fields]

        las_path = WORK / f"{area}.las"
        las_paths = write_las(las_path, tile, rgb, extras)
        # Position is always written; naming it again makes the converter store it twice.
        full = convert(las_paths, OUT / area, ["rgb", *slot_attrs])
        row = {
            "points": tile.n,
            "variants": args.variants,
            "naip_offset_m": off,
            "full": full | {"bytes_per_point": round(full["bytes"] / tile.n, 3)},
            "classes_present": present,
        }
        if not args.skip_costing:
            for label, attrs in (("position", ["gps-time"]), ("position_rgb", ["rgb"])):
                c = convert(las_paths, WORK / f"{area}_{label}", attrs)
                row[label] = c | {"bytes_per_point": round(c["bytes"] / tile.n, 3)}
                shutil.rmtree(WORK / f"{area}_{label}")
            row["marginal_bytes_per_point"] = {
                "position": row["position"]["bytes_per_point"],
                "rgb": round(
                    row["position_rgb"]["bytes_per_point"] - row["position"]["bytes_per_point"], 3
                ),
                "per_variant_class_and_confidence": round(
                    (row["full"]["bytes_per_point"] - row["position_rgb"]["bytes_per_point"])
                    / len(args.variants),
                    3,
                ),
            }
        manifest = {
            "area": area,
            "points": tile.n,
            "variants": args.variants,
            "encoding": "BROTLI",
            "city": city,
            "point_source": (
                "mirror mbendjilali/DALES-2, City of Surrey Open Government Licence"
                if args.surrey
                else "USGS 3DEP CA_SanFrancisco_1_B23, public domain"
            ),
            "colour_source": None if args.surrey else "NAIP 2022, public domain",
            "dales_labels_included": False,
            "slots": [
                {
                    "variant": v,
                    "fields": list(SLOT_FIELDS[i]),
                    "packing": "class, confidence"
                    if i == 0
                    else "class low byte, confidence high byte",
                }
                for i, v in enumerate(args.variants)
            ],
            "attributes": full["attributes_written"],
            "classes_present": present,
            "files": {
                p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in sorted((OUT / area).iterdir())
                if p.is_file()
            },
        }
        (OUT / area / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        budget["areas"][area] = row
        for lp in las_paths:
            lp.unlink()
        print(
            f"  {full['bytes'] / 1e6:.1f} MB, {row['full']['bytes_per_point']} bytes per point, "
            f"{row.get('marginal_bytes_per_point')}",
            flush=True,
        )

    budget["total_bytes"] = sum(r["full"]["bytes"] for r in budget["areas"].values())
    ARTIFACT.write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
