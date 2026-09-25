"""How much city fits in a repository? Convert a real area and measure the bytes.

The viewer has no backend. Everything it needs is committed or served as a static
file, because the two earlier projects in this portfolio that call an API are
shells when hosted, and here the viewer is the product rather than a bonus. So the
extent is decided by arithmetic before anything is converted, and the arithmetic
needs a real bytes-per-point figure rather than a guess.

Three settings matter, and they are worth a factor of eight between them:

The default encoding is uncompressed. Brotli costs a few seconds more and cuts the
octree by about five times.

A LAS point format carries attributes the viewer never reads: gps time, scan angle,
point source id. Writing position and classification only drops it further.

Display density is a free choice. The model reads whatever the delivery holds, but
a point cloud at 20 points per square metre still looks solid on screen, and each
halving of density doubles the ground the same budget covers.

Run:
    python scripts/investigations/viewer_budget.py
    python scripts/investigations/viewer_budget.py --aoi financial
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.tools import find_potree_converter, tool_report  # noqa: E402

TOOLS = ROOT / "data" / "tools"
DEFAULT_AOI = ROOT / "data" / "target" / "sf" / "mission.laz"
DEFAULT_OUT = ROOT / "data" / "artifacts" / "viewer_budget.json"
WORK = TOOLS / "budget_work"

BUDGETS_MB = (100, 250, 500)
DISPLAY_DENSITIES = (50, 20, 10)


def write_variant(src: Path, dst: Path, minimal: bool, keep_fraction: float, seed: int = 0) -> dict:
    import laspy

    las = laspy.read(src)
    n = len(las.points)
    if keep_fraction < 1.0:
        rng = np.random.default_rng(seed)
        keep = rng.random(n) < keep_fraction
    else:
        keep = np.ones(n, dtype=bool)

    if minimal:
        # Format 0 holds position and classification. The viewer reads nothing else.
        hdr = laspy.LasHeader(version="1.2", point_format=0)
    else:
        hdr = laspy.LasHeader(
            version=str(las.header.version), point_format=las.header.point_format.id
        )
    hdr.offsets, hdr.scales = las.header.offsets, las.header.scales
    try:
        crs = las.header.parse_crs()
        if crs is not None:
            hdr.add_crs(crs)
    except Exception:
        pass

    out = laspy.LasData(hdr)
    out.x = np.asarray(las.x)[keep]
    out.y = np.asarray(las.y)[keep]
    out.z = np.asarray(las.z)[keep]
    out.classification = np.asarray(las.classification)[keep]
    if not minimal and hasattr(out, "intensity"):
        out.intensity = np.asarray(las.intensity)[keep]
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.write(dst)
    return {"points": int(keep.sum()), "laz_bytes": dst.stat().st_size}


def convert(converter: Path, src: Path, out_dir: Path, encoding: str) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    t0 = time.time()
    cmd = [str(converter), str(src), "-o", str(out_dir)]
    if encoding:
        cmd += ["--encoding", encoding]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"PotreeConverter failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    octree_bytes = sum(f.stat().st_size for f in out_dir.iterdir() if f.suffix in (".bin", ".json"))
    return {
        "encoding": meta.get("encoding"),
        "attributes": [a["name"] for a in meta["attributes"]],
        "attribute_count": len(meta["attributes"]),
        "points": int(meta["points"]),
        "octree_bytes": octree_bytes,
        "bytes_per_point": round(octree_bytes / meta["points"], 3),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--budget-mb", type=int, nargs="+", default=list(BUDGETS_MB))
    args = ap.parse_args()

    if not args.aoi.exists():
        raise SystemExit(f"{args.aoi} missing, run scripts/pipeline/fetch_sf_aoi.py")
    converter = find_potree_converter()
    WORK.mkdir(parents=True, exist_ok=True)

    variants = [
        ("full attributes, uncompressed", False, 1.0, ""),
        ("full attributes, brotli", False, 1.0, "BROTLI"),
        ("position and class only, brotli", True, 1.0, "BROTLI"),
    ]
    rows = []
    for label, minimal, keep, encoding in variants:
        laz = WORK / f"{args.aoi.stem}_{'min' if minimal else 'full'}.laz"
        wrote = write_variant(args.aoi, laz, minimal, keep)
        conv = convert(converter, laz, WORK / f"octree_{len(rows)}", encoding or "DEFAULT")
        rows.append({"variant": label, **wrote, **conv})
        print(
            f"{label:34s} {conv['octree_bytes'] / 1e6:>7.1f} MB  "
            f"{conv['bytes_per_point']:>5.2f} bytes/point  "
            f"{conv['attribute_count']} attributes",
            flush=True,
        )

    best = min(rows, key=lambda r: r["bytes_per_point"])
    bpp = best["bytes_per_point"]

    budget = []
    for mb in args.budget_mb:
        points = mb * 1e6 / bpp
        entry = {"budget_mb": mb, "points": int(points), "extents": {}}
        for dens in DISPLAY_DENSITIES:
            side = (points / dens) ** 0.5
            entry["extents"][f"{dens}_pts_per_m2"] = {
                "side_m": round(side, 0),
                "area_km2": round(points / dens / 1e6, 3),
            }
        budget.append(entry)

    result = {
        "question": "how much San Francisco fits in a repository with no backend",
        "converter": "PotreeConverter 2.1.3, BSD-2-Clause",
        "tools_found": tool_report(),
        "measured_on": str(args.aoi.relative_to(ROOT)).replace("\\", "/"),
        "variants": rows,
        "best_bytes_per_point": bpp,
        "best_variant": best["variant"],
        "budget": budget,
        "notes": [
            "the default encoding is uncompressed and costs about five times brotli",
            "a LAS point format carries gps time, scan angle and point source id, "
            "none of which the viewer reads",
            "display density is a free choice and each halving doubles the ground "
            "the same budget covers",
        ],
        "seconds": round(sum(r["seconds"] for r in rows), 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\nbest: {best['variant']} at {bpp:.2f} bytes/point")
    print(
        f"{'budget':>9}{'points':>12}" + "".join(f"{d} pts/m2".rjust(16) for d in DISPLAY_DENSITIES)
    )
    for e in budget:
        line = f"{e['budget_mb']:>7} MB{e['points'] / 1e6:>10.1f}M"
        for d in DISPLAY_DENSITIES:
            v = e["extents"][f"{d}_pts_per_m2"]
            line += f"{v['side_m']:>9.0f} m sq".rjust(16)
        print(line)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
