"""How should the label-free ground surface be built? Measure it on both cities.

Height above ground is the one feature that separates a roof from a road, and it
comes from a grid over all points rather than from any label, so the same code
runs on Surrey and on San Francisco. Two choices in that grid are not details.

**Cell size.** A cell small enough to sit entirely inside a roof has no ground
point in it, so its floor is the roof, so the roof reports nearly zero height. The
feature then says almost nothing about the thing it exists to say.

**Whether the floor is a minimum or a low quantile.** A minimum is not robust.
San Francisco's financial district carries half a percent of points classified low
noise, some 70 m under the street, and one of those drags a whole cell down.

Neither choice can be made on one city. A coarse cell makes buildings visible in
flat Surrey and makes hillside ground in San Francisco read as elevated, and
hillside ground misread as building would look exactly like a domain gap. So this
measures both sides and the rule has to pass on each.

The labels used here are used *here only*, for choosing two numbers. The model
never receives them.

Run:
    python scripts/investigations/ground_cell_choice.py
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

from surrey.data import (  # noqa: E402
    CLASS_NAMES,
    height_above,
    min_z_grid,
    quantile_z_grid,
)

DEFAULT_TRAIN = ROOT / "data" / "dales2" / "train"
DEFAULT_TARGET = ROOT / "data" / "target" / "sf"
DEFAULT_OUT = ROOT / "data" / "artifacts" / "ground_cell_choice.json"

CELLS = (2.0, 5.0, 10.0, 15.0, 25.0, 50.0)
GATE_M = 2.0
QUANTILE = 0.05

GROUND = CLASS_NAMES.index("ground")
BUILDING = [CLASS_NAMES.index(n) for n in ("habitat", "complex", "annex")]
ASPRS_GROUND = 2

# The rule, stated before the numbers are looked at.
MIN_BUILDING_VISIBLE = 0.95  # of Surrey building points must clear the gate
MAX_GROUND_ELEVATED = 0.10  # of San Francisco ground points may clear it


def estimators():
    return {
        "min": min_z_grid,
        f"q{int(QUANTILE * 100):02d}": lambda x, y, z, cell_m: quantile_z_grid(
            x, y, z, cell_m=cell_m, q=QUANTILE
        ),
    }


def read(path: Path):
    import laspy

    las = laspy.read(path)
    return (
        np.asarray(las.x),
        np.asarray(las.y),
        np.asarray(las.z),
        np.asarray(las.classification, dtype=np.int16),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    train_paths = sorted(args.train.glob("*.laz"))[: args.count]
    target_paths = sorted(args.target.glob("*.laz"))
    if not train_paths:
        raise SystemExit(f"no tiles in {args.train}, run scripts/pipeline/fetch_dales2.py")
    if not target_paths:
        raise SystemExit(f"no areas in {args.target}, run scripts/pipeline/fetch_sf_aoi.py")

    t0 = time.time()
    est = estimators()
    surrey: dict[str, dict[float, list[float]]] = {k: {c: [] for c in CELLS} for k in est}
    sf: dict[str, dict[float, list[float]]] = {k: {c: [] for c in CELLS} for k in est}
    sf_rows = []

    print("Surrey, fraction of building points clearing the gate")
    for p in train_paths:
        x, y, z, cls = read(p)
        bldg = np.isin(cls, BUILDING)
        if not bldg.any():
            continue
        for name, fn in est.items():
            for cell in CELLS:
                grid, x0, y0, c = fn(x, y, z, cell_m=cell)
                hag = height_above(x, y, z, grid, x0, y0, c)
                surrey[name][cell].append(float((hag[bldg] > GATE_M).mean()))
        print(f"  {p.stem}", flush=True)

    print("San Francisco, fraction of delivery ground points falsely clearing the gate")
    for p in target_paths:
        x, y, z, cls = read(p)
        g = cls == ASPRS_GROUND
        if g.sum() < 10_000:
            sf_rows.append({"aoi": p.stem, "skipped": "no ground class in this area"})
            print(f"  {p.stem}  skipped, no ground class", flush=True)
            continue
        relief = float(np.percentile(z[g], 98) - np.percentile(z[g], 2))
        row = {
            "aoi": p.stem,
            "ground_points": int(g.sum()),
            "relief_m": round(relief, 1),
            "by_estimator": {},
        }
        for name, fn in est.items():
            row["by_estimator"][name] = {}
            for cell in CELLS:
                grid, x0, y0, c = fn(x, y, z, cell_m=cell)
                hag = height_above(x, y, z, grid, x0, y0, c)
                v = float((hag[g] > GATE_M).mean())
                sf[name][cell].append(v)
                row["by_estimator"][name][str(cell)] = round(v, 4)
        sf_rows.append(row)
        print(f"  {p.stem}  relief {relief:.1f} m", flush=True)

    pooled = []
    for name in est:
        for cell in CELLS:
            b = float(np.mean(surrey[name][cell])) if surrey[name][cell] else float("nan")
            g = float(np.mean(sf[name][cell])) if sf[name][cell] else float("nan")
            gmax = float(np.max(sf[name][cell])) if sf[name][cell] else float("nan")
            pooled.append(
                {
                    "estimator": name,
                    "cell_m": cell,
                    "surrey_building_visible": round(b, 4),
                    "sf_ground_elevated_mean": round(g, 4),
                    "sf_ground_elevated_worst": round(gmax, 4),
                    "passes": bool(b >= MIN_BUILDING_VISIBLE and gmax <= MAX_GROUND_ELEVATED),
                }
            )

    ok = [r for r in pooled if r["passes"]]
    chosen = min(ok, key=lambda r: (r["cell_m"], r["estimator"])) if ok else None

    result = {
        "question": "what ground surface keeps buildings visible in Surrey without "
        "lifting San Francisco's ground off the floor",
        "gate_m": GATE_M,
        "quantile": QUANTILE,
        "cells_tested": list(CELLS),
        "rule": {
            "surrey_building_visible_at_least": MIN_BUILDING_VISIBLE,
            "sf_ground_elevated_worst_area_at_most": MAX_GROUND_ELEVATED,
            "tie_break": "smallest cell, because a coarser grid stops following terrain",
        },
        "chosen": chosen,
        "pooled": pooled,
        "san_francisco": sf_rows,
        "surrey_tiles": [p.stem for p in train_paths],
        "labels_used_here_only": (
            "DALES labels and the delivery's ASPRS ground class "
            "choose these two numbers and are never model inputs"
        ),
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(
        f"\n{'est':>6}{'cell m':>8}{'Surrey bldg vis':>17}"
        f"{'SF gnd lifted avg':>19}{'worst':>9}{'pass':>7}"
    )
    for r in pooled:
        print(
            f"{r['estimator']:>6}{r['cell_m']:>8.0f}"
            f"{r['surrey_building_visible']:>17.1%}"
            f"{r['sf_ground_elevated_mean']:>19.1%}"
            f"{r['sf_ground_elevated_worst']:>9.1%}"
            f"{'yes' if r['passes'] else '':>7}"
        )
    if chosen:
        print(f"\nchosen: {chosen['estimator']} at {chosen['cell_m']:.0f} m cell")
    else:
        print(
            "\nnothing passes. the rule or the approach has to change, not the rule "
            "quietly loosened"
        )
    print(f"wrote {args.out}")
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
