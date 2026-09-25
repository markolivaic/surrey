"""What does the footprint metric score with no model at all?

The transfer score checks the predicted building class against Microsoft Building
Footprints, because the 3DEP delivery has no building labels. In a dense city a
large share of ALL points land inside a footprint, so a model that called every
point a building would already score well. That floor has to exist as a number
before any model does, or a good looking precision cannot be told apart from the
metric flattering itself.

This produces the baseline the README quotes. Three things about how:

It reads the same prepared caches the model reads, and uses the same label-free
height above ground. An earlier version read the raw LAZ and gated on the
delivery's own ground class, which is a different population, and a lift measured
against a null computed over a different population is not a lift at all.

It imports the same scoring functions the transfer score uses. If the null and
the measurement came from two copies of the arithmetic they could drift apart,
and the published lift would be the difference between two unrelated numbers.

It reports a named baseline, `everything_above_2m`, which is exactly the model
this metric must not reward: call every point over 2 m a building.

Run:
    python scripts/pipeline/footprint_null.py
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

from surrey.cache import CachedTile  # noqa: E402
from surrey.footprints import (  # noqa: E402
    clip_to_box,
    footprint_coverage,
    load_footprints,
    score_building,
)

DEFAULT_PREP = ROOT / "data" / "prep" / "target"
DEFAULT_FOOTPRINTS = ROOT / "data" / "target" / "sf_footprints.geojson"
DEFAULT_OUT = ROOT / "data" / "artifacts" / "footprint_null.json"

ASPRS_NOISE = (7, 18)
HAG_GATE_M = 2.0
COVERAGE_CELL_M = 1.0

# Areas whose morphology means no building should be found there. Used to say
# which numbers belong in the built-area range and which are the control.
CONTROL_AOIS = {"ggpark", "baybridge"}


def fmt(v: float | None, width: int = 9, places: int = 3) -> str:
    """Format a number, or say plainly that there was nothing to measure.

    An area with no footprints in it has no precision. Printing that as 0.000
    would read as a failed prediction rather than an absent reference.
    """
    return f"{'n/a':>{width}}" if v is None else f"{v:>{width}.{places}f}"


def measure(tile: CachedTile, polys) -> dict:
    idx = np.arange(tile.n)
    asprs = np.asarray(tile.asprs[:], dtype=np.int16)
    hag = np.asarray(tile.hag_cm[:], dtype=np.float32) / 100.0
    clean = ~np.isin(asprs, ASPRS_NOISE)
    tall = clean & (hag > HAG_GATE_M)

    ux, uy = tile.utm_xy(idx)
    clipped, box_area = clip_to_box(polys, ux, uy)

    everything = np.ones(tile.n, dtype=bool)
    all_points = score_building(clipped, ux, uy, everything, eligible=clean)
    above_2m = score_building(clipped, ux, uy, everything, eligible=tall)

    cover_null = footprint_coverage(clipped, ux, uy, tall, cell_m=COVERAGE_CELL_M)
    cover_null.pop("per_footprint", None)

    footprint_area = sum(float(p.area) for p in clipped)
    return {
        "aoi": tile.name,
        "is_control": tile.name in CONTROL_AOIS,
        "points": tile.n,
        "aoi_area_m2": round(box_area, 1),
        "footprints_in_aoi": len(clipped),
        "footprint_area_fraction": round(footprint_area / box_area, 4) if box_area else None,
        "points_over_2m_fraction": round(float(tall.mean()), 4),
        "noise_fraction": round(float((~clean).mean()), 5),
        "baseline_everything": {
            "null_precision": all_points["null_precision"],
            "eligible_points": all_points["eligible_points"],
        },
        "baseline_everything_above_2m": {
            "null_precision": above_2m["null_precision"],
            "eligible_points": above_2m["eligible_points"],
            "null_recall_touch": above_2m["null_recall"],
            "null_coverage_median": cover_null["coverage_median"],
            "null_coverage_p10": cover_null["coverage_p10"],
            "footprints_scored": cover_null["footprints_scored"],
            "footprints_too_small": cover_null["footprints_too_small"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    ap.add_argument("--footprints", type=Path, default=DEFAULT_FOOTPRINTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.footprints.exists():
        raise SystemExit(f"{args.footprints} missing, run scripts/pipeline/extract_footprints.py")
    tiles = [
        CachedTile(p, "sf")
        for p in sorted(args.prep.iterdir())
        if p.is_dir() and (p / "meta.json").exists()
    ]
    if not tiles:
        raise SystemExit(
            f"no prepared target tiles in {args.prep}, run scripts/pipeline/prep_tiles.py"
        )

    t0 = time.time()
    print(f"footprints from {args.footprints.name}, reprojecting to EPSG:{tiles[0].epsg}")
    polys = load_footprints(args.footprints, tiles[0].epsg)
    print(f"  {len(polys):,} polygons\n")

    print(
        f"{'aoi':12s}{'cover':>8}{'null all':>10}{'null >2m':>10}"
        f"{'touch':>8}{'cov med':>9}  control"
    )
    rows = []
    for tile in tiles:
        r = measure(tile, polys)
        rows.append(r)
        b = r["baseline_everything_above_2m"]
        print(
            f"{r['aoi']:12s}{fmt(r['footprint_area_fraction'], 8)}"
            f"{fmt(r['baseline_everything']['null_precision'], 10)}"
            f"{fmt(b['null_precision'], 10)}{fmt(b['null_recall_touch'], 8)}"
            f"{fmt(b['null_coverage_median'])}  {'yes' if r['is_control'] else ''}",
            flush=True,
        )

    built = [r for r in rows if not r["is_control"]]
    control = [r for r in rows if r["is_control"]]
    prec = [
        r["baseline_everything_above_2m"]["null_precision"]
        for r in built
        if r["baseline_everything_above_2m"]["null_precision"] is not None
    ]
    touch = [
        r["baseline_everything_above_2m"]["null_recall_touch"]
        for r in rows
        if r["baseline_everything_above_2m"]["null_recall_touch"] is not None
    ]

    summary = {
        "built_area_aois": [r["aoi"] for r in built],
        "control_aois": [r["aoi"] for r in control],
        "everything_above_2m_precision_range_built": [min(prec), max(prec)] if prec else None,
        "everything_above_2m_precision_control": {
            r["aoi"]: r["baseline_everything_above_2m"]["null_precision"] for r in control
        },
        "recall_touch_range": [min(touch), max(touch)] if touch else None,
        "aois_with_no_footprints": [r["aoi"] for r in rows if r["footprints_in_aoi"] == 0],
    }

    result = {
        "question": "what does this metric score with no model at all",
        "every_number_here_is_a_null": (
            "this whole artefact is the floor. Keys are "
            "named null_* so that a guard looking for an "
            "unpaired score cannot mistake one for a result"
        ),
        "footprints_source": str(args.footprints.relative_to(ROOT)).replace("\\", "/"),
        "footprints_license": "ODbL-1.0",
        "hag_gate_m": HAG_GATE_M,
        "hag_definition": (
            "label free minimum-z grid over all points, the same "
            "surface the model sees, not the delivery's ground class"
        ),
        "coverage_cell_m": COVERAGE_CELL_M,
        "asprs_noise_excluded": list(ASPRS_NOISE),
        "notes": [
            "baseline_everything_above_2m is the score of a model that calls every "
            "point over 2 m a building. Any measured precision is read against it.",
            "recall_touch counts a footprint as found when one point lands in it, "
            "which every building wins outright. It is reported to show that it is "
            "useless, and the transfer score uses area coverage instead.",
            "coverage has a high null too, because lidar covers a footprint whatever "
            "the labels say. It catches under-covering only; precision catches "
            "over-prediction. The pair covers both directions, neither alone does.",
        ],
        "summary": summary,
        "aois": rows,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(
        f"\neverything above 2 m, built areas: precision "
        f"{summary['everything_above_2m_precision_range_built'][0]:.3f} to "
        f"{summary['everything_above_2m_precision_range_built'][1]:.3f}"
    )
    for name, v in summary["everything_above_2m_precision_control"].items():
        print(f"everything above 2 m, {name} control: precision {v:.3f}")
    print(
        f"touch recall across every area: "
        f"{summary['recall_touch_range'][0]:.3f} to {summary['recall_touch_range'][1]:.3f}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
