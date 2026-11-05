"""Measure DALES-2 across every tile, not one of them.

A class distribution taken from one tile describes that tile. This reads all 40
tiles and reports the spread between them as well as the pooled total.

Run:
    python scripts/pipeline/dales_stats.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TILES = ROOT / "data" / "dales2"
DEFAULT_OUT = ROOT / "data" / "artifacts" / "dales_stats.json"

# From the DALES-2 dataset card. Ids are the values in the classification field.
CLASSES = {
    0: "ground",
    1: "vegetation",
    2: "car",
    3: "powerline",
    4: "fence",
    5: "tree",
    6: "pickup",
    7: "van-truck",
    8: "heavy-duty",
    9: "utility-pole",
    10: "light-pole",
    11: "traffic-pole",
    12: "habitat",
    13: "complex",
    14: "annex",
}
N_CLASSES = len(CLASSES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles", type=Path, default=DEFAULT_TILES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    import laspy

    paths = sorted(args.tiles.rglob("*.laz"))
    if not paths:
        raise SystemExit(f"no tiles under {args.tiles}, run scripts/pipeline/fetch_dales2.py first")

    per_tile = []
    pooled = np.zeros(N_CLASSES + 1, dtype=np.int64)  # last slot catches anything unexpected
    t0 = time.time()

    for i, p in enumerate(paths, 1):
        las = laspy.read(p)
        cls = np.asarray(las.classification, dtype=np.int64)
        n = cls.size
        counts = np.zeros(N_CLASSES + 1, dtype=np.int64)
        in_range = (cls >= 0) & (cls < N_CLASSES)
        np.add.at(counts, cls[in_range], 1)
        counts[N_CLASSES] = int((~in_range).sum())
        pooled += counts

        mins, maxs = las.header.mins, las.header.maxs
        area = float((maxs[0] - mins[0]) * (maxs[1] - mins[1]))
        per_tile.append(
            {
                "split": p.parent.name,
                "name": p.stem,
                "points": int(n),
                "area_m2": round(area, 1),
                "density_pts_per_m2": round(n / area, 1) if area > 0 else None,
                "z_min": round(float(mins[2]), 2),
                "z_max": round(float(maxs[2]), 2),
                "fractions": {CLASSES[c]: round(float(counts[c] / n), 6) for c in CLASSES},
                "out_of_range_points": int(counts[N_CLASSES]),
            }
        )
        print(
            f"  [{i:>2}/{len(paths)}] {p.parent.name:5s} {p.stem:12s} "
            f"{n:>10,} pts  {n / area:6.1f} pts/m2",
            flush=True,
        )

    total = int(pooled[:N_CLASSES].sum())
    spread = {}
    for c, name in CLASSES.items():
        vals = np.array([t["fractions"][name] for t in per_tile])
        spread[name] = {
            "pooled_fraction": round(float(pooled[c] / total), 6),
            "pooled_points": int(pooled[c]),
            "per_tile_min": round(float(vals.min()), 6),
            "per_tile_max": round(float(vals.max()), 6),
            "tiles_absent": int((vals == 0).sum()),
        }

    # Per split as well as pooled. A model that predicts only the largest class
    # scores exactly that class's fraction, so the held-out ground fraction is the
    # do-nothing baseline every later number has to beat.
    per_split = {}
    for split in sorted({t["split"] for t in per_tile}):
        rows = [t for t in per_tile if t["split"] == split]
        n = sum(t["points"] for t in rows)
        per_split[split] = {
            "tiles": len(rows),
            "points": n,
            "fractions": {
                name: round(sum(t["fractions"][name] * t["points"] for t in rows) / n, 6)
                for name in CLASSES.values()
            },
        }
        top = max(per_split[split]["fractions"], key=per_split[split]["fractions"].get)
        per_split[split]["largest_class"] = top
        per_split[split]["predict_largest_class_accuracy"] = per_split[split]["fractions"][top]

    dens = np.array([t["density_pts_per_m2"] for t in per_tile], dtype=float)
    result = {
        "tiles": len(per_tile),
        "tiles_per_split": {
            s: sum(1 for t in per_tile if t["split"] == s)
            for s in sorted({t["split"] for t in per_tile})
        },
        "points_total": total,
        "out_of_range_points": int(pooled[N_CLASSES]),
        "density_pts_per_m2": {
            "min": round(float(dens.min()), 1),
            "median": round(float(np.median(dens)), 1),
            "max": round(float(dens.max()), 1),
        },
        "classes": spread,
        "per_split": per_split,
        "seconds": round(time.time() - t0, 1),
        "per_tile": per_tile,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\n{'class':<14}{'pooled %':>10}{'min %':>9}{'max %':>9}{'absent':>8}")
    for name, s in sorted(spread.items(), key=lambda kv: -kv[1]["pooled_fraction"]):
        print(
            f"{name:<14}{100 * s['pooled_fraction']:>10.3f}"
            f"{100 * s['per_tile_min']:>9.3f}{100 * s['per_tile_max']:>9.3f}"
            f"{s['tiles_absent']:>8}"
        )
    print(
        f"\npoints {total:,}   density min/median/max "
        f"{result['density_pts_per_m2']['min']}/"
        f"{result['density_pts_per_m2']['median']}/"
        f"{result['density_pts_per_m2']['max']} pts/m2"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
