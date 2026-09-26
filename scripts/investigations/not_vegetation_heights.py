"""Where the NAIP not-vegetation cores stand, per area, with no model involved.

False vegetation is the share of the not-vegetation cores a model calls vegetation. Where the cores
are few, or stand where the photograph and the lidar see different things, that share says little
about the model. This counts, per San Francisco area and on the mask the seed spread reads
(unregistered, as scripts/pipeline/aggregate_seeds.py takes it): how many points the cores hold,
their share of the area, and their height above the ground floor the preparation stored
(`hag_cm.npy`, the robust quantile floor, never a model feature). The delivery's ASPRS classes of
those points are counted too.

Writes data/artifacts/not_vegetation_heights.json.

Run:
    python scripts/investigations/not_vegetation_heights.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / "data" / "prep" / "target"
REFERENCE = ROOT / "data" / "reference"
OUT = ROOT / "data" / "artifacts" / "not_vegetation_heights.json"
RAISED_M = 2.0


def main() -> int:
    areas = {}
    for path in sorted(REFERENCE.glob("naip_*.npz")):
        name = path.stem.removeprefix("naip_")
        core = np.load(path)["not_vegetation"]
        hag = np.load(PREP / name / "hag_cm.npy") / 100.0
        asprs = np.load(PREP / name / "asprs.npy")
        if core.size != hag.size:
            raise SystemExit(f"{name}: mask has {core.size} points, the prepared tile {hag.size}")
        n = int(core.sum())
        row = {"points": n, "area_points": int(core.size), "share_of_area": round(n / core.size, 6)}
        if n:
            h = hag[core]
            codes, counts = np.unique(asprs[core], return_counts=True)
            row |= {
                f"share_above_{RAISED_M:g}m": round(float((h > RAISED_M).mean()), 4),
                "height_above_floor_m": {
                    "p10": round(float(np.percentile(h, 10)), 2),
                    "median": round(float(np.median(h)), 2),
                    "p90": round(float(np.percentile(h, 90)), 2),
                },
                "delivery_asprs_share": {
                    str(int(c)): round(float(k) / n, 4) for c, k in zip(codes, counts, strict=True)
                },
            }
        areas[name] = row
        print(f"{name:12s} {n:>8,} points  {row['share_of_area']:.4%} of the area", flush=True)
    doc = {
        "what": "the NAIP not-vegetation cores per area, unregistered mask, no model",
        "height": "above the robust quantile ground floor stored as hag_cm.npy",
        "areas": areas,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
