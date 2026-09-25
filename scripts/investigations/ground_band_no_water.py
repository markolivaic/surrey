"""The ground band with and without water, per area.

Water is a class the DALES-2 taxonomy does not have, so the model has no way to call
it anything but its nearest class, and the nearest class is ground. Every point of
water it calls ground lands in the ground precision as a false positive. That is a
taxonomy mismatch, not a ground error, and it should be visible as one: the same
band is computed with water in and with water taken out of both the predictions and
the reference, through surrey.reference.score_ground, the scorer the transfer used.

Two prediction sets, both seeded from the same checkpoint:
    as trained        data/predictions, the return ratio channel as the delivery gives it
    channel at 0      data/predictions/ret0, the channel at its training value
                      (scripts/investigations/return_channel_check.py)

Run:
    python scripts/investigations/ground_band_no_water.py
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
from surrey.data import COARSE  # noqa: E402
from surrey.metrics import coarse_lookup  # noqa: E402
from surrey.reference import score_ground  # noqa: E402

TARGET_PREP = ROOT / "data" / "prep" / "target"
PRED = ROOT / "data" / "predictions"
OUT = ROOT / "data" / "artifacts" / "ground_band_no_water.json"
ASPRS_WATER = 9
ASPRS_NOISE_STANDIN = 7
KEEP = (
    "recall_clean",
    "precision_floor",
    "precision_ceiling",
    "predicted_points",
    "reference_points",
    "predicted_points_unjudgeable_fraction",
)


def band(asprs, x, y, z, ground_pred, drop_water: bool) -> dict:
    a = asprs.copy()
    if drop_water:
        # score_ground drops noise codes from both sides; water joins them here.
        a[a == ASPRS_WATER] = ASPRS_NOISE_STANDIN
    s = score_ground(a, x, y, z, ground_pred)
    return {k: s.get(k) for k in KEEP} if s.get("reference_points") else {"note": s.get("note")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    manifest = json.loads((PRED / "manifest.json").read_text(encoding="utf-8"))
    table = coarse_lookup()
    ground_group = list(COARSE).index("ground")
    rows = []
    for p in sorted(TARGET_PREP.iterdir()):
        if not (p.is_dir() and (p / "meta.json").exists()):
            continue
        tile = CachedTile(p, "sf")
        asprs = np.asarray(tile.asprs[:], np.int16)
        x = np.asarray(tile.xy_cm[:, 0], np.float64) / 100.0
        y = np.asarray(tile.xy_cm[:, 1], np.float64) / 100.0
        z = np.asarray(tile.z_cm[:], np.float64) / 100.0
        row = {"aoi": p.name, "water_points": int((asprs == ASPRS_WATER).sum())}
        for label, path in (
            ("as_trained", PRED / manifest["tiles"][f"sf_{p.name}"]["file"]),
            ("channel_at_0", PRED / "ret0" / f"sf_{p.name}.npz"),
        ):
            if not path.exists():
                row[label] = {"note": f"{path.name} missing"}
                continue
            g = table[np.load(path)["pred"]] == ground_group
            row[label] = {
                "with_water": band(asprs, x, y, z, g, False),
                "without_water": band(asprs, x, y, z, g, True),
            }
        rows.append(row)
        a = row["as_trained"]
        print(
            f"  {p.name:12s} water {row['water_points']:>9,}  as trained: with {a.get('with_water')}  "
            f"without {a.get('without_water')}",
            flush=True,
        )
    result = {
        "question": "the ground band with water in and with water out",
        "scorer": "surrey.reference.score_ground",
        "water_removed_from": "both the predictions and the reference, as noise is",
        "areas": rows,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
