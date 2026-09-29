"""Which NAIP vegetation mask is the reference, decided without the model.

Two masks exist per area: NAIP as sampled, and NAIP registered by the lidar-minus-NAIP
offset measured at ground level (scripts/pipeline/crs_check.py). The registered mask gave the
model lower vegetation figures in three areas, and keeping the unregistered one for
that reason would be choosing a reference by its result. So the choice is made on
something no prediction touches: the lidar's own multiple returns. A laser pulse that
comes back more than once passed through something porous, and canopy is the common
case; a roof or a road returns once.

Rule, written before this script was first run:

    per area, for each mask, the share of points with more than one return
        (number_of_returns > 1 in the original delivery) inside the vegetation core
        and inside the non-vegetation core
    the reference is the mask with the higher share in the vegetation core AND the
        lower share in the non-vegetation core
    if one mask wins on one core and loses on the other, or they tie to three
        decimals, the criterion does not decide; then the headline figure for that
        area is the LOWER of the two model figures, and both stay in the table
    an area whose two masks are the same (no offset applied) needs no decision

The number of returns is not in the prepared cache, only the ratio. It is recovered
from the delivery file by repeating the preparation's own voxel pick with the same
seed, and the recovered points are checked against the cache coordinate by coordinate
before anything is counted.

Run:
    python scripts/pipeline/naip_mask_choice.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import VOXEL_M, voxel_pick  # noqa: E402
from surrey.flags import flag_bits, kept  # noqa: E402

RAW = ROOT / "data" / "target" / "sf"
TARGET_PREP = ROOT / "data" / "prep" / "target"
REFERENCE = ROOT / "data" / "reference"
OUT = ROOT / "data" / "artifacts" / "naip_mask_choice.json"


def recover_number_of_returns(tile: CachedTile) -> tuple[np.ndarray, int]:
    """number_of_returns for every cached point, and the preparation seed that matched."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_sf_aoi import EXCLUDED

    # The same list, the same flag filter and the same seed the preparation used (prep_tiles.py).
    raw_files = [p for p in sorted(RAW.glob("*.laz")) if p.stem not in EXCLUDED]
    las = laspy.read(RAW / f"{tile.name}.laz")
    usable = np.flatnonzero(kept(flag_bits(las)))
    x = np.asarray(las.x, np.float64)[usable]
    y = np.asarray(las.y, np.float64)[usable]
    z = np.asarray(las.z, np.float64)[usable]
    first = [p.stem for p in raw_files].index(tile.name) + 1
    candidates = [first] + [s for s in range(1, len(raw_files) + 4) if s != first]
    want = np.asarray(tile.xy_cm[:], np.int64)
    for seed in candidates:
        keep = voxel_pick(x, y, z, VOXEL_M, np.random.default_rng(seed))
        if keep.size != tile.n:
            continue
        xk, yk = x[keep], y[keep]
        got = np.stack(
            [
                np.clip((xk - xk.min()) * 100.0, 0, 65535),
                np.clip((yk - yk.min()) * 100.0, 0, 65535),
            ],
            axis=1,
        ).astype(np.uint16)
        if np.array_equal(got.astype(np.int64), want):
            return np.asarray(las.number_of_returns, np.int16)[usable][keep], seed
    raise SystemExit(f"{tile.name}: no preparation seed reproduces the cached points")


def share_multi(nret: np.ndarray, mask: np.ndarray) -> float | None:
    n = int(mask.sum())
    return round(float((nret[mask] > 1).sum() / n), 4) if n else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    reg_meta = {
        a["aoi"]: a
        for a in json.loads(
            (ROOT / "data" / "artifacts" / "naip_reference_registered.json").read_text(
                encoding="utf-8"
            )
        )["areas"]
    }
    rows = []
    for p in sorted(TARGET_PREP.iterdir()):
        if not (p.is_dir() and (p / "meta.json").exists()):
            continue
        if not (REFERENCE / f"naip_{p.name}.npz").exists():
            continue
        tile = CachedTile(p, "sf")
        nret, seed = recover_number_of_returns(tile)
        masks = {
            "unregistered": np.load(REFERENCE / f"naip_{p.name}.npz"),
            "registered": np.load(REFERENCE / "registered" / f"naip_{p.name}.npz"),
        }
        row = {
            "aoi": p.name,
            "preparation_seed_recovered": seed,
            "offset_applied": reg_meta.get(p.name, {}).get("lidar_minus_naip_offset_applied"),
        }
        for name, m in masks.items():
            row[name] = {
                "vegetation_core_points": int(m["vegetation"].sum()),
                "not_vegetation_core_points": int(m["not_vegetation"].sum()),
                "multi_return_share_vegetation_core": share_multi(nret, m["vegetation"]),
                "multi_return_share_not_vegetation_core": share_multi(nret, m["not_vegetation"]),
            }
        u, r = row["unregistered"], row["registered"]
        same = all(
            np.array_equal(masks["unregistered"][k], masks["registered"][k])
            for k in ("vegetation", "not_vegetation")
        )
        if same:
            decision = "masks identical, no decision needed"
            reference = "unregistered"
        elif None in (
            u["multi_return_share_vegetation_core"],
            r["multi_return_share_vegetation_core"],
        ):
            decision = "no vegetation core to judge"
            reference = None
        else:
            veg = np.sign(
                round(
                    r["multi_return_share_vegetation_core"]
                    - u["multi_return_share_vegetation_core"],
                    3,
                )
            )
            non = np.sign(
                round(
                    u["multi_return_share_not_vegetation_core"]
                    - r["multi_return_share_not_vegetation_core"],
                    3,
                )
            )
            if veg > 0 and non > 0:
                decision, reference = "registered wins on both cores", "registered"
            elif veg < 0 and non < 0:
                decision, reference = "unregistered wins on both cores", "unregistered"
            else:
                decision, reference = (
                    "criterion does not decide: headline takes the lower model figure",
                    "lower_of_two",
                )
        row["decision"] = decision
        row["reference"] = reference
        rows.append(row)
        print(
            f"  {p.name:12s} seed {seed}  unreg veg {u['multi_return_share_vegetation_core']} "
            f"non {u['multi_return_share_not_vegetation_core']}  reg veg {r['multi_return_share_vegetation_core']} "
            f"non {r['multi_return_share_not_vegetation_core']}  -> {decision}",
            flush=True,
        )
    result = {
        "question": "which NAIP vegetation mask is the reference, decided without predictions",
        "criterion": "share of points with number_of_returns > 1, higher in vegetation cores and "
        "lower in non-vegetation cores",
        "undecided_rule": "headline takes the lower of the two model figures; both stay in the table",
        "areas": rows,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
