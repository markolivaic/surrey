"""The fourth input channel, return number over number of returns, by class, home and away.

The model reads x, y, z and this ratio. M0 drops the ratio, and the README has to be
able to say whether it mattered. Before any retraining, two things can be measured
from the data alone:

    does the ratio separate classes at home      Surrey tree against Surrey building
                                                 and ground
    does vegetation look the same away           Surrey tree against the points in a
                                                 NAIP vegetation core in each San
                                                 Francisco area, and against the
                                                 points in a non-vegetation core

Each population is a histogram over five bins: [0, 0.25), [0.25, 0.5), [0.5, 0.75),
[0.75, 1), and exactly 1, which is a last or only return. Two histograms are compared
by total variation distance, half the summed absolute difference: 0 is identical, 1
is disjoint.

Heights use the label-free floor both sides (hag_cm), in the same 2 to 25 m band as
the vegetation score, so the populations are the ones already scored.

Run:
    python scripts/investigations/return_ratio_check.py
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
from surrey.data import CLASS_NAMES  # noqa: E402

DALES_PREP = ROOT / "data" / "prep" / "dales"
TARGET_PREP = ROOT / "data" / "prep" / "target"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
REFERENCE = ROOT / "data" / "reference"
OUT = ROOT / "data" / "artifacts" / "return_ratio_check.json"
BAND_M = (2.0, 25.0)
BIN_NAMES = ["[0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1)", "1"]
GROUPS = {"tree": ["tree"], "building": ["habitat", "complex", "annex"], "ground": ["ground"]}


def binned(ret_q: np.ndarray) -> np.ndarray:
    r = ret_q.astype(np.int32)
    counts = np.array(
        [
            (r < 64).sum(),
            ((r >= 64) & (r < 128)).sum(),
            ((r >= 128) & (r < 191)).sum(),
            ((r >= 191) & (r < 255)).sum(),
            (r == 255).sum(),
        ],
        dtype=np.int64,
    )
    return counts


def as_row(counts: np.ndarray) -> dict:
    n = int(counts.sum())
    return {
        "points": n,
        "shares": {b: round(float(c / n), 4) for b, c in zip(BIN_NAMES, counts, strict=True)}
        if n
        else {},
    }


def tvd(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.sum() == 0 or b.sum() == 0:
        return None
    return round(float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum()), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    lo, hi = BAND_M

    home = {g: np.zeros(5, np.int64) for g in GROUPS}
    names = sorted(
        t["name"]
        for t in json.loads(PREP_ARTIFACT.read_text(encoding="utf-8"))["tiles"]
        if t["split"] == "test"
    )
    for n in names:
        tile = CachedTile(DALES_PREP / n, "test")
        label = np.asarray(tile.label[:], np.int16)
        hag = np.asarray(tile.hag_cm[:], np.float32) / 100.0
        ret = np.asarray(tile.ret_q[:])
        for g, classes in GROUPS.items():
            m = np.isin(label, [CLASS_NAMES.index(c) for c in classes])
            if g != "ground":
                m &= (hag > lo) & (hag <= hi)
            home[g] += binned(ret[m])

    away = []
    for p in sorted(TARGET_PREP.iterdir()):
        ref_path = REFERENCE / f"naip_{p.name}.npz"
        if not (p.is_dir() and (p / "meta.json").exists() and ref_path.exists()):
            continue
        tile = CachedTile(p, "sf")
        ref = np.load(ref_path)
        ret = np.asarray(tile.ret_q[:])
        veg = binned(ret[ref["vegetation"]])
        notveg = binned(ret[ref["not_vegetation"]])
        away.append(
            {
                "aoi": p.name,
                "naip_vegetation_core": as_row(veg),
                "naip_not_vegetation_core": as_row(notveg),
                "tvd_to_surrey_tree": tvd(veg, home["tree"]),
                "tvd_not_vegetation_core_to_surrey_building": tvd(notveg, home["building"]),
            }
        )
        print(
            f"  {p.name:12s} veg core {as_row(veg)}  tvd to surrey tree {tvd(veg, home['tree'])}",
            flush=True,
        )

    # The whole of both caches, not only the scored populations: every prepared DALES
    # tile, training and held-out, and every San Francisco area.
    census = {
        "dales_all_tiles": {"points": 0, "ratio_exactly_0": 0},
        "san_francisco_all_areas": {"points": 0, "ratio_exactly_0": 0},
    }
    for root, key in ((DALES_PREP, "dales_all_tiles"), (TARGET_PREP, "san_francisco_all_areas")):
        for p in sorted(root.iterdir()):
            if not (p.is_dir() and (p / "meta.json").exists()):
                continue
            r = np.asarray(CachedTile(p, "").ret_q[:])
            census[key]["points"] += int(r.size)
            census[key]["ratio_exactly_0"] += int((r == 0).sum())

    result = {
        "question": "does the return ratio separate classes at home, and does vegetation look the same away",
        "census": census,
        "why_dales_is_zero": "the DALES-2 files store return_number and number_of_returns as 0 on "
        "every point; preparation computes return_number / max(number_of_returns, 1)",
        "bins": BIN_NAMES,
        "height_band_m": list(BAND_M),
        "surrey_held_out": {g: as_row(c) for g, c in home.items()},
        "surrey_tvd": {
            "tree_vs_building": tvd(home["tree"], home["building"]),
            "tree_vs_ground": tvd(home["tree"], home["ground"]),
            "building_vs_ground": tvd(home["building"], home["ground"]),
        },
        "san_francisco_per_area": away,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"surrey: {result['surrey_held_out']}\nsurrey tvd: {result['surrey_tvd']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
