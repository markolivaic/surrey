"""Did confidence stop discriminating, or was water just a hard thing to separate?

The first AUROC compared the model's own error detection at home against San
Francisco, and the drop was large. But the two tasks were only the same on paper. At
home a wrong ground call was a car, low vegetation, a fence. In San Francisco it was
water, and water is flat, low, and the closest thing to ground that exists. A drop
could mean the signal vanished in the new city, or that water is simply harder to
tell from ground than a car is. Those are different claims.

So the difficulty of the error is held fixed across the two cities:

    easy errors   wrong ground calls more than 3 m above the ground surface.
                  Roofs and canopy, unambiguously not ground, geometrically plain.
    hard errors   wrong ground calls within half a metre of the ground surface.
                  At home: low vegetation, cars, fences at ground level.
                  In San Francisco: water, the only certain wrong call at that height.

At home the ground surface comes from the DALES ground labels. In San Francisco it
comes from the delivery's ground class, carried a bounded distance into cells the
vendor left blank.

Reading. If San Francisco on easy errors still sits near the water figure, the signal
is gone and the claim stands. If it climbs toward the home figure, water was a hard
task and the claim narrows to "does not tell water from ground".

Every San Francisco figure is per area. Pooling areas mixes ground from one
neighbourhood with errors from another. Steep is left out: on a 29 percent grade the
interpolated surface can miss by metres, so "more than 3 m above ground" is a guess
there. Baybridge is left out because it holds no ground class to be right about.

The Surrey figures pool the eleven held-out tiles, which share one flight, one city
and one split, and every tile is also reported alone.

Reads data/predictions, written by scripts/pipeline/predict_cache.py.

Run:
    python scripts/investigations/auroc_control.py
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
from surrey.metrics import auroc_from_histograms, auroc_row, score_histogram  # noqa: E402
from surrey.reference import SURFACE_CELL_M, SURFACE_REACH_M, vendor_ground_surface  # noqa: E402

DALES_PREP = ROOT / "data" / "prep" / "dales"
TARGET_PREP = ROOT / "data" / "prep" / "target"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
PRED = ROOT / "data" / "predictions"
OUT = ROOT / "data" / "artifacts" / "auroc_control.json"

GROUND = CLASS_NAMES.index("ground")
ASPRS_GROUND, ASPRS_UNASSIGNED, ASPRS_WATER = 2, 1, 9
ASPRS_NOISE = (7, 18)
EASY_ABOVE_M = 3.0
HARD_WITHIN_M = 0.5
EXCLUDED = {
    "steep": "29 percent grade; the interpolated surface can miss by metres, so a 3 m gate is a guess",
    "baybridge": "no delivery ground class, so there is no right call to rank",
}


def load_prediction(city: str, name: str, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    key = f"{city}_{name}"
    if key not in manifest["tiles"]:
        raise SystemExit(
            f"{key} missing from the prediction cache, run scripts/pipeline/predict_cache.py"
        )
    z = np.load(PRED / manifest["tiles"][key]["file"])
    return z["pred"], z["conf"].astype(np.float32)


def height_above(tile: CachedTile, ground_mask: np.ndarray) -> np.ndarray:
    x = np.asarray(tile.xy_cm[:, 0], np.float64) / 100.0
    y = np.asarray(tile.xy_cm[:, 1], np.float64) / 100.0
    z = np.asarray(tile.z_cm[:], np.float64) / 100.0
    grid, key = vendor_ground_surface(x, y, z, ground_mask, SURFACE_CELL_M, SURFACE_REACH_M)
    h = np.full(tile.n, np.nan)
    ok = np.isfinite(grid[key])
    h[ok] = z[ok] - grid[key[ok]]
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    manifest = json.loads((PRED / "manifest.json").read_text(encoding="utf-8"))
    t0 = time.time()

    # ---------------- Surrey ----------------
    names = sorted(
        t["name"] for t in json.loads(PREP_ARTIFACT.read_text())["tiles"] if t["split"] == "test"
    )
    pooled = {"all": ([], []), "easy": ([], []), "hard": ([], [])}
    per_tile = []
    for n in names:
        tile = CachedTile(DALES_PREP / n, "test")
        pred, conf = load_prediction("surrey", n, manifest)
        label = np.asarray(tile.label[:], dtype=np.int16)
        h = height_above(tile, label == GROUND)
        called = (pred == GROUND) & (label >= 0)
        right = called & (label == GROUND)
        wrong = called & (label != GROUND)
        easy = wrong & (h > EASY_ABOVE_M)
        hard = wrong & (np.abs(h) <= HARD_WITHIN_M)
        row = {
            "tile": n,
            "all": auroc_row(conf, right, wrong),
            "easy": auroc_row(conf, right, easy, h),
            "hard": auroc_row(conf, right, hard, h),
        }
        per_tile.append(row)
        for k, m in (("all", wrong), ("easy", easy), ("hard", hard)):
            pooled[k][0].append(score_histogram(conf[right]))
            pooled[k][1].append(score_histogram(conf[m]))
        print(
            f"  surrey {n}  all {row['all']['auroc']}  easy {row['easy']['auroc']}  "
            f"hard {row['hard']['auroc']}",
            flush=True,
        )

    surrey = {}
    for k, (rs, ws) in pooled.items():
        r, w = sum(rs), sum(ws)
        a = auroc_from_histograms(r, w)
        surrey[k] = {
            "right": int(r.sum()),
            "wrong": int(w.sum()),
            "auroc": None if a is None else round(a, 4),
        }

    # ---------------- San Francisco, per area ----------------
    relief = {}
    irc = ROOT / "data" / "artifacts" / "input_range_check.json"
    if irc.exists():
        relief = {
            k: v["median"]
            for k, v in json.loads(irc.read_text())["slope_leak"]["per_aoi_ground_relief_m"].items()
        }
    areas = []
    for p in sorted(TARGET_PREP.iterdir()):
        if not (p.is_dir() and (p / "meta.json").exists()):
            continue
        if p.name in EXCLUDED:
            areas.append({"aoi": p.name, "excluded": EXCLUDED[p.name]})
            continue
        tile = CachedTile(p, "sf")
        pred, conf = load_prediction("sf", p.name, manifest)
        asprs = np.asarray(tile.asprs[:], dtype=np.int16)
        clean = ~np.isin(asprs, ASPRS_NOISE)
        h = height_above(tile, (asprs == ASPRS_GROUND) & clean)
        called = clean & (pred == GROUND)
        right = called & (asprs == ASPRS_GROUND)
        easy = called & (asprs == ASPRS_UNASSIGNED) & (h > EASY_ABOVE_M)
        hard = called & (asprs == ASPRS_WATER)
        # The definition above says within half a metre, and the first version of this
        # script took every water point instead. Water sits below a surface carried in
        # from the shore, so the two differ; both are kept, the second being the one
        # that matches the Surrey hard row.
        hard_near = hard & (np.abs(h) <= HARD_WITHIN_M)
        row = {
            "aoi": p.name,
            "ground_relief_median_m": relief.get(p.name),
            "easy_unassigned_over_3m": auroc_row(conf, right, easy, h),
            "hard_water": auroc_row(conf, right, hard, h),
            "hard_water_within_0_5m": auroc_row(conf, right, hard_near, h),
        }
        areas.append(row)
        e = row["easy_unassigned_over_3m"]
        print(
            f"  sf {p.name:12s} easy {e['auroc']} (wrong h {e.get('wrong_height_m_p10_p50_p90')}, "
            f"conf right {e['median_conf_right']} wrong {e['median_conf_wrong']})  "
            f"hard {row['hard_water']['auroc']} within 0.5 m "
            f"{row['hard_water_within_0_5m']['auroc']} (n={row['hard_water_within_0_5m']['wrong']:,})",
            flush=True,
        )

    result = {
        "question": "is the confidence collapse a lost signal, or water being hard",
        "task": "among points the model calls ground, rank right calls above wrong ones",
        "prediction_cache": {k: manifest[k] for k in ("checkpoint_sha256_16", "seed")},
        "easy_error": f"wrong call more than {EASY_ABOVE_M} m above the ground surface",
        "hard_error": f"wrong call within {HARD_WITHIN_M} m of the ground surface",
        "surface": {
            "cell_m": SURFACE_CELL_M,
            "reach_m": SURFACE_REACH_M,
            "surrey_from": "DALES ground labels",
            "san_francisco_from": "delivery ASPRS class 2",
        },
        "surrey_pooling_note": (
            "eleven held-out tiles, one flight, one city, one split; each tile is also listed"
        ),
        "surrey": surrey,
        "surrey_per_tile": per_tile,
        "san_francisco_per_area": areas,
        "no_pooled_san_francisco": "areas are never pooled here",
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\n{'':24s}{'easy errors':>16}{'hard errors':>16}")
    print(
        f"{'Surrey, held-out':24s}{str(surrey['easy']['auroc']):>16}{str(surrey['hard']['auroc']):>16}"
        f"   (all errors {surrey['all']['auroc']})"
    )
    for a in areas:
        if "excluded" in a:
            continue
        e, hd = a["easy_unassigned_over_3m"], a["hard_water"]
        print(
            f"{'SF ' + a['aoi']:24s}{str(e['auroc']):>16}{str(hd['auroc']):>16}"
            f"   easy n={e['wrong']:,}  water n={hd['wrong']:,}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
