"""The return ratio channel was a constant at home. What happens when it is again?

scripts/investigations/return_ratio_check.py found that every one of the 395,996,290 prepared
DALES-2 points carries a return ratio of 0: the DALES-2 files store return_number
and number_of_returns as 0 for every point, and the preparation divided 0 by
max(0, 1). In San Francisco the ratio is 1/n to 1 and is 0 on three points out of
44,129,084. The network therefore trained with a fourth input that never moved, and
meets a value it has never seen on essentially every point of the new city.

A channel that never varied in training carries no learned meaning, only a fixed
offset. Setting it to 0 in San Francisco hands the network exactly the input it was
trained on, and is what a retrain without the channel would see, up to training
randomness and the density change M0 also makes. So this is run before M0, through
inference only, with the same checkpoint, seed and block order.

Per area, before (the cached predictions) and after (the channel at 0):

    what the model calls the NAIP vegetation cores, and
    what it calls the non-vegetation cores               scripts/investigations/score_vegetation.py
    wrong ground calls more than 3 m up, and their AUROC scripts/investigations/auroc_control.py
    delivery ground called ground, water called ground, bridge deck called building
    the predicted coarse class shares

Reading, fixed before the first run:
    a major cause    Golden Gate Park's vegetation cores are called vegetation above
                     0.5, or easy errors fall by at least half in at least four of
                     the six scored areas
    not the cause    Golden Gate Park stays under 0.1 and easy errors fall by less
                     than a fifth in at least four of the six
    otherwise        partial, reported as such

Run:
    python scripts/investigations/return_channel_check.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import BLOCK_M, CLASS_NAMES, COARSE, N_CLASSES, POINTS_PER_CROP  # noqa: E402
from surrey.inference import infer_tile  # noqa: E402
from surrey.metrics import auroc_row, coarse_lookup, coarse_shares  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402
from surrey.reference import SURFACE_CELL_M, SURFACE_REACH_M, vendor_ground_surface  # noqa: E402

TARGET_PREP = ROOT / "data" / "prep" / "target"
PRED = ROOT / "data" / "predictions"
REFERENCE = ROOT / "data" / "reference"
CKPT = ROOT / "data" / "checkpoints" / "dayton.pt"
OUT = ROOT / "data" / "artifacts" / "return_channel_check.json"
PRED_OUT = PRED / "ret0"

GROUND = CLASS_NAMES.index("ground")
GROUPS = list(COARSE)
SCORED = ["embarcadero", "financial", "ggpark", "mission", "soma", "sunset"]
EASY_ABOVE_M = 3.0
MIN_POINTS = 1000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def rate(mask_hit: np.ndarray, mask_all: np.ndarray) -> float | None:
    n = int(mask_all.sum())
    return round(float((mask_hit & mask_all).sum() / n), 4) if n >= MIN_POINTS else None


def describe(tile, asprs, h, pred, conf, table, ref) -> dict:
    clean = ~np.isin(asprs, (7, 18))
    coarse = table[pred]
    called = pred == GROUND
    right = called & (asprs == 2) & clean
    easy = called & (asprs == 1) & clean & (h > EASY_ABOVE_M)
    out = {
        "predicted_coarse_shares": coarse_shares(coarse[clean]),
        "delivery_ground_called_ground": rate(called, (asprs == 2) & clean),
        "water_called_ground": rate(called, asprs == 9),
        "bridge_deck_called_building": rate(coarse == GROUPS.index("building"), asprs == 17),
        "easy_errors": int(easy.sum()),
        "easy_auroc": auroc_row(conf, right, easy)["auroc"],
        "water_auroc": auroc_row(conf, right, called & (asprs == 9))["auroc"],
        "median_confidence": round(float(np.median(conf[clean])), 4),
    }
    if ref is not None:
        out["naip_vegetation_core_called"] = coarse_shares(coarse[ref["vegetation"]])
        # The other side of the same question. A model that calls everything
        # vegetation would pass the line above; this is where it would show.
        out["naip_not_vegetation_core_called"] = coarse_shares(coarse[ref["not_vegetation"]])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--width", type=int, default=128)
    args = ap.parse_args()
    t0 = time.time()

    manifest = json.loads((PRED / "manifest.json").read_text(encoding="utf-8"))
    if digest(CKPT) != manifest["checkpoint_sha256_16"]:
        raise SystemExit("checkpoint differs from the one the prediction cache came from")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(CKPT, map_location=dev, weights_only=False)
    net = SurreyNet(N_CLASSES, width=args.width).to(dev)
    net.load_state_dict(blob["model"])
    net.eval()
    table = coarse_lookup()
    PRED_OUT.mkdir(parents=True, exist_ok=True)

    areas = []
    for p in sorted(TARGET_PREP.iterdir()):
        if not (p.is_dir() and (p / "meta.json").exists()):
            continue
        tile = CachedTile(p, "sf")
        asprs = np.asarray(tile.asprs[:], np.int16)
        clean = ~np.isin(asprs, (7, 18))
        x = np.asarray(tile.xy_cm[:, 0], np.float64) / 100.0
        y = np.asarray(tile.xy_cm[:, 1], np.float64) / 100.0
        z = np.asarray(tile.z_cm[:], np.float64) / 100.0
        if ((asprs == 2) & clean).any():
            grid, key = vendor_ground_surface(
                x, y, z, (asprs == 2) & clean, SURFACE_CELL_M, SURFACE_REACH_M
            )
            h = z - grid[key]
        else:
            h = np.full(tile.n, np.nan)
        ref_path = REFERENCE / f"naip_{p.name}.npz"
        ref = np.load(ref_path) if ref_path.exists() else None

        zf = np.load(PRED / manifest["tiles"][f"sf_{p.name}"]["file"])
        pred0, conf0 = zf["pred"], zf["conf"].astype(np.float32)
        pred1, conf1, ent1, seen = infer_tile(
            net, tile, dev, BLOCK_M, POINTS_PER_CROP, None, seed=manifest["seed"], ret_value=0.0
        )
        if not seen.all():
            raise SystemExit(f"{p.name}: {int((~seen).sum())} points never predicted")
        np.savez(
            PRED_OUT / f"sf_{p.name}.npz",
            pred=pred1,
            conf=conf1.astype(np.float16),
            ent=ent1.astype(np.float16),
        )
        before = describe(tile, asprs, h, pred0, conf0, table, ref)
        after = describe(tile, asprs, h, pred1, conf1, table, ref)
        row = {
            "aoi": p.name,
            "scored_for_easy_errors": p.name in SCORED,
            "before": before,
            "after": after,
            "changed_predictions_share": round(float((pred1 != pred0).mean()), 4),
        }
        if before["easy_errors"]:
            row["easy_error_fall"] = round(1.0 - after["easy_errors"] / before["easy_errors"], 4)
        areas.append(row)
        vb = (before.get("naip_vegetation_core_called") or {}).get("vegetation")
        va = (after.get("naip_vegetation_core_called") or {}).get("vegetation")
        print(
            f"  {p.name:12s} changed {row['changed_predictions_share']}  veg core called veg {vb} -> {va}  "
            f"easy {before['easy_errors']:,} -> {after['easy_errors']:,}  auroc {before['easy_auroc']} -> "
            f"{after['easy_auroc']}  non-veg core called veg "
            f"{(before.get('naip_not_vegetation_core_called') or {}).get('vegetation')} -> "
            f"{(after.get('naip_not_vegetation_core_called') or {}).get('vegetation')}  "
            f"water->ground {before['water_called_ground']} -> {after['water_called_ground']}  "
            f"ground recall {before['delivery_ground_called_ground']} -> {after['delivery_ground_called_ground']}",
            flush=True,
        )

    scored = [a for a in areas if a["scored_for_easy_errors"]]
    halved = sum(1 for a in scored if a.get("easy_error_fall", 0) >= 0.5)
    under_fifth = sum(1 for a in scored if a.get("easy_error_fall", 0) < 0.2)
    park = next(a for a in areas if a["aoi"] == "ggpark")
    park_veg = (park["after"].get("naip_vegetation_core_called") or {}).get("vegetation")
    if (park_veg is not None and park_veg > 0.5) or halved >= 4:
        verdict = "a major cause"
    elif park_veg is not None and park_veg < 0.1 and under_fifth >= 4:
        verdict = "not the cause"
    else:
        verdict = "partial"

    result = {
        "question": "does the transfer recover when the return ratio is given its training value",
        "training_value": 0.0,
        "prediction_cache": {k: manifest[k] for k in ("checkpoint_sha256_16", "seed")},
        "predictions_written_to": "data/predictions/ret0",
        "reading_fixed_before_running": {
            "a major cause": "ggpark vegetation cores called vegetation above 0.5, or easy errors "
            "fall by half in at least four of six scored areas",
            "not the cause": "ggpark under 0.1 and easy errors fall by less than a fifth in at "
            "least four of six",
            "otherwise": "partial",
        },
        "scored_areas_easy_errors_halved": halved,
        "scored_areas_easy_errors_fell_under_a_fifth": under_fifth,
        "ggpark_vegetation_core_called_vegetation_after": park_veg,
        "verdict": verdict,
        "areas": areas,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"verdict: {verdict}  (halved {halved}/6, under a fifth {under_fifth}/6, park veg {park_veg})"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
