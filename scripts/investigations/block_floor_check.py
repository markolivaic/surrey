"""Is the roof the floor? A test of where the easy errors come from.

The model reads each point's height above its own 40 m block's floor, the 1st
percentile of the block's z. When a block lies entirely on a roof, the floor is the
roof. The roof is then a flat surface at height zero, which is exactly what a car
park looks like, and the same input carrying two labels cannot be learned away with
more data. That would explain the easy errors in San Francisco, roofs called
ground, and why the model is confident on them. It would not explain the park,
whose blocks keep ground in them. One hypothesis, one failure.

Three measurements, then one intervention through inference only, no training.

1. Per area, the share of blocks whose floor stands more than 2 m above the ground
   surface under the block. The surface is the delivery's class 2 in San Francisco
   and the DALES ground labels in Surrey, per 5 m cell, carried at most 25 m, the
   same surface as every other height here. A block is judged only when at least
   half of its points have a surface under them.
2. The share of easy errors (wrong ground calls more than 3 m above that surface,
   as in scripts/investigations/auroc_control.py) that lie in such blocks. At home, also what the
   easy errors are, by DALES class.
3. In those blocks only, the vertical reference is replaced by the surface under
   each point, and seeded inference is run again. The seed and block order are the
   same, so every other block gets the same network draws; the share of changed
   predictions outside the flagged blocks is reported as the check on that.

Reading, fixed before the first run:
    easy errors in flagged blocks fall by at least half   height normalisation is a
                                                          cause, for the share of easy
                                                          errors those blocks hold
    they fall by less than a fifth                        the hypothesis is rejected
    in between                                            partial, reported as such

The intervention uses the delivery's classification to place the surface. It is a
diagnostic and nothing more. A production model cannot take a vendor's ground class
as input; if the floor is the cause, the fix is a wider context or a terrain model.

Run:
    python scripts/investigations/block_floor_check.py
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
from surrey.metrics import auroc_row, coarse_lookup  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402
from surrey.reference import SURFACE_CELL_M, SURFACE_REACH_M, vendor_ground_surface  # noqa: E402

DALES_PREP = ROOT / "data" / "prep" / "dales"
TARGET_PREP = ROOT / "data" / "prep" / "target"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
PRED = ROOT / "data" / "predictions"
CKPT = ROOT / "data" / "checkpoints" / "dayton.pt"
OUT = ROOT / "data" / "artifacts" / "block_floor_check.json"

GROUND = CLASS_NAMES.index("ground")
GROUPS = list(COARSE)
FLOOR_ABOVE_M = 2.0
MIN_SURFACE_SHARE = 0.5
EASY_ABOVE_M = 3.0
SF_AREAS = ["embarcadero", "financial", "ggpark", "mission", "soma", "sunset"]
SF_EXCLUDED = {
    "steep": "29 percent grade; the surface can miss by metres",
    "baybridge": "no delivery ground class",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def blocks_of(tile: CachedTile, surf: np.ndarray) -> list[dict]:
    out = []
    for cx, cy in tile.block_centres(BLOCK_M):
        idx = tile.gather_block(cx, cy, BLOCK_M)
        if idx.size == 0:
            continue
        s = surf[idx]
        defined = np.isfinite(s)
        share = float(defined.mean())
        floor = tile.block_z_reference(idx)
        under = float(np.median(s[defined])) if defined.any() else float("nan")
        judged = share >= MIN_SURFACE_SHARE
        out.append(
            {
                "cx": cx,
                "cy": cy,
                "idx": idx,
                "floor": floor,
                "under": under,
                "judged": judged,
                "flagged": bool(judged and floor - under > FLOOR_ABOVE_M),
            }
        )
    return out


def coarse_shares(coarse: np.ndarray) -> dict:
    n = coarse.size
    return {g: round(float((coarse == i).sum() / n), 4) for i, g in enumerate(GROUPS)} if n else {}


def study(tile, surf, ground_true, wrong_label, pred, conf, net, dev, seed, run_intervention):
    """Everything for one area or tile. `ground_true` and `wrong_label` are point masks."""
    z = np.asarray(tile.z_cm[:], np.float64) / 100.0
    h = z - surf
    blocks = blocks_of(tile, surf)
    in_flag = np.zeros(tile.n, bool)
    flagged = {}
    for b in blocks:
        if b["flagged"]:
            in_flag[b["idx"]] = True
            flagged[(b["cx"], b["cy"])] = b
    in_unjudged = np.zeros(tile.n, bool)
    for b in blocks:
        if not b["judged"]:
            in_unjudged[b["idx"]] = True

    called = pred == GROUND
    right = called & ground_true
    easy = called & wrong_label & (h > EASY_ABOVE_M)
    judged_blocks = sum(b["judged"] for b in blocks)
    row = {
        "blocks": len(blocks),
        "blocks_judged": judged_blocks,
        "blocks_floor_over_2m": len(flagged),
        "share_of_judged_blocks_floor_over_2m": round(len(flagged) / judged_blocks, 4)
        if judged_blocks
        else None,
        "points_in_flagged_blocks_share": round(float(in_flag.mean()), 4),
        "easy_errors": int(easy.sum()),
        "easy_errors_in_flagged_blocks": int((easy & in_flag).sum()),
        "share_of_easy_errors_in_flagged_blocks": round(
            float((easy & in_flag).sum() / easy.sum()), 4
        )
        if easy.any()
        else None,
        "share_of_easy_errors_in_unjudged_blocks": round(
            float((easy & in_unjudged).sum() / easy.sum()), 4
        )
        if easy.any()
        else None,
        "easy_auroc_before": auroc_row(conf, right, easy)["auroc"],
        # The rate, not the count: of the points that are not ground and stand more
        # than 3 m above the surface, how many the model calls ground. A count alone
        # mixes how often roofs fail with how much roof an area has.
        "raised_not_ground_points": int((wrong_label & (h > EASY_ABOVE_M)).sum()),
        "raised_not_ground_called_ground_rate": (
            round(float(easy.sum() / (wrong_label & (h > EASY_ABOVE_M)).sum()), 4)
            if (wrong_label & (h > EASY_ABOVE_M)).any()
            else None
        ),
    }
    if not (run_intervention and flagged):
        return row, easy, in_flag

    def override(cx, cy, idx):
        b = flagged.get((cx, cy))
        if b is None:
            return None
        s = surf[idx]
        return np.where(np.isfinite(s), s, b["under"])

    pred2, conf2, _, seen = infer_tile(
        net, tile, dev, BLOCK_M, POINTS_PER_CROP, None, seed=seed, z_override=override
    )
    if not seen.all():
        raise SystemExit(f"{tile.name}: {int((~seen).sum())} points never predicted")
    table = coarse_lookup()
    called2 = pred2 == GROUND
    right2 = called2 & ground_true
    easy2 = called2 & wrong_label & (h > EASY_ABOVE_M)
    before_f = int((easy & in_flag).sum())
    after_f = int((easy2 & in_flag).sum())
    outside = ~in_flag
    row["intervention"] = {
        "changed_predictions_outside_flagged_blocks_share": round(
            float((pred2[outside] != pred[outside]).mean()), 6
        ),
        "changed_predictions_inside_flagged_blocks_share": round(
            float((pred2[in_flag] != pred[in_flag]).mean()), 4
        ),
        "easy_errors_in_flagged_blocks_before": before_f,
        "easy_errors_in_flagged_blocks_after": after_f,
        "fall_in_flagged_blocks": round(1.0 - after_f / before_f, 4) if before_f else None,
        "easy_errors_before": int(easy.sum()),
        "easy_errors_after": int(easy2.sum()),
        "former_easy_errors_in_flagged_blocks_now_called": coarse_shares(
            table[pred2[easy & in_flag]]
        ),
        "easy_auroc_after": auroc_row(conf2, right2, easy2)["auroc"],
        "true_ground_called_ground_before": round(float(called[ground_true].mean()), 4)
        if ground_true.any()
        else None,
        "true_ground_called_ground_after": round(float(called2[ground_true].mean()), 4)
        if ground_true.any()
        else None,
        "true_ground_in_flagged_blocks": int((ground_true & in_flag).sum()),
    }
    return row, easy, in_flag


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--skip-surrey-intervention", action="store_true")
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
    seed = manifest["seed"]

    def cached(key):
        zf = np.load(PRED / manifest["tiles"][key]["file"])
        return zf["pred"], zf["conf"].astype(np.float32)

    sf = []
    for name in SF_AREAS:
        tile = CachedTile(TARGET_PREP / name, "sf")
        asprs = np.asarray(tile.asprs[:], np.int16)
        clean = ~np.isin(asprs, (7, 18))
        x = np.asarray(tile.xy_cm[:, 0], np.float64) / 100.0
        y = np.asarray(tile.xy_cm[:, 1], np.float64) / 100.0
        z = np.asarray(tile.z_cm[:], np.float64) / 100.0
        grid, key = vendor_ground_surface(
            x, y, z, (asprs == 2) & clean, SURFACE_CELL_M, SURFACE_REACH_M
        )
        surf = grid[key]
        pred, conf = cached(f"sf_{name}")
        row, _, _ = study(
            tile, surf, (asprs == 2) & clean, (asprs == 1) & clean, pred, conf, net, dev, seed, True
        )
        sf.append({"aoi": name, **row})
        iv = row.get("intervention", {})
        print(
            f"  sf {name:12s} flagged {row['blocks_floor_over_2m']}/{row['blocks_judged']}  "
            f"easy in flagged {row['share_of_easy_errors_in_flagged_blocks']}  "
            f"fall {iv.get('fall_in_flagged_blocks')}  auroc {row['easy_auroc_before']} -> "
            f"{iv.get('easy_auroc_after')}  outside changed {iv.get('changed_predictions_outside_flagged_blocks_share')}",
            flush=True,
        )

    names = sorted(
        t["name"]
        for t in json.loads(PREP_ARTIFACT.read_text(encoding="utf-8"))["tiles"]
        if t["split"] == "test"
    )
    surrey = []
    composition = np.zeros(N_CLASSES, np.int64)
    composition_flagged = np.zeros(N_CLASSES, np.int64)
    for n in names:
        tile = CachedTile(DALES_PREP / n, "test")
        label = np.asarray(tile.label[:], np.int16)
        x = np.asarray(tile.xy_cm[:, 0], np.float64) / 100.0
        y = np.asarray(tile.xy_cm[:, 1], np.float64) / 100.0
        z = np.asarray(tile.z_cm[:], np.float64) / 100.0
        grid, key = vendor_ground_surface(x, y, z, label == GROUND, SURFACE_CELL_M, SURFACE_REACH_M)
        surf = grid[key]
        pred, conf = cached(f"surrey_{n}")
        row, easy, in_flag = study(
            tile,
            surf,
            label == GROUND,
            (label >= 0) & (label != GROUND),
            pred,
            conf,
            net,
            dev,
            seed,
            not args.skip_surrey_intervention,
        )
        composition += np.bincount(label[easy], minlength=N_CLASSES)
        composition_flagged += np.bincount(label[easy & in_flag], minlength=N_CLASSES)
        surrey.append({"tile": n, **row})
        print(
            f"  surrey {n}  flagged {row['blocks_floor_over_2m']}/{row['blocks_judged']}  "
            f"easy in flagged {row['share_of_easy_errors_in_flagged_blocks']}  "
            f"fall {row.get('intervention', {}).get('fall_in_flagged_blocks')}",
            flush=True,
        )

    total = int(composition.sum())
    result = {
        "question": "do the easy errors come from blocks whose floor is a roof",
        "prediction_cache": {k: manifest[k] for k in ("checkpoint_sha256_16", "seed")},
        "floor": "1st percentile of z in the 40 m block, as the model reads it",
        "surface": {
            "cell_m": SURFACE_CELL_M,
            "reach_m": SURFACE_REACH_M,
            "san_francisco_from": "delivery class 2",
            "surrey_from": "DALES ground labels",
        },
        "flag": f"floor more than {FLOOR_ABOVE_M} m above the median surface under the block, "
        f"judged where at least {MIN_SURFACE_SHARE} of the block's points have a surface",
        "easy_error": f"called ground, not ground, more than {EASY_ABOVE_M} m above the surface",
        "intervention": "in flagged blocks the vertical reference becomes the surface under each point",
        "reading_fixed_before_running": {
            "cause": "easy errors in flagged blocks fall by at least half",
            "rejected": "they fall by less than a fifth",
            "partial": "in between",
        },
        "san_francisco_excluded": SF_EXCLUDED,
        "san_francisco_per_area": sf,
        "surrey_per_tile": surrey,
        "surrey_easy_error_composition": {
            "points": total,
            "by_class": {
                CLASS_NAMES[i]: round(float(composition[i] / total), 4)
                for i in range(N_CLASSES)
                if composition[i]
            },
            "in_flagged_blocks_points": int(composition_flagged.sum()),
        },
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"surrey easy errors by class: {result['surrey_easy_error_composition']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
