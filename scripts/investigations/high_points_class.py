"""What class the model gives to the points that stand highest, per area.

The wide oblique hero frame over the Financial District shows its towers in the
colour the palette gives vegetation. Colour read off a render is not a finding,
so this counts the classes directly: among points standing at least N metres
above the ground surface, the share the model calls each class, per area.

Height is height above ground as prepped (`hag_cm.npy`), the label-free min-z
grid surface used everywhere else in this project, not absolute elevation. In a
city built on hills absolute z would put Sunset's terrain and Financial's towers
in the same bin and measure nothing.

Four thresholds are reported so the reading does not rest on one cut. 100 m is
the one the question was asked at, with 60 m and 150 m either side of it. The
fourth is the training block z ceiling, 121.77 m, read from the scores file
rather than typed here: that is the line the project already argues from, and
100 m sits below it.

The San Francisco delivery carries no building class (ASPRS 1, 2, 7 and 18
only), so the vendor cannot say what these points are. This measures what the
model calls them and nothing more. The ASPRS mix is reported per bin so that
limit stays visible.

No inference runs here: predictions come from the cache written by
`scripts/pipeline/predict_cache.py`, and the checkpoint digest is carried through.

Run:
    python scripts/investigations/high_points_class.py
    python scripts/investigations/high_points_class.py --variant m0_seedB
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.data import CLASS_NAMES, COARSE  # noqa: E402

ARTIFACTS = ROOT / "data" / "artifacts"
PREP = ROOT / "data" / "prep" / "target"
PREDICTIONS = ROOT / "data" / "predictions"
OUT = ARTIFACTS / "high_points_class.json"

THRESHOLDS_M = [60.0, 100.0, 150.0]
# The block z ceiling of the training set, measured before training in
# scripts/pipeline/input_range_check.py and carried in every sf_scores file. Above this
# the model is reading heights it never saw, so it is the cut the project
# already argues from; 100 m sits below it and is not that line.
CEILING_KEY = "training_block_z_max_m"
SCORES = ARTIFACTS / "sf_scores_m0_seedA.json"
ASPRS_NAMES = {1: "unclassified", 2: "ground", 7: "low-noise", 18: "high-noise"}

COARSE_OF = {}
for group, members in COARSE.items():
    for member in members:
        COARSE_OF[CLASS_NAMES.index(member)] = group


def bin_report(pred: np.ndarray, asprs: np.ndarray, keep: np.ndarray, total: int) -> dict:
    n = int(keep.sum())
    out = {"points": n, "share_of_area": round(n / total, 6)}
    if n == 0:
        out["classes"] = {}
        out["coarse"] = {}
        out["asprs"] = {}
        return out

    sel = pred[keep]
    counts = np.bincount(sel.astype(np.int64), minlength=len(CLASS_NAMES))
    out["classes"] = {
        CLASS_NAMES[i]: {"points": int(c), "share": round(float(c) / n, 6)}
        for i, c in enumerate(counts)
        if c
    }

    coarse: dict[str, int] = {}
    for i, c in enumerate(counts):
        if c:
            coarse[COARSE_OF[i]] = coarse.get(COARSE_OF[i], 0) + int(c)
    out["coarse"] = {
        k: {"points": v, "share": round(v / n, 6)}
        for k, v in sorted(coarse.items(), key=lambda kv: -kv[1])
    }

    a_vals, a_counts = np.unique(asprs[keep], return_counts=True)
    out["asprs"] = {
        ASPRS_NAMES.get(int(v), str(int(v))): {"points": int(c), "share": round(float(c) / n, 6)}
        for v, c in zip(a_vals, a_counts, strict=True)
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="m0_seedA")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    pred_dir = PREDICTIONS / args.variant
    manifest = json.loads((pred_dir / "manifest.json").read_text(encoding="utf-8"))
    ceiling = float(json.loads(SCORES.read_text(encoding="utf-8"))[CEILING_KEY])
    thresholds = sorted(THRESHOLDS_M + [ceiling])

    areas = sorted(p.name for p in PREP.iterdir() if p.is_dir())
    result = {
        "variant": args.variant,
        "checkpoint": manifest.get("checkpoint"),
        "checkpoint_sha256_16": manifest.get("checkpoint_sha256_16"),
        "checkpoint_step": manifest.get("checkpoint_step"),
        "height": "height above ground, hag_cm.npy, the min-z grid surface",
        "thresholds_m": thresholds,
        "training_block_z_max_m": ceiling,
        "delivery_has_no_building_class": True,
        "class_names": CLASS_NAMES,
        "areas": {},
    }

    for area in areas:
        npz = pred_dir / f"sf_{area}.npz"
        if not npz.exists():
            print(f"skip {area}: no cached prediction at {npz}")
            continue
        pred = np.load(npz)["pred"]
        hag = np.load(PREP / area / "hag_cm.npy").astype(np.float32) / 100.0
        asprs = np.load(PREP / area / "asprs.npy")
        if not (len(pred) == len(hag) == len(asprs)):
            raise SystemExit(
                f"{area}: length mismatch pred {len(pred)} hag {len(hag)} asprs {len(asprs)}"
            )

        total = len(pred)
        entry = {
            "points": total,
            "max_height_m": round(float(hag.max()), 2),
            "bins": {},
        }
        for t in thresholds:
            name = "above_training_ceiling" if t == ceiling else f"above_{int(t)}m"
            entry["bins"][name] = bin_report(pred, asprs, hag >= t, total)
        result["areas"][area] = entry

        top = entry["bins"]["above_training_ceiling"]
        veg = top["coarse"].get("vegetation", {}).get("share", 0.0)
        bld = top["coarse"].get("building", {}).get("share", 0.0)
        print(
            f"{area:<12} n>ceiling {top['points']:>8,}  "
            f"vegetation {veg:.4f}  building {bld:.4f}  max {entry['max_height_m']:.1f} m"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), newline="\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
