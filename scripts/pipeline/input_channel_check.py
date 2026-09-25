"""Does every channel the model reads carry information in training, and stay in range away?

The registry checked that every number was sourced and consistent. It could not check
that the input to the model was meaningful, and it was not: the return ratio was 0 on
every training point and alive on every San Francisco point. This writes the evidence
two integrity tests need so that cannot happen quietly again:

    training   per channel, the minimum, maximum and standard deviation over crops
               drawn the way training draws them (CachedTile.random_crop, seeded)
    target     per channel and per San Francisco area, the share of points outside
               the training minimum and maximum, over every point, through the same
               block features inference uses

for each input variant:

    returns_ood   the four channel checkpoint, San Francisco as delivered
    ret0          the four channel checkpoint, return ratio forced to 0
    m0            the three channel model: x, y, z above the block floor

tests/integrity/test_input_channels.py fails a published variant with a channel of
zero variance in training or out of range on most points of any area, and checks that
returns_ood would have been failed.

Run:
    python scripts/pipeline/input_channel_check.py
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
from surrey.data import BLOCK_M, POINTS_PER_CROP  # noqa: E402

DALES_PREP = ROOT / "data" / "prep" / "dales"
TARGET_PREP = ROOT / "data" / "prep" / "target"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
OUT = ROOT / "data" / "artifacts" / "input_channels.json"
NAMES = ["x", "y", "z_above_block_floor", "return_ratio"]
VARIANTS = {
    "returns_ood": {
        "channels": 4,
        "ret_value": None,
        "model": "data/checkpoints/dayton.pt, four channels",
    },
    "ret0": {
        "channels": 4,
        "ret_value": 0.0,
        "model": "data/checkpoints/dayton.pt, return ratio forced to 0 at inference",
    },
    "m0": {
        "channels": 3,
        "ret_value": None,
        "model": "three channel retrain, data/checkpoints/m0_seedA.pt and m0_seedB.pt",
    },
}
PUBLISHED = "m0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()

    train = [
        CachedTile(DALES_PREP / t["name"], "train")
        for t in json.loads(PREP_ARTIFACT.read_text())["tiles"]
        if t["split"] == "train"
    ]
    rng = np.random.default_rng(args.seed)
    lo = np.full(4, np.inf)
    hi = np.full(4, -np.inf)
    s1 = np.zeros(4)
    s2 = np.zeros(4)
    n = 0
    for _ in range(args.crops):
        tile = train[rng.integers(len(train))]
        feat, _, _ = tile.random_crop(rng, BLOCK_M, POINTS_PER_CROP, channels=4)
        f = feat.astype(np.float64)
        lo = np.minimum(lo, f.min(0))
        hi = np.maximum(hi, f.max(0))
        s1 += f.sum(0)
        s2 += (f * f).sum(0)
        n += f.shape[0]
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
    training = {
        NAMES[c]: {
            "min": round(float(lo[c]), 4),
            "max": round(float(hi[c]), 4),
            "std": round(float(std[c]), 6),
        }
        for c in range(4)
    }
    print(f"training, {n:,} points from {args.crops} crops: {training}", flush=True)

    areas = {}
    for p in sorted(TARGET_PREP.iterdir()):
        if not (p.is_dir() and (p / "meta.json").exists()):
            continue
        tile = CachedTile(p, "sf")
        outside = np.zeros((2, 4), np.int64)  # row 0 as delivered, row 1 ratio at 0
        total = 0
        for feat, _, _, valid in tile.iter_full(BLOCK_M, POINTS_PER_CROP, channels=4):
            f = feat[:valid]
            out = (f < lo.astype(np.float32)) | (f > hi.astype(np.float32))
            outside[0] += out.sum(0)
            out[:, 3] = (lo[3] > 0.0) | (hi[3] < 0.0)
            outside[1] += out.sum(0)
            total += valid
        areas[p.name] = {
            "points": total,
            "outside_training_range_share": {
                "as_delivered": {
                    NAMES[c]: round(float(outside[0, c] / total), 4) for c in range(4)
                },
                "return_ratio_at_0": {
                    NAMES[c]: round(float(outside[1, c] / total), 4) for c in range(4)
                },
            },
        }
        print(
            f"  {p.name:12s} outside range as delivered {areas[p.name]['outside_training_range_share']['as_delivered']}",
            flush=True,
        )

    variants = {}
    for name, v in VARIANTS.items():
        read = NAMES[: v["channels"]]
        key = "return_ratio_at_0" if v["ret_value"] is not None else "as_delivered"
        variants[name] = {
            **v,
            "channels_read": read,
            "training_std": {c: training[c]["std"] for c in read},
            "target_outside_share": {
                a: {c: areas[a]["outside_training_range_share"][key][c] for c in read}
                for a in areas
            },
        }

    result = {
        "question": "does every channel a model reads vary in training and stay in range on the target",
        "training_sample": {
            "crops": args.crops,
            "points": n,
            "seed": args.seed,
            "drawn_by": "CachedTile.random_crop, as scripts/pipeline/train.py draws",
        },
        "training": training,
        "san_francisco": areas,
        "variants": variants,
        "published_variant": PUBLISHED,
        "fail_rules": {
            "dead_channel": "training std of a channel the model reads is 0",
            "out_of_range": "more than half of an area's points fall outside the "
            "training range on a channel the model reads",
        },
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
