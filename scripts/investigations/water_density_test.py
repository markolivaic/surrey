"""If the input over water looks the same in both countries, does the point density explain the failure?

The side by side table in water_input_test.json says water's geometry is much the same in both places:
it sits on the block floor, mostly without land in its block, in Switzerland and in the bay alike. What
differs is density, about 17 points per square metre in the Swiss sub-tiles against about 42 in San
Francisco. This asks the model directly, with no new training: thin each San Francisco block to the
Swiss density and predict again.

Method. Per 40 m block of the two water areas, twice: once as inference always runs it, and once with
the block's points thinned at random to the density ratio first, so the distinct points a crop is drawn
from are as far apart as in Switzerland. Both runs draw the same number of points per crop, as training
did, so the difference between them is spacing and nothing else.

The reading, written before the first run (2026-09-20):

  water recall rises a lot when thinned   density is what breaks water transfer, and it is a finding
                                          that the model learned water at one spacing only
  water recall stays near zero            density is not it either; what is left is that the class was
                                          learned on sixteen tiles and does not survive off them, which
                                          the README states as an open question rather than an
                                          explanation

Control, in the same run: ground recall on the same blocks, before and after thinning. If ground falls
apart too, the thinning broke the input rather than isolating density, and neither reading holds.

Run:
    python scripts/investigations/water_density_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import BLOCK_M, POINTS_PER_CROP  # noqa: E402
from surrey.density import crop_at_density, distinct_for_density  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402
from surrey.sources import WATER  # noqa: E402

TARGET_PREP = ROOT / "data" / "prep" / "target"
INPUT_TEST = ROOT / "data" / "artifacts" / "water_input_test.json"
OUT = ROOT / "data" / "artifacts" / "water_density_test.json"
AREAS = ("embarcadero", "baybridge")
ASPRS_GROUND, ASPRS_WATER = 2, 9
GROUND_CLASS = 0
MIN_BLOCK_POINTS = 2000
READING = (
    "written 2026-09-20 before the first run: a large rise in water recall when San Francisco is "
    "thinned to the Swiss spacing means density is what breaks water transfer; recall staying near "
    "zero means density is not it and the class simply does not survive off its sixteen training "
    "tiles, which the README then states as an open question. Ground recall before and after is the "
    "control: if thinning breaks ground too, it broke the input and neither reading holds"
)


def density_ratio() -> dict:
    """The two densities, from the table already measured, never guessed here."""
    blob = json.loads(INPUT_TEST.read_text(encoding="utf-8"))["both_countries"]["countries"]
    sf = blob["san_francisco"]["points_per_square_metre"]["median"]
    ch = blob["switzerland"]["points_per_square_metre"]["median"]
    return {
        "san_francisco_points_per_m2": sf,
        "switzerland_points_per_m2": ch,
        "keep_fraction": round(min(ch / sf, 1.0), 4),
        "target_density_points_per_m2": ch,
        "what_is_varied": "how many different points of a block the crop carries, at a 40 m block",
    }


@torch.no_grad()
def run_area(
    net, tile: CachedTile, dev: str, keep: float, seed: int, fixed_floor: bool = False
) -> dict:
    """`keep` is a density in points per square metre for the thinned pass, not a fraction."""
    """Predictions for every block, at full density and thinned, scored on the points each run saw."""
    asprs = np.asarray(tile.asprs[:], np.int16)
    rng = np.random.default_rng(seed)
    # The network draws its centroids at random. Without seeding torch here the same run came back
    # with water recall 0.1276 one time and 0.1433 the next, which is the measurement moving, not the
    # model. surrey.inference does this for every other script; this one has its own loop.
    torch.manual_seed(seed)
    counts = {
        name: {
            "water_support": 0,
            "water_hit": 0,
            "water_called": 0,
            "ground_support": 0,
            "ground_hit": 0,
            "points": 0,
            "distinct_points": 0,
        }
        for name in ("full", "thinned")
    }
    crop_shape = {"full": [], "thinned": []}
    target_density = keep
    amp = dev == "cuda"
    for cx, cy in tile.block_centres(BLOCK_M, None):
        idx = tile.gather_block(cx, cy, BLOCK_M)
        if idx.size < MIN_BLOCK_POINTS:
            continue
        full_ref = tile.block_z_reference(idx)
        for name in ("full", "thinned"):
            # What is varied is how many different points the crop may carry, because that is the only
            # thing the model sees. Thinning the pool a crop is drawn from does nothing while the pool
            # stays above the crop length (surrey.density).
            distinct = (
                None if name == "full" else distinct_for_density(target_density, BLOCK_M, idx.size)
            )
            pick, what = crop_at_density(idx, POINTS_PER_CROP, rng, distinct)
            pool = np.unique(pick)
            # Thinning changes two things at once: the points in the crop, and the block's own floor, a
            # low quantile that sits higher when there are fewer points under it. With fixed_floor the
            # floor is the full block's, so only the points change.
            z_ref = full_ref if fixed_floor else tile.block_z_reference(pool)
            crop_shape[name].append(what["distinct_in_crop"])
            feat = tile.features(pick, (cx, cy), z_ref, channels=3)
            x = torch.from_numpy(feat[None]).to(dev)
            with torch.amp.autocast(dev, dtype=torch.float16, enabled=amp):
                pred = net(x).argmax(dim=-1)[0].cpu().numpy()
            truth = asprs[pick]
            water, ground = truth == ASPRS_WATER, truth == ASPRS_GROUND
            c = counts[name]
            c["points"] += pick.size
            c["distinct_points"] += int(pool.size)
            c["water_support"] += int(water.sum())
            c["water_hit"] += int((water & (pred == WATER)).sum())
            c["water_called"] += int((pred == WATER).sum())
            c["ground_support"] += int(ground.sum())
            c["ground_hit"] += int((ground & (pred == GROUND_CLASS)).sum())
    out = {}
    for name, c in counts.items():
        out[name] = c | {
            "distinct_points_in_a_crop_median": int(np.median(crop_shape[name]))
            if crop_shape[name]
            else None,
            "water_recall": round(c["water_hit"] / c["water_support"], 4)
            if c["water_support"]
            else None,
            "water_precision": round(c["water_hit"] / c["water_called"], 4)
            if c["water_called"]
            else None,
            "ground_recall": round(c["ground_hit"] / c["ground_support"], 4)
            if c["ground_support"]
            else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="m1_seedA")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fixed-floor",
        action="store_true",
        help="keep the full block's floor while thinning, to separate the two things thinning does",
    )
    ap.add_argument(
        "--sweep",
        nargs="*",
        type=float,
        default=None,
        help="densities in points per square metre to walk, widest spacing last",
    )
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ratio = density_ratio()
    ckpt = ROOT / "data" / "checkpoints" / f"{args.variant}.pt"
    blob = torch.load(ckpt, map_location=dev, weights_only=False)
    channels = int(blob.get("args", {}).get("channels", 3))
    n_out = int(blob["model"]["head.4.bias"].shape[0])
    net = SurreyNet(n_out, width=args.width, extra_features=channels - 3).to(dev)
    net.load_state_dict(blob["model"])
    net.eval()
    result = {
        "reading_written_before_running": READING,
        "variant": args.variant,
        "device": dev,
        "fixed_floor": bool(args.fixed_floor),
        "checkpoint_step": int(blob["step"]),
        "density": ratio,
        "block_m": BLOCK_M,
        "points_per_crop": POINTS_PER_CROP,
        "seed": args.seed,
        "areas": {},
    }
    t0 = time.time()
    for area in AREAS:
        result["areas"][area] = run_area(
            net,
            CachedTile(TARGET_PREP / area, "sf"),
            dev,
            ratio["target_density_points_per_m2"],
            args.seed,
            args.fixed_floor,
        )
        print(
            area,
            {
                k: {m: v[m] for m in ("water_recall", "ground_recall", "water_called")}
                for k, v in result["areas"][area].items()
            },
            flush=True,
        )
    if args.sweep:
        # One point makes a claim; a curve makes it checkable. Same blocks, same seed, wider spacing.
        result["sweep"] = {}
        for density in args.sweep:
            result["sweep"][f"{density}"] = {
                area: {
                    k: v
                    for k, v in run_area(
                        net,
                        CachedTile(TARGET_PREP / area, "sf"),
                        dev,
                        density,
                        args.seed,
                        args.fixed_floor,
                    )["thinned"].items()
                    if k
                    in (
                        "water_recall",
                        "water_precision",
                        "ground_recall",
                        "distinct_points_in_a_crop_median",
                    )
                }
                for area in AREAS
            }
            print("sweep", density, result["sweep"][f"{density}"], flush=True)
    both = result["areas"]
    result["verdict"] = {
        "water_recall_full": {a: r["full"]["water_recall"] for a, r in both.items()},
        "water_recall_thinned": {a: r["thinned"]["water_recall"] for a, r in both.items()},
        "ground_recall_full": {a: r["full"]["ground_recall"] for a, r in both.items()},
        "ground_recall_thinned": {a: r["thinned"]["ground_recall"] for a, r in both.items()},
        "control_holds": all(
            r["thinned"]["ground_recall"] is None
            or r["full"]["ground_recall"] is None
            or r["thinned"]["ground_recall"] >= 0.8 * r["full"]["ground_recall"]
            for r in both.values()
        ),
        "control_note": (
            "ground recall after thinning has to stay within a fifth of what it was, or the "
            "thinning broke the input and the water reading means nothing"
        ),
    }
    result["seconds"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["verdict"], indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
