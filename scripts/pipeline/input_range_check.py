"""Does San Francisco even fall inside the range of inputs Surrey taught?

Two questions, both answerable before a single training step, both using labels
that are never model inputs. They are here because a model can only be wrong in
two interestingly different ways, and these separate them.

**Slope leaking into the vertical origin.** Z reaches the network relative to its
own 40 m block's first percentile. That is still a coarse height above ground,
with the error moved somewhere its behaviour is consistent rather than removed. On
a 15 percent grade a 40 m block spans 6 m of fall, so ground at the top of the
block arrives 6 m up, which is the height of a house. Surrey is suburban. San
Francisco has streets past 30 percent. If the two distributions of within-block
ground relief do not overlap, the model will call hillsides buildings, and that
would be the input normalisation failing rather than the city being different.

Measured per block as the spread of z over points the delivery calls ground: DALES
class 0 on Surrey, ASPRS class 2 on San Francisco.

**Vertical extrapolation.** If no training block ever reached z above some height,
a San Francisco block that reaches 219 m is not a harder example, it is outside
the range the network's layers were ever fitted on. Networks handle that badly and
predictably, and predictably is the useful half: the blocks that exceed the
training maximum can be named in advance, so the result over them is a check on a
prediction rather than a surprise.

Run:
    python scripts/pipeline/input_range_check.py
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
from surrey.data import BLOCK_M, CLASS_NAMES  # noqa: E402

DALES_PREP = ROOT / "data" / "prep" / "dales"
TARGET_PREP = ROOT / "data" / "prep" / "target"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
DEFAULT_OUT = ROOT / "data" / "artifacts" / "input_range_check.json"

DALES_GROUND = CLASS_NAMES.index("ground")
DALES_BUILDING = [CLASS_NAMES.index(n) for n in ("habitat", "complex", "annex")]
ASPRS_GROUND = 2
ASPRS_NOISE = (7, 18)
MIN_GROUND_POINTS = 200
MIN_BLOCK_POINTS = 2000


def blocks(tile: CachedTile, block_m: float):
    ex, ey = tile.extent_m()
    for bx in range(int(np.ceil(ex / block_m))):
        for by in range(int(np.ceil(ey / block_m))):
            cx, cy = (bx + 0.5) * block_m, (by + 0.5) * block_m
            idx = tile.gather_block(cx, cy, block_m)
            if idx.size >= MIN_BLOCK_POINTS:
                yield idx


def scan(tile: CachedTile, block_m: float, labelled: bool) -> dict:
    ground_relief, z_range, building_top = [], [], []
    for idx in blocks(tile, block_m):
        z = np.asarray(tile.z_cm[idx], np.float64) / 100.0
        floor = float(np.quantile(z, 0.01))
        rel = z - floor

        if labelled:
            lab = np.asarray(tile.label[idx], np.int16)
            g = lab == DALES_GROUND
            b = np.isin(lab, DALES_BUILDING)
            clean = np.ones(idx.size, dtype=bool)
        else:
            a = np.asarray(tile.asprs[idx], np.int16)
            g = a == ASPRS_GROUND
            b = np.zeros(idx.size, dtype=bool)
            clean = ~np.isin(a, ASPRS_NOISE)

        # The vertical range the network actually sees, noise excluded so one
        # stray point cannot define the input range of a whole city.
        if clean.any():
            z_range.append(float(np.quantile(rel[clean], 0.999)))
        if g.sum() >= MIN_GROUND_POINTS:
            gz = z[g]
            ground_relief.append(float(np.quantile(gz, 0.98) - np.quantile(gz, 0.02)))
        if b.sum() >= MIN_GROUND_POINTS:
            building_top.append(float(np.quantile(rel[b], 0.98)))

    def dist(v: list[float]) -> dict | None:
        if not v:
            return None
        a = np.asarray(v)
        return {
            "blocks": int(a.size),
            "median": round(float(np.median(a)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
            "max": round(float(a.max()), 2),
        }

    return {
        "name": tile.name,
        "ground_relief_m": dist(ground_relief),
        "block_z_range_m": dist(z_range),
        "building_top_m": dist(building_top),
        "_raw": {"ground_relief": ground_relief, "z_range": z_range, "building_top": building_top},
    }


def pool(rows: list[dict], key: str) -> list[float]:
    return [v for r in rows for v in r["_raw"][key]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block-m", type=float, default=BLOCK_M)
    ap.add_argument("--train-count", type=int, default=29)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    spec = json.loads(PREP_ARTIFACT.read_text())["tiles"]
    train_names = sorted(t["name"] for t in spec if t["split"] == "train")[: args.train_count]

    t0 = time.time()
    print("Surrey, training tiles")
    surrey = []
    for n in train_names:
        surrey.append(scan(CachedTile(DALES_PREP / n, "train"), args.block_m, True))
        print(f"  {n}", flush=True)

    print("San Francisco")
    sf = []
    for p in sorted(TARGET_PREP.iterdir()):
        if p.is_dir() and (p / "meta.json").exists():
            sf.append(scan(CachedTile(p, "sf"), args.block_m, False))
            print(f"  {p.name}", flush=True)

    s_relief = np.asarray(pool(surrey, "ground_relief"))
    f_relief = np.asarray(pool(sf, "ground_relief"))
    s_zrange = np.asarray(pool(surrey, "z_range"))
    f_zrange = np.asarray(pool(sf, "z_range"))
    s_btop = np.asarray(pool(surrey, "building_top"))

    # Amendment one: how often does San Francisco's ground relief inside a single
    # block reach the height of a typical Surrey building?
    typical_building = float(np.median(s_btop)) if s_btop.size else float("nan")
    slope = {
        "surrey_ground_relief_median_m": round(float(np.median(s_relief)), 2),
        "surrey_ground_relief_p99_m": round(float(np.percentile(s_relief, 99)), 2),
        "sf_ground_relief_median_m": round(float(np.median(f_relief)), 2),
        "sf_ground_relief_p99_m": round(float(np.percentile(f_relief, 99)), 2),
        "surrey_typical_building_top_m": round(typical_building, 2),
        "sf_blocks_where_ground_relief_exceeds_a_surrey_building": round(
            float((f_relief > typical_building).mean()), 4
        ),
        "surrey_blocks_where_ground_relief_exceeds_a_surrey_building": round(
            float((s_relief > typical_building).mean()), 4
        ),
        "reading": (
            "a block whose ground alone spans more than a typical building "
            "can present a hillside to the network as a roof"
        ),
        "pooling_hides_this": (
            "the pooled median says San Francisco is no worse than "
            "Surrey, and that is an artefact of pooling: four of the "
            "eight areas are flat districts at 0.5 to 1.0 m and they "
            "drag the median under. Read the per-area column"
        ),
        "per_aoi_ground_relief_m": {},
    }

    # Amendment two: which San Francisco blocks sit outside the vertical range any
    # training block ever showed the network?
    train_max = float(s_zrange.max()) if s_zrange.size else float("nan")
    per_aoi_out = {}
    for r in sf:
        v = np.asarray(r["_raw"]["z_range"])
        per_aoi_out[r["name"]] = {
            "blocks": int(v.size),
            "blocks_above_training_max": int((v > train_max).sum()),
            "fraction_above": round(float((v > train_max).mean()), 4) if v.size else None,
            "max_m": round(float(v.max()), 2) if v.size else None,
        }
    extrapolation = {
        "surrey_training_block_z_max_m": round(train_max, 2),
        "sf_block_z_max_m": round(float(f_zrange.max()), 2),
        "sf_blocks_above_training_max": round(float((f_zrange > train_max).mean()), 4),
        "per_aoi": per_aoi_out,
        "reading": (
            "a block above the training maximum is not a harder example, it "
            "is outside the range the layers were fitted on. Naming these "
            "before scoring turns their result into a check, not a surprise"
        ),
    }

    for r in sf:
        if r["ground_relief_m"]:
            slope["per_aoi_ground_relief_m"][r["name"]] = {
                "median": r["ground_relief_m"]["median"],
                "p90": r["ground_relief_m"]["p90"],
                "max": r["ground_relief_m"]["max"],
                "exceeds_a_surrey_building_at_p90": bool(
                    r["ground_relief_m"]["p90"] > typical_building
                ),
            }
    # The confound reading written before Phase 4 (docs/phase4_criteria.md): an area is outside what
    # training showed on a measure when its median block lies above the 99th percentile of the 29
    # training tiles' blocks. Building height has no reference on the San Francisco side, which labels
    # no buildings; the block's height range carries the tall building case there.
    p99 = {
        "block_z_range_m": float(np.percentile(s_zrange, 99)),
        "ground_relief_m": float(np.percentile(s_relief, 99)),
        "building_top_m": float(np.percentile(s_btop, 99)) if s_btop.size else None,
    }
    confounds = {
        "rule": "outside when the area's median block lies above the 99th percentile of the 29 "
        "training tiles' blocks, fixed in docs/phase4_criteria.md before Phase 4",
        "training_p99": {k: None if v is None else round(v, 2) for k, v in p99.items()},
        "building_top_on_san_francisco": "not measurable: the delivery labels no buildings",
        "per_aoi": {},
    }
    for r in sf:
        row = {}
        for measure, key in (("block_z_range_m", "z_range"), ("ground_relief_m", "ground_relief")):
            v = np.asarray(r["_raw"][key])
            med = float(np.median(v)) if v.size else None
            row[measure] = {
                "median": None if med is None else round(med, 2),
                "outside": None if med is None else bool(med > p99[measure]),
            }
        confounds["per_aoi"][r["name"]] = row
    for r in surrey + sf:
        r.pop("_raw")

    result = {
        "question": "does San Francisco fall inside the input range Surrey taught",
        "block_m": args.block_m,
        "measured_before_training": True,
        "labels_used_here_only": (
            "DALES class 0 and the delivery's ASPRS class 2 "
            "choose and check these numbers, and are never inputs"
        ),
        "slope_leak": slope,
        "vertical_extrapolation": extrapolation,
        "confounds_at_the_99th_percentile": confounds,
        "surrey_tiles": surrey,
        "san_francisco": sf,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    print("\nground relief inside one 40 m block, from each side's own ground labels")
    print(
        f"  Surrey          median {slope['surrey_ground_relief_median_m']:>6.2f} m   "
        f"p99 {slope['surrey_ground_relief_p99_m']:>6.2f} m"
    )
    print(
        f"  San Francisco   median {slope['sf_ground_relief_median_m']:>6.2f} m   "
        f"p99 {slope['sf_ground_relief_p99_m']:>6.2f} m"
    )
    print(f"  a typical Surrey building stands {typical_building:.2f} m above its block floor")
    print(
        f"  San Francisco blocks whose ground alone spans more than that: "
        f"{slope['sf_blocks_where_ground_relief_exceeds_a_surrey_building']:.1%}"
    )
    print(
        f"  same figure for Surrey: "
        f"{slope['surrey_blocks_where_ground_relief_exceeds_a_surrey_building']:.1%}"
    )
    print()
    print("  per area, because pooling the eight of them hides the answer:")
    for name, v in sorted(
        slope["per_aoi_ground_relief_m"].items(), key=lambda kv: -kv[1]["median"]
    ):
        flag = (
            "  <- ground alone outreaches a Surrey building"
            if v["exceeds_a_surrey_building_at_p90"]
            else ""
        )
        print(
            f"    {name:13s} median {v['median']:>5.2f} m  p90 {v['p90']:>5.2f} m  "
            f"max {v['max']:>5.2f} m{flag}"
        )

    print(
        f"\nvertical range, tallest training block {train_max:.1f} m, "
        f"tallest San Francisco block {f_zrange.max():.1f} m"
    )
    print(
        f"  San Francisco blocks above the training maximum: "
        f"{extrapolation['sf_blocks_above_training_max']:.1%}"
    )
    for name, v in sorted(per_aoi_out.items()):
        if v["blocks_above_training_max"]:
            print(
                f"    {name:12s} {v['blocks_above_training_max']:>3}/{v['blocks']:<3} "
                f"blocks, up to {v['max_m']:.0f} m"
            )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
