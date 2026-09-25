"""One measurable reason water does not transfer: what the model is given to see on open water.

The model reads x, y and z above the floor of a 40 m block. In the middle of the bay a whole block is
water, so the floor is the water surface and every point sits flat at about zero, which is the same
input a car park gives. On a Swiss lake the shore is usually inside the block, so water there has land
to be measured against. That would also explain why the height null wins: it knows absolute NAVD88
height and the tide at the moment of flight, and the model knows neither.

Three tests, all on the data, none of them involving a model:

  a) intensity. The delivery's own intensity for class 9 against class 2, per area, with the AUROC of
     intensity alone as a water detector. Water absorbs near infrared, so it should separate.
  b) the block floor on water. Per 40 m block over the two water areas: how much of the block is water,
     whether its floor sits on the water surface, and the model's own input (z above the block floor)
     for water points against ground points, with the AUROC of that input as a water detector.
  c) the same shape of question for the Swiss sub-tiles: how much of a water point's block is land.
  d) added 2026-09-20 after a) to c) had run and both halves of the story above had failed: the same
     input for water in both countries side by side, z above the block floor, how much of the block is
     land, and the point density, so whatever does differ between Swiss water and bay water is named
     from the data rather than guessed at.

The reading, written before the first run (2026-09-20):

  a) high AUROC (say 0.9 or better) means intensity separates water almost by itself, and that the one
     channel that does so is the one this project dropped for being the most sensor-bound. That is a
     finding and a piece of future work, not an excuse.
  b) if the z above block floor distributions of water and ground sit on top of each other (AUROC near
     0.5) while most bay blocks are all water, the explanation above stops being a story and becomes a
     measurement.
  c) if Swiss water points mostly do have land inside their block and San Francisco water points mostly
     do not, the difference between the two is measured rather than asserted.
  d) reading written before d) ran, with a) to c) already known: if the z above floor of water is much
     the same in both countries and the land in its block is too, then geometry is not what changed and
     what is left is density, where the Swiss tiles are about 6.4 points per square metre and San
     Francisco about 49, and that is then the thing to name. If instead the Swiss water sits at a
     different height above its floor, the model learned a number that does not hold in the bay.

Failing any of these leaves the README with "we do not know why", which is the honest alternative.

Run:
    python scripts/investigations/water_input_test.py
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
from surrey.data import BLOCK_M  # noqa: E402
from surrey.metrics import auroc as metrics_auroc  # noqa: E402
from surrey.sources import WATER  # noqa: E402

TARGET_PREP = ROOT / "data" / "prep" / "target"
SWISS_PREP = ROOT / "data" / "prep" / "swiss"
LAZ = ROOT / "data" / "target" / "sf"
OUT = ROOT / "data" / "artifacts" / "water_input_test.json"
WATER_AREAS = ("embarcadero", "baybridge")
ASPRS_GROUND, ASPRS_WATER = 2, 9
FLOOR_NEAR_M = 0.5
MIN_BLOCK_POINTS = 2000
READING = (
    "written 2026-09-20 before the first run: intensity AUROC at or above 0.9 means the one channel "
    "that finds water is the sensor-bound one this project dropped; z above the block floor with "
    "AUROC near 0.5 while bay blocks are all water means the model's input cannot separate open "
    "water from flat ground; Swiss water having land in its block and San Francisco water not "
    "having it measures the difference between the two"
)


def auroc(positive: np.ndarray, negative: np.ndarray, cap: int = 2_000_000, seed: int = 0) -> dict:
    """Rank AUROC of a value as a detector, with ties counted at half, on a capped sample."""
    rng = np.random.default_rng(seed)
    pos = positive if positive.size <= cap else rng.choice(positive, cap, replace=False)
    neg = negative if negative.size <= cap else rng.choice(negative, cap, replace=False)
    if pos.size == 0 or neg.size == 0:
        return {"auroc": None, "n_positive": int(positive.size), "n_negative": int(negative.size)}
    value = metrics_auroc(pos, neg)
    return {
        "auroc": round(float(value), 4),
        "n_positive": int(positive.size),
        "n_negative": int(negative.size),
        "sampled_positive": int(pos.size),
        "sampled_negative": int(neg.size),
    }


def separation(value: float | None) -> dict:
    """An AUROC read as a detector either way round: how far from chance, and which side is water.

    Water absorbing near infrared would put it below ground, which is an AUROC under 0.5, and reading
    that as "no separation" would be wrong. Strength is the distance from chance, mapped back above it.
    """
    if value is None:
        return {"strength": None, "direction": None}
    return {
        "strength": round(max(value, 1.0 - value), 4),
        "direction": "water higher"
        if value > 0.5
        else ("water lower" if value < 0.5 else "no side"),
    }


def summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "n": int(values.size),
        "median": round(float(q[2]), 3),
        "p5": round(float(q[0]), 3),
        "p25": round(float(q[1]), 3),
        "p75": round(float(q[3]), 3),
        "p95": round(float(q[4]), 3),
        "mean": round(float(values.mean()), 3),
    }


def intensity_test() -> dict:
    """The delivery's intensity for its water class against its ground class, per area."""
    import laspy

    rows = {}
    for path in sorted(LAZ.glob("*.laz")):
        with laspy.open(path) as fh:
            las = fh.read()
        cls = np.asarray(las.classification)
        inten = np.asarray(las.intensity, np.float64)
        water, ground = inten[cls == ASPRS_WATER], inten[cls == ASPRS_GROUND]
        rows[path.stem] = {
            "delivery_points": int(cls.size),
            "water": summary(water),
            "ground": summary(ground),
            "separation": auroc(water, ground),
        }
        print(
            f"  intensity {path.stem} water n={water.size} ground n={ground.size} "
            f"auroc={rows[path.stem]['separation']['auroc']}",
            flush=True,
        )
    return {
        "source": "the delivery's own LAZ, before this project's voxel step",
        "note": (
            "intensity is not normalised between flights or sensors, which is why this project "
            "never trained on it"
        ),
        "areas": rows,
    }


def block_floor_test() -> dict:
    """What the model is handed over water: how much of a block is water, and z above the block floor."""
    rows = {}
    for area in WATER_AREAS:
        tile = CachedTile(TARGET_PREP / area, "sf")
        asprs = np.asarray(tile.asprs[:], np.int16)
        blocks = {"all_water": 0, "mixed": 0, "no_water": 0, "floor_on_the_water": 0, "counted": 0}
        water_above_floor, ground_above_floor = [], []
        water_points_in_all_water_blocks = 0
        water_points_counted = 0
        for cx, cy in tile.block_centres(BLOCK_M, None):
            idx = tile.gather_block(cx, cy, BLOCK_M)
            if idx.size < MIN_BLOCK_POINTS:
                continue
            blocks["counted"] += 1
            cls = asprs[idx]
            is_water, is_ground = cls == ASPRS_WATER, cls == ASPRS_GROUND
            share = is_water.mean()
            z_ref = tile.block_z_reference(idx)
            z = np.asarray(tile.z_cm[idx], np.float64) / 100.0 - z_ref
            if share >= 0.98:
                blocks["all_water"] += 1
                water_points_in_all_water_blocks += int(is_water.sum())
            elif share > 0:
                blocks["mixed"] += 1
            else:
                blocks["no_water"] += 1
            if is_water.any():
                water_points_counted += int(is_water.sum())
                # Does the block's floor sit on the water surface, or on land below it?
                if abs(float(np.median(z[is_water]))) <= FLOOR_NEAR_M:
                    blocks["floor_on_the_water"] += 1
                water_above_floor.append(z[is_water])
            if is_ground.any():
                ground_above_floor.append(z[is_ground])
        w = np.concatenate(water_above_floor) if water_above_floor else np.empty(0)
        g = np.concatenate(ground_above_floor) if ground_above_floor else np.empty(0)
        rows[area] = {
            "blocks": blocks,
            "share_of_blocks_all_water": round(blocks["all_water"] / max(blocks["counted"], 1), 4),
            "water_points": water_points_counted,
            "share_of_water_points_in_an_all_water_block": round(
                water_points_in_all_water_blocks / max(water_points_counted, 1), 4
            ),
            "z_above_block_floor_m": {"water": summary(w), "ground": summary(g)},
            "share_of_water_points_within_half_a_metre_of_the_floor": round(
                float((np.abs(w) <= FLOOR_NEAR_M).mean()), 4
            )
            if w.size
            else None,
            "share_of_ground_points_within_half_a_metre_of_the_floor": round(
                float((np.abs(g) <= FLOOR_NEAR_M).mean()), 4
            )
            if g.size
            else None,
            "separation_by_z_above_floor": auroc(w, g),
        }
        print(
            f"  blocks {area} all water {rows[area]['share_of_blocks_all_water']} "
            f"z auroc {rows[area]['separation_by_z_above_floor']['auroc']}",
            flush=True,
        )
    return {
        "block_m": BLOCK_M,
        "floor_quantile": 0.01,
        "min_block_points": MIN_BLOCK_POINTS,
        "floor_near_m": FLOOR_NEAR_M,
        "areas": rows,
    }


def swiss_block_test() -> dict:
    """How much of a Swiss water point's block is something other than water."""
    rows = {}
    total_water, water_with_land = 0, 0
    for name in sorted(p.name for p in SWISS_PREP.iterdir() if p.is_dir()):
        tile = CachedTile(SWISS_PREP / name, "train")
        label = np.asarray(tile.label[:], np.int16)
        if not (label == WATER).any():
            continue
        in_tile, with_land, land_shares = 0, 0, []
        for cx, cy in tile.block_centres(BLOCK_M, None):
            idx = tile.gather_block(cx, cy, BLOCK_M)
            if idx.size < MIN_BLOCK_POINTS:
                continue
            lab = label[idx]
            water = lab == WATER
            if not water.any():
                continue
            # Land is anything the table gave a class other than water, ignored points excluded.
            land = (lab >= 0) & (lab != WATER)
            in_tile += int(water.sum())
            land_shares.append(float(land.mean()))
            if land.mean() >= 0.05:
                with_land += int(water.sum())
        if in_tile == 0:
            continue
        rows[name] = {
            "water_points_in_counted_blocks": in_tile,
            "share_of_water_points_whose_block_is_at_least_5_percent_land": round(
                with_land / in_tile, 4
            ),
            "land_share_of_their_blocks": summary(np.array(land_shares)),
        }
        total_water += in_tile
        water_with_land += with_land
        print(
            f"  swiss {name} water in blocks {in_tile} with land "
            f"{rows[name]['share_of_water_points_whose_block_is_at_least_5_percent_land']}",
            flush=True,
        )
    return {
        "sub_tiles_with_water": len(rows),
        "water_points": total_water,
        "share_of_water_points_whose_block_is_at_least_5_percent_land": round(
            water_with_land / max(total_water, 1), 4
        ),
        "per_sub_tile": rows,
    }


def add_separation_readings(result: dict) -> dict:
    """Read every AUROC in the artefact both ways round, without measuring anything again."""
    for row in result.get("intensity", {}).get("areas", {}).values():
        row["separation"]["read_either_way"] = separation(row["separation"]["auroc"])
    for row in result.get("block_floor", {}).get("areas", {}).values():
        row["separation_by_z_above_floor"]["read_either_way"] = separation(
            row["separation_by_z_above_floor"]["auroc"]
        )
    return result


def both_countries() -> dict:
    """Water's own input in both countries: height above the block floor, land in the block, density."""
    rows = {}
    for label, root, names, water_of in (
        ("san_francisco", TARGET_PREP, WATER_AREAS, "asprs"),
        (
            "switzerland",
            SWISS_PREP,
            sorted(p.name for p in SWISS_PREP.iterdir() if p.is_dir()),
            "label",
        ),
    ):
        heights, land_shares, densities = [], [], []
        water_points, water_with_land = 0, 0
        for name in names:
            tile = CachedTile(root / name, "sf" if water_of == "asprs" else "train")
            field = np.asarray((tile.asprs if water_of == "asprs" else tile.label)[:], np.int16)
            water_id = ASPRS_WATER if water_of == "asprs" else WATER
            if not (field == water_id).any():
                continue
            ex, ey = tile.extent_m()
            densities.append(tile.n / max(ex * ey, 1.0))
            for cx, cy in tile.block_centres(BLOCK_M, None):
                idx = tile.gather_block(cx, cy, BLOCK_M)
                if idx.size < MIN_BLOCK_POINTS:
                    continue
                cls = field[idx]
                is_water = cls == water_id
                if not is_water.any():
                    continue
                # Land is any point the source gave a class other than water; unlabelled points are not land.
                land = (
                    (cls >= 0) & (cls != water_id)
                    if water_of == "label"
                    else (cls > 1) & (cls != water_id)
                )
                z_ref = tile.block_z_reference(idx)
                z = np.asarray(tile.z_cm[idx], np.float64) / 100.0 - z_ref
                heights.append(z[is_water])
                land_shares.append(float(land.mean()))
                water_points += int(is_water.sum())
                if land.mean() >= 0.05:
                    water_with_land += int(is_water.sum())
        rows[label] = {
            "tiles_with_water": len(densities),
            "water_points_in_counted_blocks": water_points,
            "z_above_block_floor_m": summary(np.concatenate(heights) if heights else np.empty(0)),
            "land_share_of_their_blocks": summary(np.array(land_shares)),
            "share_of_water_points_whose_block_is_at_least_5_percent_land": round(
                water_with_land / max(water_points, 1), 4
            ),
            "points_per_square_metre": summary(np.array(densities)),
        }
        print(
            f"  both {label} z {rows[label]['z_above_block_floor_m'].get('median')} land "
            f"{rows[label]['share_of_water_points_whose_block_is_at_least_5_percent_land']} density "
            f"{rows[label]['points_per_square_metre'].get('median')}",
            flush=True,
        )
    return {
        "note": (
            "the same measurements on both sides, so the difference between them is read off one "
            "table; land counts a delivery class above 1 in San Francisco and any labelled "
            "non-water class in Switzerland"
        ),
        "countries": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-intensity", action="store_true", help="the LAZ pass, which is the slow one"
    )
    ap.add_argument(
        "--both-countries-only",
        action="store_true",
        help="run only the side by side table added after the first three tests",
    )
    ap.add_argument(
        "--summarise-only",
        action="store_true",
        help="read the AUROCs already in the artefact both ways round, measuring nothing",
    )
    args = ap.parse_args()
    if args.summarise_only:
        blob = add_separation_readings(json.loads(OUT.read_text(encoding="utf-8")))
        OUT.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"read both ways round in {OUT}")
        return 0
    t0 = time.time()
    if args.both_countries_only:
        blob = json.loads(OUT.read_text(encoding="utf-8"))
        blob["both_countries"] = both_countries()
        blob["both_countries"]["measured_after"] = "a, b and c, whose readings are above"
        OUT.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote the side by side table into {OUT}")
        return 0
    result = {"reading_written_before_running": READING, "model_involved": False}
    result["block_floor"] = block_floor_test()
    result["swiss_blocks"] = swiss_block_test()
    if not args.skip_intensity:
        result["intensity"] = intensity_test()
    result["both_countries"] = both_countries()
    add_separation_readings(result)
    result["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
