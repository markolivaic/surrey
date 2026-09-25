"""Does the point spacing move only water, or every class?

Thinning San Francisco to the Swiss spacing lifted M1's water recall five times
(water_density_test.json). Nobody had asked what the same thinning does to the classes the project
actually reports. This asks it of M0 seed A, the published model, over all eight areas, thinning San
Francisco to the density DALES was prepared at (38.2 points per square metre, the median over the 40
DALES tiles) and scoring the same references the main table uses.

Runs per area: the crop as inference always draws it, the same crop at 0.95 of the area's own density
as a control on the thinning itself, and the crop at the DALES density. What is varied is how many
different points a 40 m crop carries, because thinning the pool a crop is drawn from changes nothing
while that pool is larger than the crop, which is the fault the first version of this measurement had
(surrey.density). Per run: ground recall against the delivery's class 2, vegetation against the NAIP
core both ways round, building precision inside the footprints over 2 m with its null and lift, water
called ground, which is what M0 does with water since it has no word for it, and the crop's own
distinct point count so the record says what was actually fed in.

The reading, written before the first run (2026-09-21):

  ground, vegetation and buildings all improve at DALES spacing  the main table's figures are held down
      by a spacing the model never trained at, and that belongs beside the table, not in a footnote
  only water moves                                              spacing is a water story and the rest
      of the table stands as it is
  the light control moves as much as the DALES thinning         the thinning itself is doing something
      and neither reading holds

Mechanism, only to be written if the data supports it: the grouping is k nearest neighbours with k=32,
so the same k reaches a smaller piece of the world when the cloud is denser. This measures the distance
to the 32nd neighbour at each density and on the held-out Surrey tiles, which is what the model trained
on, and the claim stands only if San Francisco at full density sits well below Surrey and the thinning
brings it back.

Run:
    python scripts/investigations/density_classes_test.py
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
from surrey.data import BLOCK_M, CLASS_NAMES, COARSE, POINTS_PER_CROP  # noqa: E402
from surrey.density import crop_at_density, distinct_for_density  # noqa: E402
from surrey.footprints import clip_to_box, inside_mask, load_footprints  # noqa: E402
from surrey.metrics import coarse_lookup  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402
from surrey.registration import footprint_offsets  # noqa: E402

TARGET_PREP = ROOT / "data" / "prep" / "target"
DALES_PREP = ROOT / "data" / "prep" / "dales"
ARTIFACTS = ROOT / "data" / "artifacts"
REFERENCE = ROOT / "data" / "reference"
FOOTPRINTS = ROOT / "data" / "target" / "sf_footprints.geojson"
OUT = ARTIFACTS / "density_classes_test.json"
GROUPS = list(COARSE)
AREAS = ("baybridge", "embarcadero", "financial", "ggpark", "mission", "soma", "steep", "sunset")
ASPRS_GROUND, ASPRS_WATER = 2, 9
HAG_GATE_M = 2.0
MIN_POINTS = 1000
MIN_BLOCK_POINTS = 2000
CONTROL_KEEP = 0.95
K = 32
COARSE_TABLE = coarse_lookup()
READING = (
    "written 2026-09-21 before the first run: if ground, vegetation and buildings all improve at "
    "DALES spacing, the main table is held down by a spacing the model never trained at and that "
    "belongs beside the table; if only water moves, spacing is a water story; if the 0.95 control "
    "moves as much as the DALES thinning, the thinning itself is doing something and neither "
    "reading holds"
)


def dales_density() -> float:
    tiles = json.loads((ARTIFACTS / "prep_dales.json").read_text(encoding="utf-8"))["tiles"]
    return float(np.median([t["density_kept"] for t in tiles]))


def knn_radius(xyz: np.ndarray, k: int, sample: int, rng: np.random.Generator) -> float | None:
    """Median distance to the k-th neighbour, over a sample of points; how far the grouping reaches."""
    from scipy.spatial import cKDTree

    if xyz.shape[0] <= k:
        return None
    tree = cKDTree(xyz)
    pick = xyz if xyz.shape[0] <= sample else xyz[rng.choice(xyz.shape[0], sample, replace=False)]
    d, _ = tree.query(pick, k=k + 1)
    return round(float(np.median(d[:, k])), 3)


@torch.no_grad()
def run_area(net, tile: CachedTile, dev: str, density: float | None, refs: dict, seed: int) -> dict:
    """One pass over every block at one density, scored against the same references the table uses.

    `density` is points per square metre for the crop, or None for every point the block has.
    """
    asprs = np.asarray(tile.asprs[:], np.int16)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    amp = dev == "cuda"
    n_classes = len(CLASS_NAMES)
    building_group, vegetation_group, ground_group = (
        GROUPS.index(g) for g in ("building", "vegetation", "ground")
    )
    counts = {
        "points": 0,
        "distinct_points": 0,
        "ground_support": 0,
        "ground_hit": 0,
        "water_support": 0,
        "water_called_ground": 0,
        "veg_core": 0,
        "veg_core_called_vegetation": 0,
        "not_veg_core": 0,
        "not_veg_core_called_vegetation": 0,
        "tall_eligible": 0,
        "building_called": 0,
        "building_called_inside": 0,
        "tall_inside": 0,
    }
    radii, crop_distinct = [], []
    for cx, cy in tile.block_centres(BLOCK_M, None):
        idx = tile.gather_block(cx, cy, BLOCK_M)
        if idx.size < MIN_BLOCK_POINTS:
            continue
        # A density in points per square metre, turned into how many different points the crop carries.
        distinct = None if density is None else distinct_for_density(density, BLOCK_M, idx.size)
        pick, what = crop_at_density(idx, POINTS_PER_CROP, rng, distinct)
        crop_distinct.append(what["distinct_in_crop"])
        z_ref = tile.block_z_reference(np.unique(pick))
        feat = tile.features(pick, (cx, cy), z_ref, channels=3)
        x = torch.from_numpy(feat[None]).to(dev)
        with torch.amp.autocast(dev, dtype=torch.float16, enabled=amp):
            pred = net(x).argmax(dim=-1)[0].cpu().numpy()
        coarse = COARSE_TABLE[np.clip(pred, 0, n_classes - 1)]
        truth = asprs[pick]
        counts["points"] += pick.size
        counts["distinct_points"] += int(what["distinct_in_crop"])
        ground = truth == ASPRS_GROUND
        counts["ground_support"] += int(ground.sum())
        counts["ground_hit"] += int((ground & (coarse == ground_group)).sum())
        water = truth == ASPRS_WATER
        counts["water_support"] += int(water.sum())
        counts["water_called_ground"] += int((water & (coarse == ground_group)).sum())
        if refs.get("veg") is not None:
            core, not_core = refs["veg"]["vegetation"][pick], refs["veg"]["not_vegetation"][pick]
            counts["veg_core"] += int(core.sum())
            counts["veg_core_called_vegetation"] += int((core & (coarse == vegetation_group)).sum())
            counts["not_veg_core"] += int(not_core.sum())
            counts["not_veg_core_called_vegetation"] += int(
                (not_core & (coarse == vegetation_group)).sum()
            )
        if refs.get("inside") is not None:
            hag = np.asarray(tile.hag_cm[pick], np.float64) / 100.0
            tall = hag > HAG_GATE_M
            called = tall & (coarse == building_group)
            counts["tall_eligible"] += int(tall.sum())
            counts["tall_inside"] += int(refs["inside"][pick][tall].sum())
            counts["building_called"] += int(called.sum())
            counts["building_called_inside"] += int(refs["inside"][pick][called].sum())
        if len(radii) < 12:
            radii.append(knn_radius(feat[:, :3].astype(np.float64), K, 4000, rng))
    out = dict(counts)
    out["ground_recall"] = (
        round(counts["ground_hit"] / counts["ground_support"], 4)
        if counts["ground_support"]
        else None
    )
    out["water_called_ground"] = (
        round(counts["water_called_ground"] / counts["water_support"], 4)
        if counts["water_support"]
        else None
    )
    out["vegetation_core_called_vegetation"] = (
        round(counts["veg_core_called_vegetation"] / counts["veg_core"], 4)
        if counts["veg_core"] >= MIN_POINTS
        else None
    )
    out["not_vegetation_core_called_vegetation"] = (
        round(counts["not_veg_core_called_vegetation"] / counts["not_veg_core"], 4)
        if counts["not_veg_core"] >= MIN_POINTS
        else None
    )
    if counts["tall_eligible"] >= MIN_POINTS and counts["building_called"]:
        precision = counts["building_called_inside"] / counts["building_called"]
        null = counts["tall_inside"] / counts["tall_eligible"]
        room = 1.0 - null
        out["building_precision"] = round(precision, 4)
        out["building_null_precision"] = round(null, 4)
        out["building_precision_lift_over_null"] = (
            round((precision - null) / room, 4) if room > 1e-9 else None
        )
    radii = [r for r in radii if r is not None]
    out["knn_32_radius_m_median_over_blocks"] = round(float(np.median(radii)), 3) if radii else None
    out["distinct_points_in_a_crop_median"] = (
        int(np.median(crop_distinct)) if crop_distinct else None
    )
    return out


def surrey_radius(seed: int) -> dict:
    """The same reach, on the tiles the model trained on, so San Francisco has something to be read against."""
    spec = json.loads((ARTIFACTS / "prep_dales.json").read_text(encoding="utf-8"))["tiles"]
    rng = np.random.default_rng(seed)
    radii = []
    for row in spec[:4]:
        tile = CachedTile(DALES_PREP / row["name"], "train")
        for cx, cy in list(tile.block_centres(BLOCK_M, 3)):
            idx = tile.gather_block(cx, cy, BLOCK_M)
            if idx.size < MIN_BLOCK_POINTS:
                continue
            pick = rng.choice(idx, min(POINTS_PER_CROP, idx.size), replace=False)
            feat = tile.features(pick, (cx, cy), None, channels=3)
            r = knn_radius(feat[:, :3].astype(np.float64), K, 4000, rng)
            if r is not None:
                radii.append(r)
    return {
        "tiles": [r["name"] for r in spec[:4]],
        "blocks": len(radii),
        "knn_32_radius_m_median": round(float(np.median(radii)), 3) if radii else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="m0_seedA")
    ap.add_argument("--device", default=None)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--areas", nargs="+", default=list(AREAS))
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    target = dales_density()
    ckpt = ROOT / "data" / "checkpoints" / f"{args.variant}.pt"
    blob = torch.load(ckpt, map_location=dev, weights_only=False)
    channels = int(blob.get("args", {}).get("channels", 3))
    net = SurreyNet(
        int(blob["model"]["head.4.bias"].shape[0]), width=args.width, extra_features=channels - 3
    ).to(dev)
    net.load_state_dict(blob["model"])
    net.eval()
    crs = json.loads((ARTIFACTS / "crs_check.json").read_text(encoding="utf-8"))
    offsets = footprint_offsets(crs)
    result = {
        "reading_written_before_running": READING,
        "variant": args.variant,
        "device": dev,
        "dales_density_points_per_m2": round(target, 1),
        "control_keep": CONTROL_KEEP,
        "k_of_the_grouping": K,
        "block_m": BLOCK_M,
        "seed": args.seed,
        "surrey_reach": surrey_radius(args.seed),
        "areas": {},
    }
    print("surrey reach", result["surrey_reach"], flush=True)
    t0 = time.time()
    for area in args.areas:
        tile = CachedTile(TARGET_PREP / area, "sf")
        density = float(json.loads((TARGET_PREP / area / "meta.json").read_text())["density_kept"])
        keep = round(
            min(target / density, 1.0), 4
        )  # kept for the record; the crop is set by density
        refs = {}
        veg_path = REFERENCE / f"naip_{area}.npz"
        if veg_path.exists():
            z = np.load(veg_path)
            refs["veg"] = {"vegetation": z["vegetation"], "not_vegetation": z["not_vegetation"]}
        polys = load_footprints(FOOTPRINTS, tile.epsg)
        if polys:
            import shapely.affinity

            ux, uy = tile.utm_xy(np.arange(tile.n))
            clipped, _ = clip_to_box(polys, ux, uy)
            dx, dy = offsets["global"]["east_m"], offsets["global"]["north_m"]
            moved = [shapely.affinity.translate(q, xoff=dx, yoff=dy) for q in clipped]
            refs["inside"] = inside_mask(moved, ux, uy)
        rows = {}
        for label, k in (
            ("full", None),
            ("control_light", density * CONTROL_KEEP),
            ("dales_density", target),
        ):
            rows[label] = run_area(net, tile, dev, k, refs, args.seed)
            print(
                f"  {area} {label} density {k} ground {rows[label]['ground_recall']} "
                f"veg {rows[label]['vegetation_core_called_vegetation']} "
                f"building lift {rows[label].get('building_precision_lift_over_null')} "
                f"reach {rows[label]['knn_32_radius_m_median_over_blocks']}",
                flush=True,
            )
        result["areas"][area] = {
            "density_points_per_m2": density,
            "keep_for_dales_density": keep,
            "crop_density_asked_for": {
                "full": None,
                "control_light": density * CONTROL_KEEP,
                "dales_density": target,
            },
            "footprints_used": "inside" in refs,
            "naip_used": "veg" in refs,
            "runs": rows,
        }
    result["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
