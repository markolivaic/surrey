"""Run the Surrey-trained model over San Francisco and score what can be scored.

The target has no hand labels, so there is no per-class IoU to be had. What there
is, and what each thing is worth:

**Ground, against the delivery's own ASPRS class 2.** A real per-point reference
for the largest class, though it is automated vendor output rather than hand
labels, so it measures agreement with an algorithm.

**Building, against Microsoft Building Footprints, always beside its null.**
scripts/pipeline/footprint_null.py measures what this metric scores with no model at all,
into data/artifacts/footprint_null.json. Calling every point over 2 m a building
already earns most of the available precision in a built area. So precision is
reported as lift over that floor, never alone. The floor is read from the
artefact rather than written here, because a figure typed into a docstring rots.

**Area coverage, not a touch count.** Asking what fraction of footprints receive
any predicted building point gives 1.00 to almost anything, because every
building has something above 2 m on it and one point inside a polygon takes the
whole mark. Coverage asks how much of each footprint's area is reached instead,
which separates touching a building from finding one. Its own null is high, so it
catches under-covering while precision catches over-prediction, and neither works
alone.

**Golden Gate Park and the Bay Bridge as controls.** In the park the footprint
null is 0.001, so there is nothing to hide behind.

**What the model does where it has no word for the answer.** The delivery labels
water and bridge deck. The DALES-2 taxonomy contains neither, so the model cannot
be right about the bay, only consistent. Over the Bay Bridge area most points
carry a label the model has no class for, see data/artifacts/sf_aois.json. That is
a taxonomy mismatch and not a domain gap, a different kind of error from the one
the park measures, and it is worth a number of its own.

Of fifteen classes the model knows, exactly one has a direct label on this target.
Two classes the target does label, the model cannot name at all.

Nothing is pooled. The areas were chosen for different morphologies and an average
over them would hide exactly the variation they exist to show.

Run:
    python scripts/investigations/score_transfer.py
    python scripts/investigations/score_transfer.py --write-laz
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import BLOCK_M, CLASS_NAMES, COARSE, N_CLASSES, POINTS_PER_CROP  # noqa: E402
from surrey.footprints import (  # noqa: E402
    clip_to_box,
    coverage_with_null,
    load_footprints,
    score_building,
)
from surrey.metrics import coarse_lookup  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402
from surrey.reference import score_ground  # noqa: E402

PREP = ROOT / "data" / "prep" / "target"
CKPT = ROOT / "data" / "checkpoints" / "dayton.pt"
FOOTPRINTS = ROOT / "data" / "target" / "sf_footprints.geojson"
DEFAULT_OUT = ROOT / "data" / "artifacts" / "transfer_sf.json"
LAZ_DIR = ROOT / "data" / "predictions"

ASPRS_GROUND = 2
ASPRS_WATER = 9
ASPRS_BRIDGE = 17
ASPRS_NOISE = (7, 18)
HAG_GATE_M = 2.0
COVERAGE_CELL_M = 1.0
CONTROL_AOIS = {"ggpark", "baybridge"}
COARSE_NAMES = list(COARSE)

# Classes the delivery labels, set against what the model can say about them.
TARGET_LABELLED = {
    ASPRS_GROUND: "ground",
    ASPRS_WATER: "water",
    ASPRS_BRIDGE: "bridge deck",
}
MODEL_HAS_CLASS_FOR = {ASPRS_GROUND: True, ASPRS_WATER: False, ASPRS_BRIDGE: False}


@torch.no_grad()
def predict_tile(
    net, tile: CachedTile, dev: str, block_m: float, n_points: int, batch: int = 4
) -> np.ndarray:
    out = np.full(tile.n, -1, dtype=np.int8)
    amp = dev == "cuda"
    pending: list[tuple[np.ndarray, np.ndarray, int]] = []

    def flush() -> None:
        if not pending:
            return
        x = torch.from_numpy(np.stack([p[0] for p in pending])).to(dev, non_blocking=True)
        with torch.amp.autocast(dev, dtype=torch.float16, enabled=amp):
            pred = net(x).argmax(dim=-1).cpu().numpy().astype(np.int8)
        for row, (_, chunk, valid) in enumerate(pending):
            out[chunk[:valid]] = pred[row, :valid]
        pending.clear()

    for feat, _, chunk, valid in tile.iter_full(block_m, n_points):
        pending.append((feat, chunk, valid))
        if len(pending) == batch:
            flush()
    flush()

    if (out < 0).any():
        raise RuntimeError(f"{tile.name}: {int((out < 0).sum())} points never predicted")
    return out


def taxonomy_rows(fine: np.ndarray, coarse: np.ndarray, asprs: np.ndarray) -> list[dict]:
    """What the model says where the delivery has a label, including where it cannot win."""
    rows = []
    for code, name in TARGET_LABELLED.items():
        ref = asprs == code
        if not ref.any():
            continue
        counts = np.bincount(coarse[ref], minlength=len(COARSE_NAMES))
        top = int(np.argmax(counts))
        rows.append(
            {
                "delivery_class": code,
                "delivery_name": name,
                "points": int(ref.sum()),
                "fraction_of_area": round(float(ref.mean()), 4),
                "model_has_a_class_for_it": MODEL_HAS_CLASS_FOR[code],
                "predicted_coarse": {
                    COARSE_NAMES[i]: round(float(counts[i] / counts.sum()), 4)
                    for i in range(len(COARSE_NAMES))
                    if counts[i]
                },
                "most_common_prediction": COARSE_NAMES[top],
                "most_common_share": round(float(counts[top] / counts.sum()), 4),
                "most_common_fine": CLASS_NAMES[
                    int(np.bincount(fine[ref], minlength=N_CLASSES).argmax())
                ],
            }
        )
    return rows


def score_one(tile: CachedTile, fine: np.ndarray, polys) -> dict:
    table = coarse_lookup()
    coarse = table[fine]
    idx = np.arange(tile.n)

    asprs = np.asarray(tile.asprs[:], dtype=np.int16)
    hag = np.asarray(tile.hag_cm[:], dtype=np.float32) / 100.0
    clean = ~np.isin(asprs, ASPRS_NOISE)
    tall = clean & (hag > HAG_GATE_M)

    g_pred = coarse == COARSE_NAMES.index("ground")
    b_pred = coarse == COARSE_NAMES.index("building")
    v_pred = coarse == COARSE_NAMES.index("vegetation")

    # Ground against the delivery, as a clean recall and a bounded precision. The
    # vendor left some real ground unlabelled, so a single precision would punish
    # the model for the delivery's gaps. See backend/surrey/reference.py.
    zb = np.asarray(tile.z_cm[idx], np.float64) / 100.0
    xb = np.asarray(tile.xy_cm[:, 0], np.float64) / 100.0
    yb = np.asarray(tile.xy_cm[:, 1], np.float64) / 100.0
    ground = score_ground(asprs, xb, yb, zb, g_pred)

    ux, uy = tile.utm_xy(idx)
    clipped, box_area = clip_to_box(polys, ux, uy)

    building_tall = score_building(clipped, ux, uy, b_pred, eligible=tall)
    building_tall.pop("recall", None)  # a touch count, and the null wins it
    building_tall.pop("null_recall", None)

    cover = coverage_with_null(clipped, ux, uy, b_pred, tall, cell_m=COVERAGE_CELL_M)

    return {
        "aoi": tile.name,
        "is_control": tile.name in CONTROL_AOIS,
        "points": tile.n,
        "epsg": tile.epsg,
        "aoi_area_m2": round(box_area, 1),
        "noise_excluded_fraction": round(float((~clean).mean()), 5),
        "points_over_2m_fraction": round(float(tall.mean()), 4),
        "predicted_distribution_coarse": {
            name: round(float((coarse[clean] == i).mean()), 5)
            for i, name in enumerate(COARSE_NAMES)
        },
        "ground_vs_delivery": ground,
        "building_vs_footprints_over_2m": building_tall,
        "building_area_coverage": cover,
        "building_fraction_over_2m": round(
            float(b_pred[tall].mean()) if tall.any() else float("nan"), 4
        ),
        "vegetation_fraction_over_2m": round(
            float(v_pred[tall].mean()) if tall.any() else float("nan"), 4
        ),
        "where_the_delivery_has_labels": taxonomy_rows(fine, coarse, asprs),
    }


def write_laz(tile: CachedTile, fine: np.ndarray, out: Path) -> dict:
    import laspy
    from pyproj import CRS

    idx = np.arange(tile.n)
    ux, uy = tile.utm_xy(idx)
    z = tile.z_abs(idx)

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.offsets = np.array([ux.min(), uy.min(), z.min()])
    hdr.scales = np.array([0.001, 0.001, 0.001])
    if tile.epsg:
        hdr.add_crs(CRS.from_epsg(tile.epsg))
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = ux, uy, z
    las.classification = fine.astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    las.write(out)

    check = laspy.read(out)
    return {
        "path": str(out.relative_to(ROOT)).replace("\\", "/"),
        "bytes": out.stat().st_size,
        "classification_round_trips": bool(
            np.array_equal(np.asarray(check.classification), fine.astype(np.uint8))
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--prep", type=Path, default=PREP)
    ap.add_argument("--footprints", type=Path, default=FOOTPRINTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--block-m", type=float, default=BLOCK_M)
    ap.add_argument("--points", type=int, default=POINTS_PER_CROP)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--write-laz", action="store_true")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"{args.ckpt} missing, run scripts/pipeline/train.py first")
    if not args.footprints.exists():
        raise SystemExit(f"{args.footprints} missing, run scripts/pipeline/extract_footprints.py")

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net = SurreyNet(N_CLASSES, width=args.width).to(dev)
    net.load_state_dict(blob["model"])
    net.eval()
    print(f"device {dev}, checkpoint step {blob['step']}")

    tiles = [
        CachedTile(p, "sf")
        for p in sorted(args.prep.iterdir())
        if p.is_dir() and (p / "meta.json").exists()
    ]
    if not tiles:
        raise SystemExit(f"no prepared target tiles in {args.prep}")

    print(f"footprints, reprojecting to EPSG:{tiles[0].epsg}")
    polys = load_footprints(args.footprints, tiles[0].epsg)
    print(f"  {len(polys):,} polygons\n")

    t0 = time.time()
    rows, written = [], []
    for tile in tiles:
        t1 = time.time()
        fine = predict_tile(net, tile, dev, args.block_m, args.points)
        r = score_one(tile, fine, polys)
        r["inference_seconds"] = round(time.time() - t1, 1)
        rows.append(r)

        b = r["building_vs_footprints_over_2m"]
        c = r["building_area_coverage"]
        g = r["ground_vs_delivery"]
        gr = g.get("recall_clean")
        print(
            f"{tile.name:12s} gnd recall "
            f"{'n/a' if gr is None else format(gr, '.3f')}  "
            f"bldg prec {b['precision']} null {b['null_precision']} "
            f"lift {b['precision_lift_over_null']}  "
            f"cover {c['coverage_median']} null {c['null_coverage_median']}",
            flush=True,
        )
        for t in r["where_the_delivery_has_labels"]:
            if not t["model_has_a_class_for_it"]:
                print(
                    f"             {t['delivery_name']:12s} "
                    f"{t['fraction_of_area']:.1%} of area, no class exists, "
                    f"called {t['most_common_prediction']} "
                    f"{t['most_common_share']:.1%} of the time",
                    flush=True,
                )
        if args.write_laz:
            written.append(write_laz(tile, fine, LAZ_DIR / f"{tile.name}.laz"))

    in_domain = None
    ta = ROOT / "data" / "artifacts" / "train_dales.json"
    if ta.exists():
        t = json.loads(ta.read_text())
        in_domain = {
            "iou_per_class": t["final"]["iou_per_class"],
            "accuracy": t["final"]["accuracy"],
            "steps_trained": t["steps_trained"],
            "device": t["device_name"],
        }

    result = {
        "device": dev,
        "device_name": (
            torch.cuda.get_device_name(0)
            if dev == "cuda"
            else platform.processor() or platform.machine()
        ),
        "torch": torch.__version__,
        "checkpoint_step": int(blob["step"]),
        "trained_on": "Surrey, British Columbia, Canada (DALES-2)",
        "scored_on": "San Francisco, California, USA (USGS 3DEP CA_SanFrancisco_1_B23)",
        "hag_gate_m": HAG_GATE_M,
        "hag_note": (
            "the 2 m gate uses a quantile ground floor and is a scoring gate "
            "only, never a model feature. Its error is measured in "
            "data/artifacts/ground_cell_choice.json"
        ),
        "coverage_cell_m": COVERAGE_CELL_M,
        "asprs_noise_excluded": list(ASPRS_NOISE),
        "taxonomy": {
            "model_classes": N_CLASSES,
            "delivery_labelled_classes": list(TARGET_LABELLED.values()),
            "model_has_a_class_for": [
                v for k, v in TARGET_LABELLED.items() if MODEL_HAS_CLASS_FOR[k]
            ],
            "model_has_no_class_for": [
                v for k, v in TARGET_LABELLED.items() if not MODEL_HAS_CLASS_FOR[k]
            ],
            "note": (
                "of fifteen classes the model knows, exactly one has a direct "
                "label on this target. Two classes the target labels, the model "
                "cannot name, so over water and over a bridge deck it cannot be "
                "right, only consistent."
            ),
        },
        "in_domain_reference": in_domain,
        "caveats": [
            "the target carries no hand labels, so there is no per-class IoU here",
            "ground is scored against the delivery's automated classification, which "
            "measures agreement with a vendor algorithm and not correctness",
            "the delivery leaves real ground unlabelled, so ground precision is a band "
            "between a strict floor and a generous ceiling rather than one number. "
            "See data/artifacts/ground_reference_check.json",
            "a footprint is a 2D polygon whose interior holds ground, parked cars and "
            "overhanging branches, so precision is only meaningful beside its null",
            "the fraction of footprints touched by any predicted point is not reported. "
            "It is 1.00 for almost any model and measures nothing",
            "coverage has a high null of its own, so it catches under-covering only",
        ],
        "aois": rows,
        "laz_written": written,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
