"""Every San Francisco figure, from cached predictions, for one input variant.

The return ratio was 0 on every DALES-2 training point and alive on every San
Francisco point, so every San Francisco figure from the first checkpoint measured a
model reading an input it had never seen. This scores each named variant from its cached
predictions, with no inference:

    returns_ood   data/checkpoints/dayton.pt, the input as delivered
    ret0          the same checkpoint, return ratio forced to 0 at inference
    m0_seedA      three channel retrain, seed A, the published model
    m0_seedB      three channel retrain, seed B

and writes data/artifacts/sf_scores_<variant>.json. Running it on returns_ood as
well gives the before column on the same references as the after column.

References, changed from the first scoring and applied to every variant alike:

    footprints   moved by the lidar-minus-footprint offset measured in
                 scripts/pipeline/crs_check.py: one global offset, the per-axis median over
                 areas whose correlation cleared the peak bar, with each area's own
                 offset as a sensitivity and the unmoved footprints for comparison.
                 The offset comes from lidar heights and footprint outlines, not from
                 any prediction.
    vegetation   the NAIP reference registered per area (naip_reference.py
                 --registered), beside the unregistered one.

Per area, never pooled: predicted class mix; water and bridge deck; the ground band
with and without water; building precision and lift over the null; vegetation
cores; confidence by delivery class; the easy and hard error AUROC.

The confounds named before scoring, tested again on this variant. The rule that named
them was written before the first model existed and has not changed; its first run was
on the broken input, so whether they predict degradation is open again.

    extrapolation   in Financial, blocks whose vertical range exceeds the tallest
                    training block against the rest, on building lift over the null.
                    The original wording set Financial against the other built areas;
                    that comparison is reported too.
    relief          on Steep, blocks whose ground relief exceeds a typical Surrey
                    building against the rest, on delivery ground called ground.

Reading, fixed before this script was first run on any variant:
    the named blocks score lower by at least 0.05   the confound predicts degradation
    within 0.05 either way                          it measured nothing on this variant
    higher by at least 0.05                         contrary, reported as such

Amended after the first run, on review, and recorded as an amendment. Which metric is
primary is taken from the earliest written prediction, not from this docstring:

    relief         scripts/pipeline/input_range_check.py, written before any training, says
                   verbatim: "If the two distributions of within-block ground relief
                   do not overlap, the model will call hillsides buildings". The
                   primary metric is therefore delivery ground called BUILDING, where
                   degradation means the named blocks are higher by at least 0.05.
                   Delivery ground called ground, the metric this docstring named
                   first, is reported as secondary.
    extrapolation  the rejection written before E0 compares Financial against the
                   other areas, so that is primary. The within-Financial block
                   comparison is an addition made here, and is labelled as one.

Run:
    python scripts/pipeline/score_variant.py --variant ret0
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
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import CLASS_NAMES, COARSE  # noqa: E402
from surrey.footprints import (  # noqa: E402
    clip_to_box,
    coverage_with_null,
    inside_mask,
    load_footprints,
)
from surrey.m1_criteria import (  # noqa: E402
    NULL_DESCRIPTION,
    UPPER_EDGE_EXTRA_CLASS,
    WATER_AREAS,
    WATER_BAND_WRITTEN,
    WATER_POLYGON_EDGE_WRITTEN,
)
from surrey.metrics import (  # noqa: E402
    Confusion,
    auroc_from_histograms,
    auroc_row,
    coarse_lookup,
    coarse_shares,
    score_histogram,
)
from surrey.reference import (  # noqa: E402
    GROUND_CODES,
    SURFACE_CELL_M,
    SURFACE_REACH_M,
    score_ground,
    vendor_ground_surface,
)
from surrey.registration import footprint_offsets  # noqa: E402
from surrey.sources import BRIDGE_DECK, WATER  # noqa: E402
from surrey.water_null import (  # noqa: E402
    MARGIN_M,
    SENSITIVITY_MARGINS_M,
    as_null,
    lift_over_null,
    water_scores,
)

DALES_PREP = ROOT / "data" / "prep" / "dales"
TARGET_PREP = ROOT / "data" / "prep" / "target"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
PRED = ROOT / "data" / "predictions"
REFERENCE = ROOT / "data" / "reference"
FOOTPRINTS = ROOT / "data" / "target" / "sf_footprints.geojson"
CRS = ROOT / "data" / "artifacts" / "crs_check.json"
RANGE = ROOT / "data" / "artifacts" / "input_range_check.json"
ARTIFACTS = ROOT / "data" / "artifacts"

VARIANTS = {
    "returns_ood": {
        "sf": PRED,
        "surrey": PRED,
        "note": "four channel checkpoint, input as delivered",
    },
    "ret0": {
        "sf": PRED / "ret0",
        "surrey": PRED,
        "note": "four channel checkpoint, return ratio forced to 0; on Surrey the ratio is "
        "already 0 on every point, so its Surrey predictions are the as-delivered ones",
    },
    "m0_seedA": {
        "sf": PRED / "m0_seedA",
        "surrey": PRED / "m0_seedA",
        "note": "three channel retrain, seed A",
    },
    "m0_seedB": {
        "sf": PRED / "m0_seedB",
        "surrey": PRED / "m0_seedB",
        "note": "three channel retrain, seed B",
    },
}
RECIPE_NOTES = {
    "m0": "three channel retrain on DALES-2 alone",
    "m0prime": "DALES-2 and Swiss sub-tiles, Swiss water and bridge deck ignored",
    "m1": "DALES-2 and Swiss sub-tiles, water and bridge deck as classes",
}
for _recipe, _note in RECIPE_NOTES.items():
    for _seed in ("A", "B", "C"):
        _v = f"{_recipe}_seed{_seed}"
        VARIANTS.setdefault(
            _v, {"sf": PRED / _v, "surrey": PRED / _v, "note": f"{_note}, seed {_seed}"}
        )
# The runs from the rented card, scored by the same code as everything else; only the note says where
# they were trained, and the training artefact carries the card's name and memory.
RENTED_NOTES = {
    "m0_rented_small_seedA": "the published configuration, trained on a rented RTX 4090",
    "m0_big_seedA": "width 512, batch 8, same points seen, learning rate scaled with the batch, rented card",
    "m1_big_seedA": "the same large recipe on M1 data, rented card",
    "m0_big_2x_seedA": "the large M0 at twice the points seen, a sensitivity, rented card",
    "m0_big_lr2_seedA": "the large M0 with the learning rate unscaled, a sensitivity, rented card",
    "m1_densityaug_seedA": "the small M1 with the crop density drawn at random, a diagnosis, rented card",
}
for _v, _note in RENTED_NOTES.items():
    VARIANTS.setdefault(_v, {"sf": PRED / _v, "surrey": PRED / _v, "note": _note})
# Diagnostics of new M0 seed A that separate what inference alone changed
# (scripts/investigations/building_breakdown.py).
DIAGNOSTIC_NOTES = {
    "m0_seedA_1vote": "new M0 seed A with one vote, a diagnostic",
    "m0_seedA_legacy_iter": "new M0 seed A with one vote and the first round's iterator, a diagnostic",
}
for _v, _note in DIAGNOSTIC_NOTES.items():
    VARIANTS.setdefault(_v, {"sf": PRED / _v, "surrey": PRED / _v, "note": _note})
# Phase 4's M0, trained and predicted with the fp16 neighbour search, kept aside by
# scripts/pipeline/phase6_pull.sh and scored with the same code, so Phase 6 is read against it
# (docs/phase6_criteria.md).
for _seed in ("A", "B", "C"):
    VARIANTS[f"m0phase4_seed{_seed}"] = {
        "sf": PRED / "phase4" / f"m0_seed{_seed}",
        "surrey": PRED / "phase4" / f"m0_seed{_seed}",
        "note": f"Phase 4 M0 seed {_seed}, the fp16 neighbour search",
    }
GROUPS = list(COARSE)
WATER_NULL = ROOT / "data" / "reference" / "water_null"
WATER_POLYGON = ROOT / "data" / "reference" / "water_polygon"
GROUND = CLASS_NAMES.index("ground")
ASPRS_GROUND, ASPRS_UNASSIGNED, ASPRS_WATER, ASPRS_BRIDGE = 2, 1, 9, 17
ASPRS_NOISE = (7, 18)
HAG_GATE_M = 2.0
EASY_ABOVE_M = 3.0
HARD_WITHIN_M = 0.5
EASY_EXCLUDED = {
    "steep": "29 percent grade; a 3 m gate over an interpolated surface is a guess",
    "baybridge": "no delivery ground class",
}
MIN_POINTS = 1000
MIN_BLOCK_POINTS = 2000
MIN_GROUND_POINTS = 200
MARGIN = 0.05
CONFIDENCE_GROUPS = {
    ASPRS_GROUND: "sf_ground",
    ASPRS_UNASSIGNED: "sf_unassigned",
    ASPRS_WATER: "sf_water",
    ASPRS_BRIDGE: "sf_bridge_deck",
}


def load(directory: Path, name: str, entropy_scale: float = 1.0):
    z = np.load(directory / f"{name}.npz")
    return z["pred"], z["conf"].astype(np.float32), z["ent"].astype(np.float32) * entropy_scale


def null_key(margin: float) -> str:
    return f"null_{str(margin).replace('.', '_')}m"


def water_band_truth(asprs: np.ndarray, z: np.ndarray, level: np.ndarray) -> dict[str, np.ndarray]:
    """The two edges of the water reference (surrey.m1_criteria): class 9 alone, and class 9
    with class 1 below the gauge level at the point's time plus the null's margin."""
    lower = asprs == ASPRS_WATER
    upper = lower | ((asprs == UPPER_EDGE_EXTRA_CLASS) & (z < level + MARGIN_M))
    return {"lower": lower, "upper": upper}


def missing_references(areas: list[str]) -> list[str]:
    """Every reference a column is scored against that is not on disk. Scoring stops on any of them:
    a column left out without a word reads as a column measured and found empty."""
    need = [(FOOTPRINTS, "extract_footprints.py")]
    for area in areas:
        need += [
            (REFERENCE / "registered" / f"naip_{area}.npz", "naip_reference.py"),
            (REFERENCE / f"naip_{area}.npz", "naip_reference.py"),
        ]
    for area in WATER_AREAS:
        need += [
            (WATER_NULL / f"{area}.npz", "water_null.py"),
            (WATER_POLYGON / f"{area}.npz", "water_polygon.py"),
        ]
    return [
        f"{path.relative_to(ROOT).as_posix()}, made by scripts/pipeline/{script}"
        for path, script in need
        if not path.exists()
    ]


def water_vs_null(
    name: str,
    pred: np.ndarray,
    asprs: np.ndarray,
    clean: np.ndarray,
    n_classes: int,
    z: np.ndarray | None = None,
) -> dict | None:
    """M1's water call and the physical null's, against delivery water, on the two areas
    the delivery labels water. A fifteen class model has no water output; its call is
    empty by construction and is marked so rather than scored as a failure to learn."""
    if name not in WATER_AREAS:
        return None
    ref = np.load(WATER_NULL / f"{name}.npz")
    truth = asprs == ASPRS_WATER
    has_word = n_classes > WATER
    model_call = pred == WATER if has_word else np.zeros(pred.shape, bool)
    model = water_scores(model_call, truth, clean)
    nulls = {
        null_key(m): as_null(water_scores(ref[null_key(m)], truth, clean))
        for m in (MARGIN_M, *SENSITIVITY_MARGINS_M)
    }
    head = nulls[null_key(MARGIN_M)]
    model |= {
        "null_precision": head["null_precision"],
        "null_recall": head["null_recall"],
        "null_iou": head["null_iou"],
    }
    return {
        "model_has_water_output": has_word,
        "model": model,
        "null": nulls,
        "headline_null": null_key(MARGIN_M),
        "lift_over_null": lift_over_null(model["iou"], head["null_iou"]) if has_word else None,
        "lift_over_null_sensitivity": {
            k: lift_over_null(model["iou"], v["null_iou"]) for k, v in nulls.items()
        }
        if has_word
        else None,
        "band": water_band(ref, pred, asprs, clean, has_word, z, name) if z is not None else None,
        "scope": "Embarcadero and Bay Bridge only, the areas the delivery labels water",
    }


def water_band(
    ref,
    pred: np.ndarray,
    asprs: np.ndarray,
    clean: np.ndarray,
    has_word: bool,
    z: np.ndarray,
    name: str,
) -> dict:
    edges = water_band_truth(asprs, z, ref["level_navd88_m"].astype(np.float64))
    # The edge with no height in it: class 1 inside the NHD water polygon (scripts/pipeline/water_polygon.py).
    inside = np.load(WATER_POLYGON / f"{name}.npz")["inside"]
    edges["upper_polygon"] = (asprs == ASPRS_WATER) | ((asprs == UPPER_EDGE_EXTRA_CLASS) & inside)
    model_call = pred == WATER if has_word else np.zeros(pred.shape, bool)
    out = {
        "written": WATER_BAND_WRITTEN,
        "polygon_edge_written": WATER_POLYGON_EDGE_WRITTEN,
        "null_description": NULL_DESCRIPTION,
    }
    for edge, truth in edges.items():
        null = as_null(water_scores(ref[null_key(MARGIN_M)], truth, clean))
        model = water_scores(model_call, truth, clean) | {
            k: null[k] for k in ("null_precision", "null_recall", "null_iou")
        }
        out[edge] = {
            "water_points": int((truth & clean).sum()),
            "model": model,
            "null": null,
            "lift_over_null": lift_over_null(model["iou"], null["null_iou"]) if has_word else None,
            "null_agrees_by_construction": edge == "upper",
            "uses_height": edge == "upper",
        }
    return out


def rate(hit: np.ndarray, population: np.ndarray) -> float | None:
    n = int(population.sum())
    return round(float((hit & population).sum() / n), 4) if n >= MIN_POINTS else None


def confidence(conf: np.ndarray, ent: np.ndarray) -> dict:
    if conf.size < MIN_POINTS:
        return {"points": int(conf.size)}
    return {
        "points": int(conf.size),
        "max_prob_median": round(float(np.median(conf)), 4),
        "max_prob_p10": round(float(np.percentile(conf, 10)), 4),
        "max_prob_p90": round(float(np.percentile(conf, 90)), 4),
        "max_prob_over_0_9": round(float((conf > 0.9).mean()), 4),
        "max_prob_under_0_5": round(float((conf < 0.5).mean()), 4),
        "entropy_median": round(float(np.median(ent)), 4),
    }


def building(inside: np.ndarray, predicted: np.ndarray, eligible: np.ndarray) -> dict:
    """Precision of predicted building points over 2 m against footprints, beside the
    null of calling every eligible point a building, and the lift between them as a
    share of the room above the null (surrey.footprints.score_building's definition)."""
    pred = predicted & eligible
    if eligible.sum() < MIN_POINTS:
        return {"eligible_points": int(eligible.sum())}
    null = float(inside[eligible].mean())
    prec = float(inside[pred].mean()) if pred.any() else float("nan")
    room = 1.0 - null
    # Where the null calls nothing a building there is nothing to lift over.
    lift = (prec - null) / room if room > 1e-9 and null > 0 and prec == prec else float("nan")
    return {
        "eligible_points": int(eligible.sum()),
        "predicted_building_points": int(pred.sum()),
        "precision": None if prec != prec else round(prec, 4),
        "null_precision": round(null, 4),
        "precision_lift_over_null": None if lift != lift else round(lift, 4),
    }


def blocks(tile: CachedTile, block_m: float, asprs: np.ndarray, clean: np.ndarray, z: np.ndarray):
    """Per block, as scripts/pipeline/input_range_check.py measured them: the vertical range the
    network sees, and the delivery ground's relief."""
    ex, ey = tile.extent_m()
    for bx in range(int(np.ceil(ex / block_m))):
        for by in range(int(np.ceil(ey / block_m))):
            idx = tile.gather_block((bx + 0.5) * block_m, (by + 0.5) * block_m, block_m)
            if idx.size < MIN_BLOCK_POINTS:
                continue
            zb = z[idx]
            rel = zb - float(np.quantile(zb, 0.01))
            c = clean[idx]
            zrange = float(np.quantile(rel[c], 0.999)) if c.any() else float("nan")
            g = np.isin(asprs[idx], GROUND_CODES)
            relief = (
                float(np.quantile(zb[g], 0.98) - np.quantile(zb[g], 0.02))
                if g.sum() >= MIN_GROUND_POINTS
                else float("nan")
            )
            yield idx, zrange, relief


def verdict_error(named: float | None, rest: float | None) -> str | None:
    """The same reading for an error rate, where higher is worse."""
    if named is None or rest is None:
        return None
    return verdict(rest, named)


def verdict(named: float | None, rest: float | None) -> str | None:
    if named is None or rest is None:
        return None
    d = named - rest
    if d <= -MARGIN:
        return "predicts degradation"
    if d >= MARGIN:
        return "contrary"
    return "measured nothing"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    v = VARIANTS[args.variant]
    out = args.out or ARTIFACTS / f"sf_scores_{args.variant}.json"
    t0 = time.time()

    table = coarse_lookup()
    crs = json.loads(CRS.read_text(encoding="utf-8"))
    offsets = footprint_offsets(crs)
    rng = json.loads(RANGE.read_text(encoding="utf-8"))
    train_max = float(rng["vertical_extrapolation"]["surrey_training_block_z_max_m"])
    typical_building = float(rng["slope_leak"]["surrey_typical_building_top_m"])
    block_m = float(rng["block_m"])
    import shapely.affinity

    manifest = json.loads((v["sf"] / "manifest.json").read_text(encoding="utf-8"))
    n_classes = int(manifest.get("classes", 15))
    # Predictions made before predict_cache recorded its normaliser divided entropy by log 15 for
    # every model; for a model with another class count that is rescaled here to its own.
    entropy_scale = (
        1.0 if "entropy_normalised_by_log_of" in manifest else float(np.log(15) / np.log(n_classes))
    )
    groups = list(GROUPS)
    if n_classes > len(table):
        # M1: water and bridge deck are their own coarse groups.
        table = np.concatenate([table, np.array([len(GROUPS), len(GROUPS) + 1], table.dtype)])
        groups = [*GROUPS, "water", "bridge_deck"]
    mask_choice = {
        r["aoi"]: r
        for r in json.loads((ARTIFACTS / "naip_mask_choice.json").read_text(encoding="utf-8"))[
            "areas"
        ]
    }
    areas_on_disk = sorted(
        p.name for p in TARGET_PREP.iterdir() if p.is_dir() and (p / "meta.json").exists()
    )
    missing = missing_references(areas_on_disk)
    if missing:
        raise SystemExit("references missing, nothing scored:\n  " + "\n  ".join(missing))
    polys = None
    areas = []
    for p in sorted(TARGET_PREP.iterdir()):
        if not (p.is_dir() and (p / "meta.json").exists()):
            continue
        name = p.name
        tile = CachedTile(p, "sf")
        if polys is None:
            polys = load_footprints(FOOTPRINTS, tile.epsg)
        pred, conf, ent = load(v["sf"], f"sf_{name}", entropy_scale)
        coarse = table[pred]
        asprs = np.asarray(tile.asprs[:], np.int16)
        clean = ~np.isin(asprs, ASPRS_NOISE)
        hag = np.asarray(tile.hag_cm[:], np.float32) / 100.0
        x = np.asarray(tile.xy_cm[:, 0], np.float64) / 100.0
        y = np.asarray(tile.xy_cm[:, 1], np.float64) / 100.0
        z = np.asarray(tile.z_cm[:], np.float64) / 100.0
        ux, uy = tile.utm_xy(np.arange(tile.n))
        called = pred == GROUND
        row = {
            "aoi": name,
            "points": tile.n,
            "predicted_coarse_shares": coarse_shares(coarse[clean], groups),
        }

        # Classes the model has no word for.
        row["water_called_ground"] = rate(called, asprs == ASPRS_WATER)
        row["water_called"] = coarse_shares(coarse[asprs == ASPRS_WATER], groups)
        row["bridge_deck_called_building"] = rate(
            coarse == GROUPS.index("building"), asprs == ASPRS_BRIDGE
        )
        row["bridge_deck_called"] = coarse_shares(coarse[asprs == ASPRS_BRIDGE], groups)
        # The band compares heights with the gauge level in NAVD88, so it needs absolute heights.
        row["water_vs_null"] = water_vs_null(
            name, pred, asprs, clean, n_classes, z + float(tile.origin[2])
        )
        if (asprs == ASPRS_BRIDGE).sum() >= MIN_POINTS:
            deck_call = (
                pred == BRIDGE_DECK if n_classes > BRIDGE_DECK else np.zeros(pred.shape, bool)
            )
            deck = water_scores(deck_call, asprs == ASPRS_BRIDGE, clean)
            # water_scores names its reference count for water; here the reference is the deck.
            deck["deck_points"] = deck.pop("water_points")
            row["bridge_deck_call"] = deck | {
                "model_has_bridge_deck_output": n_classes > BRIDGE_DECK,
                "null_precision": None,
                "note": "descriptive only; no simple null for a bridge deck",
            }

        # Ground band, with water and without.
        g_pred = coarse == GROUPS.index("ground")
        if np.isin(asprs, GROUND_CODES).any():
            a_nowater = asprs.copy()
            a_nowater[a_nowater == ASPRS_WATER] = ASPRS_NOISE[0]
            keep = (
                "recall_clean",
                "precision_floor",
                "precision_ceiling",
                "predicted_points_unjudgeable_fraction",
            )
            gw, gn = score_ground(asprs, x, y, z, g_pred), score_ground(a_nowater, x, y, z, g_pred)
            row["ground_band"] = {
                "with_water": {k: gw.get(k) for k in keep},
                "without_water": {k: gn.get(k) for k in keep},
            }
            surf, key = vendor_ground_surface(
                x, y, z, np.isin(asprs, GROUND_CODES) & clean, SURFACE_CELL_M, SURFACE_REACH_M
            )
            h = z - surf[key]
        else:
            row["ground_band"] = None
            h = np.full(tile.n, np.nan)

        # Buildings against footprints, three placements. What the delivery labels water or bridge
        # deck is not a candidate building; the deck keeps its own figure above. The primary gate
        # is the label-free floor; a second gate from the delivery's own ground surface stands
        # beside it for the global placement, because the floor sits on the roofs of tall blocks.
        not_building_ref = np.isin(asprs, (ASPRS_WATER, ASPRS_BRIDGE))
        tall = clean & (hag > HAG_GATE_M) & ~not_building_ref
        tall_delivery = clean & (np.nan_to_num(h, nan=-np.inf) > HAG_GATE_M) & ~not_building_ref
        b_pred = coarse == GROUPS.index("building")
        clipped, _ = clip_to_box(polys, ux, uy)
        placements = {
            "unshifted": (0.0, 0.0),
            "global_shift": (offsets["global"]["east_m"], offsets["global"]["north_m"]),
            "per_area_shift": (
                offsets["per_area"].get(name, offsets["global"])["east_m"],
                offsets["per_area"].get(name, offsets["global"])["north_m"],
            ),
        }
        inside_global = None
        row["building_vs_footprints_over_2m"] = {}
        for label, (dx, dy) in placements.items():
            moved = [shapely.affinity.translate(q, xoff=dx, yoff=dy) for q in clipped]
            inside = inside_mask(moved, ux, uy)
            row["building_vs_footprints_over_2m"][label] = {
                "offset_east_north_m": [round(dx, 3), round(dy, 3)],
                **building(inside, b_pred, tall),
            }
            if label == "global_shift":
                inside_global = inside
                row["building_area_coverage_global_shift"] = coverage_with_null(
                    moved, ux, uy, b_pred, tall
                )
                row["building_delivery_gate_global_shift"] = {
                    "gate": "height above the delivery's own ground surface, classes 2 and 20",
                    "points_the_surface_reaches": round(float(np.isfinite(h).mean()), 4),
                    **building(inside, b_pred, tall_delivery),
                    "coverage": coverage_with_null(moved, ux, uy, b_pred, tall_delivery),
                }
        lifts = {
            k: r.get("precision_lift_over_null")
            for k, r in row["building_vs_footprints_over_2m"].items()
        }
        if lifts["global_shift"] is not None and lifts["per_area_shift"] is not None:
            row["building_sign_differs_global_vs_per_area"] = (lifts["global_shift"] > 0) != (
                lifts["per_area_shift"] > 0
            )

        # Vegetation, against both references.
        row["vegetation"] = {}
        for label, path in (
            ("registered", REFERENCE / "registered" / f"naip_{name}.npz"),
            ("unregistered", REFERENCE / f"naip_{name}.npz"),
        ):
            ref = np.load(path)
            row["vegetation"][label] = {
                "vegetation_core_points": int(ref["vegetation"].sum()),
                "not_vegetation_core_points": int(ref["not_vegetation"].sum()),
                "vegetation_core_called": coarse_shares(coarse[ref["vegetation"]], groups),
                "not_vegetation_core_called_vegetation": rate(
                    coarse == GROUPS.index("vegetation"), ref["not_vegetation"]
                ),
            }

        # The headline figure follows scripts/pipeline/naip_mask_choice.py, a rule fixed before it
        # ran and decided on multiple returns, never on predictions: the chosen mask's
        # figure, or where the rule does not decide, the lower of the two.
        choice = mask_choice.get(name, {}).get("reference")
        figs = {
            k: (r.get("vegetation_core_called") or {}).get("vegetation")
            for k, r in row["vegetation"].items()
        }
        false = {
            k: r.get("not_vegetation_core_called_vegetation") for k, r in row["vegetation"].items()
        }
        if choice == "lower_of_two" and None not in figs.values() and figs:
            head, basis = min(figs.values()), "lower of the two masks: the rule did not decide"
            known = [v for v in false.values() if v is not None]
            # Both halves of the pair read the harsher mask: recall the lower, false vegetation
            # the higher.
            wrong = max(known) if known else None
        elif choice in figs:
            head, basis = figs[choice], f"{choice} mask chosen by the rule"
            wrong = false.get(choice)
        else:
            head, basis, wrong = None, "no vegetation core", None
        row["vegetation_headline"] = {
            "vegetation_core_called_vegetation": head,
            "not_vegetation_core_called_vegetation": wrong,
            "basis": basis,
        }

        # Confidence by what the delivery says.
        row["confidence"] = {
            group: confidence(conf[clean & (asprs == code)], ent[clean & (asprs == code)])
            for code, group in CONFIDENCE_GROUPS.items()
        }

        # Error detection: among ground calls, right against easy, hard and water.
        right = called & clean & np.isin(asprs, GROUND_CODES)
        easy = called & clean & (asprs == ASPRS_UNASSIGNED) & (h > EASY_ABOVE_M)
        water = called & (asprs == ASPRS_WATER)
        raised = clean & (asprs == ASPRS_UNASSIGNED) & (h > EASY_ABOVE_M)
        row["raised_unassigned_called_ground_rate"] = {
            "raised_points": int(raised.sum()),
            "rate": rate(called, raised),
            "note": "delivery unassigned more than 3 m above the delivery ground surface; the Surrey "
            "rate is over DALES non-ground labels on the same definition",
        }
        row["auroc"] = {
            "easy_unassigned_over_3m": auroc_row(conf, right, easy),
            "hard_water_within_0_5m": auroc_row(conf, right, water & (np.abs(h) <= HARD_WITHIN_M)),
            "water_any_height": auroc_row(conf, right, water),
            "excluded_from_easy": EASY_EXCLUDED.get(name),
        }

        # Confounds, block by block.
        extrap = np.zeros(tile.n, bool)
        relief = np.zeros(tile.n, bool)
        in_block = np.zeros(tile.n, bool)
        for idx, zrange, rel in blocks(tile, block_m, asprs, clean, z):
            in_block[idx] = True
            if zrange == zrange and zrange > train_max:
                extrap[idx] = True
            if rel == rel and rel > typical_building:
                relief[idx] = True
        conf_row = {
            "blocks_named_points": {
                "extrapolated": int(extrap.sum()),
                "high_relief": int(relief.sum()),
            }
        }
        if inside_global is not None and extrap.any():
            named = building(inside_global, b_pred, tall & extrap)
            rest = building(inside_global, b_pred, tall & in_block & ~extrap)
            conf_row["extrapolation_building_lift"] = {
                "status": "addition, not in the prediction written before E0",
                "named_blocks": named,
                "other_blocks": rest,
                "reading": verdict(
                    named.get("precision_lift_over_null"), rest.get("precision_lift_over_null")
                ),
            }
        if relief.any():
            g_true = clean & np.isin(asprs, GROUND_CODES)
            named = rate(called, g_true & relief)
            rest = rate(called, g_true & in_block & ~relief)
            named_b = rate(coarse == GROUPS.index("building"), g_true & relief)
            rest_b = rate(coarse == GROUPS.index("building"), g_true & in_block & ~relief)
            conf_row["relief_ground_recall"] = {
                "named_blocks": named,
                "other_blocks": rest,
                "named_blocks_ground_called_building": named_b,
                "other_blocks_ground_called_building": rest_b,
                "primary": {
                    "metric": "delivery ground called building",
                    "source": "scripts/pipeline/input_range_check.py, written before training: "
                    "'the model will call hillsides buildings'",
                    "named_minus_other": None
                    if None in (named_b, rest_b)
                    else round(named_b - rest_b, 4),
                    "reading": verdict_error(named_b, rest_b),
                },
                "secondary": {
                    "metric": "delivery ground called ground",
                    "named_minus_other": None if None in (named, rest) else round(named - rest, 4),
                    "reading": verdict(named, rest),
                },
                "reading": verdict_error(named_b, rest_b),
            }
        row["confounds"] = conf_row
        areas.append(row)

        bg = row["building_vs_footprints_over_2m"]["global_shift"]
        vr = (row["vegetation"].get("registered") or {}).get("vegetation_core_called") or {}
        print(
            f"  {name:12s} lift global {bg.get('precision_lift_over_null')} (null {bg.get('null_precision')})  "
            f"veg core called veg {vr.get('vegetation')}  easy auroc {row['auroc']['easy_unassigned_over_3m']['auroc']}  "
            f"water->ground {row['water_called_ground']}  confounds {conf_row.get('extrapolation_building_lift', {}).get('reading')} "
            f"{conf_row.get('relief_ground_recall', {}).get('reading')}",
            flush=True,
        )

    # The original wording of the extrapolation rejection: Financial against the other
    # built areas, on building lift with the global shift.
    built = ["embarcadero", "mission", "soma", "sunset"]
    lift = {
        a["aoi"]: a["building_vs_footprints_over_2m"]["global_shift"].get(
            "precision_lift_over_null"
        )
        for a in areas
    }
    others = [lift[b] for b in built if lift.get(b) is not None]
    across = {
        "financial": lift.get("financial"),
        "other_built_areas": {b: lift.get(b) for b in built},
        "other_built_median": round(float(np.median(others)), 4) if others else None,
    }
    across["reading"] = verdict(across["financial"], across["other_built_median"])
    across["status"] = "primary: the formulation written before E0"

    # Surrey, the in-domain side of the confidence and AUROC tables.
    names = sorted(
        t["name"]
        for t in json.loads(PREP_ARTIFACT.read_text(encoding="utf-8"))["tiles"]
        if t["split"] == "test"
    )
    pools = {"all": ([], []), "easy": ([], []), "hard": ([], [])}
    conf_right, conf_wrong = [], []
    # Every labelled point in one confusion. An output DALES-2 has no class for (M1's water and bridge
    # deck) is a wrong answer for the point's own class, so it gets a column of its own, not a drop.
    extra = "water or bridge deck output"
    fine = Confusion([*CLASS_NAMES, extra])
    coarse_names = list(COARSE)
    grouped = Confusion([*coarse_names, extra])
    group_of = np.append(coarse_lookup(), [len(coarse_names)] * 2)
    raised_rates = []
    correct = total = 0
    for n in names:
        tile = CachedTile(DALES_PREP / n, "test")
        pred, conf, ent = load(v["surrey"], f"surrey_{n}", entropy_scale)
        label = np.asarray(tile.label[:], np.int16)
        lab = label >= 0
        capped = np.minimum(pred.astype(np.int64), len(CLASS_NAMES))
        fine.add(label[lab], capped[lab])
        grouped.add(group_of[label[lab]], group_of[np.minimum(pred[lab].astype(np.int64), 16)])
        correct += int(((pred == label) & lab).sum())
        total += int(lab.sum())
        ok = lab & (pred == label)
        bad = lab & (pred != label)
        conf_right.append(np.stack([conf[ok], ent[ok]]))
        conf_wrong.append(np.stack([conf[bad], ent[bad]]))
        x = np.asarray(tile.xy_cm[:, 0], np.float64) / 100.0
        y = np.asarray(tile.xy_cm[:, 1], np.float64) / 100.0
        z = np.asarray(tile.z_cm[:], np.float64) / 100.0
        surf, key = vendor_ground_surface(x, y, z, label == GROUND, SURFACE_CELL_M, SURFACE_REACH_M)
        h = z - surf[key]
        called = (pred == GROUND) & lab
        right = called & (label == GROUND)
        wrong = called & (label != GROUND)
        raised = lab & (label != GROUND) & (h > EASY_ABOVE_M)
        raised_rates.append(
            {
                "tile": n,
                "raised_points": int(raised.sum()),
                "rate": round(float((called & raised).sum() / raised.sum()), 4)
                if raised.sum() >= MIN_POINTS
                else None,
            }
        )
        for k, m in (
            ("all", wrong),
            ("easy", wrong & (h > EASY_ABOVE_M)),
            ("hard", wrong & (np.abs(h) <= HARD_WITHIN_M)),
        ):
            pools[k][0].append(score_histogram(conf[right]))
            pools[k][1].append(score_histogram(conf[m]))
    cr = np.concatenate(conf_right, axis=1)
    cw = np.concatenate(conf_wrong, axis=1)
    iou_fine = {k: v for k, v in fine.iou().items() if k != extra}
    iou_coarse = {k: v for k, v in grouped.iou().items() if k != extra}
    surrey = {
        "accuracy_labelled_voxel_points": round(correct / total, 4),
        "iou_per_class": iou_fine,
        "miou_15": round(float(np.mean([v for v in iou_fine.values() if v is not None])), 4),
        "iou_per_coarse_group": iou_coarse,
        "miou_coarse": round(float(np.mean([v for v in iou_coarse.values() if v is not None])), 4),
        "points_given_an_output_dales_has_no_class_for": int(fine.m[:, -1].sum()),
        "labelled_points": total,
        "confidence": {
            "surrey_correct": confidence(cr[0], cr[1]),
            "surrey_wrong": confidence(cw[0], cw[1]),
        },
        "raised_not_ground_called_ground_rate_per_tile": raised_rates,
        "raised_rate_range": [
            min(r["rate"] for r in raised_rates if r["rate"] is not None),
            max(r["rate"] for r in raised_rates if r["rate"] is not None),
        ],
        "auroc_called_ground": {
            k: {
                "right": int(sum(h.sum() for h in r)),
                "wrong": int(sum(h.sum() for h in w)),
                "auroc": (lambda a: None if a is None else round(a, 4))(
                    auroc_from_histograms(sum(r), sum(w))
                ),
            }
            for k, (r, w) in pools.items()
        },
    }

    result = {
        "variant": args.variant,
        "variant_note": v["note"],
        "classes": n_classes,
        "votes": manifest.get("votes", 1),
        "predictions_from": {
            "san_francisco": str(v["sf"].relative_to(ROOT)).replace("\\", "/"),
            "surrey": str(v["surrey"].relative_to(ROOT)).replace("\\", "/"),
        },
        "footprint_offsets": offsets,
        "vegetation_reference": "registered: data/reference/registered; unregistered: data/reference",
        "reading_fixed_before_first_run": {
            "margin": MARGIN,
            "lower by margin": "predicts degradation",
            "within margin": "measured nothing",
            "higher by margin": "contrary",
        },
        "training_block_z_max_m": train_max,
        "typical_surrey_building_m": typical_building,
        "extrapolation_across_areas": across,
        "surrey_held_out": surrey,
        "san_francisco_per_area": areas,
        "seconds": round(time.time() - t0, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"surrey accuracy {surrey['accuracy_labelled_voxel_points']}  auroc {surrey['auroc_called_ground']}"
    )
    print(f"extrapolation across areas: {across}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
