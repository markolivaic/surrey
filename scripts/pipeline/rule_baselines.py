"""What simple rules score on the same San Francisco references, with no model.

Every model figure in San Francisco needs a rule beside it that a reader could write in one line, or
a good-looking number cannot be told apart from what the reference gives away for free. Three rules,
each scored by the same functions and on the same populations as the model (score_variant.py):

    ground      every point less than 0.5 m above the label-free floor is ground; scored like the
                model's ground calls, as recall of the delivery's ground and a precision band
    vegetation  every point 2-25 m above the label-free floor is vegetation; on the NAIP cores, which
                are themselves 2-25 m up, it scores 1.0 on both sides by construction, and is shown
                because that is what a figure on these cores has to beat
    building    every point more than 2 m above the delivery's own ground surface whose NAIP NDVI is
                below the threshold the vegetation reference uses; scored against the footprints
                with the same placement and height gate as the model, so its lift sits beside the
                model's

No prediction is read here.

Writes data/artifacts/rule_baselines.json.

Run:
    python scripts/pipeline/rule_baselines.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))

import score_variant as sv  # noqa: E402
from surrey.cache import CachedTile  # noqa: E402
from surrey.footprints import (  # noqa: E402
    clip_to_box,
    coverage_with_null,
    inside_mask,
    load_footprints,
)
from surrey.reference import (  # noqa: E402
    GROUND_CODES,
    SURFACE_CELL_M,
    SURFACE_REACH_M,
    score_ground,
    vendor_ground_surface,
)
from surrey.registration import footprint_offsets  # noqa: E402

ARTIFACTS = ROOT / "data" / "artifacts"
OUT = ARTIFACTS / "rule_baselines.json"
GROUND_BELOW_M = 0.5
VEGETATION_BAND_M = (2.0, 25.0)


def main() -> int:
    import shapely.affinity

    t0 = time.time()
    crs = json.loads((ARTIFACTS / "crs_check.json").read_text(encoding="utf-8"))
    offsets = footprint_offsets(crs)
    dx, dy = offsets["global"]["east_m"], offsets["global"]["north_m"]
    threshold = float(
        json.loads((ARTIFACTS / "naip_reference.json").read_text(encoding="utf-8"))[
            "ndvi_threshold"
        ]
    )
    mask_choice = {
        r["aoi"]: r.get("reference")
        for r in json.loads((ARTIFACTS / "naip_mask_choice.json").read_text(encoding="utf-8"))[
            "areas"
        ]
    }
    polys = None
    areas = {}
    for p in sorted(sv.TARGET_PREP.iterdir()):
        if not (p.is_dir() and (p / "meta.json").exists()):
            continue
        name = p.name
        tile = CachedTile(p, "sf")
        if polys is None:
            polys = load_footprints(sv.FOOTPRINTS, tile.epsg)
        asprs = np.asarray(tile.asprs[:], np.int16)
        clean = ~np.isin(asprs, sv.ASPRS_NOISE)
        hag = np.asarray(tile.hag_cm[:], np.float32) / 100.0
        x = np.asarray(tile.xy_cm[:, 0], np.float64) / 100.0
        y = np.asarray(tile.xy_cm[:, 1], np.float64) / 100.0
        z = np.asarray(tile.z_cm[:], np.float64) / 100.0
        ux, uy = tile.utm_xy(np.arange(tile.n))
        row: dict = {"points": tile.n}

        # Ground: the same band as the model's ground calls.
        ground_pred = hag < GROUND_BELOW_M
        if np.isin(asprs, GROUND_CODES).any():
            g = score_ground(asprs, x, y, z, ground_pred)
            row["ground_below_0_5m"] = {
                k: g.get(k)
                for k in (
                    "recall_clean",
                    "precision_floor",
                    "precision_ceiling",
                    "predicted_points_unjudgeable_fraction",
                )
            }
            surf, key = vendor_ground_surface(
                x, y, z, np.isin(asprs, GROUND_CODES) & clean, SURFACE_CELL_M, SURFACE_REACH_M
            )
            h = z - surf[key]
        else:
            row["ground_below_0_5m"] = None
            h = np.full(tile.n, np.nan)

        # Vegetation: the band the cores are drawn from, read the way the headline is read.
        lo, hi = VEGETATION_BAND_M
        veg_pred = (hag > lo) & (hag <= hi)
        per_mask = {}
        for label, path in (
            ("registered", sv.REFERENCE / "registered" / f"naip_{name}.npz"),
            ("unregistered", sv.REFERENCE / f"naip_{name}.npz"),
        ):
            ref = np.load(path)
            per_mask[label] = {
                "vegetation_core_called_vegetation": sv.rate(veg_pred, ref["vegetation"]),
                "not_vegetation_core_called_vegetation": sv.rate(veg_pred, ref["not_vegetation"]),
            }
        choice = mask_choice.get(name)
        found = {k: v["vegetation_core_called_vegetation"] for k, v in per_mask.items()}
        false = {k: v["not_vegetation_core_called_vegetation"] for k, v in per_mask.items()}
        if choice == "lower_of_two" and None not in found.values():
            known = [v for v in false.values() if v is not None]
            head = {
                "vegetation_core_called_vegetation": min(found.values()),
                "not_vegetation_core_called_vegetation": max(known) if known else None,
            }
        elif choice in per_mask:
            head = per_mask[choice]
        else:
            head = {
                "vegetation_core_called_vegetation": None,
                "not_vegetation_core_called_vegetation": None,
            }
        row["vegetation_2_25m"] = {"headline": head, "per_mask": per_mask}

        # Buildings: raised above the delivery's ground and not green, scored like the model.
        ndvi = np.asarray(np.load(sv.REFERENCE / f"naip_{name}.npz")["ndvi"], np.float32)
        raised = np.nan_to_num(h, nan=-np.inf) > sv.HAG_GATE_M
        rule = raised & (np.nan_to_num(ndvi, nan=np.inf) < threshold)
        not_building_ref = np.isin(asprs, (sv.ASPRS_WATER, sv.ASPRS_BRIDGE))
        tall = clean & (hag > sv.HAG_GATE_M) & ~not_building_ref
        clipped, _ = clip_to_box(polys, ux, uy)
        moved = [shapely.affinity.translate(q, xoff=dx, yoff=dy) for q in clipped]
        inside = inside_mask(moved, ux, uy)
        row["building_rule"] = {
            **sv.building(inside, rule, tall),
            "coverage": coverage_with_null(moved, ux, uy, rule, tall),
            "gate_points_the_surface_reaches": round(float(np.isfinite(h[tall]).mean()), 4)
            if tall.any()
            else None,
        }
        areas[name] = row
        b = row["building_rule"]
        print(
            f"  {name:12s} ground recall "
            f"{(row['ground_below_0_5m'] or {}).get('recall_clean')}  "
            f"veg {head['vegetation_core_called_vegetation']}/{head['not_vegetation_core_called_vegetation']}  "
            f"building rule lift {b.get('precision_lift_over_null')} precision {b.get('precision')}",
            flush=True,
        )

    doc = {
        "what": "simple rules scored on the San Francisco references with the model's own scoring",
        "rules": {
            "ground": f"every point less than {GROUND_BELOW_M} m above the label-free floor is ground",
            "vegetation": "every point 2-25 m above the label-free floor is vegetation",
            "building": "every point more than 2 m above the delivery's own ground surface with NAIP "
            f"NDVI below {threshold} is building",
        },
        "ndvi_threshold": threshold,
        "footprint_offset_east_north_m": [round(dx, 3), round(dy, 3)],
        "model_involved": False,
        "areas": areas,
        "seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
