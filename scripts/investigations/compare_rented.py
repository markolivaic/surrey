"""The six runs from the rented card, read by the rules written before any of them ran.

  run 1  the published configuration retrained on the rented card. If it falls inside the min to max
         range of the three laptop seeds, the machine is not a difference and the large runs may be
         read as a difference in size. If it falls outside, every large against small comparison
         carries that caveat.
  run 2  the large M0. The upgrade rule: its building lift must beat the range of the three small
         seeds in at least 4 of the 6 areas that have footprints. Read from this run only.
  run 3  the large M1, water against the null it was always judged by.
  run 4  the large M0 at twice the points seen, a sensitivity.
  run 5  the large M0 with the learning rate left unscaled, a sensitivity.
  run 6  the small M1 with the crop density drawn at random, a diagnosis on one seed.

Sensitivities are reported whatever they say and never enter the decision. Nothing here decides
anything: it prints what the rules say.

Run:
    python scripts/investigations/compare_rented.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

ARTIFACTS = ROOT / "data" / "artifacts"
OUT = ARTIFACTS / "compare_rented.json"
SMALL_SEEDS = ("m0_seedA", "m0_seedB", "m0_seedC")
M1_SEEDS = ("m1_seedA", "m1_seedB", "m1_seedC")
FOOTPRINT_AREAS = ("embarcadero", "financial", "mission", "soma", "steep", "sunset")
SF_MEASURES = (
    "building_lift_global_shift",
    "ground_recall",
    "vegetation_headline",
    "not_vegetation_core_called_vegetation",
    "raised_unassigned_called_ground_rate",
)
UPGRADE_RULE = (
    "at least 4 of the 6 areas with footprints must have the large model's building lift "
    "above the max of the three small seeds; written before the runs"
)


def scores(variant: str) -> dict | None:
    path = ARTIFACTS / f"sf_scores_{variant}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def per_area(blob: dict, aoi: str) -> dict:
    return next((r for r in blob["san_francisco_per_area"] if r["aoi"] == aoi), {})


def measure(blob: dict, aoi: str, name: str):
    """One number per area, dug out of the shape the scorer wrote it in."""
    row = per_area(blob, aoi)
    if name == "building_lift_global_shift":
        return (
            row.get("building_vs_footprints_over_2m", {})
            .get("global_shift", {})
            .get("precision_lift_over_null")
        )
    if name == "vegetation_headline":
        return (row.get("vegetation_headline") or {}).get("vegetation_core_called_vegetation")
    if name == "not_vegetation_core_called_vegetation":
        veg = row.get("vegetation") or {}
        for which in ("registered", "unregistered"):
            if which in veg and veg[which].get(name) is not None:
                return veg[which][name]
        return None
    if name == "raised_unassigned_called_ground_rate":
        return (row.get(name) or {}).get("rate")
    if name == "ground_recall":
        return (row.get("ground_band") or {}).get("with_water", {}).get("recall_clean")
    return row.get(name)


def in_domain(variant: str):
    path = ARTIFACTS / f"train_{variant}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["final"]["accuracy"]


def band(values: list[float | None]) -> dict:
    kept = [v for v in values if v is not None]
    return {"min": min(kept), "max": max(kept), "n": len(kept)} if kept else {"n": 0}


def where(value: float | None, b: dict) -> str:
    if value is None or not b.get("n"):
        return "no figure"
    if value > b["max"]:
        return "above"
    if value < b["min"]:
        return "below"
    return "inside"


def main() -> int:
    result = {
        "rules_written_before_the_runs": {
            "run_1": (
                "inside the three laptop seeds means the machine is not a difference; outside means "
                "every large against small comparison carries that caveat"
            ),
            "upgrade": UPGRADE_RULE,
            "sensitivities": "runs 4 and 5 are reported whatever they say and do not enter the decision",
            "run_6": "a diagnosis on one seed; anything claimed in the README needs three",
        },
        "cards": {},
        "run_1_machine_check": {},
        "run_2_upgrade": {},
        "run_3_water": {},
        "sensitivities": {},
        "run_6_density": {},
    }

    for variant in (
        "m0_rented_small_seedA",
        "m0_big_seedA",
        "m1_big_seedA",
        "m0_big_2x_seedA",
        "m0_big_lr2_seedA",
        "m1_densityaug_seedA",
    ):
        path = ARTIFACTS / f"train_{variant}.json"
        if path.exists():
            blob = json.loads(path.read_text(encoding="utf-8"))
            result["cards"][variant] = {
                "device_name": blob["device_name"],
                "device_total_gb": blob["device_total_gb"],
                "width": blob["config"]["width"],
                "batch": blob["config"]["batch"],
                "steps": blob["steps_trained"],
                "lr": blob["config"]["lr"],
                "parameters": blob["parameters"],
                "points_seen": blob["points_seen"],
                "crop_distinct_range": blob.get("crop_distinct_range"),
                "wall_seconds": blob["wall_seconds"],
            }

    # run 1: is the machine a difference
    small = [s for s in (scores(v) for v in SMALL_SEEDS) if s]
    rented_small = scores("m0_rented_small_seedA")
    check = {
        "in_domain": {
            "laptop_band": band([in_domain(v) for v in SMALL_SEEDS]),
            "rented": in_domain("m0_rented_small_seedA"),
        }
    }
    check["in_domain"]["verdict"] = where(
        check["in_domain"]["rented"], check["in_domain"]["laptop_band"]
    )
    check["san_francisco"] = {}
    if rented_small and small:
        for aoi in FOOTPRINT_AREAS:
            for name in SF_MEASURES:
                b = band([measure(s, aoi, name) for s in small])
                v = measure(rented_small, aoi, name)
                check["san_francisco"][f"{aoi}.{name}"] = {
                    "laptop_band": b,
                    "rented": v,
                    "verdict": where(v, b),
                }
        outside = [
            k for k, r in check["san_francisco"].items() if r["verdict"] in ("above", "below")
        ]
        check["outside_count"] = len(outside)
        check["measures_compared"] = len(check["san_francisco"])
        check["outside"] = outside
        check["reading"] = (
            "the machine is not a difference"
            if not outside and check["in_domain"]["verdict"] == "inside"
            else "the machine moves some measures; every large against small comparison "
            "carries that caveat"
        )
    result["run_1_machine_check"] = check

    # run 2: the upgrade rule, read from the large M0 only
    big = scores("m0_big_seedA")
    if big and small:
        rows = {}
        for aoi in FOOTPRINT_AREAS:
            b = band([measure(s, aoi, "building_lift_global_shift") for s in small])
            v = measure(big, aoi, "building_lift_global_shift")
            rows[aoi] = {"small_band": b, "big": v, "verdict": where(v, b)}
        above = [a for a, r in rows.items() if r["verdict"] == "above"]
        result["run_2_upgrade"] = {
            "per_area": rows,
            "areas_above": above,
            "areas_needed": 4,
            "rule": UPGRADE_RULE,
            "rule_met": len(above) >= 4,
            "in_domain": {
                "laptop_band": band([in_domain(v) for v in SMALL_SEEDS]),
                "big": in_domain("m0_big_seedA"),
            },
            "other_measures": {
                f"{aoi}.{name}": {
                    "small_band": band([measure(s, aoi, name) for s in small]),
                    "big": measure(big, aoi, name),
                }
                for aoi in FOOTPRINT_AREAS
                for name in SF_MEASURES
                if name != "building_lift_global_shift"
            },
        }

    # The comparison that carries no machine in it: the large model against the small one, both
    # trained on the same rented card, same data, same seed. Run 1 showed the machine moves measures,
    # so this is the only size comparison that is clean.
    if big and rented_small:
        rows = {}
        for aoi in FOOTPRINT_AREAS:
            for name in SF_MEASURES:
                small_v = measure(rented_small, aoi, name)
                big_v = measure(big, aoi, name)
                rows[f"{aoi}.{name}"] = {
                    "small_same_machine": small_v,
                    "big": big_v,
                    "difference": None if None in (small_v, big_v) else round(big_v - small_v, 4),
                }
        result["size_on_the_same_machine"] = {
            "note": "both on the rented card, so the machine cancels; the difference is the size",
            "in_domain": {
                "small": in_domain("m0_rented_small_seedA"),
                "big": in_domain("m0_big_seedA"),
            },
            "per_measure": rows,
            "building_lift_better_in": [
                k
                for k, r in rows.items()
                if k.endswith("building_lift_global_shift") and (r["difference"] or 0) > 0
            ],
        }

    # run 3: water on the large M1, against the null and against the three small M1 seeds
    big_m1 = scores("m1_big_seedA")
    m1s = [s for s in (scores(v) for v in M1_SEEDS) if s]
    if big_m1:
        rows = {}
        for aoi in ("embarcadero", "baybridge"):
            w = per_area(big_m1, aoi).get("water_vs_null", {})
            rows[aoi] = {
                "model": w.get("model"),
                "lift_over_null": w.get("lift_over_null"),
                "small_seed_lift_band": band(
                    [per_area(s, aoi).get("water_vs_null", {}).get("lift_over_null") for s in m1s]
                ),
                "small_seed_recall_band": band(
                    [
                        per_area(s, aoi).get("water_vs_null", {}).get("model", {}).get("recall")
                        for s in m1s
                    ]
                ),
            }
            rows[aoi]["lift_verdict"] = where(
                rows[aoi]["lift_over_null"], rows[aoi]["small_seed_lift_band"]
            )
            rows[aoi]["positive_lift"] = (rows[aoi]["lift_over_null"] or 0) > 0
        result["run_3_water"] = {
            "per_area": rows,
            "any_positive": any(r["positive_lift"] for r in rows.values()),
            "note": "a positive lift is a new story and needs three seeds before anything "
            "is claimed",
        }

    # runs 4 and 5: sensitivities
    for variant, why in (
        ("m0_big_2x_seedA", "twice the points seen"),
        ("m0_big_lr2_seedA", "the learning rate left unscaled"),
    ):
        blob = scores(variant)
        if not blob or not big:
            continue
        rows = {}
        for aoi in FOOTPRINT_AREAS:
            rows[aoi] = {
                "primary_big": measure(big, aoi, "building_lift_global_shift"),
                "this_run": measure(blob, aoi, "building_lift_global_shift"),
                "small_band": band([measure(s, aoi, "building_lift_global_shift") for s in small]),
            }
        result["sensitivities"][variant] = {
            "why": why,
            "per_area": rows,
            "in_domain": in_domain(variant),
            "in_domain_primary_big": in_domain("m0_big_seedA"),
            "enters_the_decision": False,
        }

    # run 6: the density diagnosis
    aug = scores("m1_densityaug_seedA")
    if aug and m1s:
        rows = {}
        for aoi in ("embarcadero", "baybridge"):
            w = per_area(aug, aoi).get("water_vs_null", {})
            recall_band = band(
                [
                    per_area(s, aoi).get("water_vs_null", {}).get("model", {}).get("recall")
                    for s in m1s
                ]
            )
            lift_band = band(
                [per_area(s, aoi).get("water_vs_null", {}).get("lift_over_null") for s in m1s]
            )
            rows[aoi] = {
                "recall": w.get("model", {}).get("recall"),
                "three_seed_recall_band": recall_band,
                "recall_verdict": where(w.get("model", {}).get("recall"), recall_band),
                "lift_over_null": w.get("lift_over_null"),
                "three_seed_lift_band": lift_band,
                "lift_verdict": where(w.get("lift_over_null"), lift_band),
            }
        result["run_6_density"] = {
            "per_area": rows,
            "in_domain": in_domain("m1_densityaug_seedA"),
            "in_domain_three_seeds": band([in_domain(v) for v in M1_SEEDS]),
            "reading": "one seed; above the three seed band is a proposal for two more "
            "seeds, never a claim",
        }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("run_1_machine_check", "run_2_upgrade", "run_3_water", "run_6_density")
            },
            indent=1,
        )[:4000]
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
