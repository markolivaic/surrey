"""Every figure and every claim of one recipe, read per seed and then across seeds.

The rules are in backend/surrey/seeds.py and were written before M0 seed C trained.
This applies them to whatever seeds of the recipe have been scored, and writes
data/artifacts/seeds_<recipe>.json. It never picks a seed: seed A is the published
model whatever the others show.

Claims, each read per seed with the rule it was first written with:

    extrapolation   Financial's building lift against the median of the other built
                    areas (score_variant, primary formulation written before E0)
    relief          Steep, delivery ground called building on high-relief blocks against
                    the rest (the prediction in input_range_check.py)
    easy errors     per area, raised points called ground: inside that seed's own Surrey
                    per-tile range is a weakness the model has at home, above it is
                    transfer, below it is written as below
    hard errors     home against water (Dio 2.1 of the plan): among ground calls, AUROC
                    of confidence for right ground against errors within 0.5 m of the
                    ground surface, on the Surrey held-out tiles, beside the same
                    construction for water on Embarcadero. Read with the project's
                    existing 0.05 margin: water lower by at least 0.05 is "water errors
                    are harder to see than errors at home"; within it, no difference
                    measured. Written after seeds A and B were scored and before seed C;
                    the margin is the one every other reading uses, not a new number.

A claim stands when it holds on every scored seed, otherwise its phrase is "on k of n
seeds". Every figure carries median, min and max.

Run:
    python scripts/pipeline/aggregate_seeds.py --recipe m0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.m1_criteria import SF_SIDE_EFFECT_METRICS  # noqa: E402
from surrey.reference import REFERENCE_NOT_VALID  # noqa: E402
from surrey.seeds import (  # noqa: E402
    EDGE_READINGS,
    PUBLISHED_SEED,
    SEED_LETTERS,
    holds,
    spread,
    variant,
)

ARTIFACTS = ROOT / "data" / "artifacts"
MARGIN = 0.05
SF_AOIS = ["baybridge", "embarcadero", "financial", "ggpark", "mission", "soma", "steep", "sunset"]


def sf_figure(area: dict, name: str):
    """The per-area figures the registry reads, from one score file row."""
    try:
        return {
            "building_lift_global_shift": lambda: area["building_vs_footprints_over_2m"][
                "global_shift"
            ]["precision_lift_over_null"],
            "ground_precision_floor": lambda: area["ground_band"]["with_water"]["precision_floor"],
            "ground_recall": lambda: area["ground_band"]["with_water"]["recall_clean"],
            "vegetation_headline": lambda: area["vegetation_headline"][
                "vegetation_core_called_vegetation"
            ],
            "not_vegetation_core_called_vegetation": lambda: area["vegetation"]["unregistered"][
                "not_vegetation_core_called_vegetation"
            ],
            "raised_unassigned_called_ground_rate": lambda: area[
                "raised_unassigned_called_ground_rate"
            ]["rate"],
            "easy_error_auroc": lambda: area["auroc"]["easy_unassigned_over_3m"]["auroc"],
            "hard_water_auroc": lambda: area["auroc"]["hard_water_within_0_5m"]["auroc"],
            "water_called_ground": lambda: area["water_called_ground"],
            "bridge_deck_called_building": lambda: area["bridge_deck_called_building"],
        }[name]()
    except (KeyError, TypeError):
        return None


def easy_error_reading(rate: float | None, home: list[float]) -> str | None:
    if rate is None:
        return None
    if rate > home[1]:
        return "transfer"
    if rate < home[0]:
        return "below the home range"
    return "home weakness"


def hard_reading(water: float | None, home: float | None) -> str | None:
    if water is None or home is None:
        return None
    d = water - home
    if d <= -MARGIN:
        return "water errors harder to see than errors at home"
    if d >= MARGIN:
        return "water errors easier to see than errors at home"
    return "no difference measured"


def aggregate(recipe: str, scores: dict[str, dict], training: dict[str, dict]) -> dict:
    seeds = [s for s in SEED_LETTERS if s in scores]
    rows = {s: {r["aoi"]: r for r in scores[s]["san_francisco_per_area"]} for s in seeds}

    figures: dict = {}
    for aoi in SF_AOIS:
        per = {}
        for name in SF_SIDE_EFFECT_METRICS:
            vals = {s: sf_figure(rows[s][aoi], name) for s in seeds if aoi in rows[s]}
            if any(v is not None for v in vals.values()):
                per[name] = spread(vals)
                if (aoi, name) in REFERENCE_NOT_VALID:
                    per[name]["reference_not_valid"] = REFERENCE_NOT_VALID[(aoi, name)]
        figures[aoi] = per

    surrey = {
        "accuracy_from_prediction_cache": spread(
            {s: scores[s]["surrey_held_out"]["accuracy_labelled_voxel_points"] for s in seeds}
        )
    }
    if training:
        tseeds = [s for s in seeds if s in training]
        surrey["accuracy_training_artefact"] = spread(
            {s: training[s]["final"]["accuracy"] for s in tseeds}
        )
        names = training[tseeds[0]]["final"]["iou_per_class"] if tseeds else {}
        surrey["iou_per_class"] = {
            n: spread({s: training[s]["final"]["iou_per_class"].get(n) for s in tseeds})
            for n in names
        }

    claims: dict = {}
    ex = {s: scores[s]["extrapolation_across_areas"] for s in seeds}
    claims["extrapolation"] = holds(
        {s: ex[s]["reading"] for s in seeds}, "predicts degradation"
    ) | {
        "financial_lift": spread({s: ex[s]["financial"] for s in seeds}),
        "other_built_median": spread({s: ex[s]["other_built_median"] for s in seeds}),
        "status": "primary formulation, written before E0",
    }

    rel = {s: rows[s]["steep"]["confounds"]["relief_ground_recall"]["primary"] for s in seeds}
    claims["relief"] = holds({s: rel[s]["reading"] for s in seeds}, "predicts degradation") | {
        "named_minus_other": spread({s: rel[s]["named_minus_other"] for s in seeds}),
        "metric": "delivery ground called building, high-relief blocks against the rest",
        "extension": "the rule was written for one model; read per seed with the same 0.05 margin",
        "at_the_edge": {
            s: EDGE_READINGS[(recipe, s, "relief")]
            for s in seeds
            if (recipe, s, "relief") in EDGE_READINGS
        },
    }

    easy: dict = {}
    for aoi in SF_AOIS:
        readings = {}
        for s in seeds:
            if aoi not in rows[s]:
                continue
            rate = sf_figure(rows[s][aoi], "raised_unassigned_called_ground_rate")
            readings[s] = easy_error_reading(
                rate, scores[s]["surrey_held_out"]["raised_rate_range"]
            )
        if not any(readings.values()):
            continue
        easy[aoi] = {
            "transfer": holds(readings, "transfer"),
            "home_weakness": holds(readings, "home weakness"),
            "rate": spread(
                {s: sf_figure(rows[s][aoi], "raised_unassigned_called_ground_rate") for s in seeds}
            ),
            "home_range_per_seed": {
                s: scores[s]["surrey_held_out"]["raised_rate_range"] for s in seeds
            },
        }
    claims["easy_errors"] = easy

    hard = {}
    for s in seeds:
        home = scores[s]["surrey_held_out"]["auroc_called_ground"]["hard"]
        water = rows[s]["embarcadero"]["auroc"]["hard_water_within_0_5m"]
        hard[s] = {
            "home": home,
            "embarcadero_water": water,
            "water_minus_home": None
            if None in (water.get("auroc"), home.get("auroc"))
            else round(water["auroc"] - home["auroc"], 4),
            "reading": hard_reading(water.get("auroc"), home.get("auroc")),
        }
    claims["hard_errors_home_vs_water"] = holds(
        {s: hard[s]["reading"] for s in seeds}, "water errors harder to see than errors at home"
    ) | {
        "per_seed_detail": hard,
        "home_auroc": spread({s: hard[s]["home"].get("auroc") for s in seeds}),
        "water_auroc": spread({s: hard[s]["embarcadero_water"].get("auroc") for s in seeds}),
        "construction": "among ground calls, right ground against errors within 0.5 m of the ground "
        "surface; Surrey: DALES non-ground labels; Embarcadero: delivery water",
        "margin": MARGIN,
    }

    return {
        "recipe": recipe,
        "seeds_scored": seeds,
        "published_seed": PUBLISHED_SEED,
        "published_variant": variant(recipe, PUBLISHED_SEED),
        "rules": "backend/surrey/seeds.py, written before M0 seed C",
        "san_francisco": figures,
        "surrey_held_out": surrey,
        "claims": claims,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", required=True, choices=("m0", "m0prime", "m1"))
    args = ap.parse_args()
    scores, training = {}, {}
    for s in SEED_LETTERS:
        v = variant(args.recipe, s)
        if (ARTIFACTS / f"sf_scores_{v}.json").exists():
            scores[s] = json.loads((ARTIFACTS / f"sf_scores_{v}.json").read_text(encoding="utf-8"))
        if (ARTIFACTS / f"train_{v}.json").exists():
            training[s] = json.loads((ARTIFACTS / f"train_{v}.json").read_text(encoding="utf-8"))
    if PUBLISHED_SEED not in scores:
        raise SystemExit(f"seed {PUBLISHED_SEED} of {args.recipe} is not scored")
    out = aggregate(args.recipe, scores, training)
    path = ARTIFACTS / f"seeds_{args.recipe}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    c = out["claims"]
    print(f"{args.recipe}: seeds {out['seeds_scored']}")
    print(f"  extrapolation {c['extrapolation']['phrase']} ({c['extrapolation']['expected']})")
    print(f"  relief        {c['relief']['phrase']}  gap {c['relief']['named_minus_other']}")
    for aoi, r in c["easy_errors"].items():
        print(
            f"  easy {aoi:12s} transfer {r['transfer']['phrase']}, home {r['home_weakness']['phrase']}"
        )
    h = c["hard_errors_home_vs_water"]
    print(f"  hard home {h['home_auroc']} water {h['water_auroc']} -> {h['phrase']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
