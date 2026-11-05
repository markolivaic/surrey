"""M1 against M0prime, and M0prime against M0, by the criteria in
backend/surrey/m1_criteria.py, written before either new recipe was trained.

    python scripts/pipeline/compare_recipes.py --new m1 --old m0prime
    python scripts/pipeline/compare_recipes.py --new m0prime --old m0

Writes data/artifacts/compare_<new>_vs_<old>.json:

    side_effects   every San Francisco figure with a reference, per area, and the Surrey
                   held-out accuracy and IoU per class: drop when every new seed is worse
                   than every old seed, gain by the mirror rule, otherwise neither
    water          only for --new m1: per area and seed, M1's water call and the physical
                   null's, lift over the null, and whether it is positive on all three
                   seeds; Embarcadero and Bay Bridge only
    bridge_deck    only for --new m1: recall and precision per seed, descriptive, no null

The density of the Swiss data is named as a confound for M0prime against M0 only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "archive" / "investigations")]

from aggregate_seeds import SF_AOIS, sf_figure  # noqa: E402
from surrey.m1_criteria import (  # noqa: E402
    BRIDGE_NOTE,
    DENSITY_CONFOUND,
    SF_SIDE_EFFECT_METRICS,
    WATER_AREAS,
    WATER_SCOPE_NOTE,
    water_success,
)
from surrey.reference import REFERENCE_NOT_VALID  # noqa: E402

# The two neighbour searches: the same data and recipe, only the search differs
# (docs/criteria/neighbour_search.md). The pairs m1_criteria.py defines keep their density notes.
CONFOUND = {
    **DENSITY_CONFOUND,
    ("m0", "m0fp16"): "none: the same data, weights and steps; only the neighbour search differs",
}
from surrey.seeds import (  # noqa: E402
    SEED_LETTERS,
    chance_of_a_move,
    drop,
    expected_by_chance,
    gain,
    spread,
    variant,
)

ARTIFACTS = ROOT / "data" / "artifacts"


def load(recipe: str, kind: str) -> dict[str, dict]:
    out = {}
    for s in SEED_LETTERS:
        p = ARTIFACTS / f"{kind}_{variant(recipe, s)}.json"
        if p.exists():
            out[s] = json.loads(p.read_text(encoding="utf-8"))
    return out


def compare(new: str, old: str, new_scores: dict, old_scores: dict) -> dict:
    def rows(scores):
        return {s: {r["aoi"]: r for r in sc["san_francisco_per_area"]} for s, sc in scores.items()}

    nr, orr = rows(new_scores), rows(old_scores)
    side: dict = {}
    for aoi in SF_AOIS:
        for name, higher in SF_SIDE_EFFECT_METRICS.items():
            a = {s: sf_figure(nr[s][aoi], name) for s in nr if aoi in nr[s]}
            b = {s: sf_figure(orr[s][aoi], name) for s in orr if aoi in orr[s]}
            if all(v is None for v in a.values()) or all(v is None for v in b.values()):
                continue
            if (aoi, name) in REFERENCE_NOT_VALID:
                side.setdefault(aoi, {})[name] = {
                    "reference_not_valid": REFERENCE_NOT_VALID[(aoi, name)],
                    new: spread(a),
                    old: spread(b),
                    "drop": None,
                    "gain": None,
                }
                continue
            side.setdefault(aoi, {})[name] = {
                "higher_is_better": higher,
                new: spread(a),
                old: spread(b),
                "drop": drop(a.values(), b.values(), higher),
                "gain": gain(a.values(), b.values(), higher),
            }
    # Held-out Surrey from the voted predictions, the same evaluation the README's accuracy, mean IoU
    # and per-class table come from; the training artefact's figures are a single pass.
    surrey: dict = {}
    held_new = {s: sc["surrey_held_out"] for s, sc in new_scores.items()}
    held_old = {s: sc["surrey_held_out"] for s, sc in old_scores.items()}
    if held_new and held_old:
        names = next(iter(held_old.values()))["iou_per_class"]
        for n in ["accuracy", *names]:

            def val(h, n=n):
                return (
                    h["accuracy_labelled_voxel_points"]
                    if n == "accuracy"
                    else h["iou_per_class"].get(n)
                )

            a = {s: val(h) for s, h in held_new.items()}
            b = {s: val(h) for s, h in held_old.items()}
            surrey[n] = {
                new: spread(a),
                old: spread(b),
                "drop": drop(a.values(), b.values(), True),
                "gain": gain(a.values(), b.values(), True),
            }
    out = {
        "new": new,
        "old": old,
        "seeds": {new: sorted(new_scores), old: sorted(old_scores)},
        "rule": "backend/surrey/m1_criteria.py and surrey.seeds.drop, written before training",
        "density_confound": CONFOUND.get((new, old)),
        "side_effects_san_francisco": side,
        "side_effects_surrey_held_out": surrey,
        "drops": sorted(
            f"{aoi}.{m}" for aoi, ms in side.items() for m, r in ms.items() if r["drop"]
        )
        + sorted(f"surrey.{n}" for n, r in surrey.items() if r["drop"]),
        "gains": sorted(
            f"{aoi}.{m}" for aoi, ms in side.items() for m, r in ms.items() if r["gain"]
        )
        + sorted(f"surrey.{n}" for n, r in surrey.items() if r["gain"]),
    }
    # How many of those moves chance alone would give (surrey.seeds.chance_of_a_move).
    read = sum(1 for ms in side.values() for r in ms.values() if r["drop"] is not None)
    read += sum(1 for r in surrey.values() if r["drop"] is not None)
    n_new, n_old = len(new_scores), len(old_scores)
    out["chance"] = {
        "comparisons": read,
        "two_sided_chance_of_a_move": round(chance_of_a_move(n_new, n_old), 4),
        "moves_expected_by_chance": expected_by_chance(read, n_new, n_old),
        "moves_seen": len(out["drops"]) + len(out["gains"]),
    }

    if new == "m1":
        water, lifts, bridge = {}, {}, {}
        for aoi in WATER_AREAS:
            water[aoi], lifts[aoi], bridge[aoi] = {}, {}, {}
            for s in nr:
                w = nr[s][aoi].get("water_vs_null")
                if w is None:
                    raise SystemExit(
                        f"{variant(new, s)} {aoi} has no water_vs_null block; score it with the null"
                    )
                water[aoi][s] = w
                lifts[aoi][s] = w["lift_over_null"]
                bridge[aoi][s] = nr[s][aoi].get("bridge_deck_call")
        band = {
            aoi: {
                s: {
                    edge: (water[aoi][s].get("band") or {}).get(edge, {}).get("lift_over_null")
                    for edge in ("lower", "upper", "upper_polygon")
                }
                for s in water[aoi]
            }
            for aoi in water
        }
        upper_lifts = {
            aoi: {s: v["upper"] for s, v in seeds.items()} for aoi, seeds in band.items()
        }
        polygon_lifts = {
            aoi: {s: v["upper_polygon"] for s, v in seeds.items()} for aoi, seeds in band.items()
        }
        out["water"] = {
            "criterion": water_success(lifts),
            "per_area_per_seed": water,
            "scope": WATER_SCOPE_NOTE,
            "band_sensitivity": {
                "lift_over_null_per_edge": band,
                "upper_edge_read_like_the_criterion": water_success(upper_lifts),
                "polygon_edge_read_like_the_criterion": water_success(polygon_lifts),
                "note": "sensitivity only; the criterion reads the lower edge. The upper "
                "edge uses the null's own height rule, so its lift over the null "
                "cannot be positive and is not a comparison; read M1's recall, "
                "precision and IoU there instead",
            },
        }
        out["bridge_deck"] = {
            "per_area_per_seed": bridge,
            "note": BRIDGE_NOTE,
            "scope": WATER_SCOPE_NOTE,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True, choices=("m0prime", "m1", "m0"))
    ap.add_argument("--old", required=True, choices=("m0", "m0prime", "m0fp16"))
    args = ap.parse_args()
    if (args.new, args.old) not in CONFOUND:
        raise SystemExit(
            "only m1 against m0prime, m0prime against m0 and m0 against m0fp16 are defined"
        )
    ns, os_ = load(args.new, "sf_scores"), load(args.old, "sf_scores")
    if not ns or not os_:
        raise SystemExit("both recipes need scored seeds")
    out = compare(args.new, args.old, ns, os_)
    path = ARTIFACTS / f"compare_{args.new}_vs_{args.old}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"drops {out['drops']}\ngains {out['gains']}")
    if "water" in out:
        print(json.dumps(out["water"]["criterion"], indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
