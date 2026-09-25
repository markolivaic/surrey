"""What M1 has to show, and what counts as a side effect. Written before M0prime or M1
was trained.

Primary, water, per seed and per area. Water can only be scored where the delivery
labels it: Embarcadero and Bay Bridge. That restriction is written next to every water
result.
    - recall, precision and IoU of M1's water call against delivery class 9
    - the same for the physical null in surrey.water_null
    - lift over the null, (IoU_M1 - IoU_null) / (1 - IoU_null)
    Success on an area: lift positive on all three M1 seeds.
    Success for M1: success on both areas. Anything else is written as "on k of 3 seeds"
    per area, and a failure is reported as one, with its figures.

Water band, a sensitivity beside the criterion, which it never replaces. Written
2026-09-15, after M0prime seed A was scored and before M1 seed A finished training or was
scored; no M1 water figure existed when it was written. The null called 877,553 Embarcadero
points and 951,810 Bay Bridge points water that the delivery leaves as class 1, unassigned,
at the height of the water surface: probably water the vendor did not label. Against class 9
alone, a model that misses those points gets its precision for free, because the reference
rewards the miss. So water is scored as a band, the way ground is:
    lower edge  the criterion as written: delivery class 9 is water
    upper edge  class 9, plus class 1 below the gauge level at the point's own GPS time plus
                the null's fixed margin (surrey.water_null.MARGIN_M), is water
    Both edges for the null and for M1, per seed, with the lift over the null at each edge.
    The success rule above reads the lower edge only. The upper edge is reported beside it,
    and a result that holds at one edge and not the other is written as such.
    Found on a FIXTURE run before any M1 result, and written down before M1 seed A was
    scored: the upper edge is built with the null's own height rule, so the null agrees with
    it by construction (IoU just under 1 at Embarcadero and exactly 1 at Bay Bridge on that run). The lift over the null
    at the upper edge therefore cannot be positive, and is undefined where the null's IoU is
    1. At the upper edge only M1's own recall, precision and IoU are informative; its lift is
    reported with that flag and is not read as a comparison.

A second upper edge, added 2026-09-15 on review, also before M1 seed A was scored: the
first upper edge shares the null's height rule and so cannot judge the null. This one uses no
height at all:
    upper edge, polygon   class 9, plus class 1 inside an independent water polygon (NOAA's
                          continually updated shoreline or the USGS National Hydrography
                          Dataset, whichever is checked first for licence and date and fits;
                          the choice and its date are recorded by scripts/pipeline/water_polygon.py)
    The lift over the null is readable on this edge, because the reference is no longer the null.
    Expected before any figure, so it is not read afterwards: the null will probably stay close to
    1 here too. A height referenced to the gauge's sea level at the moment of flight recognises
    open water almost perfectly. That is physics, not a fault of the metric, and it is written so.

What the water null is, for every text that mentions it: it is not a do-nothing baseline. It is a
rule that knows each point's absolute height in NAVD88 and the sea level at the moment the aircraft
flew. The model knows neither; it sees height above the floor of a 40 m block and nothing else. The
success criterion does not change.

Bridge deck, per seed, descriptive only. Recall and precision of M1's bridge deck call
against delivery class 17, on the same two areas. There is no simple null for a bridge
deck and none is invented. The Swiss decks M1 learns from are road bridges, overpasses
and footbridges; the Bay Bridge is a suspension bridge with a double deck, and that
sits beside the result.

Side effects, every metric with a reference on every area, and the Surrey held-out
figures. A drop is reported when every seed of the newer recipe is worse than every
seed of the older one (surrey.seeds.drop); the same rule reads M0prime against M0,
which is the only comparison where the density of the Swiss data is named as a
confound. M1 against M0prime has the same data and density, so it is not.

Three seeds run whatever the first seed shows. The rule that a second M1 seed runs only
for a small effect is withdrawn.
"""

from __future__ import annotations

WATER_AREAS = ("embarcadero", "baybridge")
WATER_BAND_WRITTEN = (
    "2026-09-15, after m0prime_seedA was scored and before m1_seedA was scored; "
    "a sensitivity beside the unchanged criterion"
)
UPPER_EDGE_EXTRA_CLASS = 1
WATER_POLYGON_EDGE_WRITTEN = "2026-09-15, before m1_seedA was scored; no height used"
NULL_DESCRIPTION = (
    "The water null is not a do-nothing baseline: it knows each point's absolute height in NAVD88 "
    "and the sea level at the moment of flight. The model knows neither; it sees height above "
    "the floor of a 40 m block."
)
WATER_SCOPE_NOTE = (
    "water and bridge deck scored on Embarcadero and Bay Bridge only, the areas the delivery labels"
)
BRIDGE_NOTE = (
    "no null for bridge deck; the Swiss decks are road bridges, overpasses and footbridges, "
    "the Bay Bridge is a double deck suspension bridge"
)
DENSITY_CONFOUND = {
    ("m0prime", "m0"): "Swiss data added at its own density; density is a confound here",
    ("m1", "m0prime"): None,
}

# Per-area San Francisco figures from scripts/pipeline/score_variant.py, and whether higher is
# better. Areas without the reference are skipped where the figure is absent.
SF_SIDE_EFFECT_METRICS = {
    "building_lift_global_shift": True,
    "ground_precision_floor": True,
    "ground_recall": True,
    "vegetation_headline": True,
    "not_vegetation_core_called_vegetation": False,
    "raised_unassigned_called_ground_rate": False,
    "easy_error_auroc": True,
    "hard_water_auroc": True,
    "water_called_ground": False,
    "bridge_deck_called_building": False,
}
# Surrey held-out, from each training artefact: accuracy and IoU per DALES class.
SURREY_SIDE_EFFECT_HIGHER_IS_BETTER = True


def water_success(lifts: dict[str, dict[str, float | None]]) -> dict:
    """lifts[area][seed] -> per area whether the lift is positive on every seed, and
    overall success only when both areas succeed."""
    per_area = {}
    for area in WATER_AREAS:
        seeds = lifts.get(area, {})
        # A lift is undefined when the null scores IoU 1. That seed ran and is not positive, so it
        # counts among the seeds and is named. Changed after m1_seedA was scored, in the count and
        # the phrase only: success is still three seeds positive, which undefined ones never were.
        k = sum(1 for v in seeds.values() if v is not None and v > 0)
        n = len(seeds)
        undefined = sum(1 for v in seeds.values() if v is None)
        phrase = f"positive on all {n} seeds" if n and k == n else f"positive on {k} of {n} seeds"
        if undefined:
            phrase += f", undefined on {undefined} (the null scores IoU 1)"
        per_area[area] = {
            "lift_per_seed": dict(seeds),
            "positive_on": k,
            "seeds": n,
            "undefined_on": undefined,
            "success": n == 3 and k == 3,
            "phrase": phrase,
        }
    return {
        "per_area": per_area,
        "success": all(r["success"] for r in per_area.values()),
        "scope": WATER_SCOPE_NOTE,
    }
