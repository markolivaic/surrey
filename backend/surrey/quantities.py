"""One registered source for every quantity the project reports.

A number that exists in some artefact is not yet a number that agrees with the
others about the same thing, and at one point they did not. Three Surrey ground fractions were in circulation at
once, 0.4797, 0.5016 and 0.5119, every one of them real and correctly computed, and
together impossible to read. The model accuracy 0.8948 was then paired with 0.5119
as its do-nothing baseline, which was wrong: the accuracy is over voxel-thinned
held-out points and 0.5119 is over raw ones. The right baseline on the same points
is 0.4797.

The cause is that a "quantity" was named by what it measured and not by what it was
measured over. So here every quantity carries its population: which city, which
split, which sampling, which subset. Two registered quantities may share a concept
but never a concept and a population, and a comparison between two quantities
declares whether the populations must match.

tests/integrity/test_quantities.py holds the registry to that, recomputes the
relations that have to hold between quantities, and checks the prose.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "data" / "artifacts"

COARSE = {
    "ground": ["ground"],
    "vegetation": ["vegetation", "tree"],
    "building": ["habitat", "complex", "annex"],
    "vehicle": ["car", "pickup", "van-truck", "heavy-duty"],
    "pole": ["utility-pole", "light-pole", "traffic-pole", "powerline"],
    "fence": ["fence"],
}
FINE = [
    "ground",
    "vegetation",
    "car",
    "powerline",
    "fence",
    "tree",
    "pickup",
    "van-truck",
    "heavy-duty",
    "utility-pole",
    "light-pole",
    "traffic-pole",
    "habitat",
    "complex",
    "annex",
]
SF_AOIS = ["baybridge", "embarcadero", "financial", "ggpark", "mission", "soma", "steep", "sunset"]


@dataclass(frozen=True)
class Population:
    """What a number was measured over. Two numbers over different populations are
    different quantities even when they share a name."""

    city: str  # surrey, san_francisco
    split: str  # test, train, all, or an area name, or aois
    sampling: str  # raw, voxel, voxel_spread40
    subset: str = "all"  # all, labelled, clean_over_2m, called_ground, ...
    taxonomy: str = "none"  # fine15, coarse6, or none when not class based
    # Which model and which input produced the number. "none" for a number that no
    # model touched: a class share, a footprint null, a test count. See VARIANTS.
    variant: str = "none"

    def differs_from(self, other: Population) -> list[str]:
        diffs = [
            f
            for f in ("city", "split", "sampling", "subset", "taxonomy")
            if getattr(self, f) != getattr(other, f)
        ]
        # A model figure set beside a figure no model touched, its null or its
        # baseline, is a like-for-like pair. Two model figures from different
        # variants never are.
        if "none" not in (self.variant, other.variant) and self.variant != other.variant:
            diffs.append("variant")
        return diffs


@dataclass(frozen=True)
class Quantity:
    key: str
    concept: str
    population: Population
    source: str  # artefact file name
    # Called with the loaded artefacts; many getters bind their area or variant as a default argument,
    # which a narrower Callable type cannot express.
    getter: Callable[..., float | int | None] = field(compare=False)
    doc: str = ""


# The return ratio was 0 on every DALES-2 training point and alive on every San
# Francisco point, so every San Francisco figure from the first checkpoint measured a
# model reading an input it had never seen. The variant says which model and which
# input a number came from, and only the published one may reach the README.
VARIANTS = {
    "none": "no model involved",
    "returns_ood": "four channel checkpoint data/checkpoints/dayton.pt, input as delivered; "
    "in domain its fourth channel is 0 as in training, away it is out of range",
    "ret0": "the same checkpoint with the return ratio forced to 0 at inference",
    "m0_seedA": "three channel retrain, seed A, the published model",
    "m0_seedB": "three channel retrain, seed B, for training variance only",
    "m0_seedC": "three channel retrain, seed C, for training variance only",
    "m0prime_seedA": "DALES-2 and Swiss sub-tiles with Swiss water and bridge deck ignored, seed A",
    "m0prime_seedB": "the same control, seed B",
    "m0prime_seedC": "the same control, seed C",
    "m1_seedA": "DALES-2 and Swiss sub-tiles with water and bridge deck as classes, seed A",
    "m1_seedB": "M1, seed B",
    "m1_seedC": "M1, seed C",
    # The runs on a rented RTX 4090, kept apart from the laptop's by name so no figure from one can be
    # read as the other. The first of them exists to answer whether the machine itself is a difference.
    "m0_rented_small_seedA": "the published configuration, retrained on the rented card, so the machine "
    "can be told apart from the size of the model",
    "m0_big_seedA": "width 512, batch 8, the same points seen, learning rate scaled with the batch; the "
    "run the upgrade decision is read from",
    "m1_big_seedA": "the same large recipe on the M1 data",
    "m0_big_2x_seedA": "the large M0 at twice the points seen, a sensitivity that does not enter the "
    "decision",
    "m0_big_lr2_seedA": "the large M0 with the learning rate left unscaled, a sensitivity that does not "
    "enter the decision",
    "m1_densityaug_seedA": "the published small configuration on M1 data, with each crop's density drawn "
    "at random across the gap between the Swiss and San Francisco crop densities; "
    "a diagnosis on one seed",
    "m0_seedA_1vote": "M0 seed A predicted with one pass instead of the chosen vote, a diagnosis of voting",
    "m0_seedA_legacy_iter": "M0 seed A predicted with one pass by the first round's iterator, a diagnosis "
    "of the iterator",
    "m0_seeds": "median, min and max over every scored M0 seed (surrey.seeds)",
    "m0prime_seeds": "median, min and max over every scored M0prime seed",
    "m1_seeds": "median, min and max over every scored M1 seed",
}
PUBLISHED_VARIANT = "m0_seedA"
# The spread over the published recipe's seeds may stand beside the published figure.
PUBLISHED_AGGREGATE = "m0_seeds"
RECIPES = ("m0", "m0prime", "m1")
SEEDS = ("A", "B", "C")
# Artefacts that only a model can produce. Every quantity read from one must carry a
# variant other than none.
RENTED_VARIANTS = (
    "m0_rented_small_seedA",
    "m0_big_seedA",
    "m1_big_seedA",
    "m0_big_2x_seedA",
    "m0_big_lr2_seedA",
    "m1_densityaug_seedA",
)
MODEL_ARTIFACTS = (
    {
        "auroc.json",
        "auroc_control.json",
        "confidence.json",
        "transfer_sf.json",
        "vegetation_sf.json",
        "return_channel_check.json",
        "four_class_eval.json",
        "ground_band_no_water.json",
        "sf_scores_returns_ood.json",
        "sf_scores_ret0.json",
        "sf_scores_m0_seedA.json",
        "sf_scores_m0_seedB.json",
    }
    | {f"sf_scores_{r}_seed{x}.json" for r in ("m0", "m0prime", "m1") for x in ("A", "B", "C")}
    | {
        "seeds_m0.json",
        "seeds_m0prime.json",
        "seeds_m1.json",
        "compare_m1_vs_m0prime.json",
        "compare_m0prime_vs_m0.json",
        "confidence_mapping.json",
    }
    | {f"sf_scores_{v}.json" for v in RENTED_VARIANTS}
    | {"sf_scores_m0_seedA_1vote.json", "sf_scores_m0_seedA_legacy_iter.json"}
    | {"building_breakdown.json"}
    | {f"train_{v}.json" for v in RENTED_VARIANTS}
)
MODEL_KEY_PREFIXES = ("model.", "auroc.", "conf.", "control.", "vegetation.", "ground.", "ret0.")


def variant_of(q: Quantity) -> str:
    """The variant a registered quantity was measured on, from its key.

    Old keys predate the variant and name no model, so the rule is written out here
    once rather than typed into two hundred constructors: a model figure is from the
    four channel checkpoint as delivered, except the return channel check's after
    column, which forced the ratio to 0. New quantities set their variant directly.
    """
    if q.population.variant != "none":
        return q.population.variant
    if q.key.startswith("ret0."):
        return "ret0" if (".after." in q.key or q.key.startswith("ret0.verdict")) else "returns_ood"
    if q.key.startswith(MODEL_KEY_PREFIXES):
        return "returns_ood"
    return "none"


def with_variants(qs: list[Quantity]) -> list[Quantity]:
    return [replace(q, population=replace(q.population, variant=variant_of(q))) for q in qs]


def load_artifacts(directory: Path = ARTIFACTS) -> dict:
    return {
        p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))
    }


def _row(rows: list[dict], field_name: str, value: str) -> dict:
    for r in rows:
        if r.get(field_name) == value:
            return r
    raise KeyError(f"no row with {field_name}={value!r}")


def _support(a: dict) -> dict:
    return a["train_dales.json"]["final"]["support_per_class"]


def _coarse_confusion_iou(a: dict, group: str) -> float | None:
    m = np.asarray(a["train_dales.json"]["final"]["confusion"], dtype=np.int64)
    groups = list(COARSE)
    index = {n: groups.index(g) for g, ns in COARSE.items() for n in ns}
    c = np.zeros((len(groups), len(groups)), dtype=np.int64)
    for i, ni in enumerate(FINE):
        for j, nj in enumerate(FINE):
            c[index[ni], index[nj]] += m[i, j]
    k = groups.index(group)
    tp = c[k, k]
    union = c[k, :].sum() + c[:, k].sum() - tp
    return None if c[k, :].sum() == 0 else round(float(tp / union), 4)


def _spread_of(per_seed: dict) -> dict:
    from .seeds import spread

    return spread(per_seed)


def _dig(d: dict, path: tuple):
    for k in path:
        d = d[k]
    return d


def _footprint_global(a: dict) -> dict:
    from .registration import footprint_offsets

    return footprint_offsets(a["crs_check.json"])["global"]


def build_registry() -> list[Quantity]:
    q: list[Quantity] = []

    # ---------------- Surrey ground share, three populations ----------------
    q.append(
        Quantity(
            "surrey.all.raw.ground_fraction",
            "ground_fraction",
            Population("surrey", "all", "raw", "labelled", "fine15"),
            "dales_stats.json",
            lambda a: a["dales_stats.json"]["classes"]["ground"]["pooled_fraction"],
            "ground share over all 40 tiles before thinning",
        )
    )
    q.append(
        Quantity(
            "surrey.test.raw.ground_fraction",
            "ground_fraction",
            Population("surrey", "test", "raw", "labelled", "fine15"),
            "dales_stats.json",
            lambda a: a["dales_stats.json"]["per_split"]["test"]["fractions"]["ground"],
            "ground share over the 11 held-out tiles before thinning",
        )
    )
    q.append(
        Quantity(
            "surrey.test.voxel.ground_fraction",
            "ground_fraction",
            Population("surrey", "test", "voxel", "labelled", "fine15"),
            "train_dales.json",
            lambda a: round(_support(a)["ground"] / sum(_support(a).values()), 4),
            "ground share over the held-out points the model was scored on",
        )
    )

    # ---------------- in domain accuracy and its baseline ----------------
    q.append(
        Quantity(
            "model.surrey.test.voxel.accuracy",
            "accuracy",
            Population("surrey", "test", "voxel", "labelled", "fine15"),
            "train_dales.json",
            lambda a: a["train_dales.json"]["final"]["accuracy"],
            "full pass over every held-out point, centroid draw unseeded",
        )
    )
    q.append(
        Quantity(
            "surrey.train.voxel.training_points",
            "training_points",
            Population("surrey", "train", "voxel", "labelled"),
            "train_dales.json",
            lambda a: a["train_dales.json"]["train_points"],
            "voxel-thinned points on the 29 training tiles; not the 107,947,429 scored on",
        )
    )
    q.append(
        Quantity(
            "model.surrey.test.voxel.do_nothing_accuracy",
            "do_nothing_accuracy",
            Population("surrey", "test", "voxel", "labelled", "fine15"),
            "train_dales.json",
            lambda a: round(max(_support(a).values()) / sum(_support(a).values()), 4),
            "accuracy of predicting the largest class everywhere, same points as accuracy",
        )
    )
    q.append(
        Quantity(
            "surrey.test.raw.do_nothing_accuracy",
            "do_nothing_accuracy",
            Population("surrey", "test", "raw", "labelled", "fine15"),
            "dales_stats.json",
            lambda a: a["dales_stats.json"]["per_split"]["test"]["predict_largest_class_accuracy"],
            "the same idea over raw points, NOT a baseline for any voxel accuracy",
        )
    )
    q.append(
        Quantity(
            "model.surrey.test.voxel_spread40.accuracy",
            "accuracy",
            Population("surrey", "test", "voxel_spread40", "labelled", "fine15"),
            "confidence.json",
            lambda a: (
                lambda g: round(
                    g["surrey_correct"]["points"]
                    / (g["surrey_correct"]["points"] + g["surrey_wrong"]["points"]),
                    4,
                )
            )({r["group"]: r for r in a["confidence.json"]["pooled"]}),
            "40 spread blocks per held-out tile, a subsample, not a second full pass",
        )
    )
    q.append(
        Quantity(
            "model.surrey.test.voxel.accuracy_seeded",
            "accuracy_seeded",
            Population("surrey", "test", "voxel", "labelled", "fine15"),
            "auroc.json",
            lambda a: a["auroc.json"]["surrey_full_pass"]["accuracy"],
            "the same full pass with the centroid draw seeded, for reproducibility",
        )
    )
    q.append(
        Quantity(
            "surrey.test.voxel.labelled_points",
            "labelled_points",
            Population("surrey", "test", "voxel", "labelled"),
            "train_dales.json",
            lambda a: int(sum(_support(a).values())),
        )
    )

    # ---------------- per class IoU, both taxonomies ----------------
    for name in FINE:
        q.append(
            Quantity(
                f"model.surrey.test.voxel.iou.{name}",
                "iou",
                Population("surrey", "test", "voxel", f"class:{name}", "fine15"),
                "train_dales.json",
                lambda a, n=name: a["train_dales.json"]["final"]["iou_per_class"][n],
            )
        )
    for group in COARSE:
        q.append(
            Quantity(
                f"model.surrey.test.voxel.iou_coarse.{group}",
                "iou",
                Population("surrey", "test", "voxel", f"class:{group}", "coarse6"),
                "train_dales.json",
                lambda a, g=group: _coarse_confusion_iou(a, g),
                "fine confusion regrouped, so a coarse claim is made in coarse terms",
            )
        )
        q.append(
            Quantity(
                f"surrey.test.voxel.coarse_fraction.{group}",
                "class_fraction",
                Population("surrey", "test", "voxel", f"class:{group}", "coarse6"),
                "train_dales.json",
                lambda a, g=group: round(
                    sum(_support(a)[n] for n in COARSE[g]) / sum(_support(a).values()), 4
                ),
            )
        )
        q.append(
            Quantity(
                f"surrey.all.raw.coarse_fraction.{group}",
                "class_fraction",
                Population("surrey", "all", "raw", f"class:{group}", "coarse6"),
                "dales_stats.json",
                lambda a, g=group: round(
                    sum(a["dales_stats.json"]["classes"][n]["pooled_fraction"] for n in COARSE[g]),
                    4,
                ),
            )
        )

    # ---------------- San Francisco, per area, never pooled ----------------
    for aoi in SF_AOIS:
        pop = Population("san_francisco", aoi, "voxel", "clean_over_2m")
        q.append(
            Quantity(
                f"sf.{aoi}.footprint_null_precision",
                "building_null_precision",
                pop,
                "footprint_null.json",
                lambda a, x=aoi: _row(a["footprint_null.json"]["aois"], "aoi", x)[
                    "baseline_everything_above_2m"
                ]["null_precision"],
            )
        )
        q.append(
            Quantity(
                f"model.sf.{aoi}.building_null_precision",
                "building_null_precision_in_transfer",
                pop,
                "transfer_sf.json",
                lambda a, x=aoi: _row(a["transfer_sf.json"]["aois"], "aoi", x)[
                    "building_vs_footprints_over_2m"
                ]["null_precision"],
                "the same null, as recomputed inside the transfer score",
            )
        )
        q.append(
            Quantity(
                f"model.sf.{aoi}.building_precision",
                "building_precision",
                pop,
                "transfer_sf.json",
                lambda a, x=aoi: _row(a["transfer_sf.json"]["aois"], "aoi", x)[
                    "building_vs_footprints_over_2m"
                ]["precision"],
            )
        )
        q.append(
            Quantity(
                f"model.sf.{aoi}.building_lift",
                "building_lift",
                pop,
                "transfer_sf.json",
                lambda a, x=aoi: _row(a["transfer_sf.json"]["aois"], "aoi", x)[
                    "building_vs_footprints_over_2m"
                ]["precision_lift_over_null"],
            )
        )
        for group in COARSE:
            q.append(
                Quantity(
                    f"model.sf.{aoi}.predicted_coarse_fraction.{group}",
                    "predicted_class_fraction",
                    Population("san_francisco", aoi, "voxel", f"clean:{group}", "coarse6"),
                    "transfer_sf.json",
                    lambda a, x=aoi, g=group: _row(a["transfer_sf.json"]["aois"], "aoi", x)[
                        "predicted_distribution_coarse"
                    ][g],
                )
            )

    # ---------------- confidence as a detector of error ----------------
    q.append(
        Quantity(
            "model.surrey.test.voxel.called_ground.auroc",
            "error_detection_auroc",
            Population("surrey", "test", "voxel", "called_ground"),
            "auroc.json",
            lambda a: a["auroc.json"]["in_domain"]["auroc"],
            "right: DALES says ground. wrong: DALES says anything else",
        )
    )
    # The figure pooled over areas is not registered. It mixed ground from one
    # neighbourhood with water from another, and the review dropped it.
    q.append(
        Quantity(
            "model.sf.embarcadero.voxel.called_ground.auroc",
            "error_detection_auroc",
            Population("san_francisco", "embarcadero", "voxel", "called_ground"),
            "auroc.json",
            lambda a: a["auroc.json"]["out_of_domain"]["auroc_embarcadero_only"],
            "the only area holding both ground and water, so no cross-area mixing",
        )
    )

    # ---------------- the same, with the difficulty of the error held fixed ----------------
    # auroc_control.json reads the seeded prediction cache. Easy: a wrong ground call
    # more than 3 m above the ground surface. Hard: within half a metre of it, which
    # in San Francisco is water.
    def ctrl(a: dict) -> dict:
        return a["auroc_control.json"]

    for kind, subset in (
        ("all", "called_ground"),
        ("easy", "called_ground_easy"),
        ("hard", "called_ground_hard"),
    ):
        q.append(
            Quantity(
                f"control.surrey.test.voxel.{subset}.auroc",
                "error_detection_auroc_control",
                Population("surrey", "test", "voxel", subset),
                "auroc_control.json",
                lambda a, k=kind: ctrl(a)["surrey"][k]["auroc"],
                "eleven held-out tiles, one flight, pooled; each tile is listed in the artefact",
            )
        )
    for aoi in SF_AOIS:
        if aoi in ("steep", "baybridge"):
            continue
        q.append(
            Quantity(
                f"control.sf.{aoi}.voxel.called_ground_easy.auroc",
                "error_detection_auroc_control",
                Population("san_francisco", aoi, "voxel", "called_ground_easy"),
                "auroc_control.json",
                lambda a, x=aoi: _row(ctrl(a)["san_francisco_per_area"], "aoi", x)[
                    "easy_unassigned_over_3m"
                ]["auroc"],
                "wrong: delivery unassigned more than 3 m above the delivery ground surface",
            )
        )
    q.append(
        Quantity(
            "control.sf.embarcadero.voxel.called_ground_water.auroc",
            "error_detection_auroc_control",
            Population("san_francisco", "embarcadero", "voxel", "called_ground_water"),
            "auroc_control.json",
            lambda a: _row(ctrl(a)["san_francisco_per_area"], "aoi", "embarcadero")["hard_water"][
                "auroc"
            ],
            "wrong: delivery water at any height; recomputed from the cache, must agree with "
            "auroc.json. Not the hard row: water sits below the surface carried from the shore",
        )
    )
    q.append(
        Quantity(
            "control.sf.embarcadero.voxel.called_ground_hard.auroc",
            "error_detection_auroc_control",
            Population("san_francisco", "embarcadero", "voxel", "called_ground_hard"),
            "auroc_control.json",
            lambda a: _row(ctrl(a)["san_francisco_per_area"], "aoi", "embarcadero")[
                "hard_water_within_0_5m"
            ]["auroc"],
            "wrong: delivery water within half a metre of the ground surface, as the Surrey hard row",
        )
    )

    # ---------------- vegetation, against the blind NAIP reference ----------------
    def veg(a: dict) -> dict:
        return a["vegetation_sf.json"]

    q.append(
        Quantity(
            "vegetation.surrey.test.voxel.labelled_vegetation_2_25m.called_vegetation",
            "vegetation_called_vegetation",
            Population("surrey", "test", "voxel", "labelled_vegetation_2_25m", "coarse6"),
            "vegetation_sf.json",
            lambda a: veg(a)["home_surrey"]["called_vegetation"],
            "reference: DALES tree and vegetation labels, 2 to 25 m above the label-free floor",
        )
    )
    for aoi in SF_AOIS:
        if aoi == "baybridge":
            continue
        for group in ("vegetation", "building", "ground"):
            q.append(
                Quantity(
                    f"vegetation.sf.{aoi}.voxel.naip_core_2_25m.called_{group}",
                    f"vegetation_called_{group}",
                    Population(
                        "san_francisco", aoi, "voxel", "naip_vegetation_core_2_25m", "coarse6"
                    ),
                    "vegetation_sf.json",
                    lambda a, x=aoi, g=group: _row(veg(a)["san_francisco_per_area"], "aoi", x)[
                        "vegetation_core_called"
                    ][g],
                    "reference: NAIP 2022 NDVI over the Otsu threshold, eroded for relief "
                    "displacement, 337 days before the lidar",
                )
            )

    # ---------------- the discrimination numbers, two subsets ----------------
    # confidence.json groups every prediction; auroc.json keeps only points the
    # model calls ground. The same words, "median gap" and "share under 0.5", have
    # been quoted from both, and they are not the same quantity.
    def conf(a: dict) -> dict:
        return {r["group"]: r for r in a["confidence.json"]["pooled"]}

    q.append(
        Quantity(
            "conf.surrey.test.voxel_spread40.all_predictions.median_gap",
            "median_gap",
            Population("surrey", "test", "voxel_spread40", "all_predictions"),
            "confidence.json",
            lambda a: round(
                conf(a)["surrey_correct"]["max_prob_median"]
                - conf(a)["surrey_wrong"]["max_prob_median"],
                4,
            ),
            "right minus wrong median max probability, every prediction, subsample",
        )
    )
    q.append(
        Quantity(
            "conf.sf.aois.voxel.all_predictions.median_gap",
            "median_gap",
            Population("san_francisco", "aois", "voxel", "all_predictions"),
            "confidence.json",
            lambda a: round(
                conf(a)["sf_ground"]["max_prob_median"] - conf(a)["sf_water"]["max_prob_median"], 4
            ),
            "delivery ground minus delivery water median, every prediction",
        )
    )
    q.append(
        Quantity(
            "auroc.surrey.test.voxel.called_ground.median_gap",
            "median_gap",
            Population("surrey", "test", "voxel", "called_ground"),
            "auroc.json",
            lambda a: a["auroc.json"]["in_domain"]["median_gap_right_minus_wrong"],
        )
    )
    q.append(
        Quantity(
            "auroc.sf.aois.voxel.called_ground.median_gap",
            "median_gap",
            Population("san_francisco", "aois", "voxel", "called_ground"),
            "auroc.json",
            lambda a: a["auroc.json"]["out_of_domain"]["median_gap_right_minus_wrong"],
        )
    )
    q.append(
        Quantity(
            "conf.surrey.test.voxel_spread40.all_predictions.wrong_under_half",
            "share_under_half",
            Population("surrey", "test", "voxel_spread40", "all_predictions"),
            "confidence.json",
            lambda a: conf(a)["surrey_wrong"]["max_prob_under_0_5"],
        )
    )
    q.append(
        Quantity(
            "conf.sf.aois.voxel.all_predictions.water_under_half",
            "share_under_half",
            Population("san_francisco", "aois", "voxel", "all_predictions"),
            "confidence.json",
            lambda a: conf(a)["sf_water"]["max_prob_under_0_5"],
            "rounded to three places in the artefact, so a ratio against it is fragile",
        )
    )
    q.append(
        Quantity(
            "auroc.surrey.test.voxel.called_ground.wrong_under_half",
            "share_under_half",
            Population("surrey", "test", "voxel", "called_ground"),
            "auroc.json",
            lambda a: a["auroc.json"]["in_domain"]["wrong"]["share_under_0_5"],
        )
    )
    q.append(
        Quantity(
            "auroc.sf.aois.voxel.called_ground.water_under_half",
            "share_under_half",
            Population("san_francisco", "aois", "voxel", "called_ground"),
            "auroc.json",
            lambda a: a["auroc.json"]["out_of_domain"]["wrong"]["share_under_0_5"],
            "from exact counts, so a ratio against it is not a rounding artefact",
        )
    )

    # Ratios are registered rather than divided in prose, so nobody computes one by
    # hand from two rounded figures again.
    q.append(
        Quantity(
            "auroc.under_half_ratio.surrey_wrong_over_sf_water",
            "ratio",
            Population("both", "test_and_aois", "voxel", "called_ground"),
            "auroc.json",
            lambda a: round(
                a["auroc.json"]["in_domain"]["wrong"]["share_under_0_5"]
                / a["auroc.json"]["out_of_domain"]["wrong"]["share_under_0_5"],
                1,
            ),
            "from exact counts",
        )
    )
    q.append(
        Quantity(
            "conf.under_half_ratio.surrey_wrong_over_sf_water",
            "ratio",
            Population("both", "test_and_aois", "voxel_spread40_and_voxel", "all_predictions"),
            "confidence.json",
            lambda a: round(
                conf(a)["surrey_wrong"]["max_prob_under_0_5"]
                / conf(a)["sf_water"]["max_prob_under_0_5"],
                1,
            ),
            "denominator rounded to three places in the artefact, so this is fragile",
        )
    )

    # ---------------- the return ratio channel ----------------
    # Exactly 0 on every prepared DALES point, so constant through training.
    def rr(a: dict) -> dict:
        return a["return_ratio_check.json"]

    q.append(
        Quantity(
            "surrey.all.voxel.return_ratio_exactly_0",
            "points_with_return_ratio_0",
            Population("surrey", "all", "voxel", "all"),
            "return_ratio_check.json",
            lambda a: rr(a)["census"]["dales_all_tiles"]["ratio_exactly_0"],
            "every prepared DALES tile, training and held-out",
        )
    )
    q.append(
        Quantity(
            "surrey.all.voxel.points",
            "points",
            Population("surrey", "all", "voxel", "all"),
            "return_ratio_check.json",
            lambda a: rr(a)["census"]["dales_all_tiles"]["points"],
        )
    )
    q.append(
        Quantity(
            "sf.aois.voxel.return_ratio_exactly_0",
            "points_with_return_ratio_0",
            Population("san_francisco", "aois", "voxel", "all"),
            "return_ratio_check.json",
            lambda a: rr(a)["census"]["san_francisco_all_areas"]["ratio_exactly_0"],
        )
    )
    q.append(
        Quantity(
            "sf.aois.voxel.points",
            "points",
            Population("san_francisco", "aois", "voxel", "all"),
            "return_ratio_check.json",
            lambda a: rr(a)["census"]["san_francisco_all_areas"]["points"],
            "all but ratio_exactly_0 of these carry a real return ratio",
        )
    )

    # The same checkpoint and seed with the channel at its training value, 0.
    def ret0(a: dict, aoi: str) -> dict:
        return _row(a["return_channel_check.json"]["areas"], "aoi", aoi)

    for aoi in SF_AOIS:
        if aoi == "baybridge":
            continue
        for when in ("before", "after"):
            q.append(
                Quantity(
                    f"ret0.sf.{aoi}.{when}.naip_core_called_vegetation",
                    f"vegetation_called_vegetation_channel_{when}",
                    Population(
                        "san_francisco", aoi, "voxel", "naip_vegetation_core_2_25m", "coarse6"
                    ),
                    "return_channel_check.json",
                    lambda a, x=aoi, w=when: ret0(a, x)[w]["naip_vegetation_core_called"][
                        "vegetation"
                    ],
                    "before: channel as delivered; after: channel at 0, its training value",
                )
            )
            q.append(
                Quantity(
                    f"ret0.sf.{aoi}.{when}.easy_errors",
                    f"easy_errors_channel_{when}",
                    Population("san_francisco", aoi, "voxel", "called_ground_easy"),
                    "return_channel_check.json",
                    lambda a, x=aoi, w=when: ret0(a, x)[w]["easy_errors"],
                )
            )
            q.append(
                Quantity(
                    f"ret0.sf.{aoi}.{when}.easy_auroc",
                    f"error_detection_auroc_channel_{when}",
                    Population("san_francisco", aoi, "voxel", "called_ground_easy"),
                    "return_channel_check.json",
                    lambda a, x=aoi, w=when: ret0(a, x)[w]["easy_auroc"],
                )
            )
    q.append(
        Quantity(
            "ret0.verdict_areas_easy_errors_halved",
            "areas_easy_errors_halved",
            Population("san_francisco", "aois", "voxel", "called_ground_easy"),
            "return_channel_check.json",
            lambda a: a["return_channel_check.json"]["scored_areas_easy_errors_halved"],
        )
    )

    # ---------------- four classes at home ----------------
    q.append(
        Quantity(
            "model.surrey.test.voxel.accuracy_four_class",
            "accuracy_four_class",
            Population("surrey", "test", "voxel", "labelled", "four4"),
            "four_class_eval.json",
            lambda a: a["four_class_eval.json"]["accuracy"],
            "the 0.8948 confusion regrouped into ground, vegetation, building, other",
        )
    )
    q.append(
        Quantity(
            "model.surrey.test.voxel.mean_iou_four_class",
            "mean_iou_four_class",
            Population("surrey", "test", "voxel", "labelled", "four4"),
            "four_class_eval.json",
            lambda a: a["four_class_eval.json"]["mean_iou"],
        )
    )

    # ---------------- ground band without water ----------------
    for band_end in ("precision_floor", "precision_ceiling"):
        for water in ("with_water", "without_water"):
            q.append(
                Quantity(
                    f"ground.sf.embarcadero.as_trained.{water}.{band_end}",
                    f"ground_{band_end}_{water}",
                    Population("san_francisco", "embarcadero", "voxel", "judgeable"),
                    "ground_band_no_water.json",
                    lambda a, w=water, b=band_end: _row(
                        a["ground_band_no_water.json"]["areas"], "aoi", "embarcadero"
                    )["as_trained"][w][b],
                )
            )

    # ---------------- national lidar sets, one tile each ----------------
    # The share of points more than 2 m above the delivery's own ground that the
    # delivery left unclassified, on 0.15 m voxel-thinned clouds. San Francisco is
    # measured the same way beside them.
    def nat(a: dict) -> dict:
        return a["national_tile_check.json"]

    for tile, city in (
        ("ign_lidarhd_paris", "paris"),
        ("swisssurface3d_zurich", "zurich"),
        ("ahn5_amsterdam", "amsterdam"),
    ):
        pop = Population(city, "one_tile", "voxel", "over_2m_delivery_ground")
        q.append(
            Quantity(
                f"national.{tile}.unclassified_share_over_2m",
                "unclassified_share_over_2m",
                pop,
                "national_tile_check.json",
                lambda a, t=tile: nat(a)["tiles"][t]["contamination_proxy"][
                    "unclassified_share_of_points_over_2m"
                ],
            )
        )
        q.append(
            Quantity(
                f"national.{tile}.points_per_m2",
                "points_per_m2",
                Population(city, "one_tile", "raw", "occupied_1m_cells"),
                "national_tile_check.json",
                lambda a, t=tile: nat(a)["tiles"][t]["density_raw"]["points_per_m2"],
            )
        )
    for tile, city in (("ign_lidarhd_paris", "paris"), ("swisssurface3d_zurich", "zurich")):
        q.append(
            Quantity(
                f"national.{tile}.vegetation_share_over_2m",
                "vegetation_share_over_2m",
                Population(city, "one_tile", "voxel", "over_2m_delivery_ground"),
                "national_tile_check.json",
                lambda a, t=tile: nat(a)["tiles"][t]["contamination_proxy"][
                    "vegetation_share_of_points_over_2m"
                ],
            )
        )
    for aoi in SF_AOIS:
        if aoi == "baybridge":
            continue
        q.append(
            Quantity(
                f"national.sf_yardstick.{aoi}.unclassified_share_over_2m",
                "unclassified_share_over_2m",
                Population("san_francisco", aoi, "voxel", "over_2m_delivery_ground"),
                "national_tile_check.json",
                lambda a, x=aoi: _row(nat(a)["san_francisco_same_proxy"], "aoi", x)[
                    "unclassified_share_of_points_over_2m"
                ],
            )
        )

    # ---------------- every San Francisco figure, per input variant ----------------
    # scripts/pipeline/score_variant.py, from cached predictions, on the same references for
    # every variant: footprints moved by the measured offset, NAIP both registered and
    # not. The variant is set here explicitly, never inferred from the key.
    def sc(a: dict, variant: str, aoi: str) -> dict:
        return _row(a[f"sf_scores_{variant}.json"]["san_francisco_per_area"], "aoi", aoi)

    # Registered once produced: the m0 variants join when their scoring exists.
    only = {
        "water_called_ground": {"baybridge", "embarcadero"},
        "hard_water_auroc": {"embarcadero"},
        "hard_water_auroc_wrong_n": {"embarcadero"},
        "bridge_deck_called_building": {"baybridge", "embarcadero"},
        "water_confidence_median": {"baybridge", "embarcadero"},
    }
    no_ground = {
        "ground_precision_floor",
        "ground_recall",
        "easy_error_auroc",
        "vegetation_headline",
        "raised_unassigned_called_ground_rate",
        "vegetation_core_called_vegetation",
        "vegetation_core_called_vegetation_registered",
    }
    # The rented card's runs are scored by the same code and registered the same way, under their own
    # variant names, so a figure from one of them can never be read as the published model's.
    for variant in (
        "returns_ood",
        "ret0",
        *(f"{r}_seed{x}" for r in RECIPES for x in SEEDS),
        *RENTED_VARIANTS,
    ):
        src = f"sf_scores_{variant}.json"
        if not (ARTIFACTS / src).exists():
            continue
        for aoi in SF_AOIS:

            def sf_pop(subset, taxonomy="none", x=aoi, v=variant):
                return Population("san_francisco", x, "voxel", subset, taxonomy, variant=v)

            specs = [
                (
                    "building_lift_global_shift",
                    "building_lift_shifted",
                    sf_pop("clean_over_2m"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["building_vs_footprints_over_2m"][
                        "global_shift"
                    ]["precision_lift_over_null"],
                ),
                (
                    "building_precision_global_shift",
                    "building_precision_shifted",
                    sf_pop("clean_over_2m"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["building_vs_footprints_over_2m"][
                        "global_shift"
                    ]["precision"],
                ),
                (
                    "ground_precision_floor",
                    "ground_precision_floor",
                    sf_pop("judgeable"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["ground_band"]["with_water"][
                        "precision_floor"
                    ],
                ),
                (
                    "ground_recall",
                    "ground_recall",
                    sf_pop("delivery_ground"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["ground_band"]["with_water"][
                        "recall_clean"
                    ],
                ),
                (
                    "vegetation_core_called_vegetation",
                    "vegetation_called_vegetation",
                    sf_pop("naip_vegetation_core_2_25m", "coarse6"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["vegetation"]["unregistered"][
                        "vegetation_core_called"
                    ]["vegetation"],
                ),
                (
                    "vegetation_core_called_vegetation_registered",
                    "vegetation_called_vegetation",
                    sf_pop("naip_vegetation_core_2_25m_registered", "coarse6"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["vegetation"]["registered"][
                        "vegetation_core_called"
                    ]["vegetation"],
                ),
                (
                    "not_vegetation_core_called_vegetation",
                    "not_vegetation_called_vegetation",
                    sf_pop("naip_not_vegetation_core_2_25m", "coarse6"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["vegetation"]["unregistered"][
                        "not_vegetation_core_called_vegetation"
                    ],
                ),
                (
                    "water_called_ground",
                    "water_called_ground",
                    sf_pop("delivery_water"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["water_called_ground"],
                ),
                (
                    "bridge_deck_called_building",
                    "bridge_deck_called_building",
                    sf_pop("delivery_bridge_deck"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["bridge_deck_called_building"],
                ),
                (
                    "water_confidence_median",
                    "confidence_median",
                    sf_pop("delivery_water"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["confidence"]["sf_water"][
                        "max_prob_median"
                    ],
                ),
                (
                    "unassigned_confidence_median",
                    "confidence_median",
                    sf_pop("delivery_unassigned"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["confidence"]["sf_unassigned"][
                        "max_prob_median"
                    ],
                ),
                (
                    "easy_error_auroc",
                    "error_detection_auroc",
                    sf_pop("called_ground_easy"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["auroc"]["easy_unassigned_over_3m"][
                        "auroc"
                    ],
                ),
                (
                    "predicted_vegetation_share",
                    "predicted_class_fraction",
                    sf_pop("clean:vegetation", "coarse6"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["predicted_coarse_shares"][
                        "vegetation"
                    ],
                ),
                (
                    "vegetation_headline",
                    "vegetation_called_vegetation_headline",
                    sf_pop("naip_vegetation_core_2_25m_rule", "coarse6"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["vegetation_headline"][
                        "vegetation_core_called_vegetation"
                    ],
                ),
                (
                    "raised_unassigned_called_ground_rate",
                    "raised_called_ground_rate",
                    sf_pop("unassigned_over_3m"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["raised_unassigned_called_ground_rate"][
                        "rate"
                    ],
                ),
                (
                    "hard_water_auroc",
                    "error_detection_auroc",
                    sf_pop("called_ground_hard_water_0_5m"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["auroc"]["hard_water_within_0_5m"][
                        "auroc"
                    ],
                ),
                (
                    "hard_water_auroc_wrong_n",
                    "error_detection_wrong_n",
                    sf_pop("called_ground_hard_water_0_5m"),
                    lambda a, v=variant, x=aoi: sc(a, v, x)["auroc"]["hard_water_within_0_5m"][
                        "wrong"
                    ],
                ),
            ]
            for name, concept, population, getter in specs:
                if name in only and aoi not in only[name]:
                    continue
                if aoi == "baybridge" and name in no_ground:
                    continue
                # Already registered for this variant from vegetation_sf.json and
                # transfer_sf.json; tests/integrity/test_quantities.py checks the rescore
                # agrees with them instead of registering the same quantity twice.
                if variant == "returns_ood" and name in (
                    "vegetation_core_called_vegetation",
                    "predicted_vegetation_share",
                ):
                    continue
                q.append(
                    Quantity(f"scores.{variant}.sf.{aoi}.{name}", concept, population, src, getter)
                )
        q.append(
            Quantity(
                f"scores.{variant}.sf.financial.extrapolated_blocks_lift",
                "building_lift_shifted",
                Population(
                    "san_francisco",
                    "financial",
                    "voxel",
                    "clean_over_2m:extrapolated_blocks",
                    variant=variant,
                ),
                src,
                lambda a, v=variant: sc(a, v, "financial")["confounds"][
                    "extrapolation_building_lift"
                ]["named_blocks"]["precision_lift_over_null"],
            )
        )
        q.append(
            Quantity(
                f"scores.{variant}.sf.financial.in_range_blocks_lift",
                "building_lift_shifted",
                Population(
                    "san_francisco",
                    "financial",
                    "voxel",
                    "clean_over_2m:in_range_blocks",
                    variant=variant,
                ),
                src,
                lambda a, v=variant: sc(a, v, "financial")["confounds"][
                    "extrapolation_building_lift"
                ]["other_blocks"]["precision_lift_over_null"],
            )
        )
        q.append(
            Quantity(
                f"scores.{variant}.sf.steep.high_relief_blocks_ground_recall",
                "ground_recall",
                Population(
                    "san_francisco",
                    "steep",
                    "voxel",
                    "delivery_ground:high_relief_blocks",
                    variant=variant,
                ),
                src,
                lambda a, v=variant: sc(a, v, "steep")["confounds"]["relief_ground_recall"][
                    "named_blocks"
                ],
            )
        )
        q.append(
            Quantity(
                f"scores.{variant}.sf.steep.other_blocks_ground_recall",
                "ground_recall",
                Population(
                    "san_francisco",
                    "steep",
                    "voxel",
                    "delivery_ground:other_blocks",
                    variant=variant,
                ),
                src,
                lambda a, v=variant: sc(a, v, "steep")["confounds"]["relief_ground_recall"][
                    "other_blocks"
                ],
            )
        )
        q.append(
            Quantity(
                f"scores.{variant}.surrey.test.raised_rate_min",
                "raised_called_ground_rate_min",
                Population("surrey", "test", "voxel", "not_ground_over_3m", variant=variant),
                src,
                lambda a, v=variant: a[f"sf_scores_{v}.json"]["surrey_held_out"][
                    "raised_rate_range"
                ][0],
            )
        )
        q.append(
            Quantity(
                f"scores.{variant}.surrey.test.raised_rate_max",
                "raised_called_ground_rate_max",
                Population("surrey", "test", "voxel", "not_ground_over_3m", variant=variant),
                src,
                lambda a, v=variant: a[f"sf_scores_{v}.json"]["surrey_held_out"][
                    "raised_rate_range"
                ][1],
            )
        )
        for part in ("auroc", "right", "wrong"):
            q.append(
                Quantity(
                    f"scores.{variant}.surrey.test.hard_error_auroc.{part}",
                    f"error_detection_{part}",
                    Population(
                        "surrey", "test", "voxel", "called_ground_hard_within_0_5m", variant=variant
                    ),
                    src,
                    lambda a, v=variant, pt=part: a[f"sf_scores_{v}.json"]["surrey_held_out"][
                        "auroc_called_ground"
                    ]["hard"][pt],
                    "home errors within 0.5 m of the ground surface against right ground, the water construction",
                )
            )
        q.append(
            Quantity(
                f"scores.{variant}.surrey.test.accuracy",
                "accuracy_from_prediction_cache",
                Population("surrey", "test", "voxel", "labelled", "fine15", variant=variant),
                src,
                lambda a, v=variant: a[f"sf_scores_{v}.json"]["surrey_held_out"][
                    "accuracy_labelled_voxel_points"
                ],
            )
        )

    # ---------------- M0, three channels, per seed ----------------
    for recipe, seed in ((r, x) for r in RECIPES for x in SEEDS):
        src = f"train_{recipe}_seed{seed}.json"
        if not (ARTIFACTS / src).exists():
            continue
        tag = f"{recipe}_seed{seed}"
        q.append(
            Quantity(
                f"{tag}.surrey.test.voxel.accuracy",
                "accuracy",
                Population("surrey", "test", "voxel", "labelled", "fine15", variant=tag),
                src,
                lambda a, s_=src: a[s_]["final"]["accuracy"],
                "full pass over every held-out point, three channel model",
            )
        )
        for name in ("ground", "tree", "vegetation", "habitat", "complex"):
            q.append(
                Quantity(
                    f"{tag}.surrey.test.voxel.iou.{name}",
                    "iou",
                    Population("surrey", "test", "voxel", f"class:{name}", "fine15", variant=tag),
                    src,
                    lambda a, s_=src, n=name: a[s_]["final"]["iou_per_class"][n],
                )
            )

    # ---------------- across seeds, by the rules in surrey.seeds ----------------
    stats = ("median", "min", "max")
    for recipe in RECIPES:
        src = f"seeds_{recipe}.json"
        if not (ARTIFACTS / src).exists():
            continue
        agg = f"{recipe}_seeds"
        doc = json.loads((ARTIFACTS / src).read_text(encoding="utf-8"))
        for aoi, metrics in doc["san_francisco"].items():
            for metric in metrics:
                for st in stats:
                    q.append(
                        Quantity(
                            f"seeds.{recipe}.sf.{aoi}.{metric}.{st}",
                            f"{metric}_{st}_over_seeds",
                            Population("san_francisco", aoi, "voxel", metric, variant=agg),
                            src,
                            lambda a, s_=src, x=aoi, m=metric, t=st: a[s_]["san_francisco"][x][m][
                                t
                            ],
                        )
                    )
        for part in (
            "accuracy_from_prediction_cache",
            "accuracy_training_artefact",
            "miou_15",
            "miou_coarse",
        ):
            if part not in doc["surrey_held_out"]:
                continue
            taxonomy = "coarse6" if part == "miou_coarse" else "fine15"
            for st in stats:
                q.append(
                    Quantity(
                        f"seeds.{recipe}.surrey.test.{part}.{st}",
                        f"{part}_{st}_over_seeds",
                        Population("surrey", "test", "voxel", "labelled", taxonomy, variant=agg),
                        src,
                        lambda a, s_=src, p_=part, t=st: a[s_]["surrey_held_out"][p_][t],
                    )
                )
        for claim in ("extrapolation", "relief", "hard_errors_home_vs_water"):
            for part in ("k", "n"):
                q.append(
                    Quantity(
                        f"seeds.{recipe}.claims.{claim}.{part}",
                        f"seeds_{part}",
                        Population("san_francisco", "aois", "voxel", f"claim:{claim}", variant=agg),
                        src,
                        lambda a, s_=src, c=claim, p_=part: a[s_]["claims"][c][p_],
                    )
                )
        for part, path, city, split in (
            ("relief_gap", ("relief", "named_minus_other"), "san_francisco", "steep"),
            ("hard_home_auroc", ("hard_errors_home_vs_water", "home_auroc"), "surrey", "test"),
            (
                "hard_water_auroc",
                ("hard_errors_home_vs_water", "water_auroc"),
                "san_francisco",
                "embarcadero",
            ),
            ("financial_lift", ("extrapolation", "financial_lift"), "san_francisco", "financial"),
            (
                "other_built_median_lift",
                ("extrapolation", "other_built_median"),
                "san_francisco",
                "built_areas",
            ),
        ):
            for st in stats:
                q.append(
                    Quantity(
                        f"seeds.{recipe}.claims.{part}.{st}",
                        f"{part}_{st}_over_seeds",
                        Population(city, split, "voxel", f"claim:{path[0]}:{path[1]}", variant=agg),
                        src,
                        lambda a, s_=src, pa=path, t=st: a[s_]["claims"][pa[0]][pa[1]][t],
                    )
                )
        for aoi in doc["claims"]["easy_errors"]:
            for which in ("transfer", "home_weakness"):
                for part in ("k", "n"):
                    q.append(
                        Quantity(
                            f"seeds.{recipe}.claims.easy.{aoi}.{which}.{part}",
                            f"seeds_{part}",
                            Population(
                                "san_francisco", aoi, "voxel", f"claim:easy_{which}", variant=agg
                            ),
                            src,
                            lambda a, s_=src, x=aoi, w=which, p_=part: a[s_]["claims"][
                                "easy_errors"
                            ][x][w][p_],
                        )
                    )

    # ---------------- the physical water null, no model involved ----------------
    if (ARTIFACTS / "water_null.json").exists():
        wn = json.loads((ARTIFACTS / "water_null.json").read_text(encoding="utf-8"))
        q.append(
            Quantity(
                "waternull.vertical_check.worst_abs_z_difference_m",
                "ept_minus_source_height",
                Population("san_francisco", "delivery", "source_headers"),
                "water_null.json",
                lambda a: a["water_null.json"]["vertical_check"]["worst_abs_z_difference_m"],
            )
        )
        q.append(
            Quantity(
                "waternull.margin_m",
                "water_null_margin",
                Population("san_francisco", "water_areas", "voxel"),
                "water_null.json",
                lambda a: a["water_null.json"]["margin_m"],
                "fixed before looking",
            )
        )
        for row in wn["areas"]:
            aoi = row["aoi"]
            q.append(
                Quantity(
                    f"waternull.sf.{aoi}.flight_span_minutes",
                    "flight_span_minutes",
                    Population("san_francisco", aoi, "voxel", "gps_time"),
                    "water_null.json",
                    lambda a, x=aoi: _row(a["water_null.json"]["areas"], "aoi", x)[
                        "flight_span_minutes"
                    ],
                )
            )
            for st in ("min", "max", "median"):
                q.append(
                    Quantity(
                        f"waternull.sf.{aoi}.gauge_level_navd88_m.{st}",
                        f"gauge_level_{st}",
                        Population("san_francisco", aoi, "voxel", f"gps_time:{st}"),
                        "water_null.json",
                        lambda a, x=aoi, t=st: _row(a["water_null.json"]["areas"], "aoi", x)[
                            "gauge_level_navd88_m"
                        ][t],
                    )
                )
            for margin in row["null"]:
                for part in ("null_precision", "null_recall", "null_iou", "called", "water_points"):
                    q.append(
                        Quantity(
                            f"waternull.sf.{aoi}.{margin}.{part}",
                            f"water_{part}",
                            Population("san_francisco", aoi, "voxel", f"clean:{margin}"),
                            "water_null.json",
                            lambda a, x=aoi, m=margin, pt=part: _row(
                                a["water_null.json"]["areas"], "aoi", x
                            )["null"][m][pt],
                        )
                    )

    # ---------------- the viewer's octrees, bytes, no model figure ----------------
    if (ARTIFACTS / "octree_budget.json").exists():
        ob = json.loads((ARTIFACTS / "octree_budget.json").read_text(encoding="utf-8"))
        for area, row in ob["areas"].items():
            for part, path in (
                ("bytes", ("full", "bytes")),
                ("bytes_per_point", ("full", "bytes_per_point")),
                ("points", ("points",)),
            ):
                q.append(
                    Quantity(
                        f"viewer.octree.{area}.{part}",
                        f"octree_{part}",
                        Population("san_francisco", area, "voxel", "octree_brotli"),
                        "octree_budget.json",
                        lambda a, ar=area, pa=path: _dig(a["octree_budget.json"]["areas"][ar], pa),
                    )
                )
            for part in row.get("marginal_bytes_per_point", {}):
                q.append(
                    Quantity(
                        f"viewer.octree.{area}.marginal_bytes_per_point.{part}",
                        "octree_marginal_bytes_per_point",
                        Population("san_francisco", area, "voxel", f"octree_brotli:{part}"),
                        "octree_budget.json",
                        lambda a, ar=area, pt=part: a["octree_budget.json"]["areas"][ar][
                            "marginal_bytes_per_point"
                        ][pt],
                    )
                )

    # ---------------- M1 against its control, by the criteria written before training ----------------
    if (ARTIFACTS / "compare_m1_vs_m0prime.json").exists():
        cmp_ = json.loads((ARTIFACTS / "compare_m1_vs_m0prime.json").read_text(encoding="utf-8"))
        src = "compare_m1_vs_m0prime.json"
        if "water" in cmp_:
            q.append(
                Quantity(
                    "compare.m1_vs_m0prime.water.success",
                    "criterion_met",
                    Population(
                        "san_francisco",
                        "water_areas",
                        "voxel",
                        "delivery_water_lower_edge",
                        variant="m1_seeds",
                    ),
                    src,
                    lambda a: int(
                        bool(a["compare_m1_vs_m0prime.json"]["water"]["criterion"]["success"])
                    ),
                    "1 when M1's water lift over the null is positive on all three seeds at both areas",
                )
            )
            for aoi in cmp_["water"]["criterion"]["per_area"]:
                for part in ("positive_on", "seeds"):
                    q.append(
                        Quantity(
                            f"compare.m1_vs_m0prime.water.{aoi}.{part}",
                            f"water_lift_{part}",
                            Population(
                                "san_francisco",
                                aoi,
                                "voxel",
                                "delivery_water_lower_edge",
                                variant="m1_seeds",
                            ),
                            src,
                            lambda a, x=aoi, pt=part: a["compare_m1_vs_m0prime.json"]["water"][
                                "criterion"
                            ]["per_area"][x][pt],
                        )
                    )
                for seed in ("A", "B", "C"):
                    if (
                        cmp_["water"]["criterion"]["per_area"][aoi]["lift_per_seed"].get(seed)
                        is None
                    ):
                        continue
                    q.append(
                        Quantity(
                            f"compare.m1_vs_m0prime.water.{aoi}.lift.seed_{seed.lower()}",
                            "water_lift_over_null",
                            Population(
                                "san_francisco",
                                aoi,
                                "voxel",
                                "delivery_water_lower_edge",
                                variant=f"m1_seed{seed}",
                            ),
                            src,
                            lambda a, x=aoi, sd=seed: a["compare_m1_vs_m0prime.json"]["water"][
                                "criterion"
                            ]["per_area"][x]["lift_per_seed"][sd],
                            "one seed's lift, as the README table prints it",
                        )
                    )
                for st in ("median", "min", "max"):
                    q.append(
                        Quantity(
                            f"compare.m1_vs_m0prime.water.{aoi}.lift.{st}",
                            f"water_lift_over_null_{st}",
                            Population(
                                "san_francisco",
                                aoi,
                                "voxel",
                                "delivery_water_lower_edge",
                                variant="m1_seeds",
                            ),
                            src,
                            lambda a, x=aoi, t=st: _spread_of(
                                a["compare_m1_vs_m0prime.json"]["water"]["criterion"]["per_area"][
                                    x
                                ]["lift_per_seed"]
                            )[t],
                        )
                    )

    # ---------------- water, asked on the data the class was learned from ----------------
    # In-sample by construction: every Swiss sub-tile is a training tile. The population says so, so a
    # figure from here can never be read as a held-out score.
    if (ARTIFACTS / "swiss_scores.json").exists():
        sw = json.loads((ARTIFACTS / "swiss_scores.json").read_text(encoding="utf-8"))
        src = "swiss_scores.json"
        for variant in sw.get("variants", {}):
            for cls in ("water", "bridge_deck"):
                for part in ("precision", "recall", "iou", "null_iou", "null_precision", "support"):
                    # A model with no output for the class calls it on no point, so its precision has no
                    # denominator. That is a fact about the control, not a quantity to register as empty.
                    if sw["variants"][variant]["pooled"][cls][part] is None:
                        continue
                    q.append(
                        Quantity(
                            f"swiss.{variant}.{cls}.{part}",
                            f"swiss_in_sample_{cls}_{part}",
                            Population(
                                "switzerland",
                                "sixteen_sub_tiles_all_in_training",
                                "voxel",
                                f"labelled:{cls}",
                                "m1_17",
                                variant=variant,
                            ),
                            src,
                            lambda a, v=variant, c=cls, pt=part: a["swiss_scores.json"]["variants"][
                                v
                            ]["pooled"][c][pt],
                            "in-sample: the model was trained on these tiles",
                        )
                    )
        for cls in ("water", "bridge_deck"):
            for part in ("points", "share_of_training_points"):
                q.append(
                    Quantity(
                        f"training.{cls}.{part}",
                        f"training_{cls}_{part}",
                        Population(
                            "switzerland",
                            "sixteen_sub_tiles_all_in_training",
                            "voxel",
                            f"labelled:{cls}",
                            "m1_17",
                        ),
                        src,
                        lambda a, c=cls, pt=part: a["swiss_scores.json"]["training_share"][
                            "classes"
                        ][c][pt],
                    )
                )

    # ---------------- the same classes on a Swiss tile that never trained ----------------
    # The population says held out, so this figure can never be mistaken for the in-sample one above.
    if (ARTIFACTS / "swiss_holdout_scores.json").exists():
        ho = json.loads((ARTIFACTS / "swiss_holdout_scores.json").read_text(encoding="utf-8"))
        src = "swiss_holdout_scores.json"
        for variant in ho.get("variants", {}):
            for cls in ("water", "bridge_deck"):
                for part in ("precision", "recall", "iou", "null_iou"):
                    if ho["variants"][variant]["pooled"][cls][part] is None:
                        continue
                    q.append(
                        Quantity(
                            f"swiss_holdout.{variant}.{cls}.{part}",
                            f"swiss_held_out_{cls}_{part}",
                            Population(
                                "switzerland",
                                "zug_tile_never_trained",
                                "voxel",
                                f"labelled:{cls}",
                                "m1_17",
                                variant=variant,
                            ),
                            src,
                            lambda a, v=variant, c=cls, pt=part: a["swiss_holdout_scores.json"][
                                "variants"
                            ][v]["pooled"][c][pt],
                            "a city none of the training tiles came from, chosen by the same rule afterwards",
                        )
                    )
        # The median and range over M1's three seeds, which is how a figure that exists for several
        # seeds is quoted (tests/integrity/test_seed_ranges.py).
        m1_seeds = [v for v in ("m1_seedA", "m1_seedB", "m1_seedC") if v in ho.get("variants", {})]
        if len(m1_seeds) == 3:
            for cls in ("water", "bridge_deck"):
                for part in ("precision", "recall", "iou"):
                    for st in ("median", "min", "max"):
                        q.append(
                            Quantity(
                                f"swiss_holdout.m1_seeds.{cls}.{part}.{st}",
                                f"swiss_held_out_{cls}_{part}_{st}_over_seeds",
                                Population(
                                    "switzerland",
                                    "zug_tile_never_trained",
                                    "voxel",
                                    f"labelled:{cls}",
                                    "m1_17",
                                    variant="m1_seeds",
                                ),
                                src,
                                lambda a, c=cls, pt=part, t=st, vs=tuple(m1_seeds): _spread_of(
                                    {
                                        v: a["swiss_holdout_scores.json"]["variants"][v]["pooled"][
                                            c
                                        ][pt]
                                        for v in vs
                                    }
                                )[t],
                            )
                        )

    # ---------------- what the model is handed over water, no model involved ----------------
    if (ARTIFACTS / "water_input_test.json").exists():
        wi = json.loads((ARTIFACTS / "water_input_test.json").read_text(encoding="utf-8"))
        src = "water_input_test.json"
        for aoi, row in wi.get("intensity", {}).get("areas", {}).items():
            if row["separation"]["auroc"] is None:
                continue
            q.append(
                Quantity(
                    f"water_input.intensity.{aoi}.auroc",
                    "intensity_as_a_water_detector",
                    Population(
                        "san_francisco", aoi, "raw", "delivery_water_against_delivery_ground"
                    ),
                    src,
                    lambda a, x=aoi: a["water_input_test.json"]["intensity"]["areas"][x][
                        "separation"
                    ]["auroc"],
                    "the delivery's own intensity, before this project's voxel step",
                )
            )
        for aoi, row in wi.get("block_floor", {}).get("areas", {}).items():
            q.append(
                Quantity(
                    f"water_input.block_floor.{aoi}.water_median_z",
                    "water_z_above_block_floor",
                    Population("san_francisco", aoi, "voxel", "delivery_water"),
                    src,
                    lambda a, x=aoi: a["water_input_test.json"]["block_floor"]["areas"][x][
                        "z_above_block_floor_m"
                    ]["water"]["median"],
                )
            )
            if row["separation_by_z_above_floor"]["auroc"] is not None:
                q.append(
                    Quantity(
                        f"water_input.block_floor.{aoi}.auroc",
                        "z_above_floor_as_a_water_detector",
                        Population(
                            "san_francisco", aoi, "voxel", "delivery_water_against_delivery_ground"
                        ),
                        src,
                        lambda a, x=aoi: a["water_input_test.json"]["block_floor"]["areas"][x][
                            "separation_by_z_above_floor"
                        ]["auroc"],
                    )
                )
        for country in wi.get("both_countries", {}).get("countries", {}):
            city = "san_francisco" if country == "san_francisco" else "switzerland"
            for key, getter in (
                ("z_median", ("z_above_block_floor_m", "median")),
                ("density_median", ("points_per_square_metre", "median")),
            ):
                q.append(
                    Quantity(
                        f"water_input.both.{country}.{key}",
                        f"water_{key}",
                        Population(city, "water_blocks", "voxel", "labelled_water"),
                        src,
                        lambda a, c=country, g=getter: a["water_input_test.json"]["both_countries"][
                            "countries"
                        ][c][g[0]][g[1]],
                    )
                )
            q.append(
                Quantity(
                    f"water_input.both.{country}.land_share",
                    "water_blocks_with_land",
                    Population(city, "water_blocks", "voxel", "labelled_water"),
                    src,
                    lambda a, c=country: a["water_input_test.json"]["both_countries"]["countries"][
                        c
                    ]["share_of_water_points_whose_block_is_at_least_5_percent_land"],
                )
            )

    # ---------------- how much of each area the footprints cover ----------------
    # No model in it: the share of points over 2 m that fall inside a footprint. Golden Gate Park and the
    # Bay Bridge come out at zero, which is what the README's "no buildings" says, said by the data.
    if (ARTIFACTS / "sf_scores_m0_seedA.json").exists():
        sf0 = json.loads((ARTIFACTS / "sf_scores_m0_seedA.json").read_text(encoding="utf-8"))
        for row in sf0.get("san_francisco_per_area", []):
            g = row.get("building_vs_footprints_over_2m", {}).get("global_shift", {})
            if g.get("null_precision") is None:
                continue
            q.append(
                Quantity(
                    f"footprints.{row['aoi']}.share_of_tall_points_inside",
                    "footprint_cover_of_tall_points",
                    # The figure is the reference against itself and no model enters it, but it is read from
                    # the published model's artefact, so it carries that variant rather than claiming none.
                    Population(
                        "san_francisco", row["aoi"], "voxel", "over_2m", variant=PUBLISHED_VARIANT
                    ),
                    "sf_scores_m0_seedA.json",
                    lambda a, x=row["aoi"]: next(
                        r["building_vs_footprints_over_2m"]["global_shift"]["null_precision"]
                        for r in a["sf_scores_m0_seedA.json"]["san_francisco_per_area"]
                        if r["aoi"] == x
                    ),
                    "the footprint null itself, which is zero where the area has no footprints",
                )
            )

    # ---------------- what M1 does with San Francisco water, the figure the README quotes ----------------
    for variant in ("m1_seedA", "m1_seedB", "m1_seedC"):
        art = f"sf_scores_{variant}.json"
        if not (ARTIFACTS / art).exists():
            continue
        blob = json.loads((ARTIFACTS / art).read_text(encoding="utf-8"))
        for area_row in blob.get("san_francisco_per_area", []):
            aoi = area_row["aoi"]
            band = area_row.get("water_vs_null")
            if not band or band.get("model", {}).get("recall") is None:
                continue
            for part in ("recall", "precision", "iou", "null_iou"):
                if band["model"].get(part if part != "null_iou" else "null_iou") is None:
                    continue
                q.append(
                    Quantity(
                        f"water.{variant}.{aoi}.{part}",
                        f"sf_water_{part}",
                        Population(
                            "san_francisco",
                            aoi,
                            "voxel",
                            "delivery_water_lower_edge",
                            variant=variant,
                        ),
                        art,
                        lambda a, ar=art, x=aoi, pt=part: next(
                            r["water_vs_null"]["model"][pt]
                            for r in a[ar]["san_francisco_per_area"]
                            if r["aoi"] == x
                        ),
                        "what the model itself calls water, beside the null in the same record",
                    )
                )
    m1_arts = {v: f"sf_scores_{v}.json" for v in ("m1_seedA", "m1_seedB", "m1_seedC")}
    if all((ARTIFACTS / a).exists() for a in m1_arts.values()):
        for aoi in ("embarcadero", "baybridge"):
            for part in ("recall", "precision", "iou"):
                for st in ("median", "min", "max"):
                    q.append(
                        Quantity(
                            f"water.m1_seeds.{aoi}.{part}.{st}",
                            f"sf_water_{part}_{st}_over_seeds",
                            Population(
                                "san_francisco",
                                aoi,
                                "voxel",
                                "delivery_water_lower_edge",
                                variant="m1_seeds",
                            ),
                            "sf_scores_m1_seedA.json",
                            lambda a, x=aoi, pt=part, t=st: _spread_of(
                                {
                                    v: next(
                                        r["water_vs_null"]["model"][pt]
                                        for r in a[ar]["san_francisco_per_area"]
                                        if r["aoi"] == x
                                    )
                                    for v, ar in m1_arts.items()
                                }
                            )[t],
                        )
                    )

    # ---------------- the grid the Swiss water points sit on, and whether M1 reads it ----------------
    # scripts/investigations/water_lattice.py: what the labelled points are, no model involved. scripts/investigations/water_lattice_test.py:
    # M1 on the held-out Zug tile with its water points moved off that grid, and ground moved as the control.
    if (ARTIFACTS / "water_lattice.json").exists():
        wl = json.loads((ARTIFACTS / "water_lattice.json").read_text(encoding="utf-8"))
        where = {
            "swiss_training": ("switzerland", "sixteen_sub_tiles_all_in_training"),
            "zug_held_out": ("switzerland", "zug_tile_never_trained"),
            "embarcadero": ("san_francisco", "embarcadero"),
            "baybridge": ("san_francisco", "baybridge"),
        }
        for name, row in wl["sets"].items():
            area, cls = name.split(".")
            if area not in where or "share_on_the_lattice" not in row:
                continue
            for part in (
                "share_on_the_lattice",
                "chance_share",
                "nn_cv_weighted",
                "nn_median_cm_weighted",
            ):
                q.append(
                    Quantity(
                        f"lattice.{area}.{cls}.{part}",
                        f"labelled_{cls}_{part}",
                        Population(where[area][0], where[area][1], "voxel", f"labelled:{cls}"),
                        "water_lattice.json",
                        lambda a, n=name, pt=part: a["water_lattice.json"]["sets"][n][pt],
                    )
                )
    if (ARTIFACTS / "water_lattice_test.json").exists():
        wt = json.loads((ARTIFACTS / "water_lattice_test.json").read_text(encoding="utf-8"))
        for variant, row in wt["variants"].items():
            for cond in row:
                q.append(
                    Quantity(
                        f"lattice_test.{variant}.{cond}.water.iou",
                        f"zug_water_iou_{cond}",
                        Population(
                            "switzerland",
                            "zug_tile_never_trained",
                            "voxel",
                            f"labelled:water:{cond}",
                            "m1_17",
                            variant=variant,
                        ),
                        "water_lattice_test.json",
                        lambda a, v=variant, c=cond: a["water_lattice_test.json"]["variants"][v][c][
                            "iou"
                        ],
                    )
                )
    # The density run on the rented card: one seed, read against the band the three M1 seeds span.
    if (ARTIFACTS / "compare_rented.json").exists():
        for aoi in ("embarcadero", "baybridge"):
            q.append(
                Quantity(
                    f"rented.m1_densityaug_seedA.{aoi}.water_recall",
                    "sf_water_recall",
                    Population(
                        "san_francisco",
                        aoi,
                        "voxel",
                        "delivery_water_lower_edge",
                        variant="m1_densityaug_seedA",
                    ),
                    "compare_rented.json",
                    lambda a, x=aoi: a["compare_rented.json"]["run_6_density"]["per_area"][x][
                        "recall"
                    ],
                )
            )

    # ---------------- what the verification run printed, per check ----------------
    # scripts/pipeline/verify.py runs every check the README's table lists; the table is filled from here, so a
    # count in it is what one run printed on a named machine and date.
    if (ARTIFACTS / "verification.json").exists():
        ver = json.loads((ARTIFACTS / "verification.json").read_text(encoding="utf-8"))
        for part in ("date", "machine"):
            q.append(
                Quantity(
                    f"verification.{part}",
                    f"verification_{part}",
                    Population("repository", "all", "none", "verification"),
                    "verification.json",
                    lambda a, pt=part: a["verification.json"][pt],
                )
            )
        for i, row in enumerate(ver["checks"]):
            slug = row["check"].replace(" ", "_")
            for part in ("command", "summary"):
                q.append(
                    Quantity(
                        f"verification.{slug}.{part}",
                        f"verification_{part}",
                        Population("repository", "all", "none", f"check:{slug}"),
                        "verification.json",
                        lambda a, k=i, pt=part: a["verification.json"]["checks"][k][pt],
                    )
                )

    # ---------------- the same model, on San Francisco thinned to the Swiss spacing ----------------
    if (ARTIFACTS / "water_density_test.json").exists():
        wd = json.loads((ARTIFACTS / "water_density_test.json").read_text(encoding="utf-8"))
        src = "water_density_test.json"
        variant = wd["variant"]
        for aoi in wd.get("areas", {}):
            for run in ("full", "thinned"):
                for part in ("water_recall", "water_precision", "ground_recall"):
                    # Bay Bridge carries no delivery ground class, so it has no ground recall to register.
                    if wd["areas"][aoi][run][part] is None:
                        continue
                    q.append(
                        Quantity(
                            f"density.{aoi}.{run}.{part}",
                            f"{part}_at_{run}_density",
                            Population(
                                "san_francisco",
                                aoi,
                                "voxel" if run == "full" else "voxel_thinned_to_swiss",
                                "delivery_water",
                                variant=variant,
                            ),
                            src,
                            lambda a, x=aoi, r=run, pt=part: a["water_density_test.json"]["areas"][
                                x
                            ][r][pt],
                        )
                    )
        q.append(
            Quantity(
                "density.keep_fraction",
                "keep_fraction",
                Population(
                    "san_francisco", "water_areas", "voxel_thinned_to_swiss", "delivery_water"
                ),
                src,
                lambda a: a["water_density_test.json"]["density"]["keep_fraction"],
                "how much of San Francisco survives thinning to the Swiss point spacing",
            )
        )

    # ---------------- the standard's classes, counted in the points the page draws ----------------
    # The legend on the page is the standard's own table, so its numbers have to be the model's own
    # counts over the area being drawn, per standard code rather than per this project's class names.
    if (ARTIFACTS / "octree_budget.json").exists():
        from .asprs import ASPRS_CLASSES, FINE_TO_ASPRS
        from .data import CLASS_NAMES as FINE15
        from .sources import M1_CLASS_NAMES

        budget = json.loads((ARTIFACTS / "octree_budget.json").read_text(encoding="utf-8"))
        for aoi, row in budget.get("areas", {}).items():
            for variant, counts in (row.get("classes_present") or {}).items():
                taxonomy = M1_CLASS_NAMES if variant.startswith("m1") else FINE15
                total = sum(counts.values())
                for code in ASPRS_CLASSES:
                    members = [
                        i for i, name in enumerate(taxonomy) if FINE_TO_ASPRS.get(name) == code
                    ]
                    if not members:
                        continue
                    q.append(
                        Quantity(
                            f"asprs.{aoi}.{variant}.code{code}.points",
                            f"points_called_class_{code}",
                            Population(
                                "san_francisco", aoi, "voxel", f"asprs:{code}", variant=variant
                            ),
                            "octree_budget.json",
                            lambda a, x=aoi, v=variant, ms=tuple(members): sum(
                                a["octree_budget.json"]["areas"][x]["classes_present"][v].get(
                                    str(i), 0
                                )
                                for i in ms
                            ),
                            "the model's own points under one standard code, over the area the page draws",
                        )
                    )
                    q.append(
                        Quantity(
                            f"asprs.{aoi}.{variant}.code{code}.share",
                            f"share_called_class_{code}",
                            Population(
                                "san_francisco", aoi, "voxel", f"asprs:{code}", variant=variant
                            ),
                            "octree_budget.json",
                            lambda a, x=aoi, v=variant, ms=tuple(members), t=total: (
                                round(
                                    sum(
                                        a["octree_budget.json"]["areas"][x]["classes_present"][
                                            v
                                        ].get(str(i), 0)
                                        for i in ms
                                    )
                                    / t,
                                    4,
                                )
                                if t
                                else None
                            ),
                        )
                    )

    # ---------------- how far above the training heights San Francisco stands ----------------
    # Measured before any training, from the blocks themselves: no model and no labels beyond the
    # ones that pick the blocks. The Financial District is the only area that leaves the range.
    if (ARTIFACTS / "input_range_check.json").exists():
        ve = json.loads((ARTIFACTS / "input_range_check.json").read_text(encoding="utf-8"))[
            "vertical_extrapolation"
        ]
        q.append(
            Quantity(
                "inputs.surrey.train.block_z_max_m",
                "tallest_training_block",
                Population("surrey", "train", "block", "heights"),
                "input_range_check.json",
                lambda a: a["input_range_check.json"]["vertical_extrapolation"][
                    "surrey_training_block_z_max_m"
                ],
                "the tallest block the model saw in training, before any of it was run",
            )
        )
        for aoi in ve["per_aoi"]:
            for part, field in (
                ("blocks", "blocks"),
                ("blocks_above_training_max", "blocks_above_training_max"),
                ("fraction_above_training_max", "fraction_above"),
                ("block_z_max_m", "max_m"),
            ):
                q.append(
                    Quantity(
                        f"inputs.sf.{aoi}.{part}",
                        part,
                        Population("san_francisco", aoi, "block", "heights"),
                        "input_range_check.json",
                        lambda a, x=aoi, f=field: a["input_range_check.json"][
                            "vertical_extrapolation"
                        ]["per_aoi"][x][f],
                    )
                )

    # ---------------- what the model calls the points it had no height for ----------------
    # The class of the tall points, per area and per threshold, from each seed's own calls, and the
    # median and range over the seeds. The page needs it because the colour of a tower is a claim a
    # reader can read off it, and the seeds disagree about it far more than about anything else.
    tall_files = {
        "m0_seedA": "high_points_class.json",
        "m0_seedB": "high_points_class_m0_seedB.json",
        "m0_seedC": "high_points_class_m0_seedC.json",
    }
    tall_files = {v: f for v, f in tall_files.items() if (ARTIFACTS / f).exists()}
    tall_docs = {
        v: json.loads((ARTIFACTS / f).read_text(encoding="utf-8")) for v, f in tall_files.items()
    }
    for variant, doc in tall_docs.items():
        src = tall_files[variant]
        for aoi, row in doc["areas"].items():
            for bin_name, b in row["bins"].items():
                for cls in b["coarse"]:
                    q.append(
                        Quantity(
                            f"tall.sf.{aoi}.{variant}.{bin_name}.called_{cls}",
                            f"share_called_{cls}",
                            Population(
                                "san_francisco",
                                aoi,
                                "voxel",
                                f"height:{bin_name}",
                                "coarse6",
                                variant=variant,
                            ),
                            src,
                            lambda a, s_=src, x=aoi, n=bin_name, c=cls: round(
                                a[s_]["areas"][x]["bins"][n]["coarse"][c]["share"], 4
                            ),
                        )
                    )
                q.append(
                    Quantity(
                        f"tall.sf.{aoi}.{variant}.{bin_name}.points",
                        "points_above_the_threshold",
                        Population(
                            "san_francisco", aoi, "voxel", f"height:{bin_name}", variant=variant
                        ),
                        src,
                        lambda a, s_=src, x=aoi, n=bin_name: a[s_]["areas"][x]["bins"][n]["points"],
                    )
                )
    if len(tall_docs) == 3:
        first = tall_docs["m0_seedA"]
        for aoi, row in first["areas"].items():
            for bin_name in row["bins"]:
                classes = sorted(
                    {
                        c
                        for d in tall_docs.values()
                        for c in d["areas"][aoi]["bins"][bin_name]["coarse"]
                    }
                )
                for cls in classes:
                    for st in ("median", "min", "max"):
                        # A class a seed never called at that height is a share of 0, not a gap.
                        q.append(
                            Quantity(
                                f"tall.sf.{aoi}.m0_seeds.{bin_name}.called_{cls}.{st}",
                                f"share_called_{cls}_{st}_over_seeds",
                                Population(
                                    "san_francisco",
                                    aoi,
                                    "voxel",
                                    f"height:{bin_name}",
                                    "coarse6",
                                    variant="m0_seeds",
                                ),
                                "high_points_class.json",
                                lambda a, x=aoi, n=bin_name, c=cls, t=st: _spread_of(
                                    {
                                        v: a[f]["areas"][x]["bins"][n]["coarse"].get(
                                            c, {"share": 0.0}
                                        )["share"]
                                        for v, f in tall_files.items()
                                    }
                                )[t],
                            )
                        )

    # ---------------- the model's words the standard has no code for ----------------
    # Poles, vehicles and fences are drawn on the first screen in their own colours, and the standard's
    # table has no row for any of them. They are counted the same way, per coarse group.
    if (ARTIFACTS / "octree_budget.json").exists():
        from .asprs import FINE_TO_ASPRS as _F2A
        from .data import CLASS_NAMES as _FINE
        from .data import COARSE as _COARSE

        _budget = json.loads((ARTIFACTS / "octree_budget.json").read_text(encoding="utf-8"))
        for aoi, row in _budget.get("areas", {}).items():
            for variant, counts in (row.get("classes_present") or {}).items():
                if variant.startswith("m1"):
                    continue
                total = sum(counts.values())
                for group, names in _COARSE.items():
                    # Only the members the standard has no code for: powerline is a pole in the page's
                    # colours and a wire in the standard, and is counted under its code, not here.
                    members = tuple(_FINE.index(n) for n in names if n not in _F2A)
                    if not members:
                        continue
                    q.append(
                        Quantity(
                            f"asprs.{aoi}.{variant}.nocode.{group}.points",
                            "points_called_a_class_with_no_code",
                            Population(
                                "san_francisco", aoi, "voxel", f"nocode:{group}", variant=variant
                            ),
                            "octree_budget.json",
                            lambda a, x=aoi, v=variant, ms=members: sum(
                                a["octree_budget.json"]["areas"][x]["classes_present"][v].get(
                                    str(i), 0
                                )
                                for i in ms
                            ),
                        )
                    )
                    q.append(
                        Quantity(
                            f"asprs.{aoi}.{variant}.nocode.{group}.share",
                            "share_called_a_class_with_no_code",
                            Population(
                                "san_francisco", aoi, "voxel", f"nocode:{group}", variant=variant
                            ),
                            "octree_budget.json",
                            lambda a, x=aoi, v=variant, ms=members, t=total: (
                                round(
                                    sum(
                                        a["octree_budget.json"]["areas"][x]["classes_present"][
                                            v
                                        ].get(str(i), 0)
                                        for i in ms
                                    )
                                    / t,
                                    4,
                                )
                                if t
                                else None
                            ),
                        )
                    )

    # ---------------- where the numbers on the first screen come from ----------------
    # The provenance strip names the data rather than the result: when it was flown, how far the
    # imagery is from it, how dense it is, and what the model learned from.
    if (ARTIFACTS / "naip_reference.json").exists():
        nr = json.loads((ARTIFACTS / "naip_reference.json").read_text(encoding="utf-8"))
        q.append(
            Quantity(
                "provenance.sf.flight_date",
                "lidar_flight_date",
                Population("san_francisco", "aois", "raw", "gps_time"),
                "naip_reference.json",
                lambda a: a["naip_reference.json"]["lidar_flight_dates_from_gps_time"]["first"],
                "read from the delivery's own GPS time, not from its metadata",
            )
        )
        q.append(
            Quantity(
                "provenance.sf.naip_year",
                "naip_year",
                Population("san_francisco", "aois", "raw", "imagery"),
                "naip_reference.json",
                lambda a: a["naip_reference.json"]["naip_year"],
            )
        )
        q.append(
            Quantity(
                "provenance.sf.days_from_imagery_to_lidar",
                "days_between_imagery_and_lidar",
                Population("san_francisco", "aois", "raw", "imagery_against_lidar"),
                "naip_reference.json",
                lambda a: a["naip_reference.json"]["days_from_imagery_to_lidar"],
            )
        )
        _ = nr

    # The area's density after the voxel step, from the preparation itself: the tiles every model read
    # and the page draws.
    if (ARTIFACTS / "prep_target.json").exists():
        prep_t = json.loads((ARTIFACTS / "prep_target.json").read_text(encoding="utf-8"))
        for aoi in (t["name"] for t in prep_t["tiles"]):
            q.append(
                Quantity(
                    f"provenance.sf.{aoi}.points_per_m2",
                    "points_per_m2",
                    Population("san_francisco", aoi, "voxel", "all"),
                    "prep_target.json",
                    lambda a, x=aoi: _row(a["prep_target.json"]["tiles"], "name", x)[
                        "density_kept"
                    ],
                    "after this project's voxel step, which is what the page draws",
                )
            )
            # The page draws at most its point budget at once. Spread over the whole area that is the
            # density a reader actually sees, and it is why a frame that fills the screen looks grainy.
            if (ARTIFACTS / "points_per_pixel.json").exists() and (
                ARTIFACTS / "octree_budget.json"
            ).exists():
                q.append(
                    Quantity(
                        f"provenance.sf.{aoi}.drawn_points_per_m2",
                        "points_per_m2_drawn",
                        Population("san_francisco", aoi, "voxel", "drawn_at_the_point_budget"),
                        "points_per_pixel.json",
                        lambda a, x=aoi: round(
                            _row(a["prep_target.json"]["tiles"], "name", x)["density_kept"]
                            * min(
                                1.0,
                                a["points_per_pixel.json"]["budget"]
                                / a["octree_budget.json"]["areas"][x]["points"],
                            ),
                            1,
                        ),
                        "the delivery's density scaled by the share of its points the budget lets the page draw",
                    )
                )

    if (ARTIFACTS / "water_null.json").exists():
        wn = json.loads((ARTIFACTS / "water_null.json").read_text(encoding="utf-8"))
        for row in wn.get("areas", []):
            q.append(
                Quantity(
                    f"provenance.sf.{row['aoi']}.flight_first_utc",
                    "flight_first_utc",
                    Population("san_francisco", row["aoi"], "raw", "gps_time"),
                    "water_null.json",
                    lambda a, x=row["aoi"]: next(
                        r["flight_utc_first"]
                        for r in a["water_null.json"]["areas"]
                        if r["aoi"] == x
                    ),
                )
            )
        _ = wn

    if (ARTIFACTS / "prep_dales.json").exists():
        q.append(
            Quantity(
                "provenance.dales.tiles",
                "dales_tiles",
                Population("surrey", "all", "raw", "all"),
                "prep_dales.json",
                lambda a: len(a["prep_dales.json"]["tiles"]),
                "the DALES-2 tiles this project trained and tested on",
            )
        )
        q.append(
            Quantity(
                "provenance.dales.points",
                "dales_points",
                Population("surrey", "all", "voxel", "all"),
                "prep_dales.json",
                lambda a: a["prep_dales.json"]["points_kept_total"],
                "every tile prepared, the held-out ones included; never what the model was trained on",
            )
        )
    # What the model was trained on and what was kept from it, from the training run's own tile lists.
    if (ARTIFACTS / "train_dales.json").exists():
        q.append(
            Quantity(
                "provenance.dales.train.tiles",
                "dales_tiles_trained_on",
                Population("surrey", "train", "raw", "all"),
                "train_dales.json",
                lambda a: len(a["train_dales.json"]["train_tiles"]),
            )
        )
        q.append(
            Quantity(
                "provenance.dales.test.tiles",
                "dales_tiles_held_out",
                Population("surrey", "test", "raw", "all"),
                "train_dales.json",
                lambda a: len(a["train_dales.json"]["test_tiles"]),
            )
        )

    # ---------------- the capacity control, on a rented card ----------------
    for variant in RENTED_VARIANTS:
        art = f"train_{variant}.json"
        if not (ARTIFACTS / art).exists():
            continue
        blob = json.loads((ARTIFACTS / art).read_text(encoding="utf-8"))
        for key, getter in (
            ("in_domain_accuracy", lambda b: b["final"]["accuracy"]),
            ("parameters", lambda b: b["parameters"]),
            ("points_seen", lambda b: b["points_seen"]),
            ("wall_seconds", lambda b: b["wall_seconds"]),
        ):
            q.append(
                Quantity(
                    f"rented.{variant}.{key}",
                    key,
                    Population("surrey", "test", "voxel", "labelled", "fine15", variant=variant),
                    art,
                    lambda a, ar=art, g=getter: g(a[ar]),
                    "trained on a rented RTX 4090; the artefact carries the card's name and memory",
                )
            )
        _ = blob

    if (ARTIFACTS / "compare_rented.json").exists():
        cr = json.loads((ARTIFACTS / "compare_rented.json").read_text(encoding="utf-8"))
        src = "compare_rented.json"
        if cr.get("run_1_machine_check", {}).get("outside_count") is not None:
            for key in ("outside_count", "measures_compared"):
                q.append(
                    Quantity(
                        f"rented.machine_check.{key}",
                        key,
                        Population(
                            "san_francisco",
                            "aois",
                            "voxel",
                            "same_recipe_two_machines",
                            variant="m0_rented_small_seedA",
                        ),
                        src,
                        lambda a, k=key: a["compare_rented.json"]["run_1_machine_check"][k],
                        "how far the same recipe moves when only the card changes",
                    )
                )
        if cr.get("run_2_upgrade"):
            q.append(
                Quantity(
                    "rented.upgrade.areas_above",
                    "areas_above_the_small_seed_band",
                    Population(
                        "san_francisco",
                        "footprint_areas",
                        "voxel",
                        "building_lift",
                        variant="m0_big_seedA",
                    ),
                    src,
                    lambda a: len(a["compare_rented.json"]["run_2_upgrade"]["areas_above"]),
                    "the upgrade rule needs 4 of 6; written before the runs",
                )
            )
            q.append(
                Quantity(
                    "rented.upgrade.rule_met",
                    "criterion_met",
                    Population(
                        "san_francisco",
                        "footprint_areas",
                        "voxel",
                        "building_lift",
                        variant="m0_big_seedA",
                    ),
                    src,
                    lambda a: int(bool(a["compare_rented.json"]["run_2_upgrade"]["rule_met"])),
                )
            )
    # ---------------- the density the model actually sees ----------------
    # No model in it. The crop is a fixed length, so a source denser than the crop is thinned by the
    # crop itself: this is the number that decides whether two sources look alike to the model.
    if (ARTIFACTS / "crop_density.json").exists():
        cd = json.loads((ARTIFACTS / "crop_density.json").read_text(encoding="utf-8"))
        city_of = {
            "surrey_dales_training": "surrey",
            "swiss_training": "switzerland",
            "swiss_held_out": "switzerland",
            "san_francisco": "san_francisco",
        }
        for name in cd.get("sources", {}):
            for part in (
                "source_points_per_m2_median",
                "crop_points_per_m2_median",
                "distinct_points_in_a_crop_median",
            ):
                q.append(
                    Quantity(
                        f"crop_density.{name}.{part}",
                        part,
                        Population(city_of.get(name, name), name, "voxel", "blocks_sampled"),
                        "crop_density.json",
                        lambda a, n=name, pt=part: a["crop_density.json"]["sources"][n][pt],
                        "what a 40 m crop carries, which is the density the model reads",
                    )
                )
        q.append(
            Quantity(
                "crop_density.ceiling_points_per_m2",
                "crop_ceiling",
                Population("none", "a_40m_block", "voxel", "crop_length"),
                "crop_density.json",
                lambda a: a["crop_density.json"]["crop_ceiling_points_per_m2"],
                "the most a crop can carry: its length over the block's area",
            )
        )

    # ---------------- the fifteen classes under the same thinning ----------------
    if (ARTIFACTS / "density_classes_test.json").exists():
        dc = json.loads((ARTIFACTS / "density_classes_test.json").read_text(encoding="utf-8"))
        for aoi, row in dc.get("areas", {}).items():
            for run, values in row["runs"].items():
                for part in (
                    "ground_recall",
                    "vegetation_core_called_vegetation",
                    "building_precision_lift_over_null",
                    "distinct_points_in_a_crop_median",
                    "knn_32_radius_m_median_over_blocks",
                ):
                    if values.get(part) is None:
                        continue
                    q.append(
                        Quantity(
                            f"density_classes.{aoi}.{run}.{part}",
                            f"{part}_at_{run}",
                            Population(
                                "san_francisco",
                                aoi,
                                f"voxel_crop_{run}",
                                "delivery_and_naip",
                                variant=dc["variant"],
                            ),
                            "density_classes_test.json",
                            lambda a, x=aoi, r=run, pt=part: a["density_classes_test.json"][
                                "areas"
                            ][x]["runs"][r][pt],
                            "the published model over the same blocks at a different crop density",
                        )
                    )
        q.append(
            Quantity(
                "density_classes.surrey_reach_m",
                "knn_32_radius_m_median",
                Population("surrey", "train", "voxel", "blocks_sampled"),
                "density_classes_test.json",
                lambda a: a["density_classes_test.json"]["surrey_reach"]["knn_32_radius_m_median"],
                "how far the grouping reaches on the tiles the model trained on",
            )
        )

    # ---------------- the same thinning, walked across densities ----------------
    if (ARTIFACTS / "water_density_test.json").exists():
        wd_all = json.loads((ARTIFACTS / "water_density_test.json").read_text(encoding="utf-8"))
        for keep, areas in wd_all.get("sweep", {}).items():
            for aoi, row in areas.items():
                for part in ("water_recall", "water_precision", "ground_recall"):
                    if row.get(part) is None:
                        continue
                    q.append(
                        Quantity(
                            f"density.sweep.{keep.replace('.', '_')}.{aoi}.{part}",
                            f"{part}_at_a_thinned_density",
                            Population(
                                "san_francisco",
                                aoi,
                                f"voxel_keep_{keep.replace('.', '_')}",
                                "delivery_water",
                                variant=wd_all["variant"],
                            ),
                            "water_density_test.json",
                            lambda a, kp=keep, x=aoi, pt=part: a["water_density_test.json"][
                                "sweep"
                            ][kp][x][pt],
                            "the curve behind the claim that spacing is what moved water",
                        )
                    )

    # ---------------- the confidence to chroma map, fixed on Surrey ----------------
    if (ARTIFACTS / "confidence_mapping.json").exists():
        for end in ("low", "high"):
            q.append(
                Quantity(
                    f"confidence_map.{end}",
                    f"confidence_map_{end}",
                    Population(
                        "surrey",
                        "test",
                        "voxel",
                        f"labelled:{'wrong_p5' if end == 'low' else 'right_p95'}",
                        "fine15",
                        variant="m0_seedA",
                    ),
                    "confidence_mapping.json",
                    lambda a, e=end: a["confidence_mapping.json"][e],
                    "fixed before any San Francisco frame used it; the same map for every model",
                )
            )
        cm = json.loads((ARTIFACTS / "confidence_mapping.json").read_text(encoding="utf-8"))
        for variant, rows in cm.get("san_francisco_reading", {}).items():
            for aoi in rows:
                q.append(
                    Quantity(
                        f"confidence_map.position.{variant}.sf.{aoi}",
                        "median_confidence_on_map",
                        Population(
                            "san_francisco",
                            aoi,
                            "voxel",
                            "all:median_confidence_position",
                            variant=variant,
                        ),
                        "confidence_mapping.json",
                        lambda a, v=variant, x=aoi: a["confidence_mapping.json"][
                            "san_francisco_reading"
                        ][v][x]["position_on_map"],
                    )
                )

    # ---------------- the prototype against its thresholds, bound to one device ----------------
    # Never a README figure without the device beside it: the population's city is "device"
    # and its split names the WebGL renderer the browser actually used.
    if (ARTIFACTS / "prototype_measurements.json").exists():
        pm = json.loads((ARTIFACTS / "prototype_measurements.json").read_text(encoding="utf-8"))
        renderer = (
            "webgl:" + "".join(ch if ch.isalnum() else "_" for ch in pm["webgl_renderer"])[:60]
        )
        for key in pm["runs"]:
            for part in ("median", "threshold"):
                q.append(
                    Quantity(
                        f"prototype.{key}.{part}",
                        f"prototype_{part}",
                        Population(
                            "device",
                            renderer,
                            "chromium_headless" if pm["headless"] else "chromium_headed",
                            key,
                        ),
                        "prototype_measurements.json",
                        lambda a, k=key, pt=part: a["prototype_measurements.json"]["runs"][k][pt],
                        f"device bound: {pm['device']['gpu'].split(',')[0]} laptop, WebGL on {pm['webgl_renderer']}",
                    )
                )

    # ---------------- M1 data, no model involved ----------------
    if (ARTIFACTS / "prep_swiss.json").exists():
        q.append(
            Quantity(
                "m1data.swiss.sub_tiles",
                "sub_tiles",
                Population("zurich_geneva", "train", "voxel"),
                "prep_swiss.json",
                lambda a: len(a["prep_swiss.json"]["tiles"]),
            )
        )
        q.append(
            Quantity(
                "m1data.swiss.points_kept",
                "points",
                Population("zurich_geneva", "train", "voxel"),
                "prep_swiss.json",
                lambda a: a["prep_swiss.json"]["points_kept_total"],
            )
        )
    if (ARTIFACTS / "dales_water_check.json").exists():
        q.append(
            Quantity(
                "m1data.dales.tiles_with_water",
                "tiles_with_water",
                Population("surrey", "all", "raw", "fwa_water"),
                "dales_water_check.json",
                lambda a: a["dales_water_check.json"]["tiles_with_water"],
            )
        )
    if (ARTIFACTS / "naip_mask_choice.json").exists():
        q.append(
            Quantity(
                "naipchoice.ggpark.not_vegetation_core_multi_return_unregistered",
                "multi_return_share",
                Population(
                    "san_francisco", "ggpark", "voxel", "naip_not_vegetation_core_unregistered"
                ),
                "naip_mask_choice.json",
                lambda a: _row(a["naip_mask_choice.json"]["areas"], "aoi", "ggpark")[
                    "unregistered"
                ]["multi_return_share_not_vegetation_core"],
            )
        )

    # ---------------- where the lidar really is ----------------
    q.append(
        Quantity(
            "datum.sf.zero_shift_worst_abs_difference",
            "datum_path_residual_zero_shift",
            Population("san_francisco", "delivery", "source_headers"),
            "datum_check.json",
            lambda a: a["datum_check.json"]["zero_shift_worst_abs_difference_mercator_units"],
            "EPT published bounds against a zero-shift reprojection of each source file's corners",
        )
    )
    q.append(
        Quantity(
            "datum.sf.itrf_smallest_worst_abs_difference_ground_m",
            "datum_path_residual_itrf2014_ground_m",
            Population("san_francisco", "delivery", "source_headers"),
            "datum_check.json",
            lambda a: a["datum_check.json"]["itrf2014_smallest_worst_abs_difference_ground_m"],
        )
    )
    q.append(
        Quantity(
            "datum.sf.itrf_smallest_worst_abs_difference",
            "datum_path_residual_itrf2014",
            Population("san_francisco", "delivery", "source_headers"),
            "datum_check.json",
            lambda a: a["datum_check.json"][
                "itrf2014_smallest_worst_abs_difference_mercator_units"
            ],
        )
    )
    for axis in ("east_m", "north_m"):
        q.append(
            Quantity(
                f"crs.sf.footprint_offset_global.{axis}",
                f"footprint_offset_{axis}",
                Population("san_francisco", "aois", "voxel", "footprints_vs_low_raised_lidar"),
                "crs_check.json",
                lambda a, ax=axis: round(_footprint_global(a)[ax], 3),
                "per-axis median over areas whose correlation cleared the peak bar",
            )
        )

    # ---------------- test counts, from a real collection ----------------
    for suite in ("unit", "integrity"):
        q.append(
            Quantity(
                f"tests.{suite}.functions",
                "tests_functions",
                Population("repository", suite, "none"),
                "tests.json",
                lambda a, s=suite: a["tests.json"]["suites"][s].get("functions"),
                "test functions, each counted once whatever its parametrisation",
            )
        )
        q.append(
            Quantity(
                f"tests.{suite}.collected",
                "tests_collected",
                Population("repository", suite, "none"),
                "tests.json",
                lambda a, s=suite: a["tests.json"]["suites"][s]["collected"],
            )
        )
        q.append(
            Quantity(
                f"tests.{suite}.passed",
                "tests_passed",
                Population("repository", suite, "none"),
                "tests.json",
                lambda a, s=suite: a["tests.json"]["suites"][s]["passed"],
            )
        )
    # ---------------- Phase 4: voting, real water, flags, confounds, buildings ----------------
    if (ARTIFACTS / "votes.json").exists():
        q.append(
            Quantity(
                "votes.n",
                "voting_passes",
                Population("surrey", "test", "voxel", "labelled", "coarse6"),
                "votes.json",
                lambda a: a["votes.json"]["votes"],
                "chosen by the rule in docs/phase4_criteria.md",
            )
        )
        for n in ("1", "2", "4", "8"):
            q.append(
                Quantity(
                    f"votes.agreement_{n}",
                    "vote_agreement_n_with_2n",
                    Population("surrey", "test", "voxel", f"labelled_n{n}", "coarse6"),
                    "votes.json",
                    lambda a, n=n: a["votes.json"]["agreement_n_with_2n"][n],
                )
            )
    if (ARTIFACTS / "real_water_density.json").exists():
        rw = json.loads((ARTIFACTS / "real_water_density.json").read_text(encoding="utf-8"))
        for side, city in (("swiss", "switzerland"), ("san_francisco", "san_francisco")):
            for area in rw[side]:
                # "zug 2680-1225, held out" names the tile and says it was held out; the key keeps the tile.
                slug = area.split()[0] + (area.split()[1].rstrip(",") if side == "swiss" else "")
                slug = slug.replace("-", "_")
                for part in ("source", "prepared"):
                    q.append(
                        Quantity(
                            f"realwater.{slug}.{part}.density_median",
                            "real_water_density_median",
                            Population(
                                city, slug, "raw" if part == "source" else "voxel", "water_blocks"
                            ),
                            "real_water_density.json",
                            lambda a, s_=side, ar=area, pt=part: a["real_water_density.json"][s_][
                                ar
                            ][pt]["density_p10_p50_p90"][1],
                        )
                    )
    if (ARTIFACTS / "las_flags_audit.json").exists():
        fa = json.loads((ARTIFACTS / "las_flags_audit.json").read_text(encoding="utf-8"))
        for st, pick_ in (("min", min), ("max", max)):
            q.append(
                Quantity(
                    f"flags.swiss_training.water.synthetic.{st}",
                    f"synthetic_share_of_water_{st}_over_tiles",
                    Population("switzerland", "train", "raw", "class_9"),
                    "las_flags_audit.json",
                    lambda a, f_=pick_: round(
                        f_(
                            t["per_class"]["9"]["share_synthetic"]
                            for t in a["las_flags_audit.json"]["swiss_training"][
                                "per_tile"
                            ].values()
                        ),
                        4,
                    ),
                    "over the Swiss tiles M1 was trained on",
                )
            )
        for group in ("swiss_training", "swiss_held_out"):
            for tile in fa[group]["per_tile"]:
                q.append(
                    Quantity(
                        f"flags.{tile.replace('-', '_')}.water.synthetic",
                        "synthetic_share_of_water",
                        Population("switzerland", tile, "raw", "class_9"),
                        "las_flags_audit.json",
                        lambda a, g=group, t=tile: round(
                            a["las_flags_audit.json"][g]["per_tile"][t]["per_class"]["9"][
                                "share_synthetic"
                            ],
                            4,
                        ),
                    )
                )
    for art, split in (("prep_swiss.json", "train"), ("prep_swiss_holdout.json", "held_out")):
        if (ARTIFACTS / art).exists():
            q.append(
                Quantity(
                    f"swisswater.{split}.real_points_kept",
                    "real_water_points_kept",
                    Population("switzerland", split, "voxel", "class_9_not_synthetic"),
                    art,
                    lambda a, s_=art: sum(t.get("water_points_kept", 0) for t in a[s_]["tiles"]),
                )
            )
    if (ARTIFACTS / "input_range_check.json").exists():
        q.append(
            Quantity(
                "confounds.training_p99.block_z_range_m",
                "block_height_range_p99",
                Population("surrey", "train", "voxel", "blocks"),
                "input_range_check.json",
                lambda a: a["input_range_check.json"]["confounds_at_the_99th_percentile"][
                    "training_p99"
                ]["block_z_range_m"],
            )
        )
        for aoi in SF_AOIS:
            q.append(
                Quantity(
                    f"confounds.{aoi}.block_z_range_m.median",
                    "block_height_range_median",
                    Population("san_francisco", aoi, "voxel", "blocks"),
                    "input_range_check.json",
                    lambda a, x=aoi: a["input_range_check.json"][
                        "confounds_at_the_99th_percentile"
                    ]["per_aoi"][x]["block_z_range_m"]["median"],
                )
            )
    if (ARTIFACTS / "building_breakdown.json").exists():
        bb = json.loads((ARTIFACTS / "building_breakdown.json").read_text(encoding="utf-8"))
        for aoi, row in bb["areas"].items():
            for fig, cell in row.items():
                if not isinstance(cell, dict) or "old" not in cell:
                    continue
                for when in ("old", "new"):
                    if not cell[when]:
                        continue
                    for st in ("median", "min", "max"):
                        q.append(
                            Quantity(
                                f"buildings.{aoi}.{fig}.{when}.{st}",
                                f"building_{fig}_{st}",
                                Population(
                                    "san_francisco",
                                    aoi,
                                    "voxel",
                                    f"footprints_{when}_m0",
                                    variant="m0_seeds",
                                ),
                                "building_breakdown.json",
                                lambda a, x=aoi, f=fig, w=when, t=st: a["building_breakdown.json"][
                                    "areas"
                                ][x][f][w][t],
                            )
                        )
    big = ARTIFACTS / "sf_scores_m0_big_seedA.json"
    if big.exists() and (ARTIFACTS / "seeds_m0.json").exists():

        def _capacity(a: dict) -> int:
            rows = {r["aoi"]: r for r in a["sf_scores_m0_big_seedA.json"]["san_francisco_per_area"]}
            small = a["seeds_m0.json"]["san_francisco"]
            count = 0
            for aoi, r in rows.items():
                lift = r["building_vs_footprints_over_2m"]["global_shift"][
                    "precision_lift_over_null"
                ]
                top = small.get(aoi, {}).get("building_lift_global_shift", {}).get("max")
                if lift is not None and top is not None and lift > top:
                    count += 1
            return count

        q.append(
            Quantity(
                "capacity.areas_big_above_small_max",
                "capacity_rule_count",
                Population(
                    "san_francisco", "aois", "voxel", "clean_over_2m", variant="m0_big_seedA"
                ),
                "sf_scores_m0_big_seedA.json",
                _capacity,
                "areas where the large M0's building lift exceeds the max of the three small seeds",
            )
        )
        q.append(
            Quantity(
                "capacity.areas_with_footprints",
                "capacity_rule_areas",
                Population("san_francisco", "aois", "voxel", "clean_over_2m", variant="m0_seeds"),
                "seeds_m0.json",
                lambda a: sum(
                    1
                    for r in a["seeds_m0.json"]["san_francisco"].values()
                    if (r.get("building_null_precision") or {}).get("median")
                ),
                "areas where the footprint null calls anything a building, so a lift can be read",
            )
        )
    # The two inference-only diagnoses of the building drop, beside the four-vote prediction they are
    # read against: one pass instead of the chosen vote, and one pass by the first round's iterator.
    if (ARTIFACTS / "building_breakdown.json").exists():
        seed_a = {"four_votes": "m0_seedA", **{d: d for d in bb.get("diagnostics", {})}}
        for aoi, row in bb["areas"].items():
            for fig, cell in row.items():
                if not isinstance(cell, dict) or "seed_a" not in cell:
                    continue
                for run, variant in seed_a.items():
                    if cell["seed_a"].get(run) is None:
                        continue
                    q.append(
                        Quantity(
                            f"buildings.{aoi}.{fig}.seed_a.{run}",
                            f"building_{fig}",
                            Population(
                                "san_francisco", aoi, "voxel", "footprints", variant=variant
                            ),
                            "building_breakdown.json",
                            lambda a, x=aoi, f=fig, r_=run: a["building_breakdown.json"]["areas"][
                                x
                            ][f]["seed_a"][r_],
                        )
                    )
    for variant in ("m0_seedA_1vote", "m0_seedA_legacy_iter"):
        src = f"sf_scores_{variant}.json"
        if not (ARTIFACTS / src).exists():
            continue
        for part in ("accuracy_labelled_voxel_points", "miou_15", "miou_coarse"):
            q.append(
                Quantity(
                    f"diagnostic.{variant}.surrey.test.{part}",
                    part,
                    Population(
                        "surrey",
                        "test",
                        "voxel",
                        "labelled",
                        "coarse6" if part == "miou_coarse" else "fine15",
                        variant=variant,
                    ),
                    src,
                    lambda a, s_=src, p_=part: a[s_]["surrey_held_out"][p_],
                )
            )
    # Where the Phase 4 models were trained, and how large each is, from their own training artefacts.
    for variant in ("m0_seedA", "m0_seedB", "m0_seedC", "m0_big_seedA"):
        src = f"train_{variant}.json"
        if not (ARTIFACTS / src).exists():
            continue
        q.append(
            Quantity(
                f"train.{variant}.parameters",
                "parameters",
                Population("surrey", "train", "voxel", "all", variant=variant),
                src,
                lambda a, s_=src: a[s_]["parameters"],
            )
        )
    # ---------------- the NAIP not-vegetation cores, no model involved ----------------
    if (ARTIFACTS / "not_vegetation_heights.json").exists():
        nv = json.loads((ARTIFACTS / "not_vegetation_heights.json").read_text(encoding="utf-8"))
        for aoi, row in nv["areas"].items():
            parts: list[tuple[str, tuple[str, ...]]] = [
                ("points", ("points",)),
                ("share_of_area", ("share_of_area",)),
            ]
            if row["points"]:
                parts.append(("height_median_m", ("height_above_floor_m", "median")))
            for part, path in parts:
                q.append(
                    Quantity(
                        f"notveg.sf.{aoi}.{part}",
                        f"not_vegetation_core_{part}",
                        Population("san_francisco", aoi, "voxel", "naip_not_vegetation_core"),
                        "not_vegetation_heights.json",
                        lambda a, x=aoi, p_=path: _dig(
                            a["not_vegetation_heights.json"]["areas"][x], p_
                        ),
                    )
                )
    return with_variants(q)


REGISTRY = build_registry()


@dataclass(frozen=True)
class Relation:
    """Something that must hold between registered quantities."""

    name: str
    check: Callable[[dict, dict], tuple[bool, str]]


def resolve(quantities: list[Quantity], artifacts: dict) -> dict[str, dict]:
    out = {}
    for qty in quantities:
        if qty.source not in artifacts:
            value, error = None, f"{qty.source} not present"
        else:
            try:
                value, error = qty.getter(artifacts), None
            except (KeyError, TypeError, IndexError, StopIteration) as e:
                value, error = None, f"{type(e).__name__}: {e}"
        out[qty.key] = {
            "value": value,
            "concept": qty.concept,
            "population": qty.population.__dict__,
            "source": qty.source,
            "doc": qty.doc,
            "error": error,
        }
    return out


# Pairs a reader will put side by side. "same" means the populations must be
# identical; "cross_domain" means they may differ in city and split only, because
# comparing across the two cities is the point, but never in sampling or subset.
COMPARISONS = (
    [
        ("model.surrey.test.voxel.accuracy", "model.surrey.test.voxel.do_nothing_accuracy", "same"),
        (
            "model.surrey.test.voxel.called_ground.auroc",
            "model.sf.embarcadero.voxel.called_ground.auroc",
            "cross_domain",
        ),
        (
            "control.surrey.test.voxel.called_ground_hard.auroc",
            "control.sf.embarcadero.voxel.called_ground_hard.auroc",
            "cross_domain",
        ),
    ]
    + [
        (
            "control.surrey.test.voxel.called_ground_easy.auroc",
            f"control.sf.{a}.voxel.called_ground_easy.auroc",
            "cross_domain",
        )
        for a in SF_AOIS
        if a not in ("steep", "baybridge")
    ]
    + [
        (f"model.sf.{a}.building_precision", f"sf.{a}.footprint_null_precision", "same")
        for a in SF_AOIS
    ]
)
