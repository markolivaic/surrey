"""Seed aggregation, the water null and the M1 criteria, all written before seed C."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.m1_criteria import WATER_AREAS, water_success  # noqa: E402
from surrey.seeds import PUBLISHED_SEED, drop, gain, holds, spread  # noqa: E402
from surrey.sources import BRIDGE_DECK, WATER, class_names_for, recipe_ignore_mask  # noqa: E402
from surrey.water_null import (  # noqa: E402
    gps_adjusted_to_utc,
    level_at,
    lift_over_null,
    null_water,
    water_scores,
)


def test_a_claim_stands_only_on_every_seed():
    r = holds(
        {"A": "predicts degradation", "B": "predicts degradation", "C": "measured nothing"},
        "predicts degradation",
    )
    assert (r["k"], r["n"], r["stands"], r["phrase"]) == (2, 3, False, "on 2 of 3 seeds")
    r = holds({"A": "predicts degradation", "B": "predicts degradation"}, "predicts degradation")
    assert r["stands"] and r["phrase"] == "on all 2 seeds"
    assert holds({"A": None}, "x")["phrase"] == "not measured"


def test_spread_is_median_min_max_over_seeds_that_have_a_value():
    s = spread({"A": 0.2384, "B": 0.3307, "C": None})
    assert (s["n"], s["min"], s["max"]) == (2, 0.2384, 0.3307)
    assert s["median"] == pytest.approx(0.28455, abs=1e-4)


def test_published_seed_is_fixed():
    assert PUBLISHED_SEED == "A"


def test_drop_needs_every_new_seed_below_every_old_seed():
    assert drop([0.5, 0.55, 0.59], [0.6, 0.7, 0.65], higher_is_better=True) is True
    assert drop([0.5, 0.55, 0.61], [0.6, 0.7, 0.65], higher_is_better=True) is False
    assert drop([0.3, 0.4], [0.1, 0.2], higher_is_better=False) is True
    assert drop([0.15, 0.4], [0.1, 0.2], higher_is_better=False) is False
    assert gain([0.7, 0.8], [0.5, 0.6], higher_is_better=True) is True
    assert drop([None], [0.1], True) is None


def test_m0prime_ignores_swiss_water_and_deck_and_m1_keeps_them():
    lab = np.array([0, WATER, BRIDGE_DECK, 100, -1], np.int8)
    assert recipe_ignore_mask("m0prime", lab).tolist() == [False, True, True, False, False]
    assert recipe_ignore_mask("m1", lab) is None and recipe_ignore_mask("m0", lab) is None
    assert len(class_names_for("m0prime")) == 15 and len(class_names_for("m1")) == 17
    with pytest.raises(ValueError):
        class_names_for("m2")


def test_gps_time_converts_to_the_flight_day():
    import datetime as dt

    # 2023-04-20 12:00:00 UTC in adjusted standard GPS time.
    utc = dt.datetime(2023, 4, 20, 12, tzinfo=dt.UTC).timestamp()
    gps_adj = utc - dt.datetime(1980, 1, 6, tzinfo=dt.UTC).timestamp() + 18 - 1e9
    assert gps_adjusted_to_utc(np.array([gps_adj]))[0] == pytest.approx(utc)


def test_level_interpolates_and_refuses_to_extrapolate():
    gt = np.array([0.0, 360.0, 720.0])
    gl = np.array([1.0, 1.6, 1.0])
    assert level_at(np.array([180.0, 540.0]), gt, gl).tolist() == pytest.approx([1.3, 1.3])
    with pytest.raises(ValueError):
        level_at(np.array([800.0]), gt, gl)


def test_null_and_scores_on_a_known_answer():
    z = np.array([0.0, 1.4, 1.6, 3.0, 0.2])
    level = np.full(5, 1.0)
    called = null_water(z, level, 0.5)
    assert called.tolist() == [True, True, False, False, True]
    truth = np.array([True, True, True, False, False])
    s = water_scores(called, truth, np.ones(5, bool))
    assert (s["precision"], s["recall"], s["iou"]) == (round(2 / 3, 4), round(2 / 3, 4), 0.5)
    assert lift_over_null(0.75, 0.5) == 0.5
    assert lift_over_null(0.4, 0.5) == -0.2


def test_water_success_needs_all_three_seeds_on_both_areas():
    good = {a: {"A": 0.1, "B": 0.2, "C": 0.05} for a in WATER_AREAS}
    assert water_success(good)["success"]
    two = {a: dict(v) for a, v in good.items()}
    two["baybridge"]["C"] = -0.01
    r = water_success(two)
    assert not r["success"] and r["per_area"]["baybridge"]["phrase"] == "positive on 2 of 3 seeds"
    short = {a: {"A": 0.1, "B": 0.2} for a in WATER_AREAS}
    assert not water_success(short)["success"]


def test_an_undefined_lift_is_a_seed_that_ran_and_is_named():
    lifts = {a: {"A": 0.1, "B": 0.2, "C": None} for a in WATER_AREAS}
    r = water_success(lifts)["per_area"]["baybridge"]
    assert r["seeds"] == 3 and r["undefined_on"] == 1 and not r["success"]
    assert r["phrase"] == "positive on 2 of 3 seeds, undefined on 1 (the null scores IoU 1)"


def test_counting_undefined_seeds_leaves_success_where_the_written_rule_put_it():
    import itertools

    def old_rule(seeds):
        have = [v for v in seeds.values() if v is not None]
        return len(have) == 3 and all(v > 0 for v in have)

    values = (0.1, -0.1, None)
    for n in (2, 3):
        for combo in itertools.product(values, repeat=n):
            seeds = dict(zip("ABC", combo, strict=False))
            got = water_success(dict.fromkeys(WATER_AREAS, seeds))["per_area"]["embarcadero"][
                "success"
            ]
            assert got == old_rule(seeds), seeds
