"""The scoring changes docs/phase6_criteria.md fixed before Phase 6 ran, each with the case it must not take.

Ground is judged wherever the delivery labels the point itself, the delivery's ignored ground counts as
ground, a null that calls nothing a building leaves no lift, and a move between two sets of seeds is set
beside the chance of one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.reference import score_ground  # noqa: E402
from surrey.seeds import chance_of_a_move, expected_by_chance  # noqa: E402


def load_scorer():
    sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
    spec = importlib.util.spec_from_file_location(
        "score_variant", ROOT / "scripts" / "pipeline" / "score_variant.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shore(n_side: int = 40):
    """Ground on the left half at z = 0, water on the right half at z = 0, far from any ground."""
    gx, gy = np.meshgrid(np.arange(n_side, dtype=float), np.arange(n_side, dtype=float))
    x, y = gx.ravel() * 5.0, gy.ravel() * 5.0
    z = np.zeros(x.size)
    asprs = np.where(x < n_side * 2.5, 2, 9).astype(np.int16)
    return x, y, z, asprs


def test_water_called_ground_counts_against_precision_even_beyond_the_surface():
    x, y, z, asprs = shore()
    everything_ground = np.ones(x.size, bool)
    r = score_ground(asprs, x, y, z, everything_ground, reach_m=10.0)
    # Half the calls are on water the delivery labelled: judged, and wrong.
    assert r["predicted_points_unjudgeable_fraction"] == 0.0
    assert r["precision_floor"] == pytest.approx(0.5, abs=0.01)


def test_an_unassigned_point_beyond_the_surface_is_still_not_judged():
    x, y, z, asprs = shore()
    asprs = np.where(asprs == 9, 1, asprs).astype(np.int16)
    r = score_ground(asprs, x, y, z, np.ones(x.size, bool), reach_m=10.0)
    # The unassigned points beyond reach are declared, not scored; the ones on the shore's surface
    # are wrong on the floor and right on the ceiling, which is the band's definition.
    assert r["predicted_points_unjudgeable_fraction"] == pytest.approx(0.45)
    assert r["precision_ceiling"] == pytest.approx(1.0)
    assert r["precision_floor"] < 1.0


def test_ignored_ground_counts_as_ground():
    x, y, z, asprs = shore()
    asprs = np.where(asprs == 9, 20, asprs).astype(np.int16)
    r = score_ground(asprs, x, y, z, np.ones(x.size, bool))
    assert r["recall_clean"] == pytest.approx(1.0)
    assert r["precision_floor"] == pytest.approx(1.0)


def test_no_lift_over_a_null_that_calls_nothing_a_building():
    building = load_scorer().building
    n = 5000
    eligible = np.ones(n, bool)
    predicted = np.zeros(n, bool)
    predicted[:100] = True
    none_inside = building(np.zeros(n, bool), predicted, eligible)
    assert none_inside["precision_lift_over_null"] is None
    half_inside = np.zeros(n, bool)
    half_inside[:2500] = True
    assert building(half_inside, predicted, eligible)["precision_lift_over_null"] == pytest.approx(
        1.0
    )


def test_the_chance_of_a_move_between_three_seeds_and_three():
    assert chance_of_a_move(3, 3) == pytest.approx(0.1)
    assert chance_of_a_move(1, 1) == pytest.approx(1.0)
    assert expected_by_chance(71, 3, 3) == pytest.approx(7.1)
