"""The target's ground class as a reference, and the band that honesty requires.

The delivery labels some ground and leaves other ground unlabelled, so a single
precision would punish the model for the delivery's own gaps. These pin the shape
of the answer: recall clean, precision bounded, and the bound the right way round.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.reference import score_ground, vendor_ground_surface  # noqa: E402


def flat_scene(n_side: int = 40):
    """A flat street at z=0 with a 10 m block of roof at z=12."""
    gx, gy = np.meshgrid(np.arange(n_side, dtype=float), np.arange(n_side, dtype=float))
    x, y = gx.ravel(), gy.ravel()
    z = np.zeros(x.size)
    roof = (x >= 15) & (x < 25) & (y >= 15) & (y < 25)
    z[roof] = 12.0
    return x, y, z, roof


def test_the_surface_reaches_a_short_way_into_blank_cells():
    # A roof sits a few metres from labelled street. Refusing to judge it would
    # throw away an easy call, so the surface is carried in within reach.
    x, y, z, roof = flat_scene()
    grid, key = vendor_ground_surface(x, y, z, ~roof, cell_m=2.0, reach_m=25.0)
    assert np.isfinite(grid[key[roof]]).all(), "a roof beside labelled street is judgeable"


def test_the_surface_stops_where_there_is_nothing_to_reach_from():
    # With no reach, a cell the vendor never labelled stays undefined. This is the
    # middle of the bay: there is no ground nearby to carry in.
    x, y, z, roof = flat_scene()
    grid, key = vendor_ground_surface(x, y, z, ~roof, cell_m=2.0, reach_m=0.0)
    assert np.isnan(grid[key[roof]]).any(), "blank cells beyond reach must stay undefined"
    assert np.isfinite(grid[key[~roof]]).all()


def test_recall_is_clean_and_counts_only_the_labelled_ground():
    x, y, z, roof = flat_scene()
    asprs = np.where(roof, 1, 2).astype(np.int16)
    r = score_ground(asprs, x, y, z, predicted_ground=~roof)
    assert r["recall_clean"] == pytest.approx(1.0)


def test_precision_ceiling_is_at_least_the_floor():
    x, y, z, roof = flat_scene()
    asprs = np.where(roof, 1, 2).astype(np.int16)
    r = score_ground(asprs, x, y, z, predicted_ground=~roof)
    assert r["precision_ceiling"] >= r["precision_floor"]


def test_unlabelled_ground_lifts_the_ceiling_but_not_the_floor():
    # This is the case the band exists for. The vendor labelled only every other
    # street point, so each cell still holds some class 2 and the surface exists.
    # A model that finds all the street is right, and a single precision would
    # call half its answer wrong.
    x, y, z, roof = flat_scene()
    asprs = np.full(x.size, 2, dtype=np.int16)
    asprs[roof] = 1
    street = ~roof
    unlabelled = street & (np.arange(x.size) % 2 == 0)
    asprs[unlabelled] = 1  # real ground, vendor never labelled it

    r = score_ground(asprs, x, y, z, predicted_ground=street)
    assert r["precision_floor"] < 0.75, "the floor should undercount here"
    assert r["precision_ceiling"] > 0.95, "the ceiling should recognise the real ground"
    assert r["unassigned_at_ground_level"] > 0.5


def test_a_region_the_vendor_never_touched_is_declared_rather_than_scored():
    # Where a whole area carries no class 2, there is no surface to judge against.
    # Counting those predictions wrong would charge the model for the delivery's
    # silence, so they leave the denominator and the fraction is reported.
    x, y, z, roof = flat_scene()
    asprs = np.full(x.size, 2, dtype=np.int16)
    asprs[roof] = 1
    street = ~roof
    asprs[street & (x < 20)] = 1  # an entire half left unlabelled

    r = score_ground(asprs, x, y, z, predicted_ground=street, reach_m=0.0)
    assert r["predicted_points_unjudgeable_fraction"] > 0.3
    assert r["predicted_points_judgeable"] < r["predicted_points"]
    # What remains judgeable is judged cleanly rather than dragged down.
    assert r["precision_floor"] > 0.9


def test_a_model_predicting_the_roof_as_ground_is_not_rescued_by_the_ceiling():
    # The ceiling is generous, not blind. A point 12 m up is not plausibly ground
    # however it is labelled, so predicting the roof must hurt both ends.
    x, y, z, roof = flat_scene()
    asprs = np.where(roof, 1, 2).astype(np.int16)
    r = score_ground(asprs, x, y, z, predicted_ground=np.ones(x.size, bool))
    assert r["precision_ceiling"] < 1.0
    assert r["precision_floor"] < r["recall_clean"]


def test_an_area_with_no_ground_class_says_so_rather_than_scoring_zero():
    x, y, z, _ = flat_scene()
    asprs = np.full(x.size, 1, dtype=np.int16)  # all unassigned, as over water
    r = score_ground(asprs, x, y, z, predicted_ground=np.ones(x.size, bool))
    assert r["reference_points"] == 0
    assert "nothing to score against" in r["note"]
    assert "recall_clean" not in r


def test_noise_classes_are_excluded_from_both_ends():
    x, y, z, roof = flat_scene()
    asprs = np.where(roof, 1, 2).astype(np.int16)
    z = z.copy()
    asprs[:50] = 7  # low noise
    z[:50] = -70.0
    r = score_ground(asprs, x, y, z, predicted_ground=~roof)
    assert r["reference_points"] == int((asprs == 2).sum())
    assert r["recall_clean"] == pytest.approx(1.0)
