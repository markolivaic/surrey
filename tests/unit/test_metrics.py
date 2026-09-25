"""Per-class scoring, including what it must do when a class is absent."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.data import CLASS_NAMES, COARSE  # noqa: E402
from surrey.metrics import Confusion, coarse_lookup  # noqa: E402


def test_a_perfect_prediction_scores_one_on_every_present_class():
    c = Confusion()
    truth = np.array([0, 0, 5, 12, 12, 12])
    c.add(truth, truth.copy())
    iou = c.iou()
    assert iou["ground"] == 1.0
    assert iou["tree"] == 1.0
    assert iou["habitat"] == 1.0
    assert c.accuracy() == pytest.approx(1.0)


def test_iou_is_computed_by_hand_correctly():
    # ground truth 4 points, 3 predicted ground correctly, 1 called tree.
    # One real tree point is also called ground.
    # ground: tp 3, fp 1, fn 1 -> 3 / 5 = 0.6
    # tree:   tp 0, fp 1, fn 1 -> 0 / 2 = 0.0
    c = Confusion()
    c.add(np.array([0, 0, 0, 0, 5]), np.array([0, 0, 0, 5, 0]))
    iou = c.iou()
    assert iou["ground"] == pytest.approx(0.6)
    assert iou["tree"] == pytest.approx(0.0)


def test_an_absent_class_is_none_and_not_zero():
    # This is the flattering-guard case. A class the tiles never contained has
    # not been measured, and calling that zero reads as a failed prediction.
    c = Confusion()
    c.add(np.array([0, 0]), np.array([0, 0]))
    iou = c.iou()
    assert iou["traffic-pole"] is None
    assert c.support()["traffic-pole"] == 0
    assert iou["ground"] == 1.0


def test_a_present_class_never_predicted_scores_zero_not_none():
    c = Confusion()
    c.add(np.array([9, 9, 9]), np.array([0, 0, 0]))
    iou = c.iou()
    assert iou["utility-pole"] == 0.0  # present and entirely missed
    assert c.recall()["utility-pole"] == 0.0
    assert c.precision()["utility-pole"] is None  # nothing was predicted as it


def test_ignored_labels_do_not_enter_the_matrix():
    c = Confusion()
    c.add(np.array([-1, -1, 0]), np.array([3, 7, 0]))
    assert int(c.m.sum()) == 1
    assert c.support()["ground"] == 1


def test_add_rejects_mismatched_shapes():
    c = Confusion()
    with pytest.raises(ValueError, match="shape mismatch"):
        c.add(np.zeros(3, int), np.zeros(4, int))


def test_accuracy_of_an_empty_matrix_is_not_a_number():
    c = Confusion()
    assert c.accuracy() != c.accuracy()  # NaN, rather than a misleading 0 or 1


def test_coarse_grouping_folds_the_building_classes_together():
    c = Confusion()
    # habitat truth predicted as annex is correct at coarse level, both building.
    hab = CLASS_NAMES.index("habitat")
    annex = CLASS_NAMES.index("annex")
    c.add(np.array([hab, hab]), np.array([annex, hab]))
    assert c.iou()["habitat"] < 1.0
    assert c.coarse().iou()["building"] == 1.0


def test_coarse_grouping_keeps_a_real_confusion_visible():
    c = Confusion()
    tree = CLASS_NAMES.index("tree")
    hab = CLASS_NAMES.index("habitat")
    c.add(np.array([tree, tree, hab, hab]), np.array([hab, tree, hab, hab]))
    coarse = c.coarse().iou()
    assert coarse["vegetation"] < 1.0
    assert coarse["building"] < 1.0


def test_coarse_grouping_covers_every_class():
    table = coarse_lookup()
    assert table.size == len(CLASS_NAMES)
    assert (table >= 0).all()
    assert sum(len(v) for v in COARSE.values()) == len(CLASS_NAMES)


def test_coarse_totals_match_the_fine_matrix():
    rng = np.random.default_rng(0)
    c = Confusion()
    c.add(rng.integers(0, 15, 5000), rng.integers(0, 15, 5000))
    assert int(c.coarse().m.sum()) == int(c.m.sum())
