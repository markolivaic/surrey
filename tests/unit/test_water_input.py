"""The helpers behind the two water questions: does the class exist in the model, and what does the
model's input look like over water.

Both answers are one number each (an IoU, an AUROC), so each helper is fed a case whose answer is known
by hand, including the cases that would quietly mislead: an AUROC of a variable that is all ties, and a
separator that runs the other way round.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def load(name: str):
    path = next((ROOT / "scripts").glob(f"*/{name}.py"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


swiss = load("score_swiss")
water = load("water_input_test")


def test_a_perfect_separator_scores_one_and_chance_scores_a_half():
    assert water.auroc(np.array([5.0, 6, 7]), np.array([1.0, 2, 3]))["auroc"] == 1.0
    assert water.auroc(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]))["auroc"] == 0.5


def test_self_test_a_variable_that_is_all_ties_scores_chance_not_one():
    # Ranking without averaging ties would put every positive above every negative by sort order.
    assert water.auroc(np.ones(500), np.ones(700))["auroc"] == 0.5


def test_a_separator_that_runs_the_other_way_is_read_as_separation():
    low = water.auroc(np.array([1.0, 2, 3]), np.array([5.0, 6, 7]))
    assert low["auroc"] == 0.0
    assert water.separation(low["auroc"]) == {"strength": 1.0, "direction": "water lower"}
    assert water.separation(0.9) == {"strength": 0.9, "direction": "water higher"}
    assert water.separation(None) == {"strength": None, "direction": None}


def test_an_auroc_needs_both_sides():
    assert water.auroc(np.array([1.0]), np.empty(0))["auroc"] is None


def test_class_scores_read_a_class_the_model_never_calls():
    truth = np.array([15, 15, 2, 2, 2])
    never = np.full(5, 2)
    row = swiss.scores(never, truth, 15)
    assert row["called"] == 0 and row["support"] == 2 and row["iou"] == 0.0
    assert row["precision"] is None and row["recall"] == 0.0


def test_class_scores_on_a_perfect_and_a_partial_call():
    truth = np.array([15, 15, 15, 2])
    assert swiss.scores(truth.copy(), truth, 15)["iou"] == 1.0
    half = np.array([15, 15, 2, 15])
    row = swiss.scores(half, truth, 15)
    assert (
        row["true_positive"] == 2
        and row["precision"] == round(2 / 3, 4)
        and row["recall"] == round(2 / 3, 4)
    )


def test_the_class_ids_are_the_same_number_everywhere():
    check = swiss.mapping_check()
    assert check["all_agree"], check
    assert check["water_id"] == 15 and check["bridge_deck_id"] == 16
    if check["cached_labels_checked"]:
        assert set(check["ids_in_cached_swiss_labels"]) <= set(check["ids_the_table_allows"])
    assert check["training_artefacts"], "no committed M1 training artefact to check the ids against"
