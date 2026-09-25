"""What a crop at a chosen density actually contains, including the case that fooled the first
density measurement: thinning a pool that is larger than the crop changes nothing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.density import crop_at_density, distinct_for_density  # noqa: E402


def test_the_distinct_count_is_density_times_area_capped_by_what_exists():
    assert distinct_for_density(20.0, 40.0, 10**9) == 32000
    assert distinct_for_density(38.2, 40.0, 10**9) == 61120
    assert distinct_for_density(38.2, 40.0, 5000) == 5000
    assert distinct_for_density(0.0, 40.0, 5000) == 1


def test_a_crop_carries_the_distinct_points_it_was_asked_for():
    rng = np.random.default_rng(0)
    idx = np.arange(100_000)
    pick, what = crop_at_density(idx, 32_768, rng, distinct=20_000)
    assert what["distinct_in_pool"] == 20_000
    assert what["repeats_needed"] is True
    assert pick.size == 32_768
    assert what["distinct_in_crop"] < 20_001


def test_self_test_thinning_a_pool_bigger_than_the_crop_changes_nothing_about_the_crop():
    # The fault in the first density measurement: with 70,000 points in the block and a crop of 32,768,
    # asking for 50,000 distinct points still gives a crop of 32,768 different points.
    rng = np.random.default_rng(0)
    idx = np.arange(70_000)
    _, wide = crop_at_density(idx, 32_768, np.random.default_rng(0), distinct=None)
    _, thinned = crop_at_density(idx, 32_768, rng, distinct=50_000)
    assert wide["distinct_in_crop"] == 32_768
    assert thinned["distinct_in_crop"] == 32_768


def test_a_block_smaller_than_the_crop_repeats_points_as_training_did():
    rng = np.random.default_rng(1)
    idx = np.arange(9_000)
    pick, what = crop_at_density(idx, 32_768, rng)
    # Drawing 32,768 with replacement from 9,000 leaves a few hundred of them never drawn, which is
    # what the crop really contains and what the record has to say.
    assert what["repeats_needed"] is True and pick.size == 32_768
    assert 8_500 < what["distinct_in_crop"] <= 9_000
