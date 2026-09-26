"""Voting: the mean softmax over seeded passes, each pass seeded as one pass always was."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

from surrey.cache import CachedTile, write_tile  # noqa: E402
from surrey.inference import infer_tile, vote_probabilities  # noqa: E402
from test_model_channels import small_net  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture
def tile(tmp_path: Path) -> CachedTile:
    rng = np.random.default_rng(0)
    n = 3000
    arrays = {
        "xy_cm": np.stack([rng.integers(0, 8000, n), rng.integers(0, 8000, n)], axis=1),
        "z_cm": rng.integers(0, 2000, n),
        "hag_cm": rng.integers(0, 2000, n),
        "ret_q": rng.integers(0, 256, n),
        "label": rng.integers(0, 15, n),
        "asprs": rng.integers(0, 20, n),
    }
    write_tile(tmp_path / "t", arrays, {"origin": [0.0, 0.0, 0.0]})
    return CachedTile(tmp_path / "t")


def test_two_votes_are_the_mean_of_the_two_passes_they_are_made_of(tile):
    net = small_net(0)
    p0, seen0 = vote_probabilities(net, tile, "cpu", 40.0, 256, seed=0, votes=1)
    p1, _ = vote_probabilities(net, tile, "cpu", 40.0, 256, seed=1, votes=1)
    both, seen = vote_probabilities(net, tile, "cpu", 40.0, 256, seed=0, votes=2)
    assert seen.all() and seen0.all()
    np.testing.assert_allclose(both, (p0 + p1) / 2, atol=1e-6)
    np.testing.assert_allclose(both.sum(axis=1), 1.0, atol=1e-5)


def test_the_two_passes_differ_so_the_vote_has_something_to_average(tile):
    net = small_net(0)
    p0 = vote_probabilities(net, tile, "cpu", 40.0, 256, seed=0, votes=1)[0]
    p1 = vote_probabilities(net, tile, "cpu", 40.0, 256, seed=1, votes=1)[0]
    assert not np.allclose(p0, p1)


def test_infer_tile_reads_its_classes_off_the_mean_softmax(tile):
    net = small_net(0)
    probs, _ = vote_probabilities(net, tile, "cpu", 40.0, 256, seed=3, votes=3)
    pred, conf, ent, seen = infer_tile(net, tile, "cpu", 40.0, 256, seed=3, votes=3)
    assert seen.all()
    np.testing.assert_array_equal(pred, probs.argmax(axis=1))
    np.testing.assert_allclose(conf, probs.max(axis=1), atol=1e-6)
    assert (ent >= 0).all()


def test_no_votes_is_refused(tile):
    with pytest.raises(ValueError):
        vote_probabilities(small_net(0), tile, "cpu", 40.0, 256, votes=0)


def test_the_vote_count_rule_takes_the_smallest_n_that_holds_and_caps_at_sixteen():
    sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
    from choose_votes import choose

    assert choose({1: 0.95, 2: 0.991, 4: 0.999, 8: 1.0}) == 2
    assert choose({1: 0.995, 2: 0.999, 4: 1.0, 8: 1.0}) == 1
    assert choose({1: 0.9, 2: 0.95, 4: 0.97, 8: 0.989}) == 16


def test_a_windows_path_in_a_checkpoint_is_read_as_a_plain_path():
    import io
    import pathlib

    from surrey.inference import _PathTolerantUnpickler

    unpickler = _PathTolerantUnpickler(io.BytesIO(b""))
    assert unpickler.find_class("pathlib", "WindowsPath") is pathlib.PurePath
    assert unpickler.find_class("builtins", "dict") is dict
