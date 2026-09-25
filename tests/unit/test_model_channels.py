"""The geometry-only model reads three channels and nothing else."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile, write_tile  # noqa: E402
from surrey.data import N_CLASSES  # noqa: E402
from surrey.inference import infer_tile  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402


def small_net(extra: int) -> SurreyNet:
    torch.manual_seed(0)
    net = SurreyNet(N_CLASSES, width=16, k=8, extra_features=extra)
    for sa, n in ((net.sa1, 64), (net.sa2, 32), (net.sa3, 16)):
        sa.n_sample = n
    return net.eval()


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
    write_tile(tmp_path / "t", arrays, {"origin": [0.0, 0.0, 0.0], "epsg": 32610}, cell_m=10.0)
    return CachedTile(tmp_path / "t", "test")


def test_a_three_channel_model_runs_and_has_no_feature_weights_for_a_fourth():
    net3, net4 = small_net(0), small_net(1)
    out = net3(torch.randn(2, 256, 3))
    assert out.shape == (2, 256, N_CLASSES)
    assert sum(p.numel() for p in net3.parameters()) < sum(p.numel() for p in net4.parameters())


def test_a_three_channel_model_ignores_anything_past_z():
    # Guards the published model against a stray fourth column: it must not read it.
    net = small_net(0)
    x = torch.randn(1, 256, 4)
    y = x.clone()
    y[..., 3] = 1000.0
    torch.manual_seed(1)
    a = net(x)
    torch.manual_seed(1)
    b = net(y)
    assert torch.equal(a, b)


def test_features_and_crops_honour_the_channel_count(tile: CachedTile):
    idx = tile.gather_block(40.0, 40.0, 40.0)
    assert tile.features(idx, (40.0, 40.0), channels=3).shape == (idx.size, 3)
    assert tile.features(idx, (40.0, 40.0), channels=4).shape == (idx.size, 4)
    feat, _, _ = tile.random_crop(np.random.default_rng(0), 40.0, 128, channels=3)
    assert feat.shape == (128, 3)
    with pytest.raises(ValueError):
        tile.features(idx, (40.0, 40.0), channels=5)


def test_inference_hands_a_three_channel_model_three_channels(tile: CachedTile):
    net = small_net(0)
    pred, conf, ent, seen = infer_tile(net, tile, "cpu", 40.0, 256)
    assert seen.all() and pred.min() >= 0


def test_the_return_ratio_override_refuses_a_model_without_that_channel(tile: CachedTile):
    with pytest.raises(ValueError):
        infer_tile(small_net(0), tile, "cpu", 40.0, 256, ret_value=0.0)
