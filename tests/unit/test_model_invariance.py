"""The published network reads where points lie relative to each other, never where the block is.

Set abstraction groups each neighbourhood as offsets from its centroid, and feature propagation
weights by distance, so a model with no extra input channels gives the same logits when the whole
block is moved. A shift up by a tower's height or sideways by a block width must change nothing
beyond float32 rounding.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

from surrey.data import N_CLASSES  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402


def small_net(extra: int = 0) -> SurreyNet:
    torch.manual_seed(0)
    net = SurreyNet(N_CLASSES, width=16, k=8, extra_features=extra)
    for sa, n in ((net.sa1, 256), (net.sa2, 64), (net.sa3, 16)):
        sa.n_sample = n
    return net.eval()


def block(n: int = 2048) -> torch.Tensor:
    """A 40 m block with flat ground and one 30 m box on it, coordinates in metres."""
    g = torch.Generator().manual_seed(1)
    xy = torch.rand(1, n, 2, generator=g) * 40.0 - 20.0
    z = torch.rand(1, n, 1, generator=g) * 0.2
    roof = (xy[..., 0].abs() < 6.0) & (xy[..., 1].abs() < 6.0)
    z[roof] += 30.0
    return torch.cat([xy, z], dim=-1)


def logits(net: SurreyNet, x: torch.Tensor) -> torch.Tensor:
    # The sampler draws centroids at random; the same seed draws the same indices for any shift.
    torch.manual_seed(7)
    with torch.no_grad():
        return net(x)


@pytest.mark.parametrize(
    "shift",
    [(0.0, 0.0, 80.0), (0.0, 0.0, -15.0), (35.0, -12.0, 0.0), (35.0, -12.0, 80.0)],
    ids=["z up 80 m", "z down 15 m", "xy", "xyz"],
)
def test_a_shifted_block_gives_the_same_logits(shift):
    net = small_net()
    x = block()
    moved = x + torch.tensor(shift)
    base, after = logits(net, x), logits(net, moved)
    assert torch.allclose(base, after, atol=1e-5), (base - after).abs().max().item()


def test_a_network_given_height_as_a_feature_is_not_invariant():
    """With z also fed as a per-point feature, height becomes readable and the same shift moves the
    logits by far more than the tolerance above."""
    net = small_net(extra=1)
    x = block()
    moved = x + torch.tensor((0.0, 0.0, 80.0))
    base = logits(net, torch.cat([x, x[..., 2:3]], dim=-1))
    after = logits(net, torch.cat([moved, moved[..., 2:3]], dim=-1))
    assert (base - after).abs().max().item() > 1e-2
