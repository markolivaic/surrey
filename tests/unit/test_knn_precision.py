"""The neighbour search finds the same neighbours under fp16 autocast as an exact float64 search.

Training and prediction run under fp16 autocast. The search expands |q|^2 + |r|^2 - 2 q.r, and when the
product ran in half precision, points 200 m above a block's floor got close to random neighbours. This
builds a cloud with a tower up to 250 m, asks for neighbours under CPU fp16 autocast, and compares with
an exact search; the old expansion, kept here verbatim, has to fail the same comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.model import knn_idx  # noqa: E402


def cloud(seed: int = 0) -> torch.Tensor:
    """A 40 m block: jittered ground, and a 6 m square tower of points up to 250 m, 15 cm apart."""
    g = torch.Generator().manual_seed(seed)
    ground = torch.rand(2000, 3, generator=g) * torch.tensor([40.0, 40.0, 0.3]) - torch.tensor(
        [20.0, 20.0, 0.0]
    )
    tower = torch.rand(6000, 3, generator=g) * torch.tensor([6.0, 6.0, 250.0])
    return torch.cat([ground, tower]).unsqueeze(0)


def exact(query: torch.Tensor, ref: torch.Tensor, k: int) -> torch.Tensor:
    d = torch.cdist(query.double(), ref.double(), compute_mode="donot_use_mm_for_euclid_dist")
    return d.topk(k, dim=-1, largest=False).indices


def legacy(query: torch.Tensor, ref: torch.Tensor, k: int, chunk: int = 1024) -> torch.Tensor:
    """The search before the fix: the expansion under whatever autocast is active."""
    b, s, _ = query.shape
    out = torch.empty(b, s, k, dtype=torch.long)
    ref_sq = (ref**2).sum(-1).unsqueeze(1)
    for i in range(0, s, chunk):
        j = min(i + chunk, s)
        q = query[:, i:j]
        d = (q**2).sum(-1).unsqueeze(-1) + ref_sq - 2.0 * torch.bmm(q, ref.transpose(1, 2))
        out[:, i:j] = d.topk(k, dim=-1, largest=False).indices
    return out


def kept(idx: torch.Tensor, ref_idx: torch.Tensor) -> torch.Tensor:
    return (idx.unsqueeze(-1) == ref_idx.unsqueeze(-2)).any(-1).float().mean(-1)


def high_queries(pts: torch.Tensor) -> torch.Tensor:
    return pts[:, pts[0, :, 2] > 150.0][:, :400]


def test_fp16_autocast_finds_the_exact_neighbours_high_up():
    pts = cloud()
    q = high_queries(pts)
    with torch.autocast("cpu", dtype=torch.float16):
        got = knn_idx(q, pts, 16)
    assert kept(got, exact(q, pts, 16)).mean() > 0.995


def test_the_search_is_the_same_with_and_without_autocast():
    pts = cloud(1)
    q = pts[:, ::7]
    with torch.autocast("cpu", dtype=torch.float16):
        half = knn_idx(q, pts, 8)
    assert kept(half, knn_idx(q, pts, 8)).mean() > 0.999


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the CUDA autocast path needs a CUDA card"
)
def test_cuda_fp16_autocast_finds_the_exact_neighbours_high_up():
    """The path training and prediction take: CUDA, fp16 autocast. Skipped where there is no card."""
    pts = cloud().cuda()
    q = high_queries(pts)
    with torch.autocast("cuda", dtype=torch.float16):
        got = knn_idx(q, pts, 16)
    assert kept(got.cpu(), exact(q.cpu(), pts.cpu(), 16)).mean() > 0.995


def test_self_test_the_old_expansion_fails_high_up():
    pts = cloud()
    q = high_queries(pts)
    with torch.autocast("cpu", dtype=torch.float16):
        got = legacy(q, pts, 16)
    assert kept(got, exact(q, pts, 16)).mean() < 0.5
