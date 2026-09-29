"""A PointNet++ style encoder-decoder for per-point classification.

Three set abstraction levels with k nearest neighbour grouping, three feature
propagation levels back up, and a per-point head. Written out rather than pulled
from a library because the libraries that do this well want more GPU than a
4 GB laptop card has: Pointcept's PTv3, the obvious modern choice, asks for four
RTX 3090s in its own README.

The neighbour search is chunked over queries. A full pairwise distance matrix
between 32,768 points and 4,096 centroids is 537 MB per batch item in fp32,
which is most of this card before any weights exist.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def farthest_point_sample(xyz: torch.Tensor, n_sample: int) -> torch.Tensor:
    """(B, N, 3) -> (B, n_sample) indices, spread out across the cloud.

    Correct and slow: one sequential kernel per sample, so 4,096 of them for the
    first level alone. Kept for the comparison in scripts/investigations/bench_model.py,
    not used in training. See random_sample for why.
    """
    b, n, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(b, n_sample, dtype=torch.long, device=device)
    dist = torch.full((b, n), 1e10, device=device)
    far = torch.randint(0, n, (b,), dtype=torch.long, device=device)
    ar = torch.arange(b, dtype=torch.long, device=device)
    for i in range(n_sample):
        idx[:, i] = far
        centroid = xyz[ar, far].view(b, 1, 3)
        d = torch.sum((xyz - centroid) ** 2, dim=-1)
        dist = torch.minimum(dist, d)
        far = torch.max(dist, dim=-1).indices
    return idx


def random_sample(xyz: torch.Tensor, n_sample: int) -> torch.Tensor:
    """(B, N, 3) -> (B, n_sample) indices drawn uniformly without replacement.

    Farthest point sampling exists to get even spatial coverage out of a cloud
    whose density varies. This pipeline has already removed that variation: every
    crop is voxel downsampled at a fixed size before it reaches the network, so
    the points are close to uniform in space and a uniform draw is close to an
    even one.

    scripts/investigations/bench_model.py measures what the swap saves, and the artefact it
    writes is data/artifacts/bench_model.json. Do not quote a figure here; it is
    device specific and it will rot.
    """
    b, n, _ = xyz.shape
    return torch.argsort(torch.rand(b, n, device=xyz.device), dim=-1)[:, :n_sample]


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """(B, N, C) gathered by (B, S) or (B, S, K) -> (B, S, C) or (B, S, K, C)."""
    b = points.shape[0]
    view = [b] + [1] * (idx.dim() - 1)
    ar = torch.arange(b, dtype=torch.long, device=points.device).view(view)
    return points[ar, idx]


def knn_idx(query: torch.Tensor, ref: torch.Tensor, k: int, chunk: int = 1024) -> torch.Tensor:
    """(B, S, 3) against (B, N, 3) -> (B, S, k) indices into ref.

    Chunked over S so the distance matrix never exists in full. The distance is expanded as
    |q|^2 + |r|^2 - 2 q.r, which subtracts large numbers to get a small one. Under fp16 autocast the
    q.r product ran in half precision, and with z up to 200 m above a block's floor a square near
    40,000 m^2 has a step of 32 m^2, so the nearest neighbours of a tall point were close to a random
    pick. So the search runs in fp32 with autocast off, on coordinates centred on the reference
    cloud's mean: distances do not change under a shift, and the squares being subtracted get
    smaller. scripts/investigations/precision_audit.py measures both choices on real blocks.
    """
    b, s, _ = query.shape
    out = torch.empty(b, s, k, dtype=torch.long, device=query.device)
    with torch.autocast(device_type=query.device.type, enabled=False):
        centre = ref.float().mean(dim=1, keepdim=True)
        query = query.float() - centre
        ref = ref.float() - centre
        ref_sq = (ref**2).sum(-1).unsqueeze(1)
        for i in range(0, s, chunk):
            j = min(i + chunk, s)
            q = query[:, i:j]
            d = (q**2).sum(-1).unsqueeze(-1) + ref_sq - 2.0 * torch.bmm(q, ref.transpose(1, 2))
            out[:, i:j] = d.topk(k, dim=-1, largest=False).indices
    return out


class SetAbstraction(nn.Module):
    def __init__(self, n_sample: int, k: int, in_ch: int, mlp: list[int], sampler: str = "random"):
        super().__init__()
        self.n_sample, self.k, self.sampler = n_sample, k, sampler
        layers, c = [], in_ch + 3
        for out in mlp:
            layers += [nn.Conv2d(c, out, 1, bias=False), nn.BatchNorm2d(out), nn.ReLU(True)]
            c = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor | None):
        pick = random_sample if self.sampler == "random" else farthest_point_sample
        idx = pick(xyz, self.n_sample)
        new_xyz = index_points(xyz, idx)
        nn_idx = knn_idx(new_xyz, xyz, self.k)

        grouped = index_points(xyz, nn_idx) - new_xyz.unsqueeze(2)  # (B, S, K, 3)
        if feat is not None:
            grouped = torch.cat([grouped, index_points(feat, nn_idx)], dim=-1)

        out = self.mlp(grouped.permute(0, 3, 1, 2))  # (B, C, S, K)
        return new_xyz, out.max(dim=-1).values.transpose(1, 2)  # (B, S, C)


class FeaturePropagation(nn.Module):
    def __init__(self, in_ch: int, mlp: list[int]):
        super().__init__()
        layers, c = [], in_ch
        for out in mlp:
            layers += [nn.Conv1d(c, out, 1, bias=False), nn.BatchNorm1d(out), nn.ReLU(True)]
            c = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz_to, xyz_from, feat_to, feat_from):
        idx = knn_idx(xyz_to, xyz_from, 3)
        d = torch.linalg.norm(index_points(xyz_from, idx) - xyz_to.unsqueeze(2), dim=-1)
        w = 1.0 / (d + 1e-8)
        w = w / w.sum(dim=-1, keepdim=True)
        interp = (index_points(feat_from, idx) * w.unsqueeze(-1)).sum(dim=2)
        if feat_to is not None:
            interp = torch.cat([feat_to, interp], dim=-1)
        return self.mlp(interp.transpose(1, 2)).transpose(1, 2)


class SurreyNet(nn.Module):
    """Input is (B, N, 3 + extra_features): x, y centred on the block, z above the block
    floor, then per-point features. extra_features=0 is the published geometry-only
    model; 1 is the checkpoint that read the return ratio, which is dead in DALES-2."""

    def __init__(
        self,
        n_classes: int,
        width: int = 128,
        k: int = 32,
        extra_features: int = 1,
        sampler: str = "random",
    ):
        super().__init__()
        w = width
        self.extra = extra_features
        self.sa1 = SetAbstraction(4096, k, extra_features, [w // 2, w // 2, w], sampler)
        self.sa2 = SetAbstraction(1024, k, w, [w, w, w * 2], sampler)
        self.sa3 = SetAbstraction(256, k, w * 2, [w * 2, w * 2, w * 4], sampler)
        self.fp3 = FeaturePropagation(w * 4 + w * 2, [w * 2, w * 2])
        self.fp2 = FeaturePropagation(w * 2 + w, [w, w])
        self.fp1 = FeaturePropagation(w + extra_features, [w, w])
        self.head = nn.Sequential(
            nn.Conv1d(w, w, 1, bias=False),
            nn.BatchNorm1d(w),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(w, n_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # z is carried in the coordinates as height above the block's own floor (its first
        # percentile), so the network sees a real vertical metre rather than an elevation
        # above a sea it has no reference for. No terrain model is involved.
        xyz = torch.stack([x[..., 0], x[..., 1], x[..., 2]], dim=-1)
        feat = x[..., 3 : 3 + self.extra]

        xyz1, f1 = self.sa1(xyz, feat)
        xyz2, f2 = self.sa2(xyz1, f1)
        xyz3, f3 = self.sa3(xyz2, f2)

        f2 = self.fp3(xyz2, xyz3, f2, f3)
        f1 = self.fp2(xyz1, xyz2, f1, f2)
        f0 = self.fp1(xyz, xyz1, feat, f1)
        return self.head(f0.transpose(1, 2)).transpose(1, 2)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
