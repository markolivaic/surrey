"""Tile caches that are genuinely memory mapped.

Each array is its own .npy file, opened with mmap_mode="r", so a tile costs almost no resident
memory. An .npz archive would not do: np.load accepts the mode for one and ignores it, and indexing
the archive reads the whole array.

The cell index is built at preparation time and stored the same way, rather than per tile at first
use, since it never changes.

Layout, one directory per tile:

    <tile>/
        meta.json        counts, origin, CRS, extent, the cell grid pitch
        xy_cm.npy        uint16, (N, 2), centimetres from the tile origin
        z_cm.npy         uint32, (N,), centimetres above the tile minimum
        hag_cm.npy       int16,  (N,), height for the scoring gate only, never a feature
        ret_q.npy        uint8,  (N,), return number over number of returns
        label.npy        int8,   (N,), DALES-2 class, -1 where unknown
        asprs.npy        uint8,  (N,), the delivery's own classification
        cell_order.npy   int32,  (N,), point indices sorted by cell
        cell_key.npy     int64,  (M,), the cells present, ascending
        cell_start.npy   int64,  (M + 1,), slice bounds into cell_order
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CELL_M = 10.0
CELL_STRIDE = 100000  # cells per axis in the packed key, ample for a 500 m tile

ARRAYS = {
    "xy_cm": np.uint16,
    "z_cm": np.uint32,
    "hag_cm": np.int16,
    "ret_q": np.uint8,
    "label": np.int8,
    "asprs": np.uint8,
}


def cell_key(xy_cm: np.ndarray, cell_m: float) -> np.ndarray:
    step = int(round(cell_m * 100))
    cx = xy_cm[:, 0].astype(np.int64) // step
    cy = xy_cm[:, 1].astype(np.int64) // step
    return cx * CELL_STRIDE + cy


def write_tile(
    out_dir: Path, arrays: dict[str, np.ndarray], meta: dict, cell_m: float = CELL_M
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, dtype in ARRAYS.items():
        a = np.ascontiguousarray(arrays[name], dtype=dtype)
        np.save(out_dir / f"{name}.npy", a)
    # The LAS flags of every kept point (surrey.flags), so a test can show no dropped one got through.
    if "flags" in arrays:
        np.save(out_dir / "flags.npy", np.ascontiguousarray(arrays["flags"], dtype=np.uint8))

    key = cell_key(arrays["xy_cm"], cell_m)
    order = np.argsort(key, kind="stable").astype(np.int32)
    ks = key[order]
    uniq, starts = np.unique(ks, return_index=True)
    bounds = np.append(starts, ks.size).astype(np.int64)

    np.save(out_dir / "cell_order.npy", order)
    np.save(out_dir / "cell_key.npy", uniq.astype(np.int64))
    np.save(out_dir / "cell_start.npy", bounds)

    xy = arrays["xy_cm"]
    meta = dict(meta)
    meta.update(
        {
            "points": int(xy.shape[0]),
            "cell_m": cell_m,
            "cells": int(uniq.size),
            "extent_cm": [int(xy[:, 0].max()), int(xy[:, 1].max())],
        }
    )
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    meta["bytes"] = sum(p.stat().st_size for p in out_dir.glob("*.npy"))
    return meta


class CachedTile:
    """One prepared tile, every array memory mapped."""

    # Set from ARRAYS when the tile is opened.
    xy_cm: np.ndarray
    z_cm: np.ndarray
    hag_cm: np.ndarray
    ret_q: np.ndarray
    label: np.ndarray
    asprs: np.ndarray

    def __init__(self, path: Path, split: str = ""):
        self.path = Path(path)
        self.name = self.path.name
        self.split = split
        self.meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))

        for name in ARRAYS:
            setattr(self, name, np.load(self.path / f"{name}.npy", mmap_mode="r"))
        flags = self.path / "flags.npy"
        self.flags = np.load(flags, mmap_mode="r") if flags.exists() else None
        self.cell_order = np.load(self.path / "cell_order.npy", mmap_mode="r")
        self.cell_key = np.load(self.path / "cell_key.npy", mmap_mode="r")
        self.cell_start = np.load(self.path / "cell_start.npy", mmap_mode="r")

        self.origin = np.asarray(self.meta["origin"], dtype=np.float64)
        self.epsg = self.meta.get("epsg")
        self.cell_m = float(self.meta["cell_m"])
        self._extent_m = (self.meta["extent_cm"][0] / 100.0, self.meta["extent_cm"][1] / 100.0)

    @property
    def n(self) -> int:
        return int(self.meta["points"])

    def extent_m(self) -> tuple[float, float]:
        return self._extent_m

    def cell_points(self, i: int, j: int) -> np.ndarray:
        """Point indices in one cell, sliced straight out of the mapped order."""
        k = i * CELL_STRIDE + j
        pos = int(np.searchsorted(self.cell_key, k))
        if pos >= self.cell_key.size or int(self.cell_key[pos]) != k:
            return np.empty(0, dtype=np.int64)
        lo, hi = int(self.cell_start[pos]), int(self.cell_start[pos + 1])
        return np.asarray(self.cell_order[lo:hi], dtype=np.int64)

    def gather_block(self, cx_m: float, cy_m: float, block_m: float) -> np.ndarray:
        half = block_m / 2.0
        lo_x, hi_x = cx_m - half, cx_m + half
        lo_y, hi_y = cy_m - half, cy_m + half
        i0 = int(np.floor(lo_x / self.cell_m))
        i1 = int(np.floor((hi_x - 1e-9) / self.cell_m))
        j0 = int(np.floor(lo_y / self.cell_m))
        j1 = int(np.floor((hi_y - 1e-9) / self.cell_m))

        parts = [self.cell_points(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
        parts = [p for p in parts if p.size]
        if not parts:
            return np.empty(0, dtype=np.int64)
        cand = np.concatenate(parts)
        xs = np.asarray(self.xy_cm[cand, 0], dtype=np.float64) / 100.0
        ys = np.asarray(self.xy_cm[cand, 1], dtype=np.float64) / 100.0
        keep = (xs >= lo_x) & (xs < hi_x) & (ys >= lo_y) & (ys < hi_y)
        return cand[keep]

    def block_z_reference(self, idx: np.ndarray, q: float = 0.01) -> float:
        """The block's own low height, used as the vertical origin for its points.

        A low quantile rather than the minimum: one noise point 70 m under a San
        Francisco street would otherwise set the origin for the whole block.
        """
        if idx.size == 0:
            return 0.0
        z = np.asarray(self.z_cm[idx], np.float64) / 100.0
        return float(np.quantile(z, q))

    def features(
        self,
        idx: np.ndarray,
        centre_m: tuple[float, float],
        z_ref: float | np.ndarray | None = None,
        channels: int = 4,
    ) -> np.ndarray:
        """(n, channels): x and y centred on the block, z above the block floor, and with
        four channels the return ratio.

        Physical scale is preserved on every axis, so a metre means a metre on
        both sides of the transfer. No terrain model is involved: z is measured
        from the block's own floor and the network works out what ground is.

        Three channels is the model that is published. The fourth, the return ratio,
        is exactly 0 on every prepared DALES-2 point, so a model trained with it has a
        dead input that comes alive on real lidar. Four stays
        available only so the checkpoint that had it can still be scored.
        """
        if channels not in (3, 4):
            raise ValueError(f"channels must be 3 or 4, got {channels}")
        if z_ref is None:
            z_ref = self.block_z_reference(idx)
        feat = np.empty((idx.size, channels), dtype=np.float32)
        feat[:, 0] = np.asarray(self.xy_cm[idx, 0], np.float32) / 100.0 - centre_m[0]
        feat[:, 1] = np.asarray(self.xy_cm[idx, 1], np.float32) / 100.0 - centre_m[1]
        feat[:, 2] = np.asarray(self.z_cm[idx], np.float32) / 100.0 - np.asarray(z_ref, np.float32)
        if channels == 4:
            feat[:, 3] = np.asarray(self.ret_q[idx], np.float32) / 255.0
        return feat

    def labels(self, idx: np.ndarray) -> np.ndarray:
        """Labels for these points, with any point under `ignore_mask` set to -1.

        The mask is how M1 keeps DALES water out of supervision: DALES labels it ground,
        swissSURFACE3D labels water water, and the model must not be told both.
        """
        lab = np.asarray(self.label[idx], dtype=np.int64)
        mask = getattr(self, "ignore_mask", None)
        if mask is not None:
            lab[mask[idx]] = -1
        return lab

    def utm_xy(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.xy_cm[idx, 0], np.float64) / 100.0 + self.origin[0],
            np.asarray(self.xy_cm[idx, 1], np.float64) / 100.0 + self.origin[1],
        )

    def z_abs(self, idx: np.ndarray) -> np.ndarray:
        return np.asarray(self.z_cm[idx], np.float64) / 100.0 + self.origin[2]

    def random_crop(
        self,
        rng: np.random.Generator,
        block_m: float,
        n_points: int,
        channels: int = 4,
        distinct: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """One random block, as n_points rows of features and labels.

        `distinct` caps how many different points of the block the crop may carry, for training at a
        chosen density rather than at whatever the source happens to deliver; the rest of the crop is
        made of repeats, which is what a sparse source does anyway. Left at None the draw is exactly
        what it always was, down to the order the generator is used in, so runs before this option and
        runs without it stay comparable.
        """
        ex, ey = self.extent_m()
        half = block_m / 2.0
        cx = rng.uniform(half, max(half, ex - half))
        cy = rng.uniform(half, max(half, ey - half))
        idx = self.gather_block(cx, cy, block_m)
        if idx.size == 0:
            return (
                np.zeros((n_points, channels), np.float32),
                np.full(n_points, -1, np.int64),
                True,
            )
        if distinct is not None and distinct < idx.size:
            idx = rng.choice(idx, size=distinct, replace=False)
        repeated = idx.size < n_points
        z_ref = self.block_z_reference(idx)
        pick = rng.choice(idx, size=n_points, replace=repeated)
        return self.features(pick, (cx, cy), z_ref, channels), self.labels(pick), repeated

    def block_centres(
        self, block_m: float, max_blocks: int | None = None
    ) -> list[tuple[float, float]]:
        """Centres of every block, or an evenly spread subset of them.

        Spread, not the first n. The blocks come out in bx then by order, so
        taking the first six hands back one edge strip of the tile. An eval on
        that samples a corner and calls it the tile, which is how a periodic
        score ends up moving for reasons that have nothing to do with the model.
        """
        ex, ey = self.extent_m()
        centres = [
            ((bx + 0.5) * block_m, (by + 0.5) * block_m)
            for bx in range(int(np.ceil(ex / block_m)))
            for by in range(int(np.ceil(ey / block_m)))
        ]
        if max_blocks is None or max_blocks >= len(centres) or not centres:
            return centres
        take = np.linspace(0, len(centres) - 1, max_blocks).round().astype(int)
        return [centres[i] for i in dict.fromkeys(take.tolist())]

    def iter_full(
        self,
        block_m: float,
        n_points: int,
        max_blocks: int | None = None,
        z_override=None,
        ret_value: float | None = None,
        channels: int = 4,
        seed: int = 0,
    ):
        """Every point of every block predicted exactly once, each tensor drawn as training draws a crop.

        A training crop (random_crop) reads a whole block_m square and keeps n_points of it at random,
        so the network has only ever seen a block covered end to end at that density. Here a block's
        points are shuffled and dealt into partitions of n_points, which makes every partition a random
        sample of the whole block, the same draw. The last partition is filled up with points drawn
        without replacement from the block's other partitions, and a block holding fewer than n_points
        is filled with random repeats of its own points, as random_crop fills it.

        A grid block on the tile's edge is narrower than block_m, and a crop never is: its centre stays
        half a block inside the tile. So an edge block is read through the full square a crop would read
        there, and the points it shares with its neighbour ride along without being scored.

        Yields (features, labels, chunk, valid). The first `valid` entries of chunk are the points this
        tensor predicts; the rest are context and fill. The shuffle is seeded per block from `seed` and
        the block's centre, so a block is dealt the same way whichever other blocks are visited.

        `max_blocks` restricts to an evenly spread subset for a quick look. Every
        point of the blocks it does visit is still visited exactly once.

        `z_override(cx, cy, idx)` is for diagnostics only. It may return a per-point
        vertical reference for the points of the square the block is read through, in idx
        order, to be used in place of the block floor, or None to keep the floor. Nothing
        that trains or scores the model passes it.

        `ret_value`, also for diagnostics only, replaces the return ratio channel of
        every point with one constant.
        """
        if ret_value is not None and channels < 4:
            raise ValueError("ret_value needs the four channel input")
        ex, ey = self.extent_m()
        half = block_m / 2.0
        for cx, cy in self.block_centres(block_m, max_blocks):
            owned = self.gather_block(cx, cy, block_m)
            if owned.size == 0:
                continue
            wx = min(max(cx, half), max(half, ex - half))
            wy = min(max(cy, half), max(half, ey - half))
            if (wx, wy) == (cx, cy):
                window, is_owned = owned, np.ones(owned.size, dtype=bool)
            else:
                window = np.union1d(self.gather_block(wx, wy, block_m), owned)
                is_owned = np.isin(window, owned)
            # One reference per block, not per tensor, taken over the square the crop reads.
            z_ref = self.block_z_reference(window)
            per_point = z_override(cx, cy, window) if z_override is not None else None
            rng = np.random.default_rng([seed, round(cx * 100), round(cy * 100)])
            order = rng.permutation(window.size)
            for s in range(0, window.size, n_points):
                part = order[s : s + n_points]
                scored = part[is_owned[part]]
                if scored.size == 0:
                    continue
                fill = n_points - part.size
                if fill == 0:
                    pad = order[:0]
                elif window.size > n_points:
                    pad = rng.choice(order[:s], size=fill, replace=False)
                else:
                    pad = rng.choice(window.size, size=fill, replace=True)
                pos = np.concatenate([scored, part[~is_owned[part]], pad])
                chunk = window[pos]
                ref = z_ref if per_point is None else np.asarray(per_point[pos], np.float32)
                feat = self.features(chunk, (wx, wy), ref, channels)
                if ret_value is not None:
                    feat[:, 3] = np.float32(ret_value)
                yield feat, self.labels(chunk), chunk, int(scored.size)


def load_cached(
    prep_dir: Path, names: list[str] | None = None, split: str = ""
) -> list[CachedTile]:
    dirs = sorted(p for p in Path(prep_dir).iterdir() if p.is_dir() and (p / "meta.json").exists())
    if names is not None:
        wanted = set(names)
        dirs = [p for p in dirs if p.name in wanted]
    if not dirs:
        raise SystemExit(f"no prepared tiles in {prep_dir}, run scripts/pipeline/prep_tiles.py")
    return [CachedTile(p, split) for p in dirs]
