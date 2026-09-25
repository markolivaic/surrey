"""The memory-mapped tile cache and its spatial index.

The index exists to avoid scanning a whole tile per crop, so the tests that
matter compare it against the brute-force mask it replaced. An index that is
fast and subtly wrong would move every number in the project.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from iter_full_agreement import legacy_iter_full  # noqa: E402
from surrey.cache import ARRAYS, CachedTile, write_tile  # noqa: E402


@pytest.fixture
def tile(tmp_path: Path) -> CachedTile:
    rng = np.random.default_rng(0)
    n = 20000
    xy_cm = np.stack([rng.integers(0, 30000, n), rng.integers(0, 30000, n)], axis=1)
    arrays = {
        "xy_cm": xy_cm,
        "z_cm": rng.integers(0, 5000, n),
        "hag_cm": rng.integers(-100, 3000, n),
        "ret_q": rng.integers(0, 256, n),
        "label": rng.integers(-1, 15, n),
        "asprs": rng.integers(0, 20, n),
    }
    write_tile(
        tmp_path / "t", arrays, {"origin": [500000.0, 5440000.0, 12.5], "epsg": 26910}, cell_m=10.0
    )
    return CachedTile(tmp_path / "t", "train")


CROP = 4096


@pytest.fixture
def dense(tmp_path: Path) -> CachedTile:
    """Denser than a crop, as the real sources are: about 2.15 crops of points in each 40 m block, so
    the last run of a block cut into contiguous crops is mostly padding. 120 by 100 m, so one row of
    blocks is cut by the tile's edge."""
    rng = np.random.default_rng(3)
    n = 66000
    xy_cm = np.stack([rng.integers(0, 12000, n), rng.integers(0, 10000, n)], axis=1)
    arrays = {
        "xy_cm": xy_cm,
        "z_cm": rng.integers(0, 3000, n),
        "hag_cm": rng.integers(-100, 3000, n),
        "ret_q": rng.integers(0, 256, n),
        "label": rng.integers(-1, 15, n),
        "asprs": rng.integers(0, 20, n),
    }
    write_tile(tmp_path / "d", arrays, {"origin": [0.0, 0.0, 0.0], "epsg": 26910}, cell_m=10.0)
    return CachedTile(tmp_path / "d", "train")


def brute_force_block(t: CachedTile, cx: float, cy: float, block_m: float) -> np.ndarray:
    half = block_m / 2.0
    xs = np.asarray(t.xy_cm[:, 0], np.float64) / 100.0
    ys = np.asarray(t.xy_cm[:, 1], np.float64) / 100.0
    m = (xs >= cx - half) & (xs < cx + half) & (ys >= cy - half) & (ys < cy + half)
    return np.flatnonzero(m)


def test_arrays_are_really_memory_mapped(tile: CachedTile):
    # np.load(mmap_mode="r") silently ignores the mode for an npz archive.
    for name in ARRAYS:
        assert isinstance(getattr(tile, name), np.memmap), name


def test_metadata_survives_the_round_trip(tile: CachedTile):
    assert tile.n == 20000
    assert tile.epsg == 26910
    assert tile.origin.tolist() == [500000.0, 5440000.0, 12.5]
    assert tile.cell_m == 10.0


def test_gather_block_matches_a_brute_force_mask(tile: CachedTile):
    rng = np.random.default_rng(5)
    for _ in range(25):
        cx, cy = rng.uniform(20, 280, 2)
        got = np.sort(tile.gather_block(cx, cy, 40.0))
        want = brute_force_block(tile, cx, cy, 40.0)
        assert np.array_equal(got, want)


def test_gather_block_handles_a_block_that_straddles_the_tile_edge(tile: CachedTile):
    got = np.sort(tile.gather_block(5.0, 5.0, 40.0))
    want = brute_force_block(tile, 5.0, 5.0, 40.0)
    assert np.array_equal(got, want)


def test_gather_block_off_the_tile_returns_nothing(tile: CachedTile):
    assert tile.gather_block(5000.0, 5000.0, 40.0).size == 0


def test_gather_block_on_a_non_multiple_block_size(tile: CachedTile):
    # The cell pitch is 10 m. A 25 m block does not line up with it, and the
    # exact mask after the cell gather is what keeps the answer right.
    got = np.sort(tile.gather_block(137.3, 88.9, 25.0))
    want = brute_force_block(tile, 137.3, 88.9, 25.0)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("which", ["tile", "dense"])
def test_iter_full_predicts_every_point_exactly_once(which: str, request):
    t = request.getfixturevalue(which)
    seen = np.zeros(t.n, dtype=np.int64)
    for feat, label, chunk, valid in t.iter_full(40.0, CROP):
        assert feat.shape == (CROP, 4) and label.shape == (CROP,) and 0 < valid <= CROP
        np.add.at(seen, chunk[:valid], 1)
    assert seen.min() == 1 and seen.max() == 1


def density(feat: np.ndarray, cell_m: float = 4.0) -> float:
    """Distinct points per occupied cell of the tensor's own footprint."""
    pts = np.unique(feat[:, :3], axis=0)
    cells = np.floor((pts[:, :2] - pts[:, :2].min(axis=0)) / cell_m).astype(np.int64)
    return pts.shape[0] / np.unique(cells[:, 0] * 1000 + cells[:, 1]).size


def worst_duplicate_share(tensors) -> float:
    return max(1.0 - np.unique(chunk).size / chunk.size for _, _, chunk, _ in tensors)


def test_the_full_pass_draws_tensors_as_dense_as_training_crops(dense: CachedTile):
    rng = np.random.default_rng(5)
    crops = [density(dense.random_crop(rng, 40.0, CROP)[0]) for _ in range(40)]
    full = [density(feat) for feat, _, _, _ in dense.iter_full(40.0, CROP)]
    q = [10, 50, 90]
    np.testing.assert_allclose(np.percentile(full, q), np.percentile(crops, q), rtol=0.1)


def test_no_tensor_of_a_block_denser_than_a_crop_is_mostly_repeats(dense: CachedTile):
    assert worst_duplicate_share(dense.iter_full(40.0, CROP)) <= 0.5


def test_the_duplicate_check_fails_the_iterator_it_replaced(dense: CachedTile):
    # The published predictions came from contiguous runs of each block, the last one filled with
    # copies of a single point. The check above has to catch that iterator.
    assert worst_duplicate_share(legacy_iter_full(dense, 40.0, CROP)) > 0.5


def test_an_edge_block_is_read_through_a_full_square(dense: CachedTile):
    # A crop never extends past the tile, so no tensor may cover less than a crop does.
    for feat, _, _, _ in dense.iter_full(40.0, CROP):
        span = feat[:, :2].max(axis=0) - feat[:, :2].min(axis=0)
        assert (span > 38.0).all(), span


def test_features_are_centred_on_the_block_and_carry_real_metres(tile: CachedTile):
    idx = tile.gather_block(100.0, 100.0, 40.0)
    feat = tile.features(idx, (100.0, 100.0))
    assert np.abs(feat[:, 0]).max() <= 20.0
    assert np.abs(feat[:, 1]).max() <= 20.0
    assert feat[:, 3].min() >= 0.0 and feat[:, 3].max() <= 1.0

    # z is measured from the block's own floor, not from a terrain model and not
    # from the stored gate height. See backend/surrey/data.py for why.
    z = np.asarray(tile.z_cm[idx], np.float64) / 100.0
    assert np.allclose(feat[:, 2], z - tile.block_z_reference(idx), atol=1e-3)


def test_the_block_floor_ignores_a_single_deep_outlier():
    # One point far below a block would otherwise set the vertical origin for
    # every point in it. A low quantile rather than the minimum.
    class Fake:
        z_cm = np.concatenate([np.full(999, 1000, np.uint32), np.array([0], np.uint32)])
        block_z_reference = CachedTile.block_z_reference

    ref = Fake().block_z_reference(np.arange(1000))
    assert ref == pytest.approx(10.0, abs=0.2)


def test_iter_full_uses_one_vertical_origin_per_block(tile: CachedTile):
    # A block split across several chunks must not shift its origin between them,
    # or the same roof arrives at different heights in different tensors.
    seen: dict[tuple, set] = {}
    for feat, _, chunk, valid in tile.iter_full(40.0, 512):
        z = np.asarray(tile.z_cm[chunk[:valid]], np.float64) / 100.0
        ref = z - feat[:valid, 2]
        key = (round(float(np.median(ref)), 3),)
        seen.setdefault(key, set()).add(round(float(ref.min()), 3))
        assert float(ref.max() - ref.min()) < 1e-2, "origin drifted inside one chunk"


def test_a_z_override_changes_only_the_blocks_it_names(tile: CachedTile):
    # The diagnostic hook must leave every other block byte for byte as it was, and
    # in the block it names put the given per-point reference under each point,
    # including points that land in a later chunk and in the padding.
    target = tile.block_centres(40.0)[5]
    base = {tuple(chunk[:valid]): feat for feat, _, chunk, valid in tile.iter_full(40.0, 64)}

    def override(cx, cy, idx):
        if (cx, cy) != target:
            return None
        return np.asarray(tile.z_cm[idx], np.float64) / 100.0 - 1.25

    touched = 0
    for feat, _, chunk, valid in tile.iter_full(40.0, 64, z_override=override):
        before = base[tuple(chunk[:valid])]
        xs = np.asarray(tile.xy_cm[chunk[:valid], 0], np.float64) / 100.0
        ys = np.asarray(tile.xy_cm[chunk[:valid], 1], np.float64) / 100.0
        inside = (abs(xs - target[0]) <= 20.0).all() and (abs(ys - target[1]) <= 20.0).all()
        if inside:
            touched += 1
            np.testing.assert_allclose(feat[:, 2], 1.25, atol=1e-4)
            np.testing.assert_array_equal(feat[:, [0, 1, 3]], before[:, [0, 1, 3]])
        else:
            np.testing.assert_array_equal(feat, before)
    assert touched >= 2, "the named block should span several 64-point chunks"


def test_a_constant_return_ratio_replaces_only_that_channel(tile: CachedTile):
    base = list(tile.iter_full(40.0, 256))
    swapped = list(tile.iter_full(40.0, 256, ret_value=0.0))
    assert len(base) == len(swapped)
    for (fa, _, ca, va), (fb, _, cb, vb) in zip(base, swapped, strict=True):
        assert va == vb
        np.testing.assert_array_equal(ca, cb)
        np.testing.assert_array_equal(fa[:, :3], fb[:, :3])
        assert (fb[:, 3] == 0.0).all()


def test_an_ignore_mask_turns_labels_to_minus_one_and_nothing_else(tile: CachedTile):
    idx = np.arange(200)
    before = tile.labels(idx)
    tile.ignore_mask = np.zeros(tile.n, dtype=bool)
    tile.ignore_mask[::2] = True
    after = tile.labels(idx)
    assert (after[::2] == -1).all()
    np.testing.assert_array_equal(after[1::2], before[1::2])
    np.testing.assert_array_equal(np.asarray(tile.label[idx]), before)  # the cache is untouched


def test_utm_and_absolute_z_add_the_stored_origin(tile: CachedTile):
    idx = np.array([0, 1, 2])
    ux, uy = tile.utm_xy(idx)
    assert ux[0] == pytest.approx(tile.xy_cm[0, 0] / 100.0 + 500000.0)
    assert uy[0] == pytest.approx(tile.xy_cm[0, 1] / 100.0 + 5440000.0)
    assert tile.z_abs(idx)[0] == pytest.approx(tile.z_cm[0] / 100.0 + 12.5)


def test_random_crop_returns_a_full_tensor_and_flags_repeats(tile: CachedTile):
    rng = np.random.default_rng(1)
    feat, label, repeated = tile.random_crop(rng, 40.0, 8192)
    assert feat.shape == (8192, 4) and label.shape == (8192,)
    # This fixture is sparse, so a 40 m block cannot hold 8192 distinct points.
    assert repeated is True

    feat, label, repeated = tile.random_crop(rng, 40.0, 16)
    assert repeated is False


def test_a_block_subset_is_spread_across_the_tile_not_taken_from_one_corner():
    # Blocks come out in bx then by order, so the first six are one edge strip.
    # A periodic eval on that samples a corner and calls it the tile.
    class Fake:
        _extent_m = (400.0, 400.0)
        extent_m = CachedTile.extent_m
        block_centres = CachedTile.block_centres

    t = Fake()
    every = t.block_centres(40.0)
    assert len(every) == 100

    few = t.block_centres(40.0, 6)
    assert len(few) == 6
    # The corner-taking version would return six blocks all at the same x.
    assert len({c[0] for c in few}) > 1, "subset sits in a single column"
    assert few[0] == every[0] and few[-1] == every[-1], "subset must span the tile"


def test_asking_for_more_blocks_than_exist_returns_them_all():
    class Fake:
        _extent_m = (80.0, 80.0)
        extent_m = CachedTile.extent_m
        block_centres = CachedTile.block_centres

    t = Fake()
    assert len(t.block_centres(40.0, 999)) == len(t.block_centres(40.0)) == 4


def test_iter_full_with_a_block_cap_still_visits_its_points_once(tile: CachedTile):
    seen = np.zeros(tile.n, dtype=np.int64)
    for _, _, chunk, valid in tile.iter_full(40.0, 4096, max_blocks=5):
        np.add.at(seen, chunk[:valid], 1)
    assert seen.max() == 1, "a capped pass must not double count"
    assert seen.sum() > 0
