"""Shift estimators recover a shift that was put in on purpose."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.registration import highpass, masked_ncc, phase_correlation, to_metres  # noqa: E402


def scene(n: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.normal(size=(n, n)), 1.5)


def shifted(img: np.ndarray, dr: float, dc: float) -> np.ndarray:
    """moving[r, c] = img[r - dr, c - dc], the module's convention."""
    r, c = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].astype(np.float64)
    return map_coordinates(img, [r - dr, c - dc], order=3, mode="reflect")


@pytest.mark.parametrize("dr,dc", [(0.0, 0.0), (2.3, -1.6), (-3.7, 0.4), (1.5, -1.5), (0.5, 2.5)])
def test_both_estimators_recover_a_known_shift(dr, dc):
    ref = scene()
    mov = shifted(ref, dr, dc)
    valid = np.ones_like(ref, bool)
    valid[:30] = False
    valid[:, -40:] = False
    a = highpass(mov, valid, 4.0)
    b = highpass(ref, np.ones_like(ref, bool), 4.0)
    pr, pc, _ = phase_correlation(b, a)
    nr, nc, peak, z, _ = masked_ncc(b, a, valid, 8)
    # Half-pixel shifts included: the parabola this replaced came back half a pixel
    # off on a planted shift of one and a half pixels in the real imagery.
    assert abs(nr - dr) < 0.1 and abs(nc - dc) < 0.1, (nr, nc)
    assert abs(pr - dr) < 0.5 and abs(pc - dc) < 0.5, (pr, pc)
    assert peak > 0.8 and z >= 5.0


def test_a_mask_on_the_moving_raster_does_not_pull_the_estimate_to_zero():
    # A block mask with hard edges, as a ground-only raster has. If the estimator
    # latched onto the mask outline the answer would come back near zero.
    ref = scene(seed=3)
    mov = shifted(ref, 3.0, 2.0)
    valid = np.zeros_like(ref, bool)
    valid[::16, :] = True
    valid[:, ::16] = True
    valid = gaussian_filter(valid.astype(float), 2) > 0.2
    a = highpass(mov, valid, 4.0)
    b = highpass(ref, np.ones_like(ref, bool), 4.0)
    nr, nc, _, _, _ = masked_ncc(b, a, valid, 8)
    assert abs(nr - 3.0) < 0.3 and abs(nc - 2.0) < 0.3


@pytest.mark.parametrize("seed,dr,dc", [(5, 12.0, 0.0), (7, 20.0, 9.0)])
def test_a_shift_outside_the_search_window_is_not_reported(seed, dr, dc):
    # Noise has a maximum too, and it must not be reported as a shift.
    ref = scene(seed=seed)
    mov = shifted(ref, dr, dc)
    valid = np.ones_like(ref, bool)
    nr, nc, _, z, _ = masked_ncc(highpass(ref, valid, 4.0), highpass(mov, valid, 4.0), valid, 6)
    assert np.isnan(nr) and np.isnan(nc), (nr, nc, z)


def test_rows_run_north_to_south_in_the_metre_conversion():
    # A feature one row further down is further south.
    assert to_metres(1.0, 0.0, 0.6) == (0.0, -0.6)
    assert to_metres(0.0, 1.0, 0.6) == (0.6, -0.0)


def test_pixel_noise_does_not_lock_the_estimate_to_whole_pixels():
    # The real rasters are noisy at the pixel scale, and there every estimate came
    # back in whole pixels. Noise at a third of the signal reproduces the effect.
    rng = np.random.default_rng(1)
    for dr, dc in [(1.5, -1.5), (-1.25, 0.75), (0.5, 0.3)]:
        ref = scene(seed=2)
        mov = shifted(ref, dr, dc)
        ref = ref + rng.normal(scale=0.35 * ref.std(), size=ref.shape)
        mov = mov + rng.normal(scale=0.35 * ref.std(), size=ref.shape)
        valid = np.ones_like(ref, bool)
        valid[::3, :] = False
        a = highpass(mov, valid, 4.0)
        b = highpass(ref, np.ones_like(valid), 4.0)
        nr, nc, _, _, _ = masked_ncc(b, a, valid, 6)
        assert abs(nr - dr) < 0.08 and abs(nc - dc) < 0.08, (dr, dc, nr, nc)


def test_a_snapped_window_reads_the_pixels_its_transform_describes(tmp_path):
    # The bug this pins: a fractional window was read from rounded offsets while its
    # transform kept the fraction, placing NAIP up to half a pixel off.
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import from_bounds
    from surrey.registration import snap_bounds

    data = np.arange(200 * 200, dtype=np.float32).reshape(200, 200)
    tr = from_origin(550000.0, 4190000.0, 0.6, 0.6)
    path = tmp_path / "grid.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=200,
        height=200,
        count=1,
        dtype="float32",
        transform=tr,
        crs="EPSG:26910",
    ) as ds:
        ds.write(data, 1)
    wanted = (550000.0 + 30.25, 4190000.0 - 90.4, 550000.0 + 70.33, 4190000.0 - 50.1)
    with rasterio.open(path) as ds:
        win = from_bounds(*snap_bounds(wanted, ds.transform), transform=ds.transform)
        arr = ds.read(1, window=win, boundless=True, fill_value=0)
        wtr = ds.window_transform(win)
    r0 = int(round((tr.f - wtr.f) / 0.6))
    c0 = int(round((wtr.c - tr.c) / 0.6))
    assert np.array_equal(arr, data[r0 : r0 + arr.shape[0], c0 : c0 + arr.shape[1]])
    assert wtr.c <= wanted[0] and wtr.f >= wanted[3]


def test_offset_selection_trusts_correlation_then_a_nearby_phase_then_nothing():
    from surrey.registration import footprint_offsets, naip_offsets

    def pair(ncc=None, phase=None):
        whole = {
            "ncc_east_m": None,
            "ncc_north_m": None,
            "phase_east_m": None,
            "phase_north_m": None,
        }
        if ncc:
            whole["ncc_east_m"], whole["ncc_north_m"] = ncc
        if phase:
            whole["phase_east_m"], whole["phase_north_m"] = phase
        return {"whole": whole}

    crs = {
        "search_radius_m": 6.0,
        "areas": [
            {
                "aoi": "a",
                "lidar_ground_intensity_vs_naip_red": pair(ncc=(1.0, -1.0), phase=(9.0, 9.0)),
                "footprints": {"lidar_low_raised_vs_footprints": pair(ncc=(1.6, -0.8))},
            },
            {
                "aoi": "b",
                "lidar_ground_intensity_vs_naip_red": pair(phase=(0.2, -0.3)),
                "footprints": {"lidar_low_raised_vs_footprints": pair(ncc=(2.0, -0.6))},
            },
            {
                "aoi": "c",
                "lidar_ground_intensity_vs_naip_red": pair(phase=(-88.0, -22.0)),
                "footprints": {"lidar_low_raised_vs_footprints": pair(ncc=(1.5, -0.9))},
            },
            {
                "aoi": "d",
                "lidar_ground_intensity_vs_naip_red": pair(),
                "footprints": {"lidar_low_raised_vs_footprints": pair(phase=(40.0, 3.0))},
            },
        ],
    }
    n = naip_offsets(crs)
    assert (n["a"]["east_m"], n["a"]["method"]) == (1.0, "ncc")
    assert (n["b"]["north_m"], n["b"]["method"]) == (-0.3, "phase")
    assert n["c"]["method"] == "unregistered" and n["c"]["east_m"] == 0.0
    f = footprint_offsets(crs)
    assert f["global"] == {"east_m": 1.6, "north_m": -0.8, "areas_used": 3}
    assert f["per_area"]["b"]["east_m"] == 2.0
    assert f["per_area"]["d"]["method"] == "global"
