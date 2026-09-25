"""Sub-pixel shift between two rasters of the same ground, two ways.

Used to measure, rather than assume, whether the lidar, the NAIP imagery and the
building footprints sit on the same coordinates. Metadata cannot settle it: the
EPT is published in Web Mercator, and whether the NAD83 delivery was carried into
it with a real datum shift or with none leaves no trace in the file.

One convention throughout. A shift d = (dr, dc) in pixels means

    moving[r, c]  ~  reference[r - dr, c - dc]

that is, a feature at (r, c) in the reference appears at (r + dr, c + dc) in the
moving raster. Rows run north to south, so in metres east and north the moving
raster is displaced by (dc * res, -dr * res).

Two estimators, because agreement between them is worth more than either alone:

    phase correlation   the whole spectrum at once; the moving raster is masked by
                        filling outside its valid cells with zero after centring
    masked NCC          normalised cross-correlation over the moving raster's own
                        valid cells only, on an integer grid, refined by a parabola

The reference raster is never masked. A mask applied to both would share its edges
between them and pull every estimate toward zero.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def highpass(
    img: np.ndarray, valid: np.ndarray, sigma_px: float, smooth_px: float = 1.0
) -> np.ndarray:
    """Keep the band between `smooth_px` and `sigma_px`, averaging only over valid cells.

    Normalised convolution: blur the values and the mask separately and divide, so
    a cell next to an invalid region is not dragged toward zero. Returns zero
    outside `valid`, centred and scaled to unit variance inside it.

    The small blur is what makes a fractional shift measurable. Without it the
    refinement compares a raster interpolated between pixels, and so smoothed,
    against one that is not, and on pixel-noisy rasters the unsmoothed whole-pixel
    shifts always correlate better: every estimate on the real lidar and NAIP
    rasters came back in exact multiples of the 0.6 m pixel.
    """
    v = valid.astype(np.float64)
    x = np.where(valid, img, 0.0).astype(np.float64)
    low = gaussian_filter(x, sigma_px) / np.maximum(gaussian_filter(v, sigma_px), 1e-6)
    out = np.where(valid, x - low, 0.0)
    if smooth_px > 0:
        out = np.where(
            valid,
            gaussian_filter(out, smooth_px) / np.maximum(gaussian_filter(v, smooth_px), 1e-6),
            0.0,
        )
    if valid.sum() > 1:
        m, s = out[valid].mean(), out[valid].std()
        out = np.where(valid, (out - m) / max(s, 1e-12), 0.0)
    return out


def _parabola(vm: float, v0: float, vp: float) -> float:
    denom = vm - 2.0 * v0 + vp
    return 0.0 if abs(denom) < 1e-12 else float(np.clip(0.5 * (vm - vp) / denom, -0.5, 0.5))


def phase_correlation(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    """(dr, dc, peak) with a Hann window and a parabolic peak refinement."""
    h, w = reference.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    fr = np.fft.fft2(reference * win)
    fm = np.fft.fft2(moving * win)
    cross = fm * np.conj(fr)
    cross /= np.maximum(np.abs(cross), 1e-12)
    r = np.real(np.fft.ifft2(cross))
    pr, pc = np.unravel_index(int(np.argmax(r)), r.shape)
    dr = _parabola(r[(pr - 1) % h, pc], r[pr, pc], r[(pr + 1) % h, pc])
    dc = _parabola(r[pr, (pc - 1) % w], r[pr, pc], r[pr, (pc + 1) % w])
    sr = pr if pr <= h // 2 else pr - h
    sc = pc if pc <= w // 2 else pc - w
    return float(sr + dr), float(sc + dc), float(r[pr, pc])


# A correlation surface that holds no real match still has a maximum, and with no
# guard it is reported as a shift. On synthetic scenes a true peak stood 13 to 15
# robust standard deviations above the surface median and an out-of-range shift
# 1.3 to 2.3, so the bar is set between them, before any real raster was scored.
MIN_PEAK_Z = 5.0


def peak_z(surface: np.ndarray) -> float:
    """How far the maximum stands above the rest, in robust standard deviations."""
    f = surface[np.isfinite(surface)]
    if f.size < 5:
        return float("nan")
    med = np.median(f)
    mad = 1.4826 * np.median(np.abs(f - med))
    return float((f.max() - med) / mad) if mad > 0 else float("inf")


def masked_ncc(
    reference: np.ndarray, moving: np.ndarray, valid: np.ndarray, radius_px: int
) -> tuple[float, float, float, float, np.ndarray]:
    """(dr, dc, peak, peak_z, surface) searching integer shifts within `radius_px`.

    For each shift the moving raster's valid cells are compared with the reference
    at the shifted positions, and cells that would fall outside the reference are
    dropped. The shift comes back as NaN when the peak sits on the edge of the
    search window, because a parabola through an edge says nothing about where the
    true maximum is, and when it does not stand out by MIN_PEAK_Z, because then
    there is no match to locate.
    """
    h, w = moving.shape
    size = 2 * radius_px + 1
    surface = np.full((size, size), np.nan)
    rr, cc = np.nonzero(valid)
    mv = moving[rr, cc]
    for i, dr in enumerate(range(-radius_px, radius_px + 1)):
        for j, dc in enumerate(range(-radius_px, radius_px + 1)):
            r0, c0 = rr - dr, cc - dc
            ok = (r0 >= 0) & (r0 < h) & (c0 >= 0) & (c0 < w)
            if ok.sum() < 100:
                continue
            a = mv[ok]
            b = reference[r0[ok], c0[ok]]
            a = a - a.mean()
            b = b - b.mean()
            denom = np.sqrt((a * a).sum() * (b * b).sum())
            surface[i, j] = (a * b).sum() / denom if denom > 0 else np.nan
    if not np.isfinite(surface).any():
        return float("nan"), float("nan"), float("nan"), float("nan"), surface
    i, j = np.unravel_index(int(np.nanargmax(surface)), surface.shape)
    peak = float(surface[i, j])
    z = peak_z(surface)
    if i in (0, size - 1) or j in (0, size - 1) or not z >= MIN_PEAK_Z:
        return float("nan"), float("nan"), peak, z, surface
    dr, dc = refine(reference, mv, rr, cc, float(i - radius_px), float(j - radius_px))
    return dr, dc, peak, z, surface


def _ncc_at(
    reference: np.ndarray, mv: np.ndarray, rr: np.ndarray, cc: np.ndarray, dr: float, dc: float
) -> float:
    from scipy.ndimage import map_coordinates

    h, w = reference.shape
    r0, c0 = rr - dr, cc - dc
    ok = (r0 >= 0) & (r0 <= h - 1) & (c0 >= 0) & (c0 <= w - 1)
    if ok.sum() < 100:
        return float("nan")
    a = mv[ok] - mv[ok].mean()
    b = map_coordinates(reference, [r0[ok], c0[ok]], order=1, mode="nearest")
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def refine(reference, mv, rr, cc, dr0: float, dc0: float) -> tuple[float, float]:
    """Fractional shift by evaluating the correlation itself between the pixels.

    A parabola through three integer samples locks onto whole and half pixels: a
    planted north shift of exactly one and a half pixels came back 0.3 m wrong in
    both directions on the real imagery. So the correlation is evaluated at
    fractional shifts, the reference interpolated bilinearly, first on a 0.2 px
    grid within 0.6 px of the integer peak and then on a 0.04 px grid around the
    best of those.
    """
    best = (dr0, dc0)
    for step, half in ((0.2, 0.6), (0.04, 0.2)):
        offs = np.arange(-half, half + 1e-9, step)
        cr, cc0 = best
        top = -np.inf
        for a in offs:
            for b in offs:
                v = _ncc_at(reference, mv, rr, cc, float(cr + a), float(cc0 + b))
                if v > top:
                    top, best = v, (cr + a, cc0 + b)
    return float(best[0]), float(best[1])


def to_metres(dr: float, dc: float, res_m: float) -> tuple[float, float]:
    """Pixel shift to (east, north) metres, rows running north to south."""
    return dc * res_m, -dr * res_m


def snap_bounds(bounds, transform):
    """Expand bounds outward to whole pixels of `transform`'s grid.

    rasterio reads a fractional window from offsets rounded to the nearest pixel
    while reporting the unrounded transform, so an unsnapped read puts pixels up to
    half a pixel away from their coordinates. Snapped bounds leave nothing to round.
    """
    import math

    a, e = transform.a, transform.e
    x0 = transform.c + math.floor((bounds[0] - transform.c) / a) * a
    x1 = transform.c + math.ceil((bounds[2] - transform.c) / a) * a
    y1 = transform.f + math.floor((bounds[3] - transform.f) / e) * e
    y0 = transform.f + math.ceil((bounds[1] - transform.f) / e) * e
    return (x0, y0, x1, y1)


def pick_offset(pair: dict, radius_m: float) -> tuple[float, float, str] | None:
    """One area's offset from a crs_check pair, by a fixed order of trust.

    The correlation estimate when it cleared the peak bar; otherwise the phase
    correlation when it lands inside the search radius, since a phase peak far outside
    it on a sparse raster is noise (Embarcadero's came back 88 m away); otherwise None.
    """
    whole = pair.get("whole", {})
    if whole.get("ncc_east_m") is not None and whole.get("ncc_north_m") is not None:
        return float(whole["ncc_east_m"]), float(whole["ncc_north_m"]), "ncc"
    pe, pn = whole.get("phase_east_m"), whole.get("phase_north_m")
    if pe is not None and pn is not None and (pe * pe + pn * pn) ** 0.5 <= radius_m:
        return float(pe), float(pn), "phase"
    return None


def naip_offsets(crs: dict) -> dict:
    """Per area lidar-minus-NAIP offset, east and north metres, or zero and a flag."""
    radius = float(crs["search_radius_m"])
    out = {}
    for a in crs["areas"]:
        got = pick_offset(a["lidar_ground_intensity_vs_naip_red"], radius)
        out[a["aoi"]] = (
            {"east_m": got[0], "north_m": got[1], "method": got[2]}
            if got
            else {"east_m": 0.0, "north_m": 0.0, "method": "unregistered"}
        )
    return out


def footprint_offsets(crs: dict) -> dict:
    """The global lidar-minus-footprint offset and the per-area ones behind it.

    Global is the median, per axis, over areas whose correlation estimate cleared the
    peak bar. Per area uses that area's own estimate by pick_offset, and falls back to
    the global one where it has none, which is what a sensitivity run needs.
    """
    radius = float(crs["search_radius_m"])
    ncc = []
    per = {}
    for a in crs["areas"]:
        pair = a.get("footprints", {}).get("lidar_low_raised_vs_footprints")
        got = pick_offset(pair, radius) if pair else None
        per[a["aoi"]] = got
        if got and got[2] == "ncc":
            ncc.append(got[:2])
    if not ncc:
        raise ValueError("no area has a footprint offset that cleared the peak bar")
    east = float(np.median([e for e, _ in ncc]))
    north = float(np.median([n for _, n in ncc]))
    areas = {
        aoi: (
            {"east_m": g[0], "north_m": g[1], "method": g[2]}
            if g
            else {"east_m": east, "north_m": north, "method": "global"}
        )
        for aoi, g in per.items()
    }
    return {"global": {"east_m": east, "north_m": north, "areas_used": len(ncc)}, "per_area": areas}
