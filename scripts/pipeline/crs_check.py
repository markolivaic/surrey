"""Do the lidar, the NAIP imagery and the footprints sit on the same coordinates?

Measured, because the metadata cannot say. The 3DEP EPT for CA_SanFrancisco_1_B23
is published in EPSG:3857, WGS 84. The delivery was almost certainly NAD83(2011).
A NAD83 delivery carried into Web Mercator is often treated as if NAD83 were WGS84,
with no shift at all, and then its numbers are still NAD83. NAIP is in NAD83 UTM, and
scripts/pipeline/naip_reference.py joined the two through pyproj's default, which is also a
zero shift. Two zero shifts may cancel. A correct datum transform on either side
alone would move San Francisco by about a metre and a half, which is the figure this
script compares against (computed below with PROJ's time-dependent ITRF2014 to
NAD83(2011) transformation at the flight epoch).

Lidar against NAIP. Mean intensity of the delivery's ground class, rasterised at
0.6 m, against NAIP's red band resampled to the same grid through the exact
transform the vegetation reference used. Road paint and kerbs show in both, and at
ground level NAIP has no relief displacement.

Lidar against footprints. Share of each cell's points standing 2 to 15 m above the
ground surface, against the footprint polygons rasterised on the same grid. Low
structures, so a footprint digitised from an image of a leaning tower does not pass
for a shift.

Each pair is registered twice, by masked normalised cross-correlation and by phase
correlation, over the whole area and over four interleaved subsets of it: a
checkerboard of 30 m squares split four ways, so every subset spans the whole area.
The uncertainty is the standard error of the four subset estimates. And the method
is checked on the real imagery each run: NAIP is resampled with shifts put in on
purpose, and the script reports whether it gets them back.

Changed after the first run, and why. The first run used a 3 by 3 split into 100 m
windows, and most windows held too few road markings for a peak to clear the bar,
so the standard error was missing in five of seven areas; subsets that span the
area replace them. And a planted north shift of exactly 1.5 pixels came back
0.3 m wrong, which is a parabola locking onto half pixels; the refinement now
evaluates the correlation itself at fractional shifts (surrey.registration.refine),
and a second planted shift off the half pixel is added.

And then the real cause. Even with the refinement, planted shifts came back in whole
pixels, and NAIP registered against its own planted copy did too, at a correlation of
0.9997. The NAIP read was misregistered, not the estimator: rasterio reads a
fractional window from offsets rounded to the nearest pixel while window_transform
keeps the fraction, so pixels sat up to half a pixel, 0.3 m, from their coordinates,
by an amount that changed with the bounds. The same read built the vegetation
reference in scripts/pipeline/naip_reference.py. Both now snap their bounds to whole pixels
(surrey.registration.snap_bounds), and the vegetation reference is rebuilt.

The blur that made fractional NAIP shifts measurable (surrey.registration.highpass,
smooth_px) broadened the footprint correlation until no footprint estimate cleared
the peak bar. Footprints are binary polygons and the raised share raster is not
noisy at the pixel scale, and neither passes through the NAIP read, so the footprint
pair runs without the blur, as it did in the first run, and gets its own planted
shift check by rasterising the polygons from a moved origin. Also decided after
seeing a run, and recorded here for that reason.

One more thing the first reading left out. "1.5 m west and 0.2 m north" is the lidar
relative to a raster holding NAD83 numbers, if the lidar alone had been moved into
WGS84. With the roles swapped, the lidar still holding NAD83 numbers and the other
raster truly in WGS84, the same mismatch reads 1.5 m east and 0.2 m south. Only the
first direction was written down before the first run; the second is added here,
after it, and is reported as such.

Reading, fixed before the first run:
    within 0.5 m of zero, in both axes and by both methods     consistent; the zero
                                                               shifts cancel
    near 1.5 m west and 0.2 m north                            one side took a real
                                                               datum shift
    anything else beyond the uncertainty                       reported as found, not
                                                               explained

Run:
    python scripts/pipeline/crs_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import laspy
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from surrey.footprints import clip_to_box, load_footprints  # noqa: E402
from surrey.reference import SURFACE_CELL_M, SURFACE_REACH_M, vendor_ground_surface  # noqa: E402
from surrey.registration import (  # noqa: E402
    highpass,
    masked_ncc,
    phase_correlation,
    snap_bounds,
    to_metres,
)

RAW = ROOT / "data" / "target" / "sf"
FOOTPRINTS = ROOT / "data" / "target" / "sf_footprints.geojson"
OUT = ROOT / "data" / "artifacts" / "crs_check.json"
AREAS = ["embarcadero", "financial", "ggpark", "mission", "soma", "steep", "sunset"]
RES_M = 0.6
SIGMA_PX = 4.0
SMOOTH_PX = 1.0  # lidar intensity and NAIP: noisy at the pixel scale
FOOTPRINT_SMOOTH_PX = 0.0  # binary footprints and a share raster: not
RADIUS_PX = 10  # 6 m each way
SUBSETS = 4
SUBSET_SQUARE_M = 30.0
MIN_SUBSET_VALID = 0.02
PLANTED_M = [(1.2, -0.9), (-0.75, 0.45)]  # east, north: shifts put in to check the method
LOW_BAND_M = (2.0, 15.0)
EPSG_LIDAR = 32610


def expected_datum_shift(lon: float, lat: float, epoch: float) -> dict:
    """East and north metres from NAD83(2011) to ITRF2014 at `epoch`, by PROJ."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:6319", "EPSG:7912", always_xy=True)
    lon2, lat2, _, _ = t.transform(lon, lat, 0.0, epoch)
    east = (lon2 - lon) * 111320.0 * np.cos(np.radians(lat))
    north = (lat2 - lat) * 110574.0
    return {
        "transformation": t.description,
        "epoch": epoch,
        "east_m": round(float(east), 3),
        "north_m": round(float(north), 3),
        "reading": "where a point given in NAD83(2011) lands in ITRF2014, which WGS84 tracks",
    }


def measurement_crs(name: str) -> dict:
    """The CRS the offsets are measured in, read from the file, with the scale error of
    a metre in it against a metre on the ground at the file's centre."""
    from pyproj import CRS, Proj

    with laspy.open(RAW / f"{name}.laz") as fh:
        crs = fh.header.parse_crs()
        cx = float((fh.header.mins[0] + fh.header.maxs[0]) / 2)
        cy = float((fh.header.mins[1] + fh.header.maxs[1]) / 2)
    epsg = crs.to_epsg() if crs is not None else None
    if not epsg:
        return {"epsg": None, "name": None, "metres_on_ground_per_crs_unit": None}
    proj = Proj(CRS.from_epsg(epsg))
    lon, lat = proj(cx, cy, inverse=True)
    factors = proj.get_factors(lon, lat)
    return {
        "epsg": epsg,
        "name": crs.name,
        "metres_on_ground_per_crs_unit": round(1.0 / factors.meridional_scale, 6),
    }


def read_raw(name: str):
    las = laspy.read(RAW / f"{name}.laz")
    return (
        np.asarray(las.x),
        np.asarray(las.y),
        np.asarray(las.z),
        np.asarray(las.intensity, np.float64),
        np.asarray(las.classification),
    )


def grid_for(x: np.ndarray, y: np.ndarray):
    x0 = float(np.floor(x.min()))
    y1 = float(np.ceil(y.max()))
    nc = int(np.ceil((x.max() - x0) / RES_M))
    nr = int(np.ceil((y1 - y.min()) / RES_M))
    return x0, y1, nr, nc


def cell_of(x, y, x0, y1, nr, nc):
    c = np.clip(((x - x0) / RES_M).astype(np.int64), 0, nc - 1)
    r = np.clip(((y1 - y) / RES_M).astype(np.int64), 0, nr - 1)
    return r, c


def mean_raster(r, c, values, nr, nc):
    s = np.bincount(r * nc + c, weights=values, minlength=nr * nc)
    n = np.bincount(r * nc + c, minlength=nr * nc)
    img = np.where(n > 0, s / np.maximum(n, 1), 0.0).reshape(nr, nc)
    return img, (n > 0).reshape(nr, nc)


def naip_red(x0, y1, nr, nc, planted_m=(0.0, 0.0)):
    """NAIP red on the lidar grid, through the transform naip_reference.py used.

    `planted_m` samples the imagery at cell centre minus a known offset, so the
    returned raster shows the scene displaced by that offset.
    """
    import rasterio
    from naip_reference import cached_image, naip_items
    from pyproj import Transformer
    from rasterio.merge import merge
    from rasterio.windows import from_bounds

    cols = x0 + (np.arange(nc) + 0.5) * RES_M - planted_m[0]
    rows = y1 - (np.arange(nr) + 0.5) * RES_M - planted_m[1]
    ex, ny = np.meshgrid(cols, rows)
    to_ll = Transformer.from_crs(EPSG_LIDAR, 4326, always_xy=True)
    lon, lat = to_ll.transform(ex.ravel(), ny.ravel())
    bbox = (lon.min() - 0.001, lat.min() - 0.001, lon.max() + 0.001, lat.max() + 0.001)
    items = naip_items(bbox)
    dss = [rasterio.open(cached_image(it)) for it in items]
    epsg_img = dss[0].crs.to_epsg()
    to_img = Transformer.from_crs(EPSG_LIDAR, epsg_img, always_xy=True)
    ix, iy = to_img.transform(ex.ravel(), ny.ravel())
    bounds = snap_bounds(
        (ix.min() - 20, iy.min() - 20, ix.max() + 20, iy.max() + 20), dss[0].transform
    )
    if len(dss) == 1:
        # Whole pixels, or the read and its transform disagree by the rounded fraction.
        win = from_bounds(*bounds, transform=dss[0].transform)
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        arr = dss[0].read(1, window=win, boundless=True, fill_value=0).astype(np.float64)
        tr = dss[0].window_transform(win)
    else:
        merged, tr = merge(dss, bounds=bounds, nodata=0, indexes=[1])
        arr = merged[0].astype(np.float64)
    for d in dss:
        d.close()
    pc = (ix - tr.c) / tr.a - 0.5
    pr = (iy - tr.f) / tr.e - 0.5
    red = map_coordinates(arr, [pr, pc], order=1, mode="constant", cval=0.0).reshape(nr, nc)
    return (
        red,
        red > 0,
        {"items": [it["id"] for it in items], "epsg": epsg_img, "transform": to_img.description},
    )


def register(
    reference, ref_valid, moving, mov_valid, subsets: bool = True, smooth_px: float = SMOOTH_PX
) -> dict:
    """Whole-area and subset estimates, metres east and north of moving over reference."""
    ref = highpass(reference, ref_valid, SIGMA_PX, smooth_px)
    mov = highpass(moving, mov_valid, SIGMA_PX, smooth_px)
    use = mov_valid & ref_valid

    def one(v) -> dict:
        if v.mean() < MIN_SUBSET_VALID:
            return {"skipped": f"valid share {v.mean():.3f} under {MIN_SUBSET_VALID}"}
        dr, dc, peak, z, _ = masked_ncc(ref, mov, v, RADIUS_PX)
        pr, pc, presp = phase_correlation(ref, np.where(v, mov, 0.0))
        e, n = to_metres(dr, dc, RES_M)
        pe, pn = to_metres(pr, pc, RES_M)
        return {
            "ncc_east_m": None if np.isnan(e) else round(e, 3),
            "ncc_north_m": None if np.isnan(n) else round(n, 3),
            "ncc_peak": round(peak, 3),
            "ncc_peak_z": round(z, 1),
            "phase_east_m": round(pe, 3),
            "phase_north_m": round(pn, 3),
            "phase_peak": round(presp, 4),
            "valid_share": round(float(v.mean()), 3),
        }

    out = {"whole": one(use)}
    if not subsets:
        return out
    h, w = use.shape
    sq = int(round(SUBSET_SQUARE_M / RES_M))
    br = (np.arange(h) // sq)[:, None]
    bc = (np.arange(w) // sq)[None, :]
    label = (br % 2) * 2 + (bc % 2)
    parts = [one(use & (label == k)) for k in range(SUBSETS)]
    out["subsets"] = parts
    for k in ("ncc_east_m", "ncc_north_m", "phase_east_m", "phase_north_m"):
        vals = np.array([p[k] for p in parts if p.get(k) is not None], dtype=float)
        out[f"{k}_subset_mean"] = round(float(vals.mean()), 3) if vals.size else None
        out[f"{k}_subset_se"] = (
            round(float(vals.std(ddof=1) / np.sqrt(vals.size)), 3) if vals.size > 1 else None
        )
        out[f"{k}_subsets_used"] = int(vals.size)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()

    from pyproj import Transformer

    polys_all = load_footprints(FOOTPRINTS, EPSG_LIDAR)
    rows = []
    for name in AREAS:
        x, y, z, inten, cls = read_raw(name)
        x0, y1, nr, nc = grid_for(x, y)
        r, c = cell_of(x, y, x0, y1, nr, nc)

        ground = cls == 2
        lo, hi = np.percentile(inten[ground], [1, 99])
        lidar_int, lidar_valid = mean_raster(
            r[ground], c[ground], np.clip(inten[ground], lo, hi), nr, nc
        )
        red, red_valid, naip_meta = naip_red(x0, y1, nr, nc)
        row = {
            "aoi": name,
            "measurement_crs": measurement_crs(name),
            "grid": {"res_m": RES_M, "rows": nr, "cols": nc},
            "naip": naip_meta,
            "ground_cells_share": round(float(lidar_valid.mean()), 3),
        }
        row["lidar_ground_intensity_vs_naip_red"] = register(red, red_valid, lidar_int, lidar_valid)

        # The method on this very imagery, with shifts put in on purpose. NAIP drawn
        # at centre minus the planted offset shows the scene moved by that offset, so
        # the lidar, measured against it, must come back displaced by minus it.
        base = row["lidar_ground_intensity_vs_naip_red"]["whole"]
        checks = []
        for planted in PLANTED_M:
            red_p, red_p_valid, _ = naip_red(x0, y1, nr, nc, planted)
            got = register(red_p, red_p_valid, lidar_int, lidar_valid, subsets=False)["whole"]
            have = None not in (
                got.get("ncc_east_m"),
                base.get("ncc_east_m"),
                got.get("ncc_north_m"),
                base.get("ncc_north_m"),
            )
            checks.append(
                {
                    "planted_east_north_m": list(planted),
                    "expected_change_east_north_m": [-planted[0], -planted[1]],
                    "ncc_change_east_north_m": (
                        [
                            round(got["ncc_east_m"] - base["ncc_east_m"], 3),
                            round(got["ncc_north_m"] - base["ncc_north_m"], 3),
                        ]
                        if have
                        else None
                    ),
                }
            )
        row["planted_shift_checks"] = checks

        # Footprints against low raised lidar points.
        clean = ~np.isin(cls, (7, 18))
        grid, key = vendor_ground_surface(x, y, z, ground & clean, SURFACE_CELL_M, SURFACE_REACH_M)
        h = z - grid[key]
        band = clean & np.isfinite(h) & (h > LOW_BAND_M[0]) & (h <= LOW_BAND_M[1])
        raised, _ = mean_raster(r[clean], c[clean], band[clean].astype(np.float64), nr, nc)
        any_pts = np.bincount(r[clean] * nc + c[clean], minlength=nr * nc).reshape(nr, nc) > 0
        clipped, _ = clip_to_box(polys_all, x, y)
        if clipped:
            from rasterio.features import rasterize
            from rasterio.transform import from_origin

            def footprint_raster(shift_m=(0.0, 0.0), polys=clipped, shape=(nr, nc), ox=x0, oy=y1):
                # Rasterised with the origin moved by minus the shift, the polygons land
                # displaced by plus the shift: an exact translation, no interpolation.
                return rasterize(
                    [(q, 1) for q in polys],
                    out_shape=shape,
                    transform=from_origin(ox - shift_m[0], oy - shift_m[1], RES_M, RES_M),
                    fill=0,
                    dtype="uint8",
                ).astype(np.float64)

            ones = np.ones_like(any_pts)
            reg = register(footprint_raster(), ones, raised, any_pts, smooth_px=FOOTPRINT_SMOOTH_PX)
            base_fp = reg["whole"]
            fp_checks = []
            for planted in PLANTED_M:
                got = register(
                    footprint_raster(planted),
                    ones,
                    raised,
                    any_pts,
                    subsets=False,
                    smooth_px=FOOTPRINT_SMOOTH_PX,
                )["whole"]
                have = None not in (
                    got.get("ncc_east_m"),
                    base_fp.get("ncc_east_m"),
                    got.get("ncc_north_m"),
                    base_fp.get("ncc_north_m"),
                )
                fp_checks.append(
                    {
                        "planted_east_north_m": list(planted),
                        "expected_change_east_north_m": [-planted[0], -planted[1]],
                        "ncc_change_east_north_m": (
                            [
                                round(got["ncc_east_m"] - base_fp["ncc_east_m"], 3),
                                round(got["ncc_north_m"] - base_fp["ncc_north_m"], 3),
                            ]
                            if have
                            else None
                        ),
                    }
                )
            row["footprints"] = {
                "polygons": len(clipped),
                "smooth_px": FOOTPRINT_SMOOTH_PX,
                "lidar_low_raised_vs_footprints": reg,
                "planted_shift_checks": fp_checks,
            }
        else:
            row["footprints"] = {"polygons": 0}
        rows.append(row)
        for label_, reg in (
            ("naip", row["lidar_ground_intensity_vs_naip_red"]),
            ("footprints", row["footprints"].get("lidar_low_raised_vs_footprints")),
        ):
            if reg is None:
                continue
            wh = reg["whole"]
            print(
                f"  {name:12s} {label_:10s} whole ncc E {wh.get('ncc_east_m')} N {wh.get('ncc_north_m')} "
                f"z {wh.get('ncc_peak_z')} | subsets E {reg['ncc_east_m_subset_mean']}"
                f"+-{reg['ncc_east_m_subset_se']} N {reg['ncc_north_m_subset_mean']}"
                f"+-{reg['ncc_north_m_subset_se']} (n={reg['ncc_east_m_subsets_used']}) | "
                f"phase E {wh.get('phase_east_m')} N {wh.get('phase_north_m')}",
                flush=True,
            )
        print(
            f"  {name:12s} planted naip {[c['ncc_change_east_north_m'] for c in row['planted_shift_checks']]}  "
            f"footprints {[c['ncc_change_east_north_m'] for c in row['footprints'].get('planted_shift_checks', [])]}",
            flush=True,
        )

    to_ll = Transformer.from_crs(EPSG_LIDAR, 4326, always_xy=True)
    lon, lat = to_ll.transform(551500.0, 4181000.0)
    result = {
        "question": "are lidar, NAIP and footprints on the same coordinates",
        "convention": "metres east and north by which the lidar raster is displaced "
        "relative to the other raster; positive east means lidar lies east",
        "measurement_crs": "EPSG:32610, WGS 84 / UTM zone 10N; offsets are metres in that CRS, "
        "which differ from metres on the ground by the UTM scale factor, about "
        "0.9996, recorded per area",
        "lidar": "EPT CA_SanFrancisco_1_B23, published EPSG:3857, reprojected to EPSG:32610 "
        "by scripts/pipeline/fetch_sf_aoi.py",
        "expected_if_one_side_took_a_real_datum_shift": expected_datum_shift(lon, lat, 2023.3),
        "reading_fixed_before_running": {
            "consistent": "within 0.5 m of zero in both axes and by both methods",
            "datum_shift": "near 1.5 m west and 0.2 m north",
            "otherwise": "reported as found, not explained",
        },
        "resolution_m": RES_M,
        "highpass_sigma_px": SIGMA_PX,
        "search_radius_m": RADIUS_PX * RES_M,
        "areas": rows,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"offsets measured in: {result['measurement_crs']}")
    print(f"expected datum shift: {result['expected_if_one_side_took_a_real_datum_shift']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
