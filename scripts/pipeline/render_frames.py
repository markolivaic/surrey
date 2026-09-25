"""Still frames of real predictions, for design mockups only.

Renders an oblique orthographic view of one San Francisco area from a prediction cache,
class as hue and confidence as saturation, and a top-down NAIP frame over the same
square for an orthophoto wipe. These are stills for choosing a layout. Every frame is
stamped with its input variant in the corner, because a ret0 frame is not the published
model and must not be read as a result.

No viewer code: a numpy splat with a painter's depth order, one pixel per point.

Run:
    python scripts/pipeline/render_frames.py --area ggpark --variant ret0
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.metrics import coarse_lookup  # noqa: E402

TARGET_PREP = ROOT / "data" / "prep" / "target"
PRED = ROOT / "data" / "predictions"
# Frames are working material: the page copies its stills from here, and the few that are evidence
# for the palette test were copied into docs/design/frames, which the design record names.
OUT = ROOT / "data" / "cache" / "frames"
VARIANT_DIRS = {"returns_ood": PRED, "ret0": PRED / "ret0", "m0_seedA": PRED / "m0_seedA"}
VARIANT_LABEL = {
    "returns_ood": "input as delivered (the bug)",
    "ret0": "ret0: return ratio forced to 0, not the published model",
    "m0_seedA": "M0 seed A, the published model",
}
# Hue per coarse class, in degrees. Water and bridge deck are reserved for M1.
HUES = {
    "ground": 38,
    "vegetation": 118,
    "building": 352,
    "vehicle": 292,
    "pole": 188,
    "fence": 20,
    "water": 212,
    "bridge_deck": 262,
}
GROUPS = ["ground", "vegetation", "building", "vehicle", "pole", "fence"]
PALETTE_GROUPS = GROUPS + ["water", "bridge_deck"]


def palette_colours(coarse: np.ndarray, conf: np.ndarray, theme: str) -> np.ndarray:
    """RGB per point from docs/design/palette.json: the class's OKLCH lightness and hue, chroma
    scaled by confidence through the map fixed on Surrey. No height shading, because lightness
    carries contrast and must not move."""
    from surrey.colour_vision import srgb_from_oklch
    from surrey.palette import load_palette

    pal = load_palette()
    cmap = json.loads(
        (ROOT / "data" / "artifacts" / "confidence_mapping.json").read_text(encoding="utf-8")
    )
    tokens = pal["themes"][theme]["tokens"]
    floor = pal["floor"] if pal["floor"] is not None else 0.2
    names = [g.replace("_", " ") for g in PALETTE_GROUPS]
    L = np.array([tokens[g]["L"] for g in names])[coarse]
    C = np.array([tokens[g]["C"] for g in names])[coarse]
    H = np.array([tokens[g]["h"] for g in names])[coarse]
    t = np.clip((conf - cmap["low"]) / (cmap["high"] - cmap["low"]), 0.0, 1.0)
    rgb, _ = srgb_from_oklch(L, C * (floor + (1 - floor) * t), H)
    return rgb.astype(np.float32)


def colours(coarse: np.ndarray, conf: np.ndarray, shade: np.ndarray) -> np.ndarray:
    """RGB per point: hue by class, saturation by confidence, value by a height shade."""
    hue = np.array([HUES[g] / 360.0 for g in GROUPS])[coarse]
    sat = np.clip((conf - 0.35) / 0.6, 0.04, 0.82)
    val = 0.62 + 0.26 * shade
    rgb = np.empty((coarse.size, 3), np.float32)
    # colorsys is scalar; vectorise over a small table of quantised saturation and value.
    hq = np.round(hue * 360).astype(np.int16)
    sq = np.round(sat * 20).astype(np.int16)
    vq = np.round(val * 20).astype(np.int16)
    key = (hq.astype(np.int64) * 21 + sq) * 21 + vq
    uniq, inv = np.unique(key, return_inverse=True)
    table = np.array(
        [
            colorsys.hsv_to_rgb((k // 441) / 360.0, ((k // 21) % 21) / 20.0, (k % 21) / 20.0)
            for k in uniq
        ],
        np.float32,
    )
    rgb[:] = table[inv]
    return rgb


def naip_rgb(tile: CachedTile) -> np.ndarray | None:
    """NAIP colour at every point, for the orthophoto side of a wipe in the same view."""
    import rasterio
    from pyproj import Transformer

    ux, uy = tile.utm_xy(np.arange(tile.n))
    out = np.zeros((tile.n, 3), np.float32)
    got = np.zeros(tile.n, bool)
    for path in sorted((ROOT / "data" / "cache" / "naip").glob("*.tif")):
        with rasterio.open(path) as ds:
            t = Transformer.from_crs(tile.epsg, ds.crs.to_epsg(), always_xy=True)
            ix, iy = t.transform(ux, uy)
            inv = ~ds.transform
            cols = np.floor(inv.a * ix + inv.b * iy + inv.c).astype(np.int64)
            rows = np.floor(inv.d * ix + inv.e * iy + inv.f).astype(np.int64)
            ok = (~got) & (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
            if not ok.any():
                continue
            arr = ds.read([1, 2, 3])
            out[ok] = arr[:, rows[ok], cols[ok]].T / 255.0
            got |= ok
    # NAIP over San Francisco is dark; a gamma lift keeps canopy readable on screen.
    return np.power(out, 0.7) if got.any() else None


def oblique(
    tile: CachedTile,
    coarse,
    conf,
    width: int,
    azimuth_deg: float,
    elev_deg: float,
    rgb_override: np.ndarray | None = None,
    background=(0.0, 0.0, 0.0),
    edl_strength: float = 0.9,
    theme: str | None = None,
    index_out: dict | None = None,
):
    x = np.asarray(tile.xy_cm[:, 0], np.float64) / 100.0
    y = np.asarray(tile.xy_cm[:, 1], np.float64) / 100.0
    z = np.asarray(tile.z_cm[:], np.float64) / 100.0
    z = z - np.percentile(z, 1)
    a, e = np.radians(azimuth_deg), np.radians(elev_deg)
    xr = np.cos(a) * x - np.sin(a) * y
    yr = np.sin(a) * x + np.cos(a) * y
    u = xr
    v = yr * np.sin(e) + z * np.cos(e)
    depth = yr * np.cos(e) - z * np.sin(e)
    scale = (width - 1) / (u.max() - u.min())
    px = ((u - u.min()) * scale).astype(np.int64)
    h = int((v.max() - v.min()) * scale) + 1
    py = (h - 1 - (v - v.min()) * scale).astype(np.int64)
    shade = np.clip(z / np.percentile(z, 99), 0, 1)
    if rgb_override is not None:
        rgb = rgb_override
    elif theme is not None:
        rgb = palette_colours(coarse, conf, theme)
    else:
        rgb = colours(coarse, conf, shade)
    order = np.argsort(-depth, kind="stable")  # far first, near last overwrites
    img = np.zeros((h, width, 3), np.float32)
    img[:] = np.asarray(background, np.float32)
    img[py[order], px[order]] = rgb[order]
    if index_out is not None:
        # Which point ended up in each pixel, -1 for background, for checks on the frame itself.
        ids = np.full((h, width), -1, np.int64)
        ids[py[order], px[order]] = order
        index_out["ids"] = ids
    # Eye-dome lighting: a pixel whose neighbours are nearer to the eye is an edge or a
    # crevice, and darkening it by how much nearer they are gives shape without normals.
    dimg = np.full((h, width), np.nan, np.float32)
    dimg[py[order], px[order]] = depth[order] * scale
    filled = np.isfinite(dimg)
    d = np.where(filled, dimg, np.nanmax(dimg))
    occl = np.zeros_like(d)
    for dy, dx in ((0, 2), (0, -2), (2, 0), (-2, 0), (2, 2), (-2, -2), (2, -2), (-2, 2)):
        occl += np.maximum(0.0, d - np.roll(np.roll(d, dy, 0), dx, 1))
    edl = np.exp(-edl_strength * occl / 8.0)
    img = np.where(filled[..., None], img * edl[..., None], img)
    return img


def naip_topdown(tile: CachedTile, size: int):
    import rasterio
    from pyproj import Transformer

    ex, ey = tile.extent_m()
    xs = np.linspace(0, ex, size)
    ys = np.linspace(ey, 0, size)
    gx, gy = np.meshgrid(xs, ys)
    ux, uy = gx + tile.origin[0], gy + tile.origin[1]
    for path in sorted((ROOT / "data" / "cache" / "naip").glob("*.tif")):
        with rasterio.open(path) as ds:
            t = Transformer.from_crs(tile.epsg, ds.crs.to_epsg(), always_xy=True)
            ix, iy = t.transform(ux.ravel(), uy.ravel())
            b = ds.bounds
            if not (
                b.left <= ix.min()
                and ix.max() <= b.right
                and b.bottom <= iy.min()
                and iy.max() <= b.top
            ):
                continue
            inv = ~ds.transform
            cols = np.floor(inv.a * ix + inv.b * iy + inv.c).astype(np.int64)
            rows = np.floor(inv.d * ix + inv.e * iy + inv.f).astype(np.int64)
            arr = ds.read([1, 2, 3])
            img = arr[:, np.clip(rows, 0, arr.shape[1] - 1), np.clip(cols, 0, arr.shape[2] - 1)]
            return (img.T.reshape(size, size, 3) / 255.0).astype(np.float32)
    return None


def stamp(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 9 * len(text) + 16, 26], fill=(0, 0, 0))
    d.text((8, 7), text, fill=(255, 255, 255))
    return np.asarray(im)


def main() -> int:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--area", required=True)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANT_DIRS))
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--azimuth", type=float, default=30.0)
    ap.add_argument("--elev", type=float, default=38.0)
    ap.add_argument("--naip", action="store_true")
    ap.add_argument("--colour", choices=("classes", "naip"), default="classes")
    ap.add_argument(
        "--bg", default="000000", help="background as hex, for placing on a page ground"
    )
    ap.add_argument("--no-stamp", action="store_true")
    ap.add_argument(
        "--surrey", action="store_true", help="a held-out DALES tile instead of an SF area"
    )
    ap.add_argument(
        "--theme",
        choices=("light", "dark"),
        default=None,
        help="colour from docs/design/palette.json in this theme; without it, the first palette",
    )
    ap.add_argument(
        "--edl",
        type=float,
        default=0.9,
        help="eye-dome lighting strength; the first frames used 0.9 and their halos were too dark",
    )
    args = ap.parse_args()
    bg = tuple(int(args.bg[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    OUT.mkdir(parents=True, exist_ok=True)
    if args.surrey:
        tile = CachedTile(ROOT / "data" / "prep" / "dales" / args.area, "test")
        z = np.load(VARIANT_DIRS[args.variant] / f"surrey_{args.area}.npz")
    else:
        tile = CachedTile(TARGET_PREP / args.area, "sf")
        z = np.load(VARIANT_DIRS[args.variant] / f"sf_{args.area}.npz")
    coarse = coarse_lookup()[z["pred"]]
    override = naip_rgb(tile) if args.colour == "naip" else None
    img = oblique(
        tile,
        coarse,
        z["conf"].astype(np.float32),
        args.width,
        args.azimuth,
        args.elev,
        override,
        bg,
        args.edl,
        args.theme,
    )
    label = "NAIP 2022 colour on lidar" if args.colour == "naip" else VARIANT_LABEL[args.variant]
    if args.surrey:
        label = f"Surrey held-out tile, {VARIANT_LABEL[args.variant]}"
    img = (
        (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if args.no_stamp
        else stamp(img, f"{args.area}  |  {label}")
    )
    suffix = (
        "naip"
        if args.colour == "naip"
        else (f"home_{args.variant}" if args.surrey else args.variant)
    )
    edl_tag = "" if args.edl == 0.9 else f"_edl{str(args.edl).replace('.', '')}"
    if args.theme and args.colour != "naip":
        edl_tag += "_p2"
    path = (
        OUT
        / f"{args.area}_{suffix}_oblique{'_' + args.bg if args.bg != '000000' else ''}{edl_tag}.png"
    )
    Image.fromarray(img).save(path, optimize=True)
    print(f"wrote {path} {img.shape}")
    if args.naip:
        top = naip_topdown(tile, 1000)
        if top is not None:
            p2 = OUT / f"{args.area}_naip_topdown.png"
            Image.fromarray(stamp(top, f"{args.area}  |  NAIP 2022, public domain")).save(
                p2, optimize=True
            )
            print(f"wrote {p2}")
    meta = OUT / "frames.json"
    doc = json.loads(meta.read_text()) if meta.exists() else {}
    doc[path.name] = {
        "area": args.area,
        "variant": args.variant,
        "label": label,
        "edl": args.edl,
        "palette": "docs/design/palette.json" if args.theme else "first palette",
        "note": "design still, not a result",
    }
    meta.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
