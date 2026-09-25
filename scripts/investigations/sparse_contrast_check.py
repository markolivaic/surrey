"""Contrast on real frames, at the sparsest places, in both themes: not by formula.

A palette can pass on its tokens and still fail where a point stands alone: one pixel of
colour with nothing but page ground around it. The places where that happens most are the
scattered returns over the bay and the bridge's cables. This renders Embarcadero with the
published model's predictions and the page palette, keeps the map of which point landed in
each pixel, and measures the pixels of isolated points against the ground pixels next to them.

    isolated        fewer than MAX_NEIGHBOURS other points within RADIUS_M in 3D
    over water      isolated points the delivery labels water (class 9)
    cables          isolated points more than CABLE_HEIGHT_M above the delivery's ground
                    surface (the height the cache keeps for gates), which in Embarcadero is
                    the bridge's cables and towers
    measure         WCAG contrast ratio between each such pixel and the mean colour of the
                    background pixels among its eight neighbours, in normal vision and after
                    Machado 2009 deuteranopia and protanopia

Pass when every measured pixel reaches 3:1 (rule R1 of backend/surrey/palette.py). Writes
docs/design/sparse_contrast_check.json.

Run:
    python scripts/investigations/sparse_contrast_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from render_frames import oblique  # noqa: E402
from surrey.cache import CachedTile  # noqa: E402
from surrey.colour_vision import contrast_ratio, simulate  # noqa: E402
from surrey.metrics import coarse_lookup  # noqa: E402
from surrey.palette import MIN_CONTRAST, hex_rgb, load_palette  # noqa: E402

OUT = ROOT / "docs" / "design" / "sparse_contrast_check.json"
RADIUS_M, MAX_NEIGHBOURS, CABLE_HEIGHT_M = 1.0, 3, 20.0


def main() -> int:
    from scipy.spatial import cKDTree

    pal = load_palette()
    tile = CachedTile(ROOT / "data" / "prep" / "target" / "embarcadero", "sf")
    z = np.load(ROOT / "data" / "predictions" / "m0_seedA" / "sf_embarcadero.npz")
    coarse = coarse_lookup()[z["pred"]]
    conf = z["conf"].astype(np.float32)
    xyz = np.column_stack(
        [
            np.asarray(tile.xy_cm[:, 0], np.float64) / 100,
            np.asarray(tile.xy_cm[:, 1], np.float64) / 100,
            np.asarray(tile.z_cm[:], np.float64) / 100,
        ]
    )
    counts = cKDTree(xyz).query_ball_point(xyz, RADIUS_M, return_length=True, workers=2) - 1
    isolated = counts < MAX_NEIGHBOURS
    asprs = np.asarray(tile.asprs[:], np.int16)
    hag = np.asarray(tile.hag_cm[:], np.float32) / 100
    groups = {"over_water": isolated & (asprs == 9), "cables": isolated & (hag > CABLE_HEIGHT_M)}
    result = {
        "rule": f"every pixel at least {MIN_CONTRAST}:1",
        "isolated_definition": f"fewer than {MAX_NEIGHBOURS} points within {RADIUS_M} m",
        "themes": {},
    }
    for theme, d in pal["themes"].items():
        bg = hex_rgb(d["ground"])
        store: dict = {}
        img = oblique(tile, coarse, conf, 1600, 30.0, 38.0, None, tuple(bg), 0.0, theme, store)
        ids = store["ids"]
        h, w = ids.shape
        is_bg = ids < 0
        row = {}
        for name, mask in groups.items():
            ys, xs = np.nonzero((ids >= 0) & mask[np.clip(ids, 0, None)])
            inner = (ys > 0) & (ys < h - 1) & (xs > 0) & (xs < w - 1)
            ys, xs = ys[inner], xs[inner]
            px, ground = [], []
            for y, x in zip(ys, xs, strict=True):
                win = is_bg[y - 1 : y + 2, x - 1 : x + 2]
                if win.sum() == 0:
                    continue
                px.append(img[y, x])
                ground.append(img[y - 1 : y + 2, x - 1 : x + 2][win].mean(axis=0))
            px, ground = np.array(px), np.array(ground)
            stats = {"pixels": int(len(px))}
            for vision in ("normal", "deuteranopia", "protanopia"):
                if len(px) == 0:
                    continue
                r = contrast_ratio(simulate(px, vision), simulate(ground, vision))
                stats[vision] = {
                    "min": round(float(r.min()), 2),
                    "p1": round(float(np.percentile(r, 1)), 2),
                    "median": round(float(np.median(r)), 2),
                    "share_below_3": round(float((r < MIN_CONTRAST).mean()), 4),
                }
            stats["passes"] = bool(len(px)) and all(
                stats[v]["min"] >= MIN_CONTRAST for v in ("normal", "deuteranopia", "protanopia")
            )
            row[name] = stats
        result["themes"][theme] = row
        print(theme, json.dumps(row))
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
