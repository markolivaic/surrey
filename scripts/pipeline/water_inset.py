"""Water at the scale of single points: the held-out Zug tile beside Embarcadero, as M1 reads them.

Two small top-down pictures at the same scale, of real points coloured by M1 seed A's calls. Each shows
the distinct points one crop carries, drawn the way training draws a crop: a 40 m block, 32,768 draws,
without repeats when the block holds more points than that and with repeats when it holds fewer. So the
spacing on screen is the spacing the model reads, not the spacing of the delivery.

The window, chosen by a rule written before this first ran (2026-09-23) and using labels only, never the
model's calls: 8 m squares on a 2 m step inside every 40 m block of the area; among the squares whose
labelled water share lies between 0.3 and 0.7, so that the picture holds a shoreline, the one with the
most labelled water points in the crop. Labels are swisstopo's classes for Zug and the delivery's ASPRS
class 9 for Embarcadero.

Writes the four images the page shows, light and dark, and data/artifacts/water_inset.json with the
choice and what each window holds.

Run:
    python scripts/pipeline/water_inset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import BLOCK_M, COARSE, POINTS_PER_CROP  # noqa: E402
from surrey.inference import DEFAULT_SEED, infer_tile  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402
from surrey.palette import colour, load_palette  # noqa: E402
from surrey.sources import M1_CLASS_NAMES, WATER  # noqa: E402

ZUG = ROOT / "data" / "prep" / "swiss_holdout"
SF = ROOT / "data" / "prep" / "target" / "embarcadero"
SF_PRED = ROOT / "data" / "predictions" / "m1_seedA" / "sf_embarcadero.npz"
CKPT = ROOT / "data" / "checkpoints" / "m1_seedA.pt"
STILLS = ROOT / "web" / "public" / "stills"
OUT = ROOT / "data" / "artifacts" / "water_inset.json"
WINDOW_M = 8.0
STEP_M = 2.0
SHORE = (0.3, 0.7)
PX_PER_M = 40
RADIUS_PX = 2.2
ASPRS_WATER = 9
RULE = (
    "written 2026-09-23 before the first run: 8 m squares on a 2 m step inside every 40 m block; among"
    " the squares whose labelled water share lies between 0.3 and 0.7, the one with the most labelled"
    " water points in the crop; labels only, never the model's calls"
)
GROUNDS = {"light": "e6e9e6", "dark": "131815"}


def group_of(name: str) -> str:
    for group, members in COARSE.items():
        if name in members:
            return group
    return {"water": "water", "bridge_deck": "bridge deck"}[name]


def crop(tile: CachedTile, cx: float, cy: float, rng: np.random.Generator) -> np.ndarray:
    """The distinct points one training crop of this block carries."""
    idx = tile.gather_block(cx, cy, BLOCK_M)
    if idx.size == 0:
        return idx
    pick = rng.choice(idx, size=POINTS_PER_CROP, replace=idx.size < POINTS_PER_CROP)
    return np.unique(pick)


def best_window(tiles: list[tuple[str, CachedTile]], is_water) -> dict:
    best = None
    for name, tile in tiles:
        for cx, cy in tile.block_centres(BLOCK_M):
            rng = np.random.default_rng(DEFAULT_SEED)
            idx = crop(tile, cx, cy, rng)
            if idx.size == 0:
                continue
            x = np.asarray(tile.xy_cm[idx, 0], np.float64) / 100.0
            y = np.asarray(tile.xy_cm[idx, 1], np.float64) / 100.0
            water = is_water(tile, idx)
            lo_x, lo_y = cx - BLOCK_M / 2, cy - BLOCK_M / 2
            for wx in np.arange(lo_x, lo_x + BLOCK_M - WINDOW_M + 1e-6, STEP_M):
                for wy in np.arange(lo_y, lo_y + BLOCK_M - WINDOW_M + 1e-6, STEP_M):
                    inside = (x >= wx) & (x < wx + WINDOW_M) & (y >= wy) & (y < wy + WINDOW_M)
                    n = int(inside.sum())
                    if n == 0:
                        continue
                    w = int((water & inside).sum())
                    share = w / n
                    if not (SHORE[0] <= share <= SHORE[1]):
                        continue
                    if best is None or w > best["water_points"]:
                        best = {
                            "tile": name,
                            "block_centre_m": [round(cx, 2), round(cy, 2)],
                            "window_origin_m": [round(float(wx), 2), round(float(wy), 2)],
                            "points": n,
                            "water_points": w,
                            "water_share": round(share, 4),
                            "crop_distinct_points": int(idx.size),
                            "crop_points_per_m2": round(idx.size / (BLOCK_M * BLOCK_M), 2),
                        }
    if best is None:
        raise SystemExit("no window meets the rule")
    best["window_points_per_m2"] = round(best["points"] / (WINDOW_M * WINDOW_M), 2)
    return best


def window_points(tile: CachedTile, choice: dict) -> np.ndarray:
    cx, cy = choice["block_centre_m"]
    idx = crop(tile, cx, cy, np.random.default_rng(DEFAULT_SEED))
    x = np.asarray(tile.xy_cm[idx, 0], np.float64) / 100.0
    y = np.asarray(tile.xy_cm[idx, 1], np.float64) / 100.0
    wx, wy = choice["window_origin_m"]
    inside = (x >= wx) & (x < wx + WINDOW_M) & (y >= wy) & (y < wy + WINDOW_M)
    return idx[inside]


def draw(
    tile: CachedTile, idx: np.ndarray, pred: np.ndarray, choice: dict, theme: str, pal: dict
) -> Image.Image:
    size = int(WINDOW_M * PX_PER_M)
    ground = GROUNDS[theme]
    img = Image.new("RGB", (size, size), "#" + ground)
    d = ImageDraw.Draw(img)
    tokens = pal["themes"][theme]["tokens"]
    x = np.asarray(tile.xy_cm[idx, 0], np.float64) / 100.0 - choice["window_origin_m"][0]
    y = np.asarray(tile.xy_cm[idx, 1], np.float64) / 100.0 - choice["window_origin_m"][1]
    z = np.asarray(tile.z_cm[idx], np.float64)
    for k in np.argsort(z):  # the highest point last, as seen from above
        rgb = colour(tokens[group_of(M1_CLASS_NAMES[int(pred[k])])], 1.0, pal["floor"])
        fill = tuple(int(round(float(c) * 255)) for c in rgb)
        px, py = x[k] * PX_PER_M, size - y[k] * PX_PER_M
        d.ellipse([px - RADIUS_PX, py - RADIUS_PX, px + RADIUS_PX, py + RADIUS_PX], fill=fill)
    return img


def main() -> int:
    pal = load_palette()
    zug = [(p.name, CachedTile(p, "train")) for p in sorted(ZUG.iterdir()) if p.is_dir()]
    sf = [("embarcadero", CachedTile(SF, "sf"))]
    sf_water = np.load(SF / "asprs.npy", mmap_mode="r")

    zug_choice = best_window(zug, lambda t, idx: np.asarray(t.label[idx]) == WATER)
    sf_choice = best_window(sf, lambda t, idx: np.asarray(sf_water[idx]) == ASPRS_WATER)
    print("zug", zug_choice, flush=True)
    print("embarcadero", sf_choice, flush=True)

    # M1 seed A's calls. Embarcadero's were written when the page's octrees were built; Zug's are made here,
    # over the one sub-tile the window lies in, the way scripts/pipeline/score_swiss.py scored it.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(CKPT, map_location=dev, weights_only=False)
    channels = int(blob.get("args", {}).get("channels", 3))
    n_out = int(blob["model"]["head.4.bias"].shape[0])
    net = SurreyNet(n_out, width=128, extra_features=channels - 3).to(dev)
    net.load_state_dict(blob["model"])
    net.eval()
    zug_tile = dict(zug)[zug_choice["tile"]]
    zug_pred, _c, _e, seen = infer_tile(
        net,
        zug_tile,
        dev,
        BLOCK_M,
        POINTS_PER_CROP,
        None,
        batch=4,
        seed=DEFAULT_SEED,
        channels=channels,
    )
    if not seen.all():
        raise SystemExit("some Zug points were never predicted")
    sf_pred = np.load(SF_PRED)["pred"]

    out = {
        "rule_written_before": RULE,
        "variant": "m1_seedA",
        "window_m": WINDOW_M,
        "px_per_m": PX_PER_M,
        "point_radius_px": RADIUS_PX,
        "device": dev,
        "areas": {},
    }
    STILLS.mkdir(parents=True, exist_ok=True)
    for name, tile, choice, pred, truth in (
        (
            "zug",
            zug_tile,
            zug_choice,
            zug_pred,
            lambda idx: np.asarray(zug_tile.label[idx]) == WATER,
        ),
        (
            "embarcadero",
            sf[0][1],
            sf_choice,
            sf_pred,
            lambda idx: np.asarray(sf_water[idx]) == ASPRS_WATER,
        ),
    ):
        idx = window_points(tile, choice)
        p = np.asarray(pred[idx]).astype(int)
        water = truth(idx)
        called = p == WATER
        choice |= {
            "labelled_water_called_water": round(
                float((called & water).sum() / max(water.sum(), 1)), 4
            ),
            "labelled_water_called_ground": round(
                float(((p == M1_CLASS_NAMES.index("ground")) & water).sum() / max(water.sum(), 1)),
                4,
            ),
        }
        out["areas"][name] = choice
        for theme, ground in GROUNDS.items():
            draw(tile, idx, p, choice, theme, pal).save(
                STILLS / f"water_inset_{name}_{ground}.webp", lossless=True, method=6
            )
        print(name, choice, flush=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
