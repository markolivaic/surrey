"""How much of the first screen is still the page showing through, and how loud that is.

Two readings of one rendered frame, both over the scene, which is everything right of the document
column:

- square: the share of background inside one 450 by 450 square, centred on what is drawn. It counts
  the sky as well as the holes, which is fair for a frame the cloud fills and unfair for a skyline.
- grain: the share of background among the pixels the cloud encloses, after a closing and a fill, so
  the sky around a silhouette is left out and only what a reader reads as grain remains.

It also reports how visible a hole is: the contrast ratio between the mean drawn pixel and the
ground it sits on. The same hole is quiet on a light ground and loud on a dark one.

Last, it puts candidate dark grounds through the palette's own rules, because lifting the ground is
the obvious way to quieten the grain and the rules decide whether it is available.

Run (the Python that has Playwright, Pillow, SciPy):
    python scripts/investigations/hero_emptiness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.palette import check_theme, load_palette  # noqa: E402

SHOTS = ROOT / "data" / "cache" / "hero"
OUT = ROOT / "data" / "artifacts" / "hero_emptiness.json"
GROUND = {"light": (230, 233, 230), "dark": (19, 24, 21)}
BLOCK = 450  # the side of the square the share is measured in
SCENE_X0 = 560  # right of the document column, which is 38% of 1440
NEAR = 12  # a pixel is the ground if its channels differ by less than this in total
CANDIDATE_GROUNDS = ["#131815", "#1a201c", "#1f2723", "#242d28", "#2a332e"]


def luminance(rgb) -> float:
    c = np.asarray(rgb, dtype=float) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[..., 0].mean() + 0.7152 * c[..., 1].mean() + 0.0722 * c[..., 2].mean())


def contrast(a: float, b: float) -> float:
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def measure(shot: Path, theme: str) -> dict:
    a = np.asarray(Image.open(shot).convert("RGB")).astype(int)[:, SCENE_X0:]
    ground = np.array(GROUND[theme])
    bg = np.abs(a - ground).sum(axis=2) < NEAR
    ys, xs = np.nonzero(~bg)
    if len(xs) == 0:
        return {"square": 1.0, "grain": 1.0, "note": "nothing drawn"}
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = max(0, min(cy - BLOCK // 2, bg.shape[0] - BLOCK))
    x0 = max(0, min(cx - BLOCK // 2, bg.shape[1] - BLOCK))
    solid = binary_fill_holes(binary_closing(~bg, structure=np.ones((9, 9))))
    drawn_l = luminance(a[~bg])
    ground_l = luminance(ground)
    return {
        "square": round(float(bg[y0 : y0 + BLOCK, x0 : x0 + BLOCK].mean()), 4),
        "grain": round(float((bg & solid).sum() / max(solid.sum(), 1)), 4),
        "mean_drawn_luminance": round(drawn_l, 4),
        "ground_luminance": round(ground_l, 4),
        "hole_contrast": round(contrast(drawn_l, ground_l), 2),
        "block": BLOCK,
        "block_origin_in_scene": [x0, y0],
        "scene_x0": SCENE_X0,
    }


def grounds(reference_drawn_luminance: float | None) -> list[dict]:
    """Every candidate dark ground against the palette rules, and what it buys.

    What it buys is read off the frame with the most grain in it: how loud a hole in that frame
    would be if the ground were lifted to the candidate.
    """
    pal = load_palette()
    dark = pal["themes"]["dark"]
    floor = pal["floor"] if pal["floor"] is not None else 0.2
    rows = []
    for hexs in CANDIDATE_GROUNDS:
        result = check_theme(dark["tokens"], hexs, floor)
        rgb = np.array([int(hexs[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
        row = {
            "ground": hexs,
            "luminance": round(luminance(rgb), 4),
            "worst_class_contrast": round(min(result["contrast_min"].values()), 2),
            "failures": result["failures"],
        }
        if reference_drawn_luminance is not None:
            row["hole_contrast_in_the_top_down_frame"] = round(
                contrast(reference_drawn_luminance, luminance(rgb)), 2
            )
        rows.append(row)
    return rows


def main() -> int:
    shots = {}
    for shot in sorted(SHOTS.glob("hero_*_desktop_*.png")):
        theme = shot.stem.rsplit("_", 1)[1]
        if theme not in GROUND:
            continue
        shots[shot.name] = measure(shot, theme) | {"theme": theme}
        r = shots[shot.name]
        print(
            f"{shot.name}: square {r['square']} grain {r['grain']} hole contrast {r['hole_contrast']}"
        )
    reference = shots.get("hero_financial_order1_desktop_dark.png", {}).get("mean_drawn_luminance")
    doc = {
        "what": "background showing through the first screen, per rendered frame",
        "near": NEAR,
        "shots": shots,
        "dark_grounds": grounds(reference),
        "dark_grounds_reference_frame": "hero_financial_order1_desktop_dark.png",
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
