"""What the model calls the tall points in the Financial District, and what the picture shows.

Two readings of the same claim. The first is the model's own calls, read out of
`high_points_class*.json`: the share of points above a height threshold that carry each coarse
class, for every seed of the published recipe. The second is the rendered first screen itself: every
drawn pixel in the upper quarter of the cloud is matched to the palette hue it is nearest, which is
what a reader's eye does. Confidence scales chroma and leaves hue alone, so hue is the honest thing
to match on, and nearly grey pixels are left out because they carry no hue to match.

Run:
    python scripts/investigations/tower_colour_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.palette import CLASSES, colour, load_palette, oklab_from_srgb  # noqa: E402

ARTIFACTS = ROOT / "data" / "artifacts"
OUT = ARTIFACTS / "tower_colour_check.json"
FRAME = ROOT / "data" / "design_archive" / "hero" / "hero_financial_order2wide_desktop_dark.png"
GROUND_DARK = (19, 24, 21)
SCENE_X0 = 560
TOP_FRACTION = 0.25  # the upper quarter of what is drawn, which is the towers
MIN_CHROMA = 0.02  # below this a pixel is grey and has no hue to match


def model_calls() -> dict:
    out = {}
    for name in (
        "high_points_class.json",
        "high_points_class_m0_seedB.json",
        "high_points_class_m0_seedC.json",
    ):
        doc = json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))
        per_area = {}
        for area, v in doc["areas"].items():
            bins = {}
            for key, b in v["bins"].items():
                bins[key] = {
                    "points": b["points"],
                    "share_of_area": b["share_of_area"],
                    "coarse": {k: round(x["share"], 4) for k, x in b["coarse"].items()},
                }
            per_area[area] = {"max_height_m": v["max_height_m"], "bins": bins}
        out[doc["variant"]] = {
            "training_block_z_max_m": doc["training_block_z_max_m"],
            "areas": per_area,
        }
    return out


def pixels() -> dict:
    pal = load_palette()
    tokens = pal["themes"]["dark"]["tokens"]
    hue = {}
    for c in CLASSES:
        lab = oklab_from_srgb(np.array([float(x) for x in colour(tokens[c], 1.0, pal["floor"])]))
        hue[c] = float(np.degrees(np.arctan2(lab[2], lab[1])) % 360)

    a = np.asarray(Image.open(FRAME).convert("RGB")).astype(float)[:, SCENE_X0:]
    bg = np.abs(a - np.array(GROUND_DARK, dtype=float)).sum(axis=2) < 12
    ys, _ = np.nonzero(~bg)
    cut = ys.min() + (ys.max() - ys.min()) * TOP_FRACTION
    sel = (~bg) & (np.arange(a.shape[0])[:, None] < cut)
    lab = np.stack([oklab_from_srgb(p) for p in a[sel] / 255.0])
    h = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    names = list(hue)
    gaps = np.stack([np.minimum(np.abs(h - hue[n]), 360 - np.abs(h - hue[n])) for n in names])
    near = np.array(names)[gaps.argmin(axis=0)]
    keep = chroma >= MIN_CHROMA
    shares = {
        n: round(float(((near == n) & keep).sum()) / max(int(keep.sum()), 1), 4) for n in names
    }
    return {
        "frame": str(FRAME.relative_to(ROOT)).replace("\\", "/"),
        "pixels_judged": int(sel.sum()),
        "grey_pixels_left_out": int((~keep).sum()),
        "top_fraction_of_the_cloud": TOP_FRACTION,
        "palette_hue_degrees": hue,
        "share_nearest_each_hue": shares,
    }


def main() -> int:
    doc = {
        "question": "does the model call the tall towers vegetation",
        "height_is": "height above ground, the min-z grid surface",
        "model": model_calls(),
        "picture": pixels(),
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    p = doc["picture"]["share_nearest_each_hue"]
    print(
        "picture, upper quarter:",
        {k: v for k, v in sorted(p.items(), key=lambda kv: -kv[1]) if v > 0.005},
    )
    for variant, d in doc["model"].items():
        b = d["areas"]["financial"]["bins"]
        print(
            variant,
            "above 100 m",
            b["above_100m"]["coarse"],
            "| above 150 m",
            b["above_150m"]["coarse"],
        )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
