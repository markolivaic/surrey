"""The page's class palette and the rules it has to pass. The rules were written before any
palette was tried against them.

What the first palette got wrong, measured: confidence moved every colour toward one light
grey, so unsure points vanished on the light ground and, on the dark ground, the grey was
lighter than any class colour, making the least sure points the loudest on screen. And its
hues put building (red) and vegetation (green) on the axis red-green dichromacy removes:
on the bug frame pair only 29 percent of the visible difference survives deuteranopia and
53 percent protanopia (scripts/investigations/cvd_bug_frames.py, reproducing the measurement that found it).

How a point is coloured now. Each class has, per theme, one OKLCH lightness L, one hue h and
a full chroma C. Confidence maps to t in [0, 1] by the map fixed on Surrey
(data/artifacts/confidence_mapping.json) and scales chroma only: C(t) = C * (f + (1 - f) t),
f being the chroma floor. Lightness never changes with confidence, so contrast against the
ground is set by the class's lightness and a point neither vanishes nor jumps out when the
model is unsure.

Rules. Colour differences are CIELAB delta E76, the unit the bug frame measurement uses.
Simulation is Machado 2009 at full severity (surrey.colour_vision), deuteranopia and
protanopia.

R1 contrast      every class colour, at full chroma and at the floor, stands off its theme's
                 ground by at least 3:1, the WCAG 2.1 success criterion 1.4.11 ratio for
                 graphical objects needed to understand the content. Checked in normal
                 vision and in both simulations.
R2 story pairs   the pairs the page's findings rest on differ by at least 20 in normal vision
                 and in both simulations, at full chroma: building and vegetation (the bug),
                 ground and vegetation (the park), ground and building (roofs called ground),
                 water and ground (the bay), water and building, bridge deck and building.
                 20 because the bug frame measurement counts a pixel as changed class colour
                 above 20; a story pair has to stay a change of class colour for every reader.
R3 other pairs   every other pair of classes differs by at least 10 in all three visions.
                 Points are one to six pixels wide and small marks lose colour discrimination,
                 so the bar is several times the 2.3 of one just noticeable difference; the
                 minor classes (vehicle, pole, fence) are named in the legend.
R4 lightness     building and vegetation differ in OKLab lightness by at least 0.15 in both
                 themes. Dichromacy leaves lightness nearly intact, and 0.15 in OKLab is about
                 15 in CIELAB lightness, three quarters of R2 on lightness alone, so the bug
                 reads as light against dark even if hue collapses.
R5 bug frames    on the committed pair of bug frames, in both themes, at least 60 percent of
                 the normal vision median difference remains in each simulation
                 (scripts/investigations/cvd_bug_frames.py). The bar was set before any palette was tried.
R6 floor         the chroma floor f is the smallest of 0.2, 0.25 to 0.6 at which every class at
                 the floor is still nearer, in OKLab, to its own full colour than to any other
                 class's full colour, and keeps an OKLCH chroma of at least 0.03, a visible
                 tint rather than grey. If no f in that list passes, the legend says instead
                 that the lowest confidence hides the class.
R7 gamut         every token is inside sRGB at full chroma, so no colour is clipped.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .colour_vision import contrast_ratio, delta_e76, oklab_from_srgb, simulate, srgb_from_oklch

ROOT = Path(__file__).resolve().parents[2]
PALETTE_FILE = ROOT / "docs" / "design" / "palette.json"
CLASSES = ("ground", "vegetation", "building", "vehicle", "pole", "fence", "water", "bridge deck")
STORY_PAIRS = {
    ("building", "vegetation"),
    ("ground", "vegetation"),
    ("ground", "building"),
    ("water", "ground"),
    ("water", "building"),
    ("bridge deck", "building"),
    # The Financial card says the towers are called poles; a pole has to stand apart from
    # the building it is mistaken for and from the trees around it.
    ("pole", "building"),
    ("pole", "vegetation"),
}
GROUNDS = {"light": "#e6e9e6", "dark": "#131815"}
VISIONS = ("normal", "deuteranopia", "protanopia")
MIN_CONTRAST = 3.0
MIN_STORY_DE = 20.0
MIN_OTHER_DE = 10.0
MIN_BUG_LIGHTNESS = 0.15
MIN_BUG_RETENTION = 0.60
FLOORS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
MIN_FLOOR_CHROMA = 0.03


def hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)])


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if (a, b) in STORY_PAIRS else (b, a)


def colour(token: dict, t: float, floor: float) -> np.ndarray:
    rgb, _ = srgb_from_oklch(token["L"], token["C"] * (floor + (1 - floor) * t), token["h"])
    return rgb


def check_theme(tokens: dict, ground_hex: str, floor: float, shade: float = 1.0) -> dict:
    """R1, R2, R3, R4, R6 (at a given floor) and R7 for one theme.

    `shade` is what eye-dome lighting multiplies a drawn colour by, 1 for none. The page's pass
    scales the displayed colour, so the check scales the same values before it simulates vision.
    """
    ground = hex_rgb(ground_hex)
    full = {c: colour(tokens[c], 1.0, floor) * shade for c in CLASSES}
    low = {c: colour(tokens[c], 0.0, floor) * shade for c in CLASSES}
    out: dict[str, Any] = {"contrast_min": {}, "pairs": {}, "failures": []}
    for vision in VISIONS:
        g = simulate(ground, vision)
        worst = min(
            float(contrast_ratio(simulate(x[c], vision), g)) for x in (full, low) for c in CLASSES
        )
        out["contrast_min"][vision] = round(worst, 2)
        if worst < MIN_CONTRAST:
            out["failures"].append(f"R1 {vision}: contrast {worst:.2f}")
    for a, b in itertools.combinations(CLASSES, 2):
        story = (a, b) in STORY_PAIRS or (b, a) in STORY_PAIRS
        bar = MIN_STORY_DE if story else MIN_OTHER_DE
        des = {
            v: round(float(delta_e76(simulate(full[a], v), simulate(full[b], v))), 1)
            for v in VISIONS
        }
        out["pairs"][f"{a} / {b}"] = des | {"bar": bar}
        for v, d in des.items():
            if d < bar:
                out["failures"].append(f"{'R2' if story else 'R3'} {a}/{b} {v}: {d}")
    dl = abs(tokens["building"]["L"] - tokens["vegetation"]["L"])
    out["building_vegetation_lightness_gap"] = round(dl, 3)
    if dl < MIN_BUG_LIGHTNESS:
        out["failures"].append(f"R4 lightness gap {dl:.3f}")
    for c in CLASSES:
        if not bool(srgb_from_oklch(tokens[c]["L"], tokens[c]["C"], tokens[c]["h"])[1]):
            out["failures"].append(f"R7 {c} out of gamut")
    return out


def floor_ok(tokens: dict, floor: float) -> bool:
    """R6: at the floor every class is nearest its own full colour, and still tinted."""
    full = {c: oklab_from_srgb(colour(tokens[c], 1.0, floor)) for c in CLASSES}
    for c in CLASSES:
        if tokens[c]["C"] * floor < MIN_FLOOR_CHROMA:
            return False
        low = oklab_from_srgb(colour(tokens[c], 0.0, floor))
        nearest = min(CLASSES, key=lambda k: float(np.linalg.norm(full[k] - low)))
        if nearest != c:
            return False
    return True


def choose_floor(palette: dict) -> float | None:
    for f in FLOORS:
        if all(floor_ok(palette["themes"][th]["tokens"], f) for th in palette["themes"]):
            return f
    return None


def load_palette(path: Path = PALETTE_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def old_palette() -> dict:
    """The first palette, as the frames and the prototype drew it, for the self-test: HSV
    hue per class, confidence pulling saturation toward grey, value 0.68 light and 0.95 dark.
    Approximated in OKLCH at full saturation so the same checks can fail it."""
    import colorsys

    hues = {
        "ground": 38,
        "vegetation": 118,
        "building": 352,
        "vehicle": 292,
        "pole": 188,
        "fence": 20,
        "water": 212,
        "bridge deck": 262,
    }
    themes = {}
    for theme, value in (("light", 0.68), ("dark", 0.95)):
        tokens = {}
        for c, h in hues.items():
            rgb = np.array(colorsys.hsv_to_rgb(h / 360, 0.9, value))
            lab = oklab_from_srgb(rgb)
            tokens[c] = {
                "L": float(lab[0]),
                "C": float(np.hypot(lab[1], lab[2])),
                "h": float(np.degrees(np.arctan2(lab[2], lab[1])) % 360),
            }
        themes[theme] = {"ground": GROUNDS[theme], "tokens": tokens}
    return {"themes": themes, "floor": 0.04 / 0.9}
