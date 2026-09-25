"""Find a class palette that passes the rules in backend/surrey/palette.py, and write it.

The rules were written first; this only searches. One hue per class for both themes, so a
class keeps its identity when the reader switches theme, and one OKLCH lightness per class
per theme. Hues are kept in ranges that still read as the thing: ground warm (50 to 95
degrees), vegetation green to teal (125 to 180), building violet to magenta (260 to 345,
away from the red that red-green dichromacy merges with green), water blue (200 to 250);
pole cyan (180 to 230); vehicle, fence and bridge deck are free. Full chroma is 90 percent of the most sRGB
allows at that lightness and hue, capped at 0.16 so no class shouts over the others, and
at least 0.10, because a class whose full colour is nearly grey cannot keep a tint at the
floor (R6). Lightness stays in 0.35 to 0.62 on the light ground and 0.55 to 0.9 on the dark:
a first run without these limits passed R1 to R4 with near-black trees and grey water and
failed R6 at every floor. The search also counts R6 at a floor of 0.35 as a failure.

R5 is measured on rendered frames, which a search cannot afford, so the search uses a proxy:
the building to vegetation difference in each simulation must keep PROXY_RETENTION of its
normal vision size, at full chroma and at 65 percent. The second run without the proxy passed
every token rule and kept only 0.507 in deuteranopia on the frames, because magenta against
teal differs mostly on the red-green axis the simulation removes.

Pole is held to cyan (180 to 230). The third run left it free and put it at hue 76, next to
ground at 72; it passed R3 on its lightness, but on the dark ground the bridge's cables read
as ground. The threshold did not change, only the range the search may use.

A seeded random search with local moves minimises the number of rule failures and then
maximises the smallest margin. It is design tuning against rules fixed beforehand, not a
measurement, and the seed and step count are written into the palette file.

Pole joined the story pairs later (pole against building, and pole against vegetation), and the
palette above failed pole against vegetation in the light theme. With --only pole the search starts
from the palette as it stands and moves pole alone: one hue for both themes within its range, a
lightness per theme within that theme's range, chroma by the same rule. A colour has to pass every rule
in backend/surrey/palette.py in both themes and all three visions, with no shade and at every median
shade the eye-dome sweep measured (data/artifacts/edl_sweep.json) on both scenes: in the light theme
for every strength up to the largest any pole colour allows, in the dark theme for every strength up to
the one its rule chose. Every candidate is held at the floor R6 then chooses for it. Of the colours that
pass, the nearest to the pole as it stands is taken, lightness first. Every other class keeps its
colour, and no threshold moves. The sweep then chooses each theme's strength again by its own rule
(scripts/investigations/edl_sweep.py --rechoose).

Run:
    python scripts/investigations/palette_search.py
    python scripts/investigations/palette_search.py --only pole
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.colour_vision import (  # noqa: E402
    MACHADO_2009,
    lab,
    oklab_from_srgb,
    srgb_from_oklch,
    to_linear,
    to_srgb,
)
from surrey.palette import (  # noqa: E402
    CLASSES,
    GROUNDS,
    MIN_BUG_LIGHTNESS,
    MIN_CONTRAST,
    MIN_OTHER_DE,
    MIN_STORY_DE,
    PALETTE_FILE,
    STORY_PAIRS,
    check_theme,
    choose_floor,
    hex_rgb,
)

SEED, STEPS, RESTARTS = 20260914, 4000, 16
EDL_SWEEP = ROOT / "data" / "artifacts" / "edl_sweep.json"
HUE_RANGE = {
    "ground": (50, 95),
    "vegetation": (125, 180),
    "building": (260, 345),
    "water": (200, 250),
    "vehicle": (0, 360),
    "pole": (180, 230),
    "fence": (0, 360),
    "bridge deck": (0, 360),
}
L_GRID = np.round(np.arange(0.25, 0.951, 0.01), 2)
H_GRID = np.arange(0, 360, 2)
C_CAP, C_SHARE, SEARCH_FLOOR, C_MIN = 0.16, 0.9, 0.35, 0.10
L_RANGE = {"light": (0.35, 0.62), "dark": (0.55, 0.90)}
PROXY_RETENTION = 0.65


def max_chroma_table() -> np.ndarray:
    cs = np.arange(0, 0.37, 0.002)
    ll, hh, cc = np.meshgrid(L_GRID, H_GRID, cs, indexing="ij")
    _, inside = srgb_from_oklch(ll, cc, hh)
    # the largest chroma before the first out-of-gamut step
    first_out = np.argmax(~inside, axis=2)
    first_out[inside.all(axis=2)] = cs.size
    return cs[np.maximum(first_out - 1, 0)]


def adjust_pole() -> int:
    """Move pole alone to the nearest colour that passes every rule, unshaded and at the measured
    eye-dome shades, keeping the light theme's strength as high as any pole colour allows."""
    from surrey.palette import FLOORS, floor_ok

    sweep = json.loads(EDL_SWEEP.read_text(encoding="utf-8"))

    def shades(theme: str, up_to: float) -> list[float]:
        cells = sweep["cells"].values()
        # Darkest first: a colour that fails mostly fails there, and the check stops at a failure.
        return sorted(
            [1.0]
            + [
                c["shade_median"]
                for c in cells
                if c["theme"] == theme and c["strength"] <= up_to + 1e-9
            ]
        )

    cmax = max_chroma_table()
    li = {round(float(v), 2): i for i, v in enumerate(L_GRID)}
    palette = json.loads(PALETTE_FILE.read_text(encoding="utf-8"))
    old = {th: dict(d["tokens"]["pole"]) for th, d in palette["themes"].items()}
    lo_h, hi_h = HUE_RANGE["pole"]
    dark_up_to = sweep["chosen"]["dark"]["strength"]

    def pole(L: float, h: int) -> dict:
        chroma = max(C_MIN, min(C_CAP, C_SHARE * cmax[li[round(float(L), 2)], h // 2]))
        return {"L": float(L), "C": round(float(chroma), 4), "h": float(h)}

    def passes(th: str, tokens: dict, floor: float, up_to: float) -> bool:
        ground = palette["themes"][th]["ground"]
        return all(
            not check_theme(tokens, ground, floor, sh)["failures"] for sh in shades(th, up_to)
        )

    def nearest(th: str, h: int, floor: float, up_to: float) -> dict | None:
        lo, hi = L_RANGE[th]
        tokens = dict(palette["themes"][th]["tokens"])
        for L in sorted(
            L_GRID[(lo - 1e-9 <= L_GRID) & (hi + 1e-9 >= L_GRID)],
            key=lambda v, o=old[th]["L"]: abs(v - o),
        ):
            tokens["pole"] = pole(L, h)
            if floor_ok(tokens, floor) and passes(th, tokens, floor, up_to):
                return tokens["pole"]
        return None

    best = None
    for light_up_to in sorted(sweep["grid"], reverse=True):
        for h in range(lo_h, hi_h + 1, 2):
            for f in FLOORS:
                light = nearest("light", h, f, light_up_to)
                dark = nearest("dark", h, f, dark_up_to) if light else None
                if not (light and dark):
                    continue
                trial = json.loads(json.dumps(palette))
                trial["themes"]["light"]["tokens"]["pole"] = light
                trial["themes"]["dark"]["tokens"]["pole"] = dark
                floor = choose_floor(trial)
                if floor is None:
                    continue
                if not (
                    passes("light", trial["themes"]["light"]["tokens"], floor, light_up_to)
                    and passes("dark", trial["themes"]["dark"]["tokens"], floor, dark_up_to)
                ):
                    continue
                change = (
                    abs(light["L"] - old["light"]["L"])
                    + abs(dark["L"] - old["dark"]["L"])
                    + abs(h - old["light"]["h"]) / 360
                )
                if best is None or change < best[0]:
                    best = (change, {"light": light, "dark": dark}, light_up_to)
        if best is not None:
            break
    if best is None:
        print("no pole colour passes every rule with the other classes as they are")
        return 1
    for th, tok in best[1].items():
        palette["themes"][th]["tokens"]["pole"] = tok
    floor = choose_floor(palette)
    palette["floor"] = floor
    use = floor if floor is not None else 0.2
    palette["checks"] = {
        th: check_theme(d["tokens"], d["ground"], use) for th, d in palette["themes"].items()
    }
    palette["search"]["pole_adjusted"] = (
        "pole moved alone for the story pairs pole/building and pole/vegetation, by --only pole, held "
        "to every rule at the eye-dome shades measured in data/artifacts/edl_sweep.json"
    )
    PALETTE_FILE.write_text(json.dumps(palette, indent=2) + "\n", encoding="utf-8", newline="\n")
    for th in best[1]:
        print(th, "pole", old[th], "->", best[1][th], "failures", palette["checks"][th]["failures"])
    print("floor", floor, "light passes up to", best[2], "dark up to", dark_up_to)
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=("pole",), default=None)
    if ap.parse_args().only == "pole":
        return adjust_pole()
    rng = np.random.default_rng(SEED)
    cmax = max_chroma_table()
    li = {round(float(v), 2): i for i, v in enumerate(L_GRID)}
    idx = {c: i for i, c in enumerate(CLASSES)}
    story = np.zeros((8, 8), bool)
    for a, b in STORY_PAIRS:
        story[idx[a], idx[b]] = story[idx[b], idx[a]] = True
    bar = np.where(story, MIN_STORY_DE, MIN_OTHER_DE)
    iu = np.triu_indices(8, 1)
    grounds = {th: hex_rgb(g) for th, g in GROUNDS.items()}
    mats = [np.eye(3), MACHADO_2009["deuteranopia"], MACHADO_2009["protanopia"]]

    def tokens_of(h, ls):
        out = {}
        for i, c in enumerate(CLASSES):
            chroma = min(C_CAP, C_SHARE * cmax[li[round(float(ls[i]), 2)], int(h[i]) // 2])
            out[c] = {"L": float(ls[i]), "C": round(float(chroma), 4), "h": float(h[i])}
        return out

    def cost(h, lights):
        fails, margin = 0, np.inf
        for th, ls in lights.items():
            tok = tokens_of(h, ls)
            L = np.array([tok[c]["L"] for c in CLASSES])
            C = np.array([tok[c]["C"] for c in CLASSES])
            H = np.array([tok[c]["h"] for c in CLASSES])
            full, _ = srgb_from_oklch(L, C, H)
            low, _ = srgb_from_oklch(L, C * SEARCH_FLOOR, H)
            g = grounds[th]
            for m in mats:
                sim = lambda x, m=m: to_srgb(to_linear(x) @ m.T)  # noqa: E731
                gl = to_linear(sim(g[None]))[0] @ np.array([0.2126, 0.7152, 0.0722])
                for x in (full, low):
                    yl = to_linear(sim(x)) @ np.array([0.2126, 0.7152, 0.0722])
                    ratio = (np.maximum(yl, gl) + 0.05) / (np.minimum(yl, gl) + 0.05)
                    fails += int((ratio < MIN_CONTRAST).sum())
                    margin = min(margin, float((ratio / MIN_CONTRAST).min()) - 1)
                labs = lab(sim(full))
                de = np.linalg.norm(labs[:, None] - labs[None], axis=-1)[iu]
                fails += int((de < bar[iu]).sum())
                margin = min(margin, float((de / bar[iu]).min()) - 1)
            fails += int((C < C_MIN).sum())
            ok_full = oklab_from_srgb(full)
            ok_low = oklab_from_srgb(low)
            dist = np.linalg.norm(ok_low[:, None] - ok_full[None], axis=-1)
            fails += int((dist.argmin(axis=1) != np.arange(8)).sum())
            bi, vi = idx["building"], idx["vegetation"]
            for share in (1.0, 0.65):
                pair, _ = srgb_from_oklch(L[[bi, vi]], C[[bi, vi]] * share, H[[bi, vi]])
                base = float(np.linalg.norm(lab(pair[0]) - lab(pair[1])))
                for m in mats[1:]:
                    sp = to_srgb(to_linear(pair) @ m.T)
                    kept = float(np.linalg.norm(lab(sp[0]) - lab(sp[1]))) / base
                    fails += int(kept < PROXY_RETENTION)
                    margin = min(margin, kept / PROXY_RETENTION - 1)
            gap = abs(L[idx["building"]] - L[idx["vegetation"]])
            fails += int(gap < MIN_BUG_LIGHTNESS)
            margin = min(margin, gap / MIN_BUG_LIGHTNESS - 1)
        return fails, margin

    def random_hues():
        return np.array(
            [rng.integers(lo // 2, hi // 2) * 2 % 360 for lo, hi in (HUE_RANGE[c] for c in CLASSES)]
        )

    def random_lights():
        return {
            th: rng.choice(L_GRID[(lo - 1e-9 <= L_GRID) & (hi + 1e-9 >= L_GRID)], 8)
            for th, (lo, hi) in L_RANGE.items()
        }

    best = None
    for restart in range(RESTARTS):
        h, lights = random_hues(), random_lights()
        cur = cost(h, lights)
        for _ in range(STEPS):
            h2, l2 = h.copy(), {k: v.copy() for k, v in lights.items()}
            i = rng.integers(8)
            if rng.random() < 0.5:
                lo, hi = HUE_RANGE[CLASSES[i]]
                h2[i] = int(np.clip(h2[i] + rng.integers(-3, 4) * 2, lo, hi - 2)) % 360
            else:
                th = "light" if rng.random() < 0.5 else "dark"
                lo, hi = L_RANGE[th]
                l2[th][i] = float(np.clip(round(l2[th][i] + rng.integers(-3, 4) * 0.01, 2), lo, hi))
            new = cost(h2, l2)
            if (new[0], -new[1]) <= (cur[0], -cur[1]):
                h, lights, cur = h2, l2, new
        print(f"restart {restart}: failures {cur[0]}, margin {cur[1]:.3f}", flush=True)
        if best is None or (cur[0], -cur[1]) < (best[2][0], -best[2][1]):
            best = (h, lights, cur)

    h, lights, cur = best
    palette = {
        "search": {
            "seed": SEED,
            "steps": STEPS,
            "restarts": RESTARTS,
            "hue_ranges": HUE_RANGE,
            "chroma_cap": C_CAP,
            "chroma_min": C_MIN,
            "chroma_share_of_gamut": C_SHARE,
            "lightness_ranges": L_RANGE,
            "r5_proxy_retention": PROXY_RETENTION,
        },
        "rules": "backend/surrey/palette.py",
        "confidence_map": "data/artifacts/confidence_mapping.json",
        "themes": {
            th: {"ground": GROUNDS[th], "tokens": tokens_of(h, ls)} for th, ls in lights.items()
        },
    }
    floor = choose_floor(palette)
    palette["floor"] = floor
    palette["floor_note"] = (
        "every class stays nearest its own colour at the lowest confidence"
        if floor is not None
        else "no floor keeps every class recognisable: the lowest "
        "confidence hides the class, and the legend says so"
    )
    use = floor if floor is not None else 0.2
    palette["checks"] = {
        th: check_theme(d["tokens"], d["ground"], use) for th, d in palette["themes"].items()
    }
    PALETTE_FILE.write_text(json.dumps(palette, indent=2) + "\n", encoding="utf-8", newline="\n")
    for th, r in palette["checks"].items():
        print(
            th,
            "failures",
            r["failures"],
            "contrast",
            r["contrast_min"],
            "gap",
            r["building_vegetation_lightness_gap"],
        )
    print(
        "floor",
        floor,
        {
            th: {c: (t["L"], t["C"], t["h"]) for c, t in d["tokens"].items()}
            for th, d in palette["themes"].items()
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
