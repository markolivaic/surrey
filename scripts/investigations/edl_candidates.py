"""Eye-dome lighting on the live page, at the candidate strengths, measured.

For each scene and strength, in light and in dark:

- the frame itself, composed into a sheet light beside dark;
- grain, by `hero_emptiness.measure`, the method every earlier sheet was measured with;
- how much the pass darkens: the same camera drawn with the pass off and on, and the ratio of the two
  per drawn pixel is the shade the pass multiplied it by. Median and the darker tenth are kept;
- the palette's own rules at the median shade, because the pass darkens class colours and the rules
  were written for undarkened ones; and the share of drawn pixels darker than the shade at which the
  dark theme's contrast rule first fails;
- frame rate the way the prototype was held to it: a 10 s camera orbit and 10 s of wheel scrolling,
  median frame gap, three repeats, 30 fps the threshold.

Sheets go to data/design_archive/hero/edl, kept locally with the design rounds; single shots and the pass-off
twins go to data/cache/edl, which is not committed.

Run (the Python that has Playwright, Pillow and SciPy):
    npm --prefix web run build:investigate
    python scripts/investigations/edl_candidates.py --strengths 0 0.3 0.6
    python scripts/investigations/edl_candidates.py --strengths 0.25 0.5 1.0 --probe      shade only, no sheets or fps
    python scripts/investigations/edl_candidates.py --page      the page as it ships: each theme at its own strength
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from hero_emptiness import GROUND, measure  # noqa: E402
from shoot_hero import compose  # noqa: E402
from surrey.palette import MIN_CONTRAST, check_theme, load_palette  # noqa: E402
from surrey.static_server import serve  # noqa: E402

SHEETS = ROOT / "data" / "design_archive" / "hero" / "edl"
CACHE = ROOT / "data" / "cache" / "edl"
OUT = ROOT / "data" / "artifacts" / "edl_candidates.json"
OUT_PAGE = ROOT / "data" / "artifacts" / "edl_chosen.json"
SWEEP = ROOT / "data" / "artifacts" / "edl_sweep.json"
PORT = 8785
SIZE = {"width": 1440, "height": 900}
SCENES = {"sunset_mid": "area=sunset&open=mid", "ggpark_cards": "area=ggpark&open=cards"}
REPEATS = 3
SCENE_X0 = 560
ARGS = [
    "--enable-gpu",
    "--ignore-gpu-blocklist",
    "--use-angle=d3d11",
    "--disable-domain-blocking-for-3d-apis",
]


def settled(page) -> None:
    page.wait_for_function(
        "() => { const m = window.__surrey; if (!m || m.firstFrameMs === null) return false;"
        " const a = Object.values(m.areas); return a.length && a.every(r => r.loading === 0 && r.idleSince !== null); }",
        timeout=90000,
    )
    page.wait_for_timeout(900)


def median_fps(stamps: list[float]) -> float:
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False) if b > a]
    return round(1000.0 / statistics.median(gaps), 1) if gaps else 0.0


def shade_of(on: Path, off: Path, theme: str) -> np.ndarray:
    """Per drawn pixel, what the pass multiplied it by: the brightest channel, on over off."""
    a = np.asarray(Image.open(on).convert("RGB")).astype(float)[:, SCENE_X0:]
    b = np.asarray(Image.open(off).convert("RGB")).astype(float)[:, SCENE_X0:]
    drawn = np.abs(b - np.array(GROUND[theme])).sum(axis=2) >= 12
    ch = b.argmax(axis=2)
    bm = np.take_along_axis(b, ch[..., None], axis=2)[..., 0]
    am = np.take_along_axis(a, ch[..., None], axis=2)[..., 0]
    ok = drawn & (bm >= 40)
    return np.clip(am[ok] / bm[ok], 0.0, 1.0)


def failing_shade(tokens: dict, ground: str, floor: float) -> float:
    """The largest shade at which the contrast rule fails, found by halving; 0 if none does."""

    def fails(s: float) -> bool:
        return any(f.startswith("R1") for f in check_theme(tokens, ground, floor, s)["failures"])

    if not fails(0.0):
        return 0.0
    if fails(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fails(mid) else (lo, mid)
    return round(lo, 4)


def edl_param(strength: float | None) -> str:
    """No parameter at all when the page is to use its own strength per theme."""
    return "" if strength is None else f"&edl={strength}"


def shoot(p, query: str, strength: float | None, theme: str, stem: str) -> dict:
    browser = p.chromium.launch(args=ARGS)
    page = browser.new_page(viewport=SIZE)
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"http://127.0.0.1:{PORT}/index.html?{query}{edl_param(strength)}")
    settled(page)
    if theme == "dark":
        page.click("#theme")
        page.wait_for_timeout(1200)
    on = CACHE / f"{stem}_{theme}.png"
    off = CACHE / f"{stem}_{theme}_off.png"
    page.screenshot(path=str(on))
    page.evaluate("window.__surreyEdl(0)")
    page.wait_for_timeout(400)
    page.screenshot(path=str(off))
    browser.close()
    return {"on": on, "off": off, "errors": errors}


def fps(p, query: str, strength: float | None) -> dict:
    orbit, scroll = [], []
    for _ in range(REPEATS):
        browser = p.chromium.launch(args=ARGS)
        page = browser.new_page(viewport=SIZE)
        page.goto(f"http://127.0.0.1:{PORT}/index.html?{query}{edl_param(strength)}")
        settled(page)
        page.wait_for_timeout(2000)
        page.evaluate("window.__surrey.frames.length = 0; window.__surreyOrbit(true)")
        page.wait_for_timeout(10000)
        orbit.append(median_fps(page.evaluate("window.__surrey.frames")))
        page.evaluate("window.__surreyOrbit(false); window.__surrey.frames.length = 0")
        t_end = time.time() + 10
        while time.time() < t_end:
            page.mouse.wheel(0, 40)
            page.wait_for_timeout(16)
        scroll.append(median_fps(page.evaluate("window.__surrey.frames")))
        browser.close()
    return {
        "orbit_median_fps": orbit,
        "scroll_median_fps": scroll,
        "orbit_passes": min(orbit) >= 30.0,
        "scroll_passes": min(scroll) >= 30.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strengths", type=float, nargs="+")
    ap.add_argument(
        "--page", action="store_true", help="each theme at the page's own strength, from the sweep"
    )
    ap.add_argument(
        "--probe", action="store_true", help="shade only: no sheets, no frame rate, no artefact"
    )
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    args = ap.parse_args()
    if args.page:
        args.strengths = [None]
        chosen = {
            t: c["strength"]
            for t, c in json.loads(SWEEP.read_text(encoding="utf-8"))["chosen"].items()
        }
    elif not args.strengths:
        ap.error("give --strengths or --page")
    from playwright.sync_api import sync_playwright

    pal = load_palette()
    floor = pal["floor"] if pal["floor"] is not None else 0.2
    CACHE.mkdir(parents=True, exist_ok=True)
    SHEETS.mkdir(parents=True, exist_ok=True)
    srv = serve(
        str(ROOT / "web" / "dist-investigate"),
        PORT,
        mounts={"/data": str(ROOT / "data" / "viewer")},
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    rows, errors = {}, []
    with sync_playwright() as p:
        for scene in args.scenes:
            for strength in args.strengths:
                stem = f"edl_{scene}_" + ("page" if strength is None else f"{strength:g}")
                row = {"scene": scene, "strength": strength, "themes": {}}
                shots = []
                for theme in ("light", "dark"):
                    got = shoot(p, SCENES[scene], strength, theme, stem)
                    errors += got["errors"]
                    shade = shade_of(got["on"], got["off"], theme)
                    d = pal["themes"][theme]
                    median_shade = float(np.median(shade)) if shade.size else 1.0
                    t = {
                        "strength": chosen[theme] if strength is None else strength,
                        "shade_median": round(median_shade, 4),
                        "shade_darkest_tenth": round(float(np.percentile(shade, 10)), 4)
                        if shade.size
                        else 1.0,
                    }
                    if theme == "dark":
                        cut = failing_shade(d["tokens"], d["ground"], floor)
                        t["contrast_fails_below_shade"] = cut
                        t["share_of_pixels_below_it"] = (
                            round(float((shade < cut).mean()), 4) if shade.size else 0.0
                        )
                    if not args.probe:
                        g = measure(got["on"], theme)
                        rules = check_theme(d["tokens"], d["ground"], floor, median_shade)
                        t |= {
                            "grain": g["grain"],
                            "square": g["square"],
                            "rules_at_median_shade": {
                                "failures": rules["failures"],
                                "contrast_min": rules["contrast_min"],
                                "min_contrast_rule": MIN_CONTRAST,
                            },
                        }
                    row["themes"][theme] = t
                    shots.append(got["on"])
                    print(stem, theme, t, flush=True)
                if not args.probe:
                    if strength is None:
                        labels = [
                            f"{scene}, eye-dome {chosen['light']:g}, light",
                            f"{scene}, eye-dome {chosen['dark']:g}, dark",
                        ]
                    else:
                        label = "off" if strength == 0 else f"strength {strength:g}"
                        labels = [
                            f"{scene}, eye-dome {label}, light",
                            f"{scene}, eye-dome {label}, dark",
                        ]
                    compose(shots, SHEETS / f"{stem}.png", labels)
                    row["fps"] = fps(p, SCENES[scene], strength)
                    print(stem, "fps", row["fps"], flush=True)
                rows[stem] = row
    srv.shutdown()
    if args.probe:
        return 0
    doc = {
        "what": "eye-dome lighting on the live page, per scene and strength"
        + (", each theme at the strength the sweep chose" if args.page else ""),
        "viewport": SIZE,
        "repeats": REPEATS,
        "fps_threshold": 30.0,
        "rows": rows,
        "console_errors": errors,
    }
    out = OUT_PAGE if args.page else OUT
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("console errors:", errors or "none")
    print(f"wrote {out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
