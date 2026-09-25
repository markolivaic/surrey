"""Eye-dome strength per theme, chosen by the palette's rules on a fine grid.

The rule, written before this first ran: for each theme, the largest strength in the
grid 0, 0.05 ... 0.6 at which every palette rule passes on both scenes (Sunset in the middle framing,
Golden Gate Park in the cards' framing) in all three simulated visions, at the median shade the pass
leaves on the drawn pixels, provided every smaller strength in the grid passes too.

One page load per scene and theme: the camera settles with the pass off, that frame is kept, and then
each strength is set in turn on the same camera and compared with it, so the shade is the pass's alone.

The shade a strength leaves depends on the pass and the scene, not on the palette. So when the palette
changes, --rechoose reads each rule again at the shades already measured, with the palette as it now
is, and chooses again by the same rule; nothing is rendered.

Run (the Python that has Playwright, Pillow and SciPy):
    npm --prefix web run build:investigate
    python scripts/investigations/edl_sweep.py
    python scripts/investigations/edl_sweep.py --rechoose
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from edl_candidates import ARGS, SCENES, SIZE, settled, shade_of  # noqa: E402
from surrey.palette import check_theme, load_palette  # noqa: E402
from surrey.static_server import serve  # noqa: E402

CACHE = ROOT / "data" / "cache" / "edl_sweep"
OUT = ROOT / "data" / "artifacts" / "edl_sweep.json"
PORT = 8786
GRID = [round(0.05 * i, 2) for i in range(13)]
RULE = (
    "for each theme, the largest strength in the grid at which every palette rule passes on both scenes"
    " in all three visions at the median shade, provided every smaller strength in the grid passes too;"
    " written on 2026-09-23 before the sweep ran"
)


def choose(passes: dict[float, bool]) -> float:
    """The largest strength such that it and every smaller one pass."""
    chosen = 0.0
    for s in sorted(passes):
        if not passes[s]:
            break
        chosen = s
    return chosen


def chosen_by_rule(cells: dict) -> dict:
    chosen = {}
    for theme in ("light", "dark"):
        passes = {
            s: all(not cells[f"{theme}|{scene}|{s:g}"]["failures"] for scene in SCENES)
            for s in GRID
        }
        chosen[theme] = {
            "strength": choose(passes),
            "passes_by_strength": {f"{s:g}": v for s, v in passes.items()},
        }
    return chosen


def rechoose() -> int:
    """The rule again, at the measured shades, with the palette as it now is."""
    pal = load_palette()
    floor = pal["floor"] if pal["floor"] is not None else 0.2
    doc = json.loads(OUT.read_text(encoding="utf-8"))
    for cell in doc["cells"].values():
        d = pal["themes"][cell["theme"]]
        rules = check_theme(d["tokens"], d["ground"], floor, cell["shade_median"])
        cell["failures"] = rules["failures"]
        cell["contrast_min"] = rules["contrast_min"]
    doc["chosen"] = chosen_by_rule(doc["cells"])
    doc["palette_read"] = "docs/design/palette.json as it stood when --rechoose last ran"
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("chosen:", {t: c["strength"] for t, c in doc["chosen"].items()})
    return 0


def main() -> int:
    if "--rechoose" in sys.argv[1:]:
        return rechoose()
    from playwright.sync_api import sync_playwright

    pal = load_palette()
    floor = pal["floor"] if pal["floor"] is not None else 0.2
    CACHE.mkdir(parents=True, exist_ok=True)
    srv = serve(
        str(ROOT / "web" / "dist-investigate"),
        PORT,
        mounts={"/data": str(ROOT / "data" / "viewer")},
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cells: dict[str, dict] = {}
    errors: list[str] = []
    with sync_playwright() as p:
        for theme in ("light", "dark"):
            d = pal["themes"][theme]
            for scene, query in SCENES.items():
                browser = p.chromium.launch(args=ARGS)
                page = browser.new_page(viewport=SIZE)
                page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(f"http://127.0.0.1:{PORT}/index.html?{query}&edl=0")
                settled(page)
                if theme == "dark":
                    page.click("#theme")
                    page.wait_for_timeout(1200)
                page.evaluate("window.__surreyEdl(0)")
                page.wait_for_timeout(400)
                off = CACHE / f"{scene}_{theme}_off.png"
                page.screenshot(path=str(off))
                for s in GRID:
                    page.evaluate(f"window.__surreyEdl({s})")
                    page.wait_for_timeout(400)
                    on = CACHE / f"{scene}_{theme}_{s:g}.png"
                    page.screenshot(path=str(on))
                    shade = shade_of(on, off, theme)
                    median = float(np.median(shade)) if shade.size else 1.0
                    rules = check_theme(d["tokens"], d["ground"], floor, median)
                    cells[f"{theme}|{scene}|{s:g}"] = {
                        "theme": theme,
                        "scene": scene,
                        "strength": s,
                        "shade_median": round(median, 4),
                        "failures": rules["failures"],
                        "contrast_min": rules["contrast_min"],
                    }
                    print(
                        theme, scene, s, round(median, 4), rules["failures"] or "pass", flush=True
                    )
                browser.close()
    srv.shutdown()
    chosen = chosen_by_rule(cells)
    doc = {
        "rule_written_before": RULE,
        "grid": GRID,
        "scenes": SCENES,
        "viewport": SIZE,
        "cells": cells,
        "chosen": chosen,
        "console_errors": errors,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("chosen:", {t: c["strength"] for t, c in chosen.items()})
    print("console errors:", errors or "none")
    print(f"wrote {OUT}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
