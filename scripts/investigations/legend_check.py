"""The first screen's legend at the size a reader sees it, and whether the strip under it fits.

Three checks on the built page, all in a real browser:

- layout: at four common desktop sizes, where the provenance strip ends against the bottom of the
  first screen, whether any of its cells is clipped, and whether it runs under the model label;
- the free view at the same four sizes: the controls panel in the bottom right, the legend in the bottom
  left, that the two do not touch, and that the panel leaves the middle of the scene free, the middle
  being the central half of the width and height;
- the legend in the light theme, cut out of a screenshot at 1440 by 900 and put through the same
  colour vision simulation the palette's rules use, deuteranopia and protanopia beside normal
  vision, so the rows can be compared by eye at the size they are drawn. The mean colour of each
  swatch is measured off the picture and the smallest differences between swatches are written, per
  vision, with the colour difference the palette's rules use.

Run (the Python that has Playwright and Pillow):
    python scripts/investigations/legend_check.py
"""

from __future__ import annotations

import itertools
import json
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.colour_vision import delta_e76, simulate  # noqa: E402
from surrey.static_server import serve  # noqa: E402

OUT = ROOT / "data" / "artifacts" / "legend_check.json"
SHEET = ROOT / "data" / "design_archive" / "hero" / "legend_colour_vision.png"
CACHE = ROOT / "data" / "cache" / "legend"
PORT = 8788
SIZES = ((1440, 900), (1920, 1080), (1280, 800), (1366, 768))
VISIONS = ("normal", "deuteranopia", "protanopia")

LAYOUT_JS = """() => {
  const r = document.getElementById('strip').getBoundingClientRect();
  const c = document.getElementById('chip').getBoundingClientRect();
  const overlap = !(r.right <= c.left || r.left >= c.right || r.bottom <= c.top || r.top >= c.bottom);
  const clipped = [...document.querySelectorAll('#strip .cell')].some(e => e.scrollWidth > e.clientWidth + 1);
  const t = document.getElementById('asprs').getBoundingClientRect();
  return {strip_bottom: Math.round(r.bottom), viewport_height: innerHeight,
          fits: r.bottom <= innerHeight, overlaps_label: overlap, clipped,
          table_height: Math.round(t.height), table_share_of_height: Math.round(t.height / innerHeight * 1000) / 1000};
}"""

FREE_JS = """() => {
  const box = (sel) => { const e = document.querySelector(sel); if (!e) return null;
    const s = getComputedStyle(e); if (s.display === 'none' || s.visibility === 'hidden') return null;
    const b = e.getBoundingClientRect(); return {x: b.left, y: b.top, r: b.right, b: b.bottom}; };
  const touch = (a, c) => !!(a && c) && !(a.r <= c.x || a.x >= c.r || a.b <= c.y || a.y >= c.b);
  const panel = box('.free .panel'), legend = box('#legend'), chip = box('#chip');
  const mid = {x: innerWidth * 0.25, r: innerWidth * 0.75, y: innerHeight * 0.25, b: innerHeight * 0.75};
  return {panel_shown: !!panel, panel_bottom_gap: panel ? Math.round(innerHeight - panel.b) : null,
          panel_right_gap: panel ? Math.round(innerWidth - panel.r) : null,
          panel_touches_legend: touch(panel, legend), panel_touches_chip: touch(panel, chip),
          panel_covers_middle: touch(panel, mid), chip: document.getElementById('chip').textContent};
}"""

SWATCH_JS = """() => [...document.querySelectorAll('#asprs tr')].map(tr => {
  const i = tr.querySelector('td.swatch i'); const th = tr.querySelector('th');
  if (!i || !th || i.classList.contains('none')) return null;
  const b = i.getBoundingClientRect();
  return {row: th.textContent.trim(), x: b.left, y: b.top, w: b.width, h: b.height};
}).filter(Boolean)"""


def main() -> int:
    from playwright.sync_api import sync_playwright

    CACHE.mkdir(parents=True, exist_ok=True)
    srv = serve(str(ROOT / "web" / "dist"), PORT, mounts={"/data": str(ROOT / "data" / "viewer")})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    layout, swatches, free = {}, [], {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for w, h in SIZES:
            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(f"http://127.0.0.1:{PORT}/index.html")
            page.wait_for_selector("#strip .cell")
            page.wait_for_timeout(500)
            layout[f"{w}x{h}"] = page.evaluate(LAYOUT_JS)
            page.evaluate("document.getElementById('free').scrollIntoView({block: 'center'})")
            page.wait_for_timeout(1500)
            free[f"{w}x{h}"] = page.evaluate(FREE_JS)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(600)
            if (w, h) == (1440, 900):
                box = page.evaluate(
                    "() => { const b = document.getElementById('asprs').getBoundingClientRect();"
                    " return {x: b.left, y: b.top, w: b.width, h: b.height}; }"
                )
                swatches = page.evaluate(SWATCH_JS)
                page.screenshot(
                    path=str(CACHE / "legend_light.png"),
                    clip={"x": box["x"], "y": box["y"], "width": box["w"], "height": box["h"]},
                )
                legend_origin = (box["x"], box["y"])
            page.close()
        browser.close()
    srv.shutdown()

    crop = np.asarray(Image.open(CACHE / "legend_light.png").convert("RGB")).astype(float) / 255.0
    panels, per_vision = [], {}
    for vision in VISIONS:
        sim = (
            crop
            if vision == "normal"
            else np.clip(simulate(crop.reshape(-1, 3), vision).reshape(crop.shape), 0, 1)
        )
        panels.append((vision, Image.fromarray((sim * 255).round().astype(np.uint8))))
        means = {}
        for s in swatches:
            x0 = int(round(s["x"] - legend_origin[0])) + 1
            y0 = int(round(s["y"] - legend_origin[1])) + 1
            x1, y1 = x0 + max(int(s["w"]) - 2, 1), y0 + max(int(s["h"]) - 2, 1)
            means[s["row"]] = sim[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
        pairs = sorted(
            (
                (round(float(delta_e76(means[a], means[b])), 1), a, b)
                for a, b in itertools.combinations(means, 2)
            )
        )
        per_vision[vision] = [{"rows": [a, b], "delta_e76": d} for d, a, b in pairs[:6]]

    w, h = panels[0][1].size
    sheet = Image.new("RGB", (w * len(panels), h + 26), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("segoeui.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for k, (vision, img) in enumerate(panels):
        sheet.paste(img, (k * w, 26))
        draw.text((k * w + 8, 4), f"legend, light theme, {vision}", fill="black", font=font)
    sheet.save(SHEET)

    OUT.write_text(
        json.dumps(
            {
                "layout": layout,
                "free_view": free,
                "swatches_measured": [s["row"] for s in swatches],
                "closest_swatch_pairs": per_vision,
                "note": "swatch colour is the mean of the swatch's own pixels, border excluded,"
                " so a patterned swatch reads as its mix with the card behind it",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for size, row in layout.items():
        print(size, row)
    for size, row in free.items():
        print("free view", size, row)
    for vision, rows in per_vision.items():
        print(vision, rows[:4])
    print(f"wrote {OUT} and {SHEET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
