"""Why one framing is grainier than another: how many points the camera draws, over how many pixels.

The loader spends a fixed budget of points on whatever the camera sees, finest where the screen
needs it. So grain is not only the distance: a frame that shows the whole area on a small patch of
screen is dense, a frame that shows part of it closely is dense, and a frame that shows all of it
over most of the screen spreads the same budget thinnest. This counts, per framing, the points drawn
and the pixels the cloud covers, and writes both.

Run (the Python that has Playwright):
    npm --prefix web run build:investigate
    python scripts/investigations/points_per_pixel.py
"""

from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.static_server import serve  # noqa: E402

OUT = ROOT / "data" / "artifacts" / "points_per_pixel.json"
SHOTS = ROOT / "data" / "cache" / "ppp"
PORT = 8783
GROUND = (230, 233, 230)
FRAMINGS = (("sunset", "wide"), ("sunset", "mid"), ("sunset", "cards"), ("ggpark", "cards"))
# The budget is the page's own constant, read from the page, so the artefact cannot disagree with it.
BUDGET = int(
    re.search(
        r"POINT_BUDGET = ([\d_]+)", (ROOT / "web" / "src" / "viewer.js").read_text(encoding="utf-8")
    )
    .group(1)
    .replace("_", "")
)


def main() -> int:
    from playwright.sync_api import sync_playwright

    srv = serve(
        str(ROOT / "web" / "dist-investigate"),
        PORT,
        mounts={"/data": str(ROOT / "data" / "viewer")},
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    SHOTS.mkdir(parents=True, exist_ok=True)
    rows = {}
    with sync_playwright() as p:
        for area, framing in FRAMINGS:
            browser = p.chromium.launch(
                args=[
                    "--enable-gpu",
                    "--ignore-gpu-blocklist",
                    "--use-angle=d3d11",
                    "--disable-domain-blocking-for-3d-apis",
                ]
            )
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            url = f"http://127.0.0.1:{PORT}/index.html?hero=b&area={area}&order=2"
            page.goto(url + f"&open={framing}")
            page.wait_for_function(
                "() => { const m = window.__surrey; if (!m || m.firstFrameMs === null) return false;"
                " const a = Object.values(m.areas); return a.length && a.every(r => r.loading === 0 && r.idleSince !== null); }",
                timeout=90000,
            )
            page.wait_for_timeout(900)
            shown = page.evaluate("() => Object.values(window.__surrey.areas)[0].shown")
            dpr = page.evaluate("() => window.devicePixelRatio")
            shot = SHOTS / f"{area}_{framing}.png"
            # Only the canvas: the points drawn are counted over the whole viewport, so the pixels
            # have to be too, without the text column and the strip standing in for cloud.
            page.evaluate(
                "() => document.querySelectorAll('body > *:not(#stage)')"
                ".forEach((e) => { e.style.visibility = 'hidden'; })"
            )
            page.wait_for_timeout(300)
            page.screenshot(path=str(shot))
            browser.close()
            a = np.asarray(Image.open(shot).convert("RGB")).astype(int)
            bg = np.abs(a - np.array(GROUND)).sum(axis=2) < 12
            solid = binary_fill_holes(binary_closing(~bg, structure=np.ones((9, 9))))
            covered = int(solid.sum())
            name = f"{area}_{framing}"
            rows[name] = {
                "area": area,
                "framing": framing,
                "points_drawn": int(shown),
                "pixels_covered": covered,
                "device_pixel_ratio": dpr,
                "points_per_covered_pixel": round(shown / max(covered, 1), 3),
            }
            print(name, rows[name], flush=True)
    srv.shutdown()
    OUT.write_text(
        json.dumps(
            {
                "what": "points drawn against pixels the cloud covers, per framing, light theme,"
                " 1440 by 900, whole viewport",
                "budget": BUDGET,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
