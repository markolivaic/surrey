"""The same frame at each point budget, side by side, so the choice can be made by eye as well as by clock.

The load time thresholds can be met by drawing fewer points. That trade is not something to settle
quietly inside a measurement script: this renders the park and Embarcadero at each candidate budget,
from the same camera, over the same throttled link, and writes one image with the budget and the
measured time under each frame.

Each frame is taken once loading has settled, so it shows what a reader ends up looking at, not what
flashes past on the way. The time under it is the same f figure the thresholds use: how long until 90
percent of the nodes that settled camera shows have arrived, at 20 Mbit/s.

Needs the card to itself, like every other measurement here.

Run:
    python scripts/investigations/budget_frames.py
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

import measure_prototype as mp  # noqa: E402
from surrey.static_server import serve  # noqa: E402
from surrey.tools import gpu_busy  # noqa: E402

OUT_IMAGE = ROOT / "data" / "design_archive" / "prototype" / "budget_frames.png"
OUT = ROOT / "data" / "artifacts" / "budget_frames.json"
BUDGETS = mp.VISIBILITY_BUDGETS
AREAS = (("ggpark", 0), ("embarcadero", 3))
LABEL_H = 46


def compose(cells: list[dict], path: Path, columns: int) -> None:
    """One image: a row per area, a column per budget, each frame labelled with its budget and time."""
    from PIL import Image, ImageDraw, ImageFont

    shots = [Image.open(c["file"]) for c in cells]
    w, h = shots[0].size
    scale = 560 / w
    w, h = int(w * scale), int(h * scale)
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (w * columns, (h + LABEL_H) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("segoeui.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for i, (cell, shot) in enumerate(zip(cells, shots, strict=True)):
        x, y = (i % columns) * w, (i // columns) * (h + LABEL_H)
        sheet.paste(shot.resize((w, h), Image.LANCZOS), (x, y))
        seconds = cell["seconds_to_90_percent"]
        line1 = f"{cell['area']}, budget {cell['budget']:,} points"
        line2 = (
            "90 percent of its nodes in "
            + (f"{seconds:.3f} s" if seconds is not None else "no figure")
            + f" at 20 Mbit/s, {cell['nodes']} nodes"
        )
        draw.text((x + 8, y + h + 6), line1, fill="black", font=font)
        draw.text((x + 8, y + h + 24), line2, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-busy-gpu", action="store_true", help="only for trying the plumbing")
    args = ap.parse_args()
    busy, gpu = gpu_busy()
    if busy and not args.allow_busy_gpu:
        raise SystemExit(f"GPU busy ({gpu}); measure only while nothing trains")
    page_srv = serve(str(ROOT / "web" / "dist-prototype"), mp.PAGE_PORT)
    data_srv = serve(str(ROOT / "data" / "viewer"), mp.DATA_PORT)
    for s in (page_srv, data_srv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    shots_dir = ROOT / "data" / "design_archive" / "prototype" / "budget"
    shots_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    with sync_playwright() as p:
        renderer = None
        for area, card in AREAS:
            for budget in BUDGETS:
                # A browser per frame: this laptop's integrated GPU stops handing out WebGL contexts
                # after a few heavy ones in the same browser, which fails the harness, not the page.
                browser = p.chromium.launch(headless=True, args=mp.CHROMIUM_ARGS)
                ctx = browser.new_context(viewport=mp.VIEWPORT, device_scale_factor=1)
                page = ctx.new_page()
                cdp = ctx.new_cdp_session(page)
                cdp.send("Network.enable")
                rate = 20_000_000 / 8
                cdp.send(
                    "Network.emulateNetworkConditions",
                    {
                        "offline": False,
                        "latency": 0,
                        "downloadThroughput": rate,
                        "uploadThroughput": rate,
                    },
                )
                page.goto(mp.url(edl=0, budget=budget, order="visibility"))
                t0 = 0.0
                if card:
                    mp.settle(page, "ggpark")
                    page.evaluate(
                        f"document.querySelectorAll('.card')[{card}].scrollIntoView({{block: 'center'}})"
                    )
                    start = time.time()
                    sw = None
                    while time.time() - start < 60:
                        got = page.evaluate(
                            f"window.__surrey.switches.filter(s => s && s.to === '{area}')"
                        )
                        if got and got[-1]["firstFrameMs"] is not None:
                            sw = got[-1]
                            break
                        page.wait_for_timeout(25)
                    t0 = sw["t0"] if sw else 0.0
                rec = mp.settle(page, area)
                renderer = renderer or page.evaluate("window.__surrey.renderer")
                shot = shots_dir / f"{area}_{budget}.png"
                page.screenshot(path=str(shot))
                cells.append(
                    {
                        "area": area,
                        "budget": budget,
                        "seconds_to_90_percent": mp.time_to_share(rec, t0) if rec else None,
                        "nodes": len(rec["visibleNames"]) if rec else 0,
                        "octree_bytes": rec.get("bytes") if rec else None,
                        "file": str(shot),
                    }
                )
                print(cells[-1], flush=True)
                ctx.close()
                browser.close()
    page_srv.shutdown()
    data_srv.shutdown()
    compose(cells, OUT_IMAGE, columns=len(BUDGETS))
    OUT.write_text(
        json.dumps(
            {
                "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "webgl_renderer": renderer,
                "throttle_megabit_per_second": 20,
                "order": "visibility",
                "image": str(OUT_IMAGE.relative_to(ROOT)).replace("\\", "/"),
                "one_run_per_cell": True,
                "cells": [
                    {k: v for k, v in c.items() if k != "file"}
                    | {"file": str(Path(c["file"]).relative_to(ROOT)).replace("\\", "/")}
                    for c in cells
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUT} and {OUT_IMAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
