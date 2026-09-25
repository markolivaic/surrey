"""The first screen in its three arrangements, so the choice can be made by eye.

Each arrangement is shot at 1440 by 900 and at a phone width, in light and in dark, over the real
Golden Gate Park octree with the published model's classes. The four shots of one arrangement are
composed into two sheets, desktop and phone, light beside dark, which is what gets sent.

Not a measurement: no timing, no threshold. It needs the card free only because a browser drawing
three million points beside a training run would measure neither well.

Run (the Python that has Playwright):
    npm --prefix web run build:investigate
    python scripts/investigations/shoot_hero.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.static_server import serve  # noqa: E402

OUT = ROOT / "data" / "design_archive" / "hero"  # the sheets, kept locally with the design rounds
SINGLES = ROOT / "data" / "cache" / "hero"  # one theme at a time, working material
PORT = 8778
VARIANTS = ("b",)
# Each shot is an area and its opening: the order, and the framing ("mid", "wide" or None for the
# cards' own).
SHOTS = (("sunset", 2, "mid"), ("ggpark", 2, None))
SIZES = {"desktop": {"width": 1440, "height": 900}, "phone": {"width": 390, "height": 844}}
THEMES = ("light", "dark")


def wait_settled(page, timeout_ms: int = 90000, live_scene: bool = True) -> None:
    """A phone draws no live scene by design, so there it waits for the still instead."""
    if not live_scene:
        page.wait_for_function(
            "() => [...document.querySelectorAll('img.still')].some(i => i.complete"
            " && i.naturalWidth > 0)",
            timeout=timeout_ms,
        )
        page.wait_for_timeout(600)
        return
    page.wait_for_function(
        "() => { const m = window.__dayton || window.__surrey; if (!m || m.firstFrameMs === null) return false;"
        " const a = Object.values(m.areas); return a.length && a.every(r => r.loading === 0 && r.idleSince !== null); }",
        timeout=timeout_ms,
    )
    page.wait_for_timeout(900)


GROUND = {"light": (230, 233, 230), "dark": (19, 24, 21)}
BLOCK = 450  # the side of the square the share is measured in
SCENE_X0 = 560  # right of the document column, which is 38% of 1440


def empty_share(shot: Path, theme: str) -> dict:
    """The share of pixels still showing the page ground, in one square of the scene.

    The same square everywhere: BLOCK by BLOCK, centred on the middle of what is drawn, so that a
    frame whose cloud does not fill the window is not charged for the window it does not fill.
    """
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(shot).convert("RGB")).astype(int)
    scene = a[:, SCENE_X0:]
    bg = np.abs(scene - np.array(GROUND[theme])).sum(axis=2) < 12
    ys, xs = np.nonzero(~bg)
    if len(xs) == 0:
        return {"empty_share": 1.0, "block": BLOCK, "note": "nothing drawn"}
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = max(0, min(cy - BLOCK // 2, scene.shape[0] - BLOCK))
    x0 = max(0, min(cx - BLOCK // 2, scene.shape[1] - BLOCK))
    block = bg[y0 : y0 + BLOCK, x0 : x0 + BLOCK]
    # The share above counts sky as well as holes, which is fair for a frame the cloud fills and
    # unfair for a skyline. The second number counts only background that the cloud encloses: the
    # holes a reader sees as grain, with the sky around the silhouette left out.
    from scipy.ndimage import binary_closing, binary_fill_holes

    drawn = ~bg
    solid = binary_fill_holes(binary_closing(drawn, structure=np.ones((9, 9))))
    grain = float((bg & solid).sum()) / max(float(solid.sum()), 1.0)
    return {
        "empty_share": round(float(block.mean()), 4),
        "block": BLOCK,
        "grain_inside_the_cloud": round(grain, 4),
        "block_origin_in_scene": [x0, y0],
        "scene_x0": SCENE_X0,
    }


def compose(paths: list[Path], out: Path, labels: list[str]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    shots = [Image.open(p) for p in paths]
    w, h = shots[0].size
    label_h = 30
    sheet = Image.new("RGB", (w * len(shots), h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("segoeui.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, (shot, label) in enumerate(zip(shots, labels, strict=True)):
        sheet.paste(shot, (i * w, label_h))
        draw.text((i * w + 10, 7), label, fill="black", font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main() -> int:
    from playwright.sync_api import sync_playwright

    srv = serve(
        str(ROOT / "web" / "dist-investigate"),
        PORT,
        mounts={"/data": str(ROOT / "data" / "viewer")},
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    OUT.mkdir(parents=True, exist_ok=True)
    SINGLES.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    measured: dict[str, dict] = {}
    with sync_playwright() as p:
        for area, order, framing in SHOTS:
            for variant in VARIANTS:
                for size_name, size in SIZES.items():
                    shots = []
                    for theme in THEMES:
                        browser = p.chromium.launch(
                            args=[
                                "--enable-gpu",
                                "--ignore-gpu-blocklist",
                                "--use-angle=d3d11",
                                "--disable-domain-blocking-for-3d-apis",
                            ]
                        )
                        page = browser.new_page(viewport=size)
                        page.on(
                            "console",
                            lambda m: errors.append(m.text) if m.type == "error" else None,
                        )
                        page.on("pageerror", lambda e: errors.append(str(e)))
                        page.goto(
                            f"http://127.0.0.1:{PORT}/index.html"
                            f"?hero={variant}&area={area}&order={order}"
                            f"{'&open=' + framing if framing else ''}"
                        )
                        wait_settled(page, live_scene=size_name == "desktop")
                        if theme == "dark":
                            page.click("#theme")
                            page.wait_for_timeout(1200)
                        tag = f"order{order}{framing or ''}"
                        shot = SINGLES / f"hero_{area}_{tag}_{size_name}_{theme}.png"
                        page.screenshot(path=str(shot))
                        if size_name == "desktop":
                            measured[shot.name] = empty_share(shot, theme) | {
                                "area": area,
                                "order": order,
                                "framing": framing or "cards",
                                "theme": theme,
                            }
                            print(
                                f"    {shot.name}: empty {measured[shot.name]['empty_share']}",
                                flush=True,
                            )
                        shots.append(shot)
                        browser.close()
                    frame = {"wide": "whole block", "mid": "middle framing"}.get(
                        framing, "the cards' framing"
                    )
                    sheet = OUT / f"hero_{area}_{tag}_{size_name}.png"
                    compose(
                        shots,
                        sheet,
                        [
                            f"{area}, order {order}, {frame}, {size_name}, light",
                            f"{area}, order {order}, {frame}, {size_name}, dark",
                        ],
                    )
                    print(f"  {area} {tag} {size_name} done", flush=True)
    srv.shutdown()
    import json

    (SINGLES / "emptiness.json").write_text(
        json.dumps(
            {
                "method": "share of pixels equal to the page ground inside one 450 by 450 square of the"
                " scene, centred on what is drawn; the scene is everything right of x=560 at"
                " 1440 by 900",
                "shots": measured,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("console errors:", errors or "none")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
