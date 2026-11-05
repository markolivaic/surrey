"""Record the walkthrough of the real page, and encode it.

One uninterrupted take of the built page over the real octrees: the first screen with the lidar
standard's table; the cards from Golden Gate Park under the published model to the same park under the
return-ratio model; then the free view of the Financial District, orbiting slowly, holding on the legend.
Only the seconds before the first screen has drawn are cut from the front.

The page is recorded at 1100 by 780, a real desktop layout small enough that the card text stays
readable in the 880 px wide README column. make_walkthrough.py then encodes the MP4, the animation for
the README and the poster.

If headless Chromium falls back to a software renderer (SwiftShader), the run switches to a headed
browser, and it refuses to record on a software renderer at all. It also refuses while the GPU trains.
`--dry-run` walks the same scenario without recording, and fails on any console error or on a card that
never settles.

Needs web/dist and data/viewer, Playwright's Chromium (python -m playwright install chromium) and
ffmpeg (on PATH, or named by SURREY_FFMPEG in .env.example).

Run:
    python scripts/pipeline/record_walkthrough.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from surrey.static_server import serve  # noqa: E402
from surrey.tools import gpu_busy  # noqa: E402

VIEWPORT = {"width": 1100, "height": 780}
WORK = ROOT / "data" / "cache" / "walkthrough_recording"
PORT = 8776
MAX_SECONDS = 25.0


def settle(page, area: str, timeout_ms: int = 60000) -> None:
    """Wait until one area has every point it wants. An area left before it finished keeps its last
    state, so only the one on screen is asked."""
    page.wait_for_function(
        "(area) => { const m = window.__surrey; if (!m || m.firstFrameMs === null) return false;"
        " const r = m.areas[area]; return !!r && r.loading === 0 && r.idleSince !== null; }",
        arg=area,
        timeout=timeout_ms,
    )


def renderer_of(browser) -> str:
    page = browser.new_page()
    page.goto(f"http://127.0.0.1:{PORT}/index.html")
    page.wait_for_function(
        "() => window.__surrey && window.__surrey.firstFrameMs !== null", timeout=60000
    )
    name = page.evaluate("""() => { const c = document.createElement('canvas').getContext('webgl');
      const e = c.getExtension('WEBGL_debug_renderer_info'); return e ? c.getParameter(e.UNMASKED_RENDERER_WEBGL) : ''; }""")
    page.close()
    return name


def show_card(page, number: int, area: str, hold_ms: int) -> None:
    """Scroll one card to the middle of the screen, wait for its points, and hold on it."""
    page.evaluate(
        f"document.querySelector('.card[data-card=\"{number}\"]')"
        ".scrollIntoView({block: 'center', behavior: 'smooth'})"
    )
    page.wait_for_timeout(900)
    settle(page, area)
    page.wait_for_timeout(hold_ms)
    title = page.locator(f'.card[data-card="{number}"] h2').text_content()
    print(f"card {number}: {title} | {page.text_content('#chip')}")


def scenario(page) -> None:
    """Everything after the first screen has settled."""
    page.wait_for_timeout(2500)
    print(f"first screen: {page.text_content('#chip')}")
    show_card(page, 1, "ggpark", 3000)
    show_card(page, 2, "5135_54430", 1000)
    show_card(page, 3, "ggpark", 3000)
    # Short of the section's full height, so the credits under it stay off the screen and the free
    # view's panel and legend stay on it.
    page.evaluate(
        "window.scrollTo({top: scrollY + document.getElementById('free').getBoundingClientRect().top - 24,"
        " behavior: 'smooth'})"
    )
    page.wait_for_timeout(1000)
    page.click('#areas button[data-area="financial"]')
    page.wait_for_timeout(300)
    settle(page, "financial")
    box = page.locator("#stage").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    for k in range(120):
        page.mouse.move(x - k, y)
        page.wait_for_timeout(33)
    page.mouse.up()
    page.wait_for_timeout(1500)
    print(f"free view: {page.text_content('#chip')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run", action="store_true", help="walk the scenario, record nothing, fail on errors"
    )
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    busy, gpu = gpu_busy()
    if busy:
        raise SystemExit(f"GPU busy ({gpu}); record only while nothing trains")
    srv = serve(str(ROOT / "web" / "dist"), PORT, mounts={"/data": str(ROOT / "data" / "viewer")})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    WORK.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=d3d11"]
        )
        renderer = renderer_of(browser)
        if "swiftshader" in renderer.lower():
            browser.close()
            browser = p.chromium.launch(headless=False)
            renderer = renderer_of(browser)
        print(f"renderer: {renderer}")
        if "swiftshader" in renderer.lower():
            raise SystemExit(
                "the page is on a software renderer; a walkthrough of that is not the page"
            )
        record = (
            {} if args.dry_run else {"record_video_dir": str(WORK), "record_video_size": VIEWPORT}
        )
        ctx = browser.new_context(viewport=VIEWPORT, color_scheme="light", **record)
        page = ctx.new_page()
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{PORT}/index.html")
        settle(page, "sunset")
        first_screen = time.monotonic()
        scenario(page)
        # The video starts late against the clock, so the cut is measured back from its end.
        keep = time.monotonic() - first_screen
        video = None if args.dry_run else Path(page.video.path())
        ctx.close()
        browser.close()
    srv.shutdown()
    time.sleep(1)
    print(f"{keep:.1f} s from the first screen to the end")
    if errors:
        print("console errors:", *errors, sep="\n  ")
        return 1
    print("console errors: none")
    if keep > MAX_SECONDS:
        print(f"longer than {MAX_SECONDS:.0f} s; shorten the holds in scenario()")
        return 1
    if args.dry_run:
        print(f"dry run passed on {renderer}")
        return 0
    print(f"recorded {video}")
    return subprocess.call(
        [
            sys.executable,
            str(ROOT / "scripts" / "pipeline" / "make_walkthrough.py"),
            "--source",
            str(video),
            "--keep",
            f"{keep:.2f}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
