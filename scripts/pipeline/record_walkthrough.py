"""Record the walkthrough of the real page, then hand the recording to make_walkthrough.py.

The scenario, written before any recording and brought up to date with the page on 2026-09-23: open
the page at 1440 by 900 and let the first screen settle; scroll it away slowly, so the colour crosses
from the model's classes to the photograph; go through the six cards one at a time and let each
settle; arrive in the free view, turn the NAIP divider on, and drag the handle once across the scene.
One uninterrupted recording, no stitched takes; only the seconds before the first screen has drawn
are cut from its front.

`--dry-run` walks the same scenario without recording and without encoding, and fails on any console
error, on a card that never settles, or on a software renderer: the check before the real take.

Playwright's recordVideo drives the page. The WebGL renderer is read first: if headless Chromium
falls back to a software renderer (SwiftShader), the recording is slow and ugly, so the run
switches to a headed browser. make_walkthrough.py then locates ffmpeg, encodes the MP4, the GIF and
the poster, and reads codec, pixel format and size back from the files with ffprobe.

Not run until the page has M1 in it. It refuses to run while the GPU is training.

Run (the Python that has Playwright):
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
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from surrey.static_server import serve  # noqa: E402
from surrey.tools import gpu_busy  # noqa: E402

VIEWPORT = {"width": 1440, "height": 900}
WORK = ROOT / "data" / "cache" / "walkthrough_recording"
PORT = 8776


def settle(page, timeout_ms: int = 60000) -> None:
    page.wait_for_function(
        "() => { const m = window.__surrey; if (!m || m.firstFrameMs === null) return false;"
        " const a = Object.values(m.areas); return a.length && a.every(r => r.loading === 0 && r.idleSince !== null); }",
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
        ctx = browser.new_context(viewport=VIEWPORT, **record)
        page = ctx.new_page()
        # The recording starts with the page, before it has drawn anything; the white seconds of loading
        # are cut from the front, and only those.
        recording_started = time.monotonic()
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{PORT}/index.html")
        settle(page)
        first_screen_at = time.monotonic() - recording_started
        page.wait_for_timeout(2500)
        print(f"first screen: {page.text_content('#chip')}, settled {first_screen_at:.1f} s in")
        # The first screen's own reveal, driven by the reader's scrolling: classes, then the photograph.
        # Far enough for the colour to have crossed, and not so far that the first card takes the scene.
        for _ in range(13):
            page.mouse.wheel(0, 30)
            page.wait_for_timeout(60)
        page.wait_for_timeout(1800)
        print(f"first screen, photograph: {page.text_content('#chip')}")
        for _ in range(11):
            page.mouse.wheel(0, 30)
            page.wait_for_timeout(60)
        page.wait_for_timeout(1200)
        cards = page.locator(".card").count()
        for i in range(cards):
            page.evaluate(
                f"document.querySelectorAll('.card')[{i}].scrollIntoView({{block: 'center', behavior: 'smooth'}})"
            )
            page.wait_for_timeout(1200)
            settle(page)
            page.wait_for_timeout(2500)
            title = page.locator(".card").nth(i).locator("h2").text_content()
            print(f"card {i + 1} of {cards}: {title} | {page.text_content('#chip')}")
        page.evaluate(
            "document.getElementById('free').scrollIntoView({block: 'end', behavior: 'smooth'})"
        )
        page.wait_for_timeout(1500)
        settle(page)
        page.click("#wipe")
        page.wait_for_timeout(800)
        box = page.locator("#grip").bounding_box()
        y = box["y"] + box["height"] / 2
        page.mouse.move(box["x"] + box["width"] / 2, y)
        page.mouse.down()
        for k in range(60):
            page.mouse.move(VIEWPORT["width"] * (0.5 - 0.35 * k / 59), y)
            page.wait_for_timeout(16)
        for k in range(90):
            page.mouse.move(VIEWPORT["width"] * (0.15 + 0.7 * k / 89), y)
            page.wait_for_timeout(16)
        page.mouse.up()
        page.wait_for_timeout(1500)
        print(
            f"free view: divider at {page.evaluate('window.__surrey && document.getElementById("handle") && !document.getElementById("handle").hidden')}"
        )
        video = None if args.dry_run else Path(page.video.path())
        ctx.close()
        browser.close()
    srv.shutdown()
    time.sleep(1)
    if errors:
        print("console errors:", *errors, sep="\n  ")
        return 1
    print("console errors: none")
    if args.dry_run:
        print(f"dry run passed: first screen, {cards} cards and the free view, on {renderer}")
        return 0
    print(f"recorded {video}")
    return subprocess.call(
        [
            sys.executable,
            str(ROOT / "scripts" / "pipeline" / "make_walkthrough.py"),
            "--source",
            str(video),
            "--start",
            f"{first_screen_at:.2f}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
