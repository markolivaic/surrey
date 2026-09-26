"""The two pictures at the top of the README, shot from the real page.

The first screen as a reader meets it, and cards 1 and 3 side by side: the same Golden Gate Park under
the published model and under the old model with its input as delivered. Both are shot at 1440 by 900
in the light theme, from web/dist over the real octrees, once every area on screen has finished
loading. The renderer is read first, as scripts/pipeline/record_walkthrough.py does: a software
renderer is not the page, so the shot moves to a headed browser, and refuses if that is software too.

Writes docs/design/readme/first_screen.webp and docs/design/readme/cards_1_and_3.webp.

Run (the Python that has Playwright), after `npm run build` in web/:
    python scripts/pipeline/readme_images.py
"""

from __future__ import annotations

import io
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))

from record_walkthrough import PORT, VIEWPORT, renderer_of, settle  # noqa: E402
from surrey.static_server import serve  # noqa: E402

OUT = ROOT / "docs" / "design" / "readme"
GAP_PX = 16


def shot(page) -> bytes:
    settle(page)
    page.wait_for_timeout(2500)
    return page.screenshot()


def main() -> int:
    from PIL import Image
    from playwright.sync_api import sync_playwright

    srv = serve(str(ROOT / "web" / "dist"), PORT, mounts={"/data": str(ROOT / "data" / "viewer")})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=d3d11"]
        )
        renderer = renderer_of(browser)
        if "swiftshader" in renderer.lower():
            browser.close()
            browser = p.chromium.launch(headless=False)
            renderer = renderer_of(browser)
        if "swiftshader" in renderer.lower():
            raise SystemExit(
                "the page is on a software renderer; a picture of that is not the page"
            )
        print(f"renderer: {renderer}")
        page = browser.new_page(viewport=VIEWPORT, color_scheme="light")
        page.goto(f"http://127.0.0.1:{PORT}/index.html")
        first = shot(page)
        cards = []
        for n in (1, 3):
            page.evaluate(
                f"document.querySelector('.card[data-card=\"{n}\"]').scrollIntoView({{block: 'center'}})"
            )
            page.wait_for_timeout(1200)
            cards.append(shot(page))
            print(f"card {n}: {page.text_content('#chip')}")
        browser.close()
    srv.shutdown()

    Image.open(io.BytesIO(first)).convert("RGB").save(
        OUT / "first_screen.webp", "WEBP", quality=82, method=6
    )
    a, b = (Image.open(io.BytesIO(c)).convert("RGB") for c in cards)
    pair = Image.new("RGB", (a.width + GAP_PX + b.width, max(a.height, b.height)), "white")
    pair.paste(a, (0, 0))
    pair.paste(b, (a.width + GAP_PX, 0))
    pair.save(OUT / "cards_1_and_3.webp", "WEBP", quality=82, method=6)
    for f in sorted(OUT.glob("*.webp")):
        print(f"wrote {f.relative_to(ROOT).as_posix()} ({f.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
