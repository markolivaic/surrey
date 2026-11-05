"""Smoke tests: the built page loads the FIXTURE octree, draws it and shows the FIXTURE banner, and a
card keeps the model's colours when the reader scrolls off the first screen.

Runs in CI on Linux with a software WebGL renderer, which is fine for "does it draw" and useless
for speed, so nothing here measures time. Skips, with the reason in the output, when Playwright or
the built page is missing.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

from surrey.static_server import serve  # noqa: E402

DIST = ROOT / "web" / "dist"


def serve_page():
    playwright = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    if not (DIST / "index.html").exists():
        pytest.skip("web/dist is not built; run npm run build in web/")
    srv = serve(str(DIST), 0, mounts={"/fixtures": str(ROOT / "web" / "fixtures")})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return playwright, srv


def open_fixture(browser, srv, errors: list[str]):
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(f"http://127.0.0.1:{srv.server_address[1]}/index.html?fixture=1")
    page.wait_for_function(
        "() => window.__surrey && window.__surrey.firstFrameMs !== null", timeout=60000
    )
    return page


def test_the_page_draws_the_fixture_and_labels_it():
    playwright, srv = serve_page()
    errors: list[str] = []
    try:
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch()
            page = open_fixture(browser, srv, errors)
            banner = page.text_content("#fixture-banner")
            chip = page.text_content("#chip")
            first_card_figure = page.text_content("[data-q]")
            browser.close()
    finally:
        srv.shutdown()
    assert not errors, errors
    assert banner and "FIXTURE" in banner
    assert chip
    assert first_card_figure and first_card_figure != "missing"


def test_scrolling_off_the_first_screen_leaves_the_card_in_the_model_colours(tmp_path):
    # The first screen's reveal crosses to the photograph as the reader scrolls. A card that has taken
    # the scene must not be left in the photograph's colours by the scroll events still arriving while
    # its area loads. The fixture under real area names, so each card has an area of its own to load.
    playwright = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    if not (DIST / "index.html").exists():
        pytest.skip("web/dist is not built; run npm run build in web/")
    for area in ("sunset", "ggpark", "5135_54430"):
        shutil.copytree(ROOT / "web" / "fixtures" / "FIXTURE", tmp_path / area)
    srv = serve(str(DIST), 0, mounts={"/data": str(tmp_path)})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    held = []
    try:
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.route("**/data/ggpark/manifest.json", lambda route: held.append(route))
            page.goto(f"http://127.0.0.1:{srv.server_address[1]}/index.html")
            page.wait_for_function(
                "() => window.__surrey && window.__surrey.firstFrameMs !== null", timeout=60000
            )
            page.mouse.move(640, 400)
            while page.evaluate(
                "document.querySelector('.card[data-card=\"1\"]').getBoundingClientRect().top > 150"
            ):
                page.mouse.wheel(0, 60)
                page.wait_for_timeout(30)
            page.wait_for_timeout(300)
            assert held, "the card never asked for its area"
            held[0].continue_()
            page.wait_for_function("() => document.body.dataset.onHero === ''", timeout=30000)
            page.wait_for_timeout(300)
            wipe = page.evaluate("window.__surrey.wipe")
            browser.close()
    finally:
        srv.shutdown()
    assert wipe == 0
