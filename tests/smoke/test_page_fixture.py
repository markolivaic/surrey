"""Smoke test: the built page loads the FIXTURE octree, draws it, and shows the FIXTURE banner.

Runs in CI on Linux with a software WebGL renderer, which is fine for "does it draw" and useless
for speed, so nothing here measures time. Skips, with the reason in the output, when Playwright or
the built page is missing.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.static_server import serve  # noqa: E402

DIST = ROOT / "web" / "dist"


def test_the_page_draws_the_fixture_and_labels_it():
    playwright = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    if not (DIST / "index.html").exists():
        pytest.skip("web/dist is not built; run npm run build in web/")
    srv = serve(str(DIST), 0, mounts={"/fixtures": str(ROOT / "web" / "fixtures")})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    errors: list[str] = []
    try:
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(f"http://127.0.0.1:{srv.server_address[1]}/index.html?fixture=1")
            page.wait_for_function(
                "() => window.__surrey && window.__surrey.firstFrameMs !== null", timeout=60000
            )
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
