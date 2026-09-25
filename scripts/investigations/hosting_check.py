"""Can a static page read byte ranges from each candidate host? Tested, not read off docs.

A browser, not curl: CORS and redirects are the browser's rules, and a Range header on
a cross-origin fetch that is redirected to a CDN is exactly where a host passes with
curl and fails in a page. So a headless Chromium page on a local origin fetches the
first and a middle kilobyte of a real public file on each host with a Range header,
and records what the page could see.

The files are existing public files on each host, because putting a file of our own on
any of them needs an account, and this project creates none:

    github_pages   a three.js example model on threejs.org, served by GitHub Pages
    huggingface    a DALES-2 tile in the dataset repo mbendjilali/DALES-2, which
                   redirects to the Hugging Face CDN
    cloudflare_r2  a Protomaps sample on r2-public.protomaps.com, the public R2 URL
                   the Protomaps documentation gives
    local          surrey.static_server on another local port, the clone-and-run path

No WebGL runs here, so it may run beside training.

Run (the Python that has Playwright):
    python scripts/investigations/hosting_check.py
"""

from __future__ import annotations

import datetime as dt
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.static_server import serve  # noqa: E402

OUT = ROOT / "data" / "artifacts" / "hosting_check.json"
FILES = {
    "github_pages": "https://threejs.org/examples/models/gltf/LittlestTokyo.glb",
    "huggingface": "https://huggingface.co/datasets/mbendjilali/DALES-2/resolve/main/train/5080_54435.laz",
    "cloudflare_r2": "https://r2-public.protomaps.com/protomaps-sample-datasets/cb_2018_us_zcta510_500k.pmtiles",
}
PAGE_PORT, DATA_PORT = 8771, 8772

PROBE = """async ([url, start, end]) => {
  const t0 = performance.now();
  try {
    const r = await fetch(url, {headers: {Range: `bytes=${start}-${end}`}, cache: "no-store"});
    const buf = await r.arrayBuffer();
    return {status: r.status, bytes: buf.byteLength, redirected: r.redirected,
            final_host: new URL(r.url).host, content_range_visible: r.headers.get("content-range"),
            ms: Math.round(performance.now() - t0)};
  } catch (e) {
    return {error: String(e), ms: Math.round(performance.now() - t0)};
  }
}"""


def main() -> int:
    from playwright.sync_api import sync_playwright

    tmp = ROOT / "data" / "cache" / "hosting_check"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "index.html").write_text(
        "<!doctype html><title>hosting check</title>", encoding="utf-8", newline="\n"
    )
    (tmp / "probe.bin").write_bytes(bytes(range(256)) * 4096)
    page_srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", PAGE_PORT),
        functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp)),
    )
    data_srv = serve(str(tmp), DATA_PORT)
    for s in (page_srv, data_srv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    files = dict(FILES, local=f"http://127.0.0.1:{DATA_PORT}/probe.bin")

    rows = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{PAGE_PORT}/index.html")
        for host, url in files.items():
            first = page.evaluate(PROBE, [url, 0, 1023])
            middle = page.evaluate(PROBE, [url, 500000, 501023])
            ok = all(r.get("status") == 206 and r.get("bytes") == 1024 for r in (first, middle))
            rows[host] = {
                "url": url,
                "first_kib": first,
                "middle_kib": middle,
                "range_and_cors_work_in_a_page": ok,
            }
            print(f"{host:14s} {'ok' if ok else 'FAILS'}  {first}  {middle}", flush=True)
        browser.close()
    page_srv.shutdown()
    data_srv.shutdown()
    result = {
        "checked_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "page_origin": f"http://127.0.0.1:{PAGE_PORT}",
        "method": "headless Chromium fetch with a Range header",
        "hosts": rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
