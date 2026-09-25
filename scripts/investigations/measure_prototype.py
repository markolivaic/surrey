"""Measure the technical prototype against the thresholds written before measuring.

The thresholds are design targets for this laptop, not README claims. Every figure this
writes is tied to the device it was measured on and says so.

    a) first 3D frame, local static server                   at most 3 s
    b) first 3D frame, network held to 20 Mbit/s              at most 8 s
    c) camera orbit, median frame rate                        at least 30 fps
    d) pinned view while the cards scroll, median             at least 30 fps
    e) card change to another area, to its first frame        at most 2 s
       (preloading the next area is allowed)

Added 2026-09-15, before any of them was measured, because "first frame" is the coarse root
and a visitor sees a blurred scene until the detail arrives (found in review). All at
20 Mbit/s:

    f) the card's starting camera, time until 90 percent of the nodes the point budget
       shows once loading has settled are loaded                  at most 5 s
    g) card change to another area when only its coarse level (hierarchy and root node) was
       preloaded: first frame                                     at most 2 s
       and 90 percent of the settled nodes                        at most 5 s
    h) bytes the preloading of an area actually downloads, per area   reported, no threshold

"Settled" is three seconds with nothing loading and nothing more wanted by the budget for
that camera; the nodes drawn then are the set the 90 percent is taken of, and each node's
load time is recorded as it arrives. If f or g fails, the levers go in this order, written
before measuring: first a smaller point budget for the starting camera, then loading order by
visibility, and only then fewer bits for confidence. The thresholds were taken as
proposed in review; none was changed.

The first lever, written 2026-09-15 after f and g failed at the page's budget of 3,000,000 points and
before any smaller budget was measured: f and g are measured again at 1,500,000, 1,000,000 and 750,000
points, three runs each, in one pause. The budget taken is the largest at which f and both parts of g
pass on the median. If none passes, the budget stays at 3,000,000 and the second lever, loading order
by visibility, is next. A run that returns no figure is kept as None and printed with the page's
console, and the median is over the runs that returned one, with their count stated.

Result of the first lever, 2026-09-15: no smaller budget passed (g's 90 percent grew as the budget
shrank), so the budget stayed at 3,000,000. The second lever, written the same evening before it was
measured: the page plans the node set from the hierarchy for the camera and fetches only that set,
largest on screen first (order=visibility in web/src/potree2/octree.js). f and g are measured with it at
3,000,000, 1,500,000, 1,000,000 and 750,000 points, three runs each, in one pause; the largest budget at
which f and both parts of g pass is taken. If none passes, the third lever, fewer bits for confidence, is
next. From this run on Chromium starts with --disable-domain-blocking-for-3d-apis: in the first pause one
run returned nothing because Chromium blocked WebGL for the page after repeated context losses from the
closed contexts before it, which measures the harness and not the page.

What the diagnosis found, 2026-09-16, and what follows from it. At the 90 percent mark the page had
pulled 26,668,192 bytes for the park and 14,841,931 for Embarcadero, at 16.59 and 16.2 Mbit/s of the
20 Mbit/s the link was held to. The link is close to full, so f and g are the number of bytes the
budget asks for, and no loading order can shorten them: 3,000,000 points at 8.6 bytes each is 25 MB,
which is 10 s at this rate whatever order they arrive in. Two things came out of the same run and are
fixes rather than levers: every node's byte count read zero, because the decoder is handed the buffer
and the transfer empties it, and the area a reader leaves kept fetching, so its nodes shared the link
with the area now on screen (octree.js stopLoading, tested in web/tests/octree_order.test.mjs).

Written before the last measurement, which runs once the chain has freed the card: f and g with those
fixes and the four variant data at 3,000,000, 1,500,000, 1,000,000 and 750,000 points, and the largest
budget where f and both parts of g pass becomes the page's budget; then a to e are measured again at
that budget, since a smaller budget changes what is drawn. If none passes, the third lever, fewer bits
for confidence, is measured next, and its ceiling is already known: the four variants are 4.1 of the
8.6 bytes a point costs, so even dropping them entirely would leave f above the threshold at
3,000,000 points, which is itself the answer to whether the threshold can be met at that budget.

Fallback rule, also written before measuring: if d) or e) fails after reasonable tuning
(point budget, preloading, area size), design A is built instead of B. If a), b) or c)
fails, neither is safe: reduce the point budget or the area first, and report.

Method. Chromium through Playwright at 1440 by 900, device scale 1, each case in a fresh
browser context so nothing is cached. The page is web/dist-prototype served by
surrey.static_server on one port and the octrees on another, both answering byte ranges.
Throttling is Chrome's own network emulation at 20 Mbit/s down and up with no added
latency, since the threshold names a rate and no latency. Frame rate is the median of
1000 over requestAnimationFrame intervals. Each timed case runs three times and the
median is the figure. Before anything runs, the WebGL renderer string is read: a
software renderer (SwiftShader) invalidates every figure, and the run switches to a
headed browser.

It refuses to run while the GPU is busy, because a training run beside it would measure
the training.

Run (the Python that has Playwright), after building the prototype on its own:
    cd web && npx vite build --config vite.prototype.config.js && cd ..
    python scripts/investigations/measure_prototype.py
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.static_server import serve  # noqa: E402
from surrey.tools import gpu_busy  # noqa: E402

OUT = ROOT / "data" / "artifacts" / "prototype_measurements.json"
SHOTS = ROOT / "data" / "design_archive" / "prototype"
PAGE_PORT, DATA_PORT = 8770, 8772
VIEWPORT = {"width": 1440, "height": 900}
THRESHOLDS = {
    "a_first_frame_local_s": ("max", 3.0),
    "b_first_frame_20mbit_s": ("max", 8.0),
    "c_orbit_median_fps": ("min", 30.0),
    "d_scroll_median_fps": ("min", 30.0),
    "e_area_change_first_frame_s": ("max", 2.0),
}
DETAIL_THRESHOLDS = {
    "f_start_camera_90pct_nodes_20mbit_s": ("max", 5.0),
    "g_area_change_coarse_preload_first_frame_20mbit_s": ("max", 2.0),
    "g_area_change_coarse_preload_90pct_nodes_20mbit_s": ("max", 5.0),
}
SETTLE_MS = 3000
REPEATS = 3
LEVER_BUDGETS = (1_500_000, 1_000_000, 750_000)
VISIBILITY_BUDGETS = (3_000_000, 1_500_000, 1_000_000, 750_000)
CHROMIUM_ARGS = [
    "--enable-gpu",
    "--ignore-gpu-blocklist",
    "--use-angle=d3d11",
    "--disable-domain-blocking-for-3d-apis",
]


def url(**params) -> str:
    q = "&".join(f"{k}={v}" for k, v in params.items())
    return f"http://127.0.0.1:{PAGE_PORT}/prototype.html?data=http://127.0.0.1:{DATA_PORT}&{q}"


def median_fps(stamps: list[float]) -> float:
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False) if b > a]
    return round(1000.0 / statistics.median(gaps), 1) if gaps else 0.0


def wait_first_frame(page, timeout_s: float = 60.0) -> float | None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        v = page.evaluate("window.__surrey && window.__surrey.firstFrameMs")
        if v:
            return v / 1000.0
        page.wait_for_timeout(50)
    return None


def settle(page, area: str, timeout_s: float = 180.0) -> dict | None:
    """Wait until the area has had nothing loading or wanted for SETTLE_MS, and return its record."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rec = page.evaluate(
            f"(() => {{ const r = window.__surrey.areas['{area}']; if (!r) return null;"
            f" return {{idleSince: r.idleSince, now: performance.now(), visibleNames: r.visibleNames,"
            f" loadTimes: r.loadTimes, bytes: r.bytes, preloadBytes: r.preloadBytes,"
            f" nodeBytes: r.nodeBytes, bytesAtLoad: r.bytesAtLoad}}; }})()"
        )
        if rec and rec["idleSince"] is not None and rec["now"] - rec["idleSince"] >= SETTLE_MS:
            return rec
        page.wait_for_timeout(100)
    return None


def time_to_share(rec: dict, since_ms: float, share: float = 0.9) -> float | None:
    names = rec["visibleNames"]
    times = sorted(rec["loadTimes"][n] for n in names if n in rec["loadTimes"])
    if not names or len(times) < len(names):
        return None
    k = max(1, int(-(-share * len(names) // 1)))
    return round((times[k - 1] - since_ms) / 1000.0, 3)


def diagnose(rec: dict, since_ms: float, share: float = 0.9) -> dict:
    """Why a load time is what it is: how many nodes, how many bytes, and the rate those bytes arrived at.

    If the rate sits at the throttled link's own rate, the time is the link and no loading order can
    shorten it; only fewer bytes can.
    """
    names = [n for n in rec["visibleNames"] if n in rec["loadTimes"]]
    if not names:
        return {"nodes": 0}
    order = sorted(names, key=lambda n: rec["loadTimes"][n])
    k = max(1, int(-(-share * len(names) // 1)))
    last = order[k - 1]
    seconds = (rec["loadTimes"][last] - since_ms) / 1000.0
    bytes_at = rec["bytesAtLoad"].get(last, 0)
    settled_bytes = sum(rec["nodeBytes"].get(n, 0) for n in names)
    return {
        "nodes": len(names),
        "nodes_missing_a_load_time": len(rec["visibleNames"]) - len(names),
        "settled_set_bytes": settled_bytes,
        "share": share,
        "node_at_share": last,
        "seconds_to_share": round(seconds, 3),
        "octree_bytes_by_then": bytes_at,
        "megabit_per_second_by_then": round(bytes_at * 8 / 1e6 / seconds, 2)
        if seconds > 0
        else None,
        "preload_bytes": rec.get("preloadBytes"),
        "slowest_five": [
            {
                "node": n,
                "second": round((rec["loadTimes"][n] - since_ms) / 1000.0, 3),
                "bytes": rec["nodeBytes"].get(n),
            }
            for n in order[-5:]
        ],
    }


def detail_runs(
    browser, fresh, repeats: int, budget: int | None = None, order: str | None = None, relaunch=None
) -> dict:
    """Repeats of f, g and h at one budget.

    `relaunch`, when given, starts a fresh browser for every repeat and closes the last one. The third
    repeat of a budget kept coming back empty with "could not create a WebGL context" on this laptop's
    integrated GPU: contexts closed inside one browser did not give their GPU memory back fast enough,
    so the harness, not the page, was what failed. A browser per repeat costs a few seconds and removes
    that failure mode.
    """
    runs = {k: [] for k in DETAIL_THRESHOLDS} | {"h_preload_bytes": [], "page_order": []}
    params = (
        {"edl": 0} | ({"budget": budget} if budget else {}) | ({"order": order} if order else {})
    )
    for _ in range(repeats):
        if relaunch is not None:
            browser = relaunch()
        c, pg_ = fresh(throttle=True, browser=browser)
        pg_.goto(url(**params))
        rec = settle(pg_, "ggpark")
        runs["page_order"].append(
            pg_.evaluate("window.__surrey && [window.__surrey.order, window.__surrey.budget]")
        )
        if rec is None:
            print(
                "no settled record for ggpark; page console:",
                CONSOLE.get(id(pg_), [])[-15:],
                flush=True,
            )
            print(
                "page state:",
                pg_.evaluate(
                    "JSON.stringify(window.__surrey && {first: window.__surrey.firstFrameMs,"
                    " areas: Object.keys(window.__surrey.areas || {})})"
                ),
                flush=True,
            )
        runs["f_start_camera_90pct_nodes_20mbit_s"].append(time_to_share(rec, 0.0) if rec else None)
        # Card 3 is the same park and preloads Embarcadero's coarse level for card 4.
        pg_.evaluate("document.querySelectorAll('.card')[2].scrollIntoView({block: 'center'})")
        t0 = time.time()
        pre = None
        while time.time() - t0 < 60:
            pre = pg_.evaluate(
                "window.__surrey.areas['embarcadero'] && window.__surrey.areas['embarcadero'].preloadBytes"
            )
            if pre:
                break
            pg_.wait_for_timeout(50)
        runs["h_preload_bytes"].append({"embarcadero": pre})
        pg_.wait_for_timeout(1000)
        pg_.evaluate("document.querySelectorAll('.card')[3].scrollIntoView({block: 'center'})")
        t0 = time.time()
        sw = None
        while time.time() - t0 < 60:
            got = pg_.evaluate("window.__surrey.switches.filter(s => s && s.to === 'embarcadero')")
            if got and got[-1]["firstFrameMs"] is not None:
                sw = got[-1]
                break
            pg_.wait_for_timeout(25)
        runs["g_area_change_coarse_preload_first_frame_20mbit_s"].append(
            round(sw["firstFrameMs"] / 1000, 3) if sw else None
        )
        rec2 = settle(pg_, "embarcadero")
        runs["g_area_change_coarse_preload_90pct_nodes_20mbit_s"].append(
            time_to_share(rec2, sw["t0"]) if (rec2 and sw) else None
        )
        c.close()
        if relaunch is not None:
            browser.close()
        print({"budget": budget} | {k: v[-1] for k, v in runs.items()}, flush=True)
    return runs


def summarise_detail(detail: dict) -> dict:
    runs = {}
    for key, (kind, limit) in DETAIL_THRESHOLDS.items():
        vals = [v for v in detail[key] if v is not None]
        med = round(statistics.median(vals), 3) if vals else None
        passed = None if med is None else (med <= limit if kind == "max" else med >= limit)
        runs[key] = {
            "per_run": detail[key],
            "median": med,
            "runs_with_a_figure": len(vals),
            "threshold": limit,
            "kind": kind,
            "passes": passed,
        }
    runs["h_preload_bytes"] = {"per_run": detail["h_preload_bytes"]}
    runs["page_order_and_budget"] = detail.get("page_order")
    return runs


def budget_choice(
    by_budget: dict[int, dict],
    if_none: str = "no smaller budget passes; the second lever, loading order by visibility",
) -> dict:
    """The largest measured budget at which f and both parts of g pass; none means the next lever."""
    passing = [
        b
        for b, runs in by_budget.items()
        if all(runs[k]["passes"] is True for k in DETAIL_THRESHOLDS)
    ]
    if passing:
        return {
            "budget": max(passing),
            "next": "set the page's point budget to this and re-measure a to e",
        }
    return {"budget": None, "next": if_none}


CONSOLE: dict[int, list[str]] = {}


def main() -> int:
    from playwright.sync_api import sync_playwright

    busy, gpu = gpu_busy()
    if busy:
        raise SystemExit(f"GPU busy ({gpu}); measure only while nothing trains")
    page_srv, data_srv = (
        serve(str(ROOT / "web" / "dist-prototype"), PAGE_PORT),
        serve(str(ROOT / "data" / "viewer"), DATA_PORT),
    )
    for s in (page_srv, data_srv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    SHOTS.mkdir(parents=True, exist_ok=True)
    result = {
        "measured_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "device_bound": True,
        "device": {"gpu": gpu, "cpu": platform.processor(), "os": platform.platform()},
        "viewport": VIEWPORT,
        "thresholds": {k: {"kind": a, "value": b} for k, (a, b) in THRESHOLDS.items()},
        "runs": {},
    }

    with sync_playwright() as p:
        headless = True
        browser = p.chromium.launch(headless=headless, args=CHROMIUM_ARGS)
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        pg = ctx.new_page()
        pg.goto(url(edl=0.5))
        wait_first_frame(pg)
        renderer = pg.evaluate("window.__surrey.renderer")
        ctx.close()
        if "swiftshader" in renderer.lower() or "software" in renderer.lower():
            browser.close()
            headless = False
            browser = p.chromium.launch(headless=False, args=["--ignore-gpu-blocklist"])
            ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
            pg = ctx.new_page()
            pg.goto(url(edl=0.5))
            wait_first_frame(pg)
            renderer = pg.evaluate("window.__surrey.renderer")
            ctx.close()
        result["webgl_renderer"] = renderer
        result["headless"] = headless
        result["browser"] = f"chromium {browser.version}"
        outer_browser = [browser]
        print(f"renderer {renderer}, headless {headless}", flush=True)

        def launch():
            return p.chromium.launch(headless=headless, args=CHROMIUM_ARGS)

        def fresh(throttle: bool = False, browser=None):
            browser = browser or outer_browser[0]
            c = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
            pg_ = c.new_page()
            CONSOLE[id(pg_)] = []
            pg_.on(
                "console", lambda m, key=id(pg_): CONSOLE[key].append(f"{m.type}: {m.text}"[:300])
            )
            pg_.on("pageerror", lambda e, key=id(pg_): CONSOLE[key].append(f"pageerror: {e}"[:300]))
            if throttle:
                cdp = c.new_cdp_session(pg_)
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
            return c, pg_

        runs = {k: [] for k in THRESHOLDS}
        if "--diagnose-g" in sys.argv:
            c, pg_ = fresh(throttle=True)
            pg_.goto(url(edl=0, order="visibility"))
            first = settle(pg_, "ggpark")
            diag = {"start_camera_ggpark": diagnose(first, 0.0) if first else None}
            pg_.evaluate("document.querySelectorAll('.card')[2].scrollIntoView({block: 'center'})")
            pg_.wait_for_timeout(4000)
            pg_.evaluate("document.querySelectorAll('.card')[3].scrollIntoView({block: 'center'})")
            sw = None
            t0 = time.time()
            while time.time() - t0 < 60:
                got = pg_.evaluate(
                    "window.__surrey.switches.filter(s => s && s.to === 'embarcadero')"
                )
                if got and got[-1]["firstFrameMs"] is not None:
                    sw = got[-1]
                    break
                pg_.wait_for_timeout(25)
            second = settle(pg_, "embarcadero")
            diag["area_change_embarcadero"] = (
                diagnose(second, sw["t0"]) if (second and sw) else None
            )
            c.close()
            browser.close()
            page_srv.shutdown()
            data_srv.shutdown()
            out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else result
            out["g_diagnosis"] = {
                "measured_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "webgl_renderer": renderer,
                "device": result["device"],
                "order": "visibility",
                "budget": 3_000_000,
                "throttle_megabit_per_second": 20,
                "one_run_only": True,
                "runs": diag,
            }
            OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
            print(json.dumps(out["g_diagnosis"], indent=1))
            return 0
        if "--visibility-lever" in sys.argv:
            by_budget = {
                b: summarise_detail(
                    detail_runs(browser, fresh, REPEATS, b, "visibility", relaunch=launch)
                )
                for b in VISIBILITY_BUDGETS
            }
            browser.close()
            page_srv.shutdown()
            data_srv.shutdown()
            out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else result
            out["visibility_lever"] = {
                "measured_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "webgl_renderer": renderer,
                "device": result["device"],
                "order": "visibility",
                "chromium_flags_added": ["--disable-domain-blocking-for-3d-apis"],
                "renderer_note": "device.gpu is the CUDA card checked idle; WebGL ran on webgl_renderer",
                "budgets": {str(b): r for b, r in by_budget.items()},
                "choice": budget_choice(
                    by_budget,
                    "no budget passes with visibility order; the third lever, "
                    "fewer bits for confidence",
                ),
            }
            OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
            print(json.dumps(out["visibility_lever"], indent=1))
            return 0
        if "--only-detail" in sys.argv or "--budget-lever" in sys.argv:
            lever = "--budget-lever" in sys.argv
            if lever:
                by_budget = {
                    b: summarise_detail(detail_runs(browser, fresh, REPEATS, b))
                    for b in LEVER_BUDGETS
                }
            else:
                detail = summarise_detail(detail_runs(browser, fresh, REPEATS))
            browser.close()
            page_srv.shutdown()
            data_srv.shutdown()
            out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else result
            stamp = {
                "measured_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "webgl_renderer": renderer,
                "device": result["device"],
                "renderer_note": "device.gpu is the CUDA card checked idle; WebGL ran on webgl_renderer",
            }
            if lever:
                out["budget_lever"] = stamp | {
                    "budgets": {str(b): r for b, r in by_budget.items()},
                    "choice": budget_choice(by_budget),
                }
                print(json.dumps(out["budget_lever"], indent=1))
            else:
                out["detail"] = stamp | {"runs": detail}
                print(json.dumps(out["detail"]["runs"], indent=1))
            OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
            return 0
        for _ in range(REPEATS):
            c, pg_ = fresh()
            pg_.goto(url(edl=0.5))
            runs["a_first_frame_local_s"].append(wait_first_frame(pg_))
            pg_.wait_for_timeout(5000)
            pg_.evaluate("window.__surrey.frames.length = 0; window.__surreyOrbit(true)")
            pg_.wait_for_timeout(10000)
            runs["c_orbit_median_fps"].append(median_fps(pg_.evaluate("window.__surrey.frames")))
            pg_.evaluate("window.__surreyOrbit(false)")
            # Scroll through the first cards with the wheel, as a reader would.
            pg_.evaluate("window.__surrey.frames.length = 0")
            t_end = time.time() + 10
            while time.time() < t_end:
                pg_.mouse.wheel(0, 40)
                pg_.wait_for_timeout(16)
            runs["d_scroll_median_fps"].append(median_fps(pg_.evaluate("window.__surrey.frames")))
            c.close()

            c, pg_ = fresh(throttle=True)
            pg_.goto(url(edl=0.5))
            runs["b_first_frame_20mbit_s"].append(wait_first_frame(pg_, 120))
            c.close()

            # Area change: settle on card 3 (the park, which preloads Embarcadero for card 4),
            # then go to card 4 and time its first frame.
            c, pg_ = fresh()
            pg_.goto(url(edl=0.5))
            wait_first_frame(pg_)
            pg_.evaluate("document.querySelectorAll('.card')[2].scrollIntoView({block: 'center'})")
            pg_.wait_for_timeout(4000)
            pg_.evaluate("document.querySelectorAll('.card')[3].scrollIntoView({block: 'center'})")
            t0 = time.time()
            got = None
            while time.time() - t0 < 30:
                sw = pg_.evaluate(
                    "window.__surrey.switches.filter(s => s && s.to === 'embarcadero')"
                )
                if sw and sw[-1]["firstFrameMs"] is not None:
                    got = sw[-1]["firstFrameMs"] / 1000.0
                    break
                pg_.wait_for_timeout(25)
            runs["e_area_change_first_frame_s"].append(got)
            c.close()
            print({k: v[-1] for k, v in runs.items()}, flush=True)

        # EDL at three strengths, free view with the divider in the middle.
        for strength in (0.0, 0.5, 1.0):
            c, pg_ = fresh()
            pg_.goto(url(edl=strength))
            wait_first_frame(pg_)
            pg_.evaluate("document.querySelectorAll('.card')[5].scrollIntoView({block: 'center'})")
            pg_.wait_for_timeout(6000)
            pg_.screenshot(path=str(SHOTS / f"edl_{str(strength).replace('.', '_')}.png"))
            c.close()

        # Ten seconds of video through the cards.
        c = browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=1,
            record_video_dir=str(ROOT / "data" / "cache" / "prototype"),
            record_video_size=VIEWPORT,
        )
        pg_ = c.new_page()
        pg_.goto(url(edl=0.5))
        wait_first_frame(pg_)
        t_end = time.time() + 10
        while time.time() < t_end:
            pg_.mouse.wheel(0, 60)
            pg_.wait_for_timeout(16)
        video = pg_.video.path()
        c.close()
        browser.close()
        result["video"] = str(Path(video).relative_to(ROOT)).replace("\\", "/")

    page_srv.shutdown()
    data_srv.shutdown()
    for key, (kind, limit) in THRESHOLDS.items():
        vals = [v for v in runs[key] if v is not None]
        med = round(statistics.median(vals), 3) if vals else None
        passed = None if med is None else (med <= limit if kind == "max" else med >= limit)
        result["runs"][key] = {
            "per_run": runs[key],
            "median": med,
            "threshold": limit,
            "kind": kind,
            "passes": passed,
        }
    r = result["runs"]
    if (
        r["d_scroll_median_fps"]["passes"] is False
        or r["e_area_change_first_frame_s"]["passes"] is False
    ):
        result["fallback_reading"] = "d or e fails: tune, and if it still fails build design A"
    elif any(
        r[k]["passes"] is False
        for k in ("a_first_frame_local_s", "b_first_frame_20mbit_s", "c_orbit_median_fps")
    ):
        result["fallback_reading"] = (
            "a, b or c fails: neither design is safe; reduce budget or area and report"
        )
    else:
        result["fallback_reading"] = "all thresholds pass: design B stands"
    # The lever runs and the diagnosis live in this same file and were measured separately. A full run
    # used to drop them, which is how the first choice of budget nearly lost its evidence; anything this
    # run did not measure is carried over.
    if OUT.exists():
        kept = {
            k: v for k, v in json.loads(OUT.read_text(encoding="utf-8")).items() if k not in result
        }
        result = kept | result
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["runs"], indent=1))
    print(result["fallback_reading"])
    return 0


def run_releasing_hold() -> int:
    """In the pause mode the chain is waiting on data/logs/HOLD; whatever happens here, it is released."""
    hold = ROOT / "data" / "logs" / "HOLD"
    try:
        return main()
    finally:
        if (
            any(
                a in sys.argv
                for a in ("--only-detail", "--budget-lever", "--visibility-lever", "--diagnose-g")
            )
            and hold.exists()
        ):
            hold.unlink()
            print("released data/logs/HOLD", flush=True)


if __name__ == "__main__":
    raise SystemExit(run_releasing_hold())
