"""Which swissSURFACE3D tiles M1 learns water and bridge deck from, chosen before any
San Francisco scoring of M1 and without looking at San Francisco at all.

DALES-2 has no water and no bridge deck, so over the bay the model can only be
consistent, never right. swissSURFACE3D labels both (9 and 17) in its 2024 and later
classification, is anonymous over STAC, and is manually reviewed. Candidates are the
2024 or later COPC tiles over two cities where a lake meets a river crossed by bridges,
Zurich and Geneva.

Each candidate is characterised from its own coarse octree levels (0 and 1), read by
range request: a spatially even subsample, so class shares come without downloading
the tile.

Rule, written before this script was first run:
    eligible   water (9) at least 0.05 of the sampled points, and building roofs plus
               facades (6 and 26) at least 0.10, so the tile teaches water beside a city
               rather than open lake
    selected   per city, the two eligible tiles with the highest bridge deck (17) share;
               if fewer than two are eligible, the rest by water share among all tiles
    four tiles in total, a little under half of the DALES training area, which is the
    mixture M1 trains on

Run:
    python scripts/pipeline/swiss_tile_selection.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
import time
from pathlib import Path

import laspy
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "artifacts" / "swiss_tile_selection.json"
STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swisssurface3d/items"
CITIES = {
    "zurich": (8.50, 47.33, 8.58, 47.39),
    "geneva": (6.11, 46.19, 6.19, 46.23),
}
# A third city, for a tile that is never trained on. Same rule, same reading, chosen for the same
# reason as the first two: a lake meeting a river that is crossed by bridges, with a city around it.
# Lucerne, where the Reuss leaves the Vierwaldstaettersee under the Kapellbruecke.
# Order written down before any eligibility was read, so the third city is not the one that flattered
# the result: lakeside towns with a river outflow and bridges, in the order they were listed. The first
# city with an eligible tile is taken, which is also why swissSURFACE3D coverage decides it: the class
# table this project uses is the 2024 one, and Lucerne, Bern, Basel, Biel, Lugano and Solothurn have no
# 2024 or later tiles at all (read from the STAC service, 2026-09-21).
HOLDOUT_CITIES = {
    "lucerne": (8.28, 47.03, 8.34, 47.07),
    "zug": (8.49, 47.16, 8.54, 47.19),
    "rapperswil": (8.79, 47.21, 8.85, 47.24),
    "interlaken": (7.83, 46.67, 7.90, 46.70),
    "neuchatel": (6.91, 46.98, 6.97, 47.01),
    "lausanne": (6.60, 46.50, 6.66, 46.53),
}
PER_CITY = 2
MIN_WATER = 0.05
MIN_BUILDING = 0.10
LEVELS = range(0, 2)


def gps_day(t: float) -> str | None:
    if not t or t < 1e6:
        return None
    return (dt.datetime(1980, 1, 6) + dt.timedelta(seconds=float(t) + 1e9)).strftime("%Y-%m-%d")


def characterise(url: str) -> dict:
    with laspy.CopcReader.open(url) as r:
        pts = r.query(level=LEVELS)
        cls = np.asarray(pts.classification)
        n = cls.size
        share = lambda codes: round(float(np.isin(cls, codes).sum() / n), 4) if n else 0.0  # noqa: E731
        gps = np.asarray(pts.gps_time)
        return {
            "points_in_tile": int(r.header.point_count),
            "sampled": int(n),
            "water": share([9]),
            "bridge_deck": share([17]),
            "building": share([6, 26]),
            "ground": share([2]),
            "vegetation": share([3]),
            "flight_day": gps_day(float(np.median(gps))) if gps.size else None,
        }


def main() -> int:
    # laspy's range reads carry no timeout of their own; the first run hung on one read
    # for a quarter of an hour. A stalled tile now fails and is reported.
    socket.setdefaulttimeout(60)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--holdout",
        action="store_true",
        help="the same rule over a city that never enters training, taking one tile",
    )
    args = ap.parse_args()
    cities, per_city = (HOLDOUT_CITIES, 1) if args.holdout else (CITIES, PER_CITY)
    t0 = time.time()
    result = {
        "held_out": bool(args.holdout),
        "held_out_note": (
            "chosen after M1 was trained and scored, by the rule the training tiles were "
            "chosen by, in a city none of them came from; never trained on"
        )
        if args.holdout
        else None,
        "rule": {
            "min_water": MIN_WATER,
            "min_building": MIN_BUILDING,
            "per_city": per_city,
            "ranking": "bridge deck share among eligible, then water share",
        },
        "levels_read": list(LEVELS),
        "cities": {},
        "selected": [],
    }
    for city, bbox in cities.items():
        params = {"bbox": ",".join(map(str, bbox)), "limit": 100}
        if args.holdout:
            # The class table this project maps is the 2024 one, so an older flight would be labelled by
            # a different scheme and quietly mean something else.
            params["datetime"] = "2024-01-01T00:00:00Z/.."
        items = requests.get(STAC, params=params, timeout=60).json()["features"]
        rows = []
        for it in items:
            href = next(
                (a["href"] for a in it["assets"].values() if a["href"].endswith(".copc.laz")), None
            )
            if href is None:
                continue
            print(f"  {city:7s} reading {it['id']}", flush=True)
            try:
                c = characterise(href)
            except Exception as e:  # a tile that cannot be read is reported, not skipped silently
                rows.append({"id": it["id"], "url": href, "error": f"{type(e).__name__}: {e}"})
                continue
            rows.append(
                {
                    "id": it["id"],
                    "url": href,
                    **c,
                    "eligible": c["water"] >= MIN_WATER and c["building"] >= MIN_BUILDING,
                }
            )
            print(
                f"  {city:7s} {it['id']:40s} water {c['water']:.3f} bridge {c['bridge_deck']:.4f} "
                f"building {c['building']:.3f} day {c['flight_day']}",
                flush=True,
            )
            time.sleep(0.2)
        ok = [r for r in rows if "error" not in r]
        eligible = sorted(
            [r for r in ok if r["eligible"]], key=lambda r: (-r["bridge_deck"], -r["water"])
        )
        chosen = eligible[:per_city]
        if len(chosen) < per_city:
            rest = sorted([r for r in ok if r not in chosen], key=lambda r: -r["water"])
            chosen += rest[: per_city - len(chosen)]
        result["cities"][city] = {
            "candidates": rows,
            "eligible": len(eligible),
            "selected": [r["id"] for r in chosen],
        }
        if args.holdout and not eligible:
            # No tile here meets the rule the training tiles met, so this city is passed over whole.
            print(
                f"  {city}: nothing eligible, on to the next city in the written order", flush=True
            )
            continue
        result["selected"] += [
            {
                "city": city,
                **{
                    k: r[k]
                    for k in (
                        "id",
                        "url",
                        "water",
                        "bridge_deck",
                        "building",
                        "flight_day",
                        "points_in_tile",
                    )
                },
            }
            for r in chosen
        ]
        if args.holdout and result["selected"]:
            # The first city in the written order that has an eligible tile is the one taken.
            break
    result["seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        "selected:",
        [(s["city"], s["id"], s["water"], s["bridge_deck"]) for s in result["selected"]],
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
