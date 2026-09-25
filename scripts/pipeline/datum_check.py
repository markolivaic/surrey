"""What datum the San Francisco lidar is really in, read from its own source files.

The EPT is published as EPSG:3857, WGS 84. The metadata of the delivery's original LAZ
files survives in the EPT's ept-sources, and each file's header states its
coordinate system. And each source record carries the file's bounds in both systems:
its native header extent and its extent in Web Mercator after Entwine's conversion. So
the conversion can be identified, not assumed: reproject the native corners by a
zero-shift path (NAD83(2011) taken as WGS 84, PROJ's default) and by a real datum path
(NAD83(2011) to ITRF2014 at the flight epoch), and see which lands on the published
bounds.

If the zero shift lands, the lidar holds NAD83(2011) numbers labelled WGS 84, and a
dataset truly in WGS 84 will sit about a metre and a half away from it.

Run:
    python scripts/pipeline/datum_check.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "artifacts" / "datum_check.json"
BUCKET = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"
DELIVERY = "CA_SanFrancisco_1_B23"
EPOCH = 2023.3


def list_sources(limit: int) -> list[str]:
    r = requests.get(
        BUCKET, params={"prefix": f"{DELIVERY}/ept-sources/", "max-keys": 1000}, timeout=60
    )
    keys = re.findall(r"<Key>([^<]+\.json)</Key>", r.text)
    step = max(1, len(keys) // limit)
    return keys[::step][:limit]


def main() -> int:
    from pyproj import Transformer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", type=int, default=12)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()

    zero = Transformer.from_crs("EPSG:7131", "EPSG:3857", always_xy=True)
    to_geo = Transformer.from_crs("EPSG:7131", "EPSG:6318", always_xy=True)
    itrf = Transformer.from_crs("EPSG:6319", "EPSG:7912", always_xy=True)
    merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def bbox(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def itrf_path(x, y):
        lon, lat = to_geo.transform(x, y)
        lo, la, _, _ = itrf.transform(lon, lat, 0.0, EPOCH)
        return merc.transform(lo, la)

    rows, horizontal, lats = [], set(), []
    for key in list_sources(args.files):
        src = requests.get(f"{BUCKET}/{key}", timeout=60).json()
        las = src["metadata"]["readers.las"]
        horizontal.add(las["srs"]["horizontal"].split('"')[1])
        lats.append(
            to_geo.transform((las["minx"] + las["maxx"]) / 2, (las["miny"] + las["maxy"]) / 2)[1]
        )
        corners = [(x, y) for x in (las["minx"], las["maxx"]) for y in (las["miny"], las["maxy"])]
        b = src["bounds"]
        published = (b[0], b[1], b[3], b[4])
        z = bbox([zero.transform(x, y) for x, y in corners])
        i = bbox([itrf_path(x, y) for x, y in corners])
        rows.append(
            {
                "file": key.rsplit("/", 1)[-1],
                "zero_shift_minus_published": [
                    round(a - p, 3) for a, p in zip(z, published, strict=True)
                ],
                "itrf2014_minus_published": [
                    round(a - p, 3) for a, p in zip(i, published, strict=True)
                ],
            }
        )
        time.sleep(0.2)

    worst_zero = max(abs(v) for r in rows for v in r["zero_shift_minus_published"])
    worst_itrf = min(max(abs(v) for v in r["itrf2014_minus_published"]) for r in rows)
    # Web Mercator stretches distance by 1/cos(latitude). At San Francisco a metre on the
    # ground is about 1.265 Mercator units, so the ITRF miss is quoted in both.
    import math

    units_per_metre = 1.0 / math.cos(math.radians(float(np.mean(lats))))
    result = {
        "question": "which datum path produced the published Web Mercator bounds",
        "source": f"{BUCKET}/{DELIVERY}/ept-sources, readers.las header of each original file",
        "source_horizontal_crs": sorted(horizontal),
        "zero_shift_path": zero.description,
        "itrf_path": f"EPSG:7131 to NAD83(2011) to ITRF2014 at epoch {EPOCH} to Web Mercator",
        "files_checked": len(rows),
        "zero_shift_worst_abs_difference_mercator_units": worst_zero,
        "itrf2014_smallest_worst_abs_difference_mercator_units": worst_itrf,
        "mercator_units_per_ground_metre": round(units_per_metre, 4),
        "itrf2014_smallest_worst_abs_difference_ground_m": round(worst_itrf / units_per_metre, 3),
        "reading": (
            "the published bounds are the zero-shift reprojection: the lidar holds "
            "NAD83(2011) coordinates labelled WGS 84"
            if worst_zero < 0.01 < worst_itrf
            else "neither path reproduces the bounds; not settled"
        ),
        "files": rows,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "source_horizontal_crs",
                    "files_checked",
                    "zero_shift_worst_abs_difference_mercator_units",
                    "itrf2014_smallest_worst_abs_difference_mercator_units",
                    "mercator_units_per_ground_metre",
                    "itrf2014_smallest_worst_abs_difference_ground_m",
                    "reading",
                )
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
