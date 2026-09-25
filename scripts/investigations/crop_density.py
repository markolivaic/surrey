"""What density the model actually sees, per source, once a crop is drawn.

A crop is 32,768 points over a 40 m block whatever the source holds, so a block with more points than
that is thinned by the crop itself and a block with fewer is padded with repeats. That makes the source
density a misleading number: two sources at 38 and 42 points per square metre both arrive at the model
as the same 20.5, while a source at 17 arrives sparser. This measures the distinct points a crop
carries, per source, with no model involved.

Run:
    python scripts/investigations/crop_density.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import BLOCK_M, POINTS_PER_CROP  # noqa: E402
from surrey.density import crop_at_density  # noqa: E402

ARTIFACTS = ROOT / "data" / "artifacts"
OUT = ARTIFACTS / "crop_density.json"
SOURCES = {
    "surrey_dales_training": (ROOT / "data" / "prep" / "dales", "train"),
    "swiss_training": (ROOT / "data" / "prep" / "swiss", "train"),
    "swiss_held_out": (ROOT / "data" / "prep" / "swiss_holdout", "train"),
    "san_francisco": (ROOT / "data" / "prep" / "target", "sf"),
}
MIN_BLOCK_POINTS = 2000
BLOCKS_PER_TILE = 12
TILES = 6


def main() -> int:
    rng = np.random.default_rng(0)
    result = {
        "block_m": BLOCK_M,
        "points_per_crop": POINTS_PER_CROP,
        "crop_ceiling_points_per_m2": round(POINTS_PER_CROP / (BLOCK_M * BLOCK_M), 2),
        "note": (
            "the crop is the same length everywhere, so a denser source is thinned by the crop "
            "and a sparser one is padded with repeats; what the model sees is the distinct "
            "count below"
        ),
        "sources": {},
    }
    for name, (root, split) in SOURCES.items():
        if not root.exists():
            continue
        tiles = sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists())[
            :TILES
        ]
        distinct, source_density = [], []
        for path in tiles:
            tile = CachedTile(path, split)
            ex, ey = tile.extent_m()
            source_density.append(tile.n / max(ex * ey, 1.0))
            for cx, cy in list(tile.block_centres(BLOCK_M, BLOCKS_PER_TILE)):
                idx = tile.gather_block(cx, cy, BLOCK_M)
                if idx.size < MIN_BLOCK_POINTS:
                    continue
                _, what = crop_at_density(idx, POINTS_PER_CROP, rng)
                distinct.append(what["distinct_in_crop"])
        if not distinct:
            continue
        median = float(np.median(distinct))
        result["sources"][name] = {
            "tiles_read": [p.name for p in tiles],
            "blocks": len(distinct),
            "source_points_per_m2_median": round(float(np.median(source_density)), 2),
            "distinct_points_in_a_crop_median": int(median),
            "crop_points_per_m2_median": round(median / (BLOCK_M * BLOCK_M), 2),
        }
        print(
            f"  {name:24s} source {result['sources'][name]['source_points_per_m2_median']:>6} "
            f"crop {result['sources'][name]['crop_points_per_m2_median']:>6} "
            f"({result['sources'][name]['distinct_points_in_a_crop_median']:,} distinct)",
            flush=True,
        )
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
