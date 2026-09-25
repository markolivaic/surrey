"""The four LAS point flags, per class, in every source the project reads.

LAS 1.4 gives every point four flags besides its class: synthetic (the point was made, not measured),
key point (kept by a thinning for a surface model), withheld (to be left out of use) and overlap (from
the overlapping edge of a flight line). Formats 0 to 5 carry the first three in the classification
byte and have no overlap bit. A field that is present is not a field that means something, so this
reads what each source actually sets, class by class.

Sources:
- mbendjilali/DALES-2: all 40 tiles, training and held out apart, from data/dales2.
- swissSURFACE3D: the four training tiles and the held-out Zug tile, from data/cache/swiss.
- USGS 3DEP San Francisco: read from the delivery's own EPT, because the areas this project fetched
  were written without the flags. A seeded sample of the nodes overlapping each area, and only the
  points inside the area.

Writes data/artifacts/las_flags_audit.json, with the decision for each flag in each source from
surrey.flags beside the counts.

Run:
    python scripts/investigations/las_flags_audit.py
    python scripts/investigations/las_flags_audit.py --decisions-only    # rewrite the decisions, keep the counts
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from surrey.data import CLASS_NAMES  # noqa: E402
from surrey.ept import EptSource, _get, overlaps_xy  # noqa: E402
from surrey.flags import DECISIONS, SOURCES  # noqa: E402

OUT = ROOT / "data" / "artifacts" / "las_flags_audit.json"
FLAGS = ("synthetic", "key_point", "withheld", "overlap")
SWISS_TRAINING = ("2683-1246", "2683-1243", "2500-1118", "2500-1119")
SWISS_HELD_OUT = ("2680-1225",)
SWISS_NAMES = {
    1: "unclassified",
    2: "ground",
    3: "vegetation",
    6: "building roofs",
    9: "water",
    14: "power line",
    15: "masts",
    17: "bridge deck",
    26: "building facades",
    27: "bridge piers",
}
ASPRS_NAMES = {
    1: "unclassified",
    2: "ground",
    7: "low noise",
    9: "water",
    17: "bridge deck",
    18: "high noise",
}
SF_NODES_PER_AREA = 60
SF_AREAS = ("financial", "soma", "mission", "sunset", "ggpark", "embarcadero", "baybridge", "steep")


class Tally:
    """Points per class, and how many of them carry each flag."""

    def __init__(self) -> None:
        self.points: dict[int, int] = {}
        self.flagged: dict[int, dict[str, int]] = {}
        self.has_overlap_bit = True

    def add(self, las) -> None:
        cls = np.asarray(las.classification, np.int64)
        present = {
            f: np.asarray(getattr(las, f), bool)
            if f in las.point_format.dimension_names or f in {"synthetic", "key_point", "withheld"}
            else None
            for f in FLAGS
        }
        if present["overlap"] is None:
            self.has_overlap_bit = False
        codes, counts = np.unique(cls, return_counts=True)
        for c, n in zip(codes.tolist(), counts.tolist(), strict=True):
            self.points[c] = self.points.get(c, 0) + n
            row = self.flagged.setdefault(c, dict.fromkeys(FLAGS, 0))
            m = cls == c
            for f, arr in present.items():
                if arr is not None:
                    row[f] += int(arr[m].sum())

    def table(self, names: dict) -> dict:
        out = {}
        total = sum(self.points.values())
        for c in sorted(self.points):
            n = self.points[c]
            out[str(c)] = {
                "name": names.get(c, ""),
                "points": n,
                "share_of_source": round(n / total, 6),
                **{
                    f"share_{f}": (round(self.flagged[c][f] / n, 6) if n else None)
                    if (f != "overlap" or self.has_overlap_bit)
                    else None
                    for f in FLAGS
                },
            }
        all_flags = {
            f"share_{f}": (
                round(sum(self.flagged[c][f] for c in self.points) / total, 6)
                if (f != "overlap" or self.has_overlap_bit)
                else None
            )
            for f in FLAGS
        }
        return {
            "points": total,
            "overlap_bit_in_format": self.has_overlap_bit,
            "all_classes": all_flags,
            "per_class": out,
        }


def dales(split: str) -> dict:
    import laspy

    t = Tally()
    files = sorted((ROOT / "data" / "dales2" / split).glob("*.laz"))
    for p in files:
        t.add(laspy.read(p))
    names = dict(enumerate(CLASS_NAMES))
    return {"tiles": len(files)} | t.table(names)


def swiss(ids: tuple[str, ...]) -> dict:
    import laspy

    per_tile, total = {}, Tally()
    for tile_id in ids:
        path = next((ROOT / "data" / "cache" / "swiss").glob(f"*{tile_id}*.laz"))
        las = laspy.read(path)
        t = Tally()
        t.add(las)
        total.add(las)
        per_tile[tile_id] = t.table(SWISS_NAMES)
    return {"tiles": list(ids), "per_tile": per_tile, "pooled": total.table(SWISS_NAMES)}


def san_francisco(seed: int = 0) -> dict:
    import laspy
    from fetch_sf_aoi import AOIS, DEFAULT_SIZE_M, DELIVERY
    from pyproj import Transformer

    src = EptSource.open(DELIVERY)
    to_wm = Transformer.from_crs(4326, 3857, always_xy=True)
    rng = np.random.default_rng(seed)
    per_area, total = {}, Tally()
    for name in SF_AREAS:
        lat, lon, _ = AOIS[name]
        cx, cy = to_wm.transform(lon, lat)
        half = (DEFAULT_SIZE_M / 2.0) / math.cos(math.radians(lat))
        aoi = (cx - half, cy - half, cx + half, cy + half)
        nodes = sorted(src.walk(keep=lambda k, a=aoi: overlaps_xy(k.bounds(src.cube), a)), key=str)
        pick = [
            nodes[i]
            for i in sorted(
                rng.choice(len(nodes), min(SF_NODES_PER_AREA, len(nodes)), replace=False)
            )
        ]
        t = Tally()
        for key in pick:
            las = laspy.read(io.BytesIO(_get(f"{src.base}/ept-data/{key}.laz")))
            x, y = np.asarray(las.x), np.asarray(las.y)
            inside = (x >= aoi[0]) & (x < aoi[2]) & (y >= aoi[1]) & (y < aoi[3])
            if inside.any():
                sub = las[inside]
                t.add(sub)
                total.add(sub)
        per_area[name] = {"nodes_overlapping": len(nodes), "nodes_read": len(pick)} | t.table(
            ASPRS_NAMES
        )
        print(
            f"  {name}: {per_area[name]['points']} points from {len(pick)} of {len(nodes)} nodes",
            flush=True,
        )
    return {
        "sample": f"a seeded sample of up to {SF_NODES_PER_AREA} nodes overlapping each area, points inside it",
        "per_area": per_area,
        "pooled": total.table(ASPRS_NAMES),
    }


def decisions() -> dict:
    return {
        src: {
            flag: {"decision": DECISIONS[flag][src][0], "reason": DECISIONS[flag][src][1]}
            for flag in FLAGS
        }
        for src in SOURCES
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions-only", action="store_true")
    args = ap.parse_args()
    if args.decisions_only:
        doc = json.loads(OUT.read_text(encoding="utf-8"))
        doc["decisions"] = decisions()
        OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote the decisions into {OUT}")
        return 0
    t0 = time.time()
    doc = {"what": "LAS point flags per class in every source", "flags": list(FLAGS)}
    doc["dales2_train"] = dales("train")
    doc["dales2_test"] = dales("test")
    print("DALES-2 done", flush=True)
    doc["swiss_training"] = swiss(SWISS_TRAINING)
    doc["swiss_held_out"] = swiss(SWISS_HELD_OUT)
    print("Swiss done", flush=True)
    doc["san_francisco"] = san_francisco()
    doc["decisions"] = decisions()
    doc["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
