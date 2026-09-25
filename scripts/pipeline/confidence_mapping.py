"""The map from confidence to chroma, fixed on Surrey before any San Francisco frame uses it.

How saturated a point is decides how unsure San Francisco looks, so the map is a claim and is
chosen the way seeds, references and thresholds are: mechanically, from data the finding
does not come from, and written down before it is applied.

Rule, written before this ran:
    source      the published model, M0 seed A, on every labelled point of the eleven
                held-out Surrey tiles (data/predictions/m0_seedA/surrey_*.npz), confidence
                being the top class probability
    low end     the 5th percentile of confidence over points the model got WRONG
    high end    the 95th percentile of confidence over points the model got RIGHT
    map         t = clip((confidence - low) / (high - low), 0, 1); chroma rises from the
                palette's floor at t = 0 to the class's full chroma at t = 1, linearly

The same map, unchanged, colours San Francisco, the old four channel model, M0 and M1. The
legend quotes it. After it is written, the script also reports where each San Francisco area's
median confidence falls on it, for every variant with predictions; those positions are a
reading of the map, never an input to it.

Run:
    python scripts/pipeline/confidence_mapping.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402

PRED = ROOT / "data" / "predictions"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
OUT = ROOT / "data" / "artifacts" / "confidence_mapping.json"
LOW_PERCENTILE, HIGH_PERCENTILE = 5, 95
READ_VARIANTS = {
    "returns_ood": PRED,
    "ret0": PRED / "ret0",
    "m0_seedA": PRED / "m0_seedA",
    "m0_seedB": PRED / "m0_seedB",
    "m0_seedC": PRED / "m0_seedC",
}
SF_AOIS = ["baybridge", "embarcadero", "financial", "ggpark", "mission", "soma", "steep", "sunset"]


def main() -> int:
    names = sorted(
        t["name"] for t in json.loads(PREP_ARTIFACT.read_text())["tiles"] if t["split"] == "test"
    )
    right, wrong = [], []
    for n in names:
        tile = CachedTile(ROOT / "data" / "prep" / "dales" / n, "test")
        z = np.load(PRED / "m0_seedA" / f"surrey_{n}.npz")
        label = np.asarray(tile.label[:], np.int16)
        lab = label >= 0
        conf = z["conf"].astype(np.float32)
        ok = lab & (z["pred"] == label)
        right.append(conf[ok])
        wrong.append(conf[lab & ~ok])
    right, wrong = np.concatenate(right), np.concatenate(wrong)
    low = float(np.percentile(wrong, LOW_PERCENTILE))
    high = float(np.percentile(right, HIGH_PERCENTILE))
    doc = {
        "rule": __doc__.split("\n\n")[2],
        "source": "m0_seedA, held-out Surrey tiles, labelled points",
        "points_right": int(right.size),
        "points_wrong": int(wrong.size),
        "low": round(low, 4),
        "high": round(high, 4),
        "low_definition": f"p{LOW_PERCENTILE} of confidence on wrong points",
        "high_definition": f"p{HIGH_PERCENTILE} of confidence on right points",
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"map fixed: low {doc['low']}  high {doc['high']}  from {right.size:,} right and {wrong.size:,} wrong"
    )

    # Only now, a reading of San Francisco on the fixed map.
    readings = {}
    for v, d in READ_VARIANTS.items():
        rows = {}
        for aoi in SF_AOIS:
            path = d / f"sf_{aoi}.npz"
            if not path.exists():
                continue
            c = np.load(path)["conf"].astype(np.float32)
            med = float(np.median(c))
            rows[aoi] = {
                "median_confidence": round(med, 4),
                "position_on_map": round(min(max((med - low) / (high - low), 0.0), 1.0), 3),
                "share_at_or_below_low": round(float((c <= low).mean()), 4),
                "share_at_or_above_high": round(float((c >= high).mean()), 4),
            }
        readings[v] = rows
    surrey = {
        "median_confidence_right": round(float(np.median(right)), 4),
        "median_confidence_wrong": round(float(np.median(wrong)), 4),
    }
    surrey["position_right"] = round((surrey["median_confidence_right"] - low) / (high - low), 3)
    surrey["position_wrong"] = round((surrey["median_confidence_wrong"] - low) / (high - low), 3)
    doc["surrey_reading"] = surrey
    doc["san_francisco_reading"] = readings
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    for v, rows in readings.items():
        print(v, {a: r["position_on_map"] for a, r in rows.items()})
    print("surrey", surrey)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
