"""What does the model call the trees? Scored against the blind NAIP reference.

"The vegetation class does not fire" was measured. "The model calls the park's trees
buildings" was inferred from class shares, because San Francisco had no tree
reference. scripts/pipeline/naip_reference.py built one without looking at any prediction.
This reads it next to the cached predictions and turns the inference into a number.

For each area, over points in a NAIP vegetation core that stand 2 to 25 m above
ground, the share the model assigns to each coarse class. The same over points in a
NAIP non-vegetation core, which is where a vegetation call would be wrong.

The home figure is built over the same height band from the DALES labels: of the
held-out Surrey points labelled tree or vegetation between 2 and 25 m, the share the
model calls vegetation. Same height gate, same label-free ground, so the two
figures differ in the city and the reference, and in nothing chosen here.

Per area, never pooled.

Run:
    python scripts/investigations/score_vegetation.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import CLASS_NAMES, COARSE  # noqa: E402
from surrey.metrics import (
    coarse_lookup,  # noqa: E402
    coarse_shares,  # noqa: E402
)

DALES_PREP = ROOT / "data" / "prep" / "dales"
TARGET_PREP = ROOT / "data" / "prep" / "target"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
PRED = ROOT / "data" / "predictions"
REFERENCE = ROOT / "data" / "reference"
NAIP_ARTIFACT = ROOT / "data" / "artifacts" / "naip_reference.json"
OUT = ROOT / "data" / "artifacts" / "vegetation_sf.json"

GROUPS = list(COARSE)
VEG_FINE = [CLASS_NAMES.index("tree"), CLASS_NAMES.index("vegetation")]
MIN_POINTS = 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    naip = json.loads(NAIP_ARTIFACT.read_text(encoding="utf-8"))
    manifest = json.loads((PRED / "manifest.json").read_text(encoding="utf-8"))
    lo, hi = naip["height_gate_m"]
    table = coarse_lookup()
    t0 = time.time()

    areas = []
    for p in sorted(TARGET_PREP.iterdir()):
        if not (p.is_dir() and (p / "meta.json").exists()):
            continue
        ref_path = REFERENCE / f"naip_{p.name}.npz"
        key = f"sf_{p.name}"
        if not ref_path.exists() or key not in manifest["tiles"]:
            areas.append({"aoi": p.name, "skipped": "no NAIP reference or no cached prediction"})
            continue
        ref = np.load(ref_path)
        pred = np.load(PRED / manifest["tiles"][key]["file"])["pred"]
        coarse = table[pred]
        veg, notveg = ref["vegetation"], ref["not_vegetation"]
        row = {
            "aoi": p.name,
            "points_in_vegetation_core": int(veg.sum()),
            "points_in_not_vegetation_core": int(notveg.sum()),
        }
        if veg.sum() >= MIN_POINTS:
            row["vegetation_core_called"] = coarse_shares(coarse[veg], min_points=1)
        else:
            row["vegetation_core_called"] = None
            row["vegetation_core_note"] = f"fewer than {MIN_POINTS} points, not scored"
        row["not_vegetation_core_called_vegetation"] = (
            round(float((coarse[notveg] == GROUPS.index("vegetation")).mean()), 4)
            if notveg.sum() >= MIN_POINTS
            else None
        )
        areas.append(row)
        s = row["vegetation_core_called"]
        if s:
            print(
                f"  {p.name:12s} veg core {veg.sum():>9,}  called vegetation {s['vegetation']:.4f}  "
                f"building {s['building']:.4f}  ground {s['ground']:.4f}",
                flush=True,
            )
        else:
            print(f"  {p.name:12s} veg core {veg.sum():>9,}  not scored", flush=True)

    # Home figure: the same height band, from the DALES labels.
    names = sorted(
        t["name"]
        for t in json.loads(PREP_ARTIFACT.read_text(encoding="utf-8"))["tiles"]
        if t["split"] == "test"
    )
    total = hit_veg = hit_bld = 0
    missing = []
    for n in names:
        key = f"surrey_{n}"
        if key not in manifest["tiles"]:
            missing.append(n)
            continue
        tile = CachedTile(DALES_PREP / n, "test")
        pred = np.load(PRED / manifest["tiles"][key]["file"])["pred"]
        label = np.asarray(tile.label[:], dtype=np.int16)
        hag = np.asarray(tile.hag_cm[:], dtype=np.float32) / 100.0
        m = np.isin(label, VEG_FINE) & (hag > lo) & (hag <= hi)
        c = table[pred[m]]
        total += int(m.sum())
        hit_veg += int((c == GROUPS.index("vegetation")).sum())
        hit_bld += int((c == GROUPS.index("building")).sum())
    if missing:
        raise SystemExit(f"Surrey tiles missing from the prediction cache: {missing}")
    home = {
        "population": "held-out Surrey points labelled tree or vegetation, 2 to 25 m above ground",
        "points": total,
        "called_vegetation": round(hit_veg / total, 4) if total else None,
        "called_building": round(hit_bld / total, 4) if total else None,
    }

    result = {
        "question": "what the model calls vegetation that NAIP sees",
        "reference": "scripts/pipeline/naip_reference.py, NDVI threshold fixed before any prediction",
        "ndvi_threshold": naip["ndvi_threshold"],
        "height_band_m": [lo, hi],
        "prediction_cache": {k: manifest[k] for k in ("checkpoint_sha256_16", "seed")},
        "home_surrey": home,
        "san_francisco_per_area": areas,
        "no_pooling": "areas are reported one by one",
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nhome: {home}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
