"""How much the vegetation figures move when the NDVI threshold moves.

The vegetation reference calls a NAIP pixel green above an NDVI threshold fixed by Otsu's method
(naip_reference.py), and the imagery is 337 days older than the lidar. This rebuilds both reference
masks at a few fixed thresholds around the Otsu value, into data/reference/ndvi_sensitivity/, and reads
the published recipe's three seeds against each the way the headline is read (the mask rule's choice,
or where it does not decide, the lower of the two masks for vegetation found and the higher for false
vegetation). The committed reference is not touched.

Writes data/artifacts/ndvi_sensitivity.json.

Run:
    python scripts/pipeline/ndvi_sensitivity.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))

import score_variant as sv  # noqa: E402
from surrey.metrics import coarse_lookup  # noqa: E402

ARTIFACTS = ROOT / "data" / "artifacts"
OUT = ARTIFACTS / "ndvi_sensitivity.json"
WORK = ROOT / "data" / "reference" / "ndvi_sensitivity"
THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
SEEDS = ("A", "B", "C")


def build(threshold: float) -> Path:
    out = WORK / f"{threshold:.2f}"
    for extra in ([], ["--registered"]):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "pipeline" / "naip_reference.py"),
                "--threshold",
                str(threshold),
                "--out-dir",
                str(out),
                *extra,
            ],
            check=True,
            cwd=ROOT,
        )
    return out


def headline(base: Path, name: str, called_veg: np.ndarray, choice: str | None) -> dict:
    per = {}
    for label, path in (
        ("registered", base / "registered" / f"naip_{name}.npz"),
        ("unregistered", base / f"naip_{name}.npz"),
    ):
        ref = np.load(path)
        per[label] = (
            sv.rate(called_veg, ref["vegetation"]),
            sv.rate(called_veg, ref["not_vegetation"]),
        )
    if choice == "lower_of_two" and None not in (v[0] for v in per.values()):
        false = [v[1] for v in per.values() if v[1] is not None]
        return {
            "found": min(v[0] for v in per.values()),
            "false": max(false) if false else None,
        }
    if choice in per:
        return {"found": per[choice][0], "false": per[choice][1]}
    return {"found": None, "false": None}


def main() -> int:
    t0 = time.time()
    otsu = json.loads((ARTIFACTS / "naip_reference.json").read_text(encoding="utf-8"))
    choice = {
        r["aoi"]: r.get("reference")
        for r in json.loads((ARTIFACTS / "naip_mask_choice.json").read_text(encoding="utf-8"))[
            "areas"
        ]
    }
    bases = {f"{otsu['ndvi_threshold']:.3f} (Otsu)": sv.REFERENCE}
    for t in THRESHOLDS:
        bases[f"{t:.3f}"] = build(t)
    table = coarse_lookup()
    vegetation = sv.GROUPS.index("vegetation")
    areas = sorted(
        p.name for p in sv.TARGET_PREP.iterdir() if p.is_dir() and (p / "meta.json").exists()
    )
    result: dict = {}
    for name in areas:
        called = {
            s: table[sv.load(sv.PRED / f"m0_seed{s}", f"sf_{name}")[0]] == vegetation for s in SEEDS
        }
        result[name] = {}
        for label, base in bases.items():
            per_seed = {s: headline(base, name, called[s], choice.get(name)) for s in SEEDS}
            row = {}
            for fig in ("found", "false"):
                vals = [per_seed[s][fig] for s in SEEDS if per_seed[s][fig] is not None]
                row[fig] = (
                    {
                        "median": round(float(np.median(vals)), 4),
                        "min": min(vals),
                        "max": max(vals),
                        "seed_a": per_seed["A"][fig],
                    }
                    if vals
                    else None
                )
            result[name][label] = row
        print(
            f"  {name:12s} "
            + "  ".join(
                f"{k}: {(v['found'] or {}).get('median')}/{(v['false'] or {}).get('median')}"
                for k, v in result[name].items()
            ),
            flush=True,
        )
    doc = {
        "what": "the vegetation headline of the published recipe's seeds against NAIP references "
        "rebuilt at fixed NDVI thresholds around the Otsu value",
        "otsu_threshold": otsu["ndvi_threshold"],
        "thresholds": list(bases),
        "read_as": "the mask rule's choice, or the lower of the two masks for vegetation found and the "
        "higher for false vegetation where it does not decide",
        "areas": result,
        "seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
