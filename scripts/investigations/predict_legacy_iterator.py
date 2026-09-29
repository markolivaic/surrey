"""Predict with the iterator the first round used, to measure what the new iterator changed on its own.

The same checkpoint and one vote, predicted once with `legacy_iter_full` (scripts/investigations/
iter_full_agreement.py) instead of CachedTile.iter_full, on the same tiles predict_cache.py covers.
The output has predict_cache.py's layout, so score_variant.py scores it as the variant
m0_seedA_legacy_iter.

Writes data/predictions/m0_seedA_legacy_iter/.

Run:
    python scripts/investigations/predict_legacy_iterator.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from iter_full_agreement import LegacyTile  # noqa: E402
from surrey.data import BLOCK_M, POINTS_PER_CROP  # noqa: E402
from surrey.inference import DEFAULT_SEED, infer_tile, load_checkpoint  # noqa: E402

CKPT = ROOT / "data" / "checkpoints" / "m0_seedA.pt"
OUT = ROOT / "data" / "predictions" / "m0_seedA_legacy_iter"
DALES_PREP = ROOT / "data" / "prep" / "dales"
TARGET_PREP = ROOT / "data" / "prep" / "target"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, blob, channels = load_checkpoint(CKPT, dev, allow_pickle=True)
    jobs = [
        ("surrey", t["name"], DALES_PREP / t["name"])
        for t in json.loads(PREP_ARTIFACT.read_text(encoding="utf-8"))["tiles"]
        if t["split"] == "test"
    ]
    jobs += [("sf", p.name, p) for p in sorted(TARGET_PREP.iterdir()) if (p / "meta.json").exists()]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": CKPT.relative_to(ROOT).as_posix(),
        "iterator": "legacy_iter_full",
        "votes": 1,
        "seed": DEFAULT_SEED,
        "channels": channels,
        "classes": int(blob["model"]["head.4.bias"].shape[0]),
        "device": dev,
        "tiles": {},
    }
    t0 = time.time()
    for city, name, path in sorted(jobs):
        tile = LegacyTile(path, city)
        pred, conf, ent, seen = infer_tile(
            net, tile, dev, BLOCK_M, POINTS_PER_CROP, None, batch=4, seed=DEFAULT_SEED, votes=1
        )
        if not seen.all():
            raise SystemExit(f"{name}: {int((~seen).sum())} points were never predicted")
        target = OUT / f"{city}_{name}.npz"
        np.savez(target, pred=pred, conf=conf.astype(np.float16), ent=ent.astype(np.float16))
        manifest["tiles"][f"{city}_{name}"] = {"points": tile.n, "file": target.name}
        print(f"  {city:6s} {name:12s} {tile.n:>11,} points", flush=True)
    manifest["seconds"] = round(time.time() - t0, 1)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {OUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
