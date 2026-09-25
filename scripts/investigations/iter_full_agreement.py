"""How much the prediction moves when the full pass draws its tensors the way training draws crops.

The published predictions came from an iterator that cut each block into contiguous runs of its
points, at the source's full density, and filled the last run with copies of one point. A training
crop is a random sample of the whole block. `CachedTile.iter_full` now deals each block into random
partitions of the whole block; `legacy_iter_full` below is the old iterator, kept to measure the
difference and for the test that shows the duplicate check catches it.

On a spread sample of blocks per area, with the same checkpoint, this reports the share of points
whose predicted class is the same under the old and the new iterator. Two base rates sit beside it:
the new iterator against itself with a different seed (a different deal and different network
draws), and the old iterator against itself with different network draws. A change larger than
those is the iterator, not the noise. On the held-out Surrey tiles, where labels exist, it also
reports accuracy under each.

Writes data/artifacts/iter_full_agreement.json.

Run:
    python scripts/investigations/iter_full_agreement.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import BLOCK_M, N_CLASSES, POINTS_PER_CROP  # noqa: E402
from surrey.inference import infer_tile  # noqa: E402
from surrey.metrics import coarse_lookup  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402

CKPT = ROOT / "data" / "checkpoints" / "m0_seedA.pt"
TARGET_PREP = ROOT / "data" / "prep" / "target"
DALES_PREP = ROOT / "data" / "prep" / "dales"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
OUT = ROOT / "data" / "artifacts" / "iter_full_agreement.json"
SF_AREAS = ("financial", "soma", "mission", "sunset", "ggpark", "embarcadero", "baybridge", "steep")
SURREY_TILES = 3
BLOCKS = 12


def legacy_iter_full(
    self, block_m, n_points, max_blocks=None, z_override=None, ret_value=None, channels=4, seed=0
):
    """The iterator the published predictions were made with, without its diagnostic hooks. It dealt
    nothing at random, so `seed` is accepted and has no effect."""
    if z_override is not None or ret_value is not None:
        raise ValueError("the legacy iterator is kept without its diagnostic hooks")
    for cx, cy in self.block_centres(block_m, max_blocks):
        idx = self.gather_block(cx, cy, block_m)
        if idx.size == 0:
            continue
        z_ref = self.block_z_reference(idx)
        for s in range(0, idx.size, n_points):
            chunk = idx[s : s + n_points]
            valid = chunk.size
            if valid < n_points:
                chunk = np.concatenate([chunk, np.repeat(chunk[:1], n_points - valid)])
            yield self.features(chunk, (cx, cy), z_ref, channels), self.labels(chunk), chunk, valid


class LegacyTile(CachedTile):
    iter_full = legacy_iter_full


def same(a: np.ndarray, b: np.ndarray) -> float:
    return round(float((a == b).mean()), 4)


def predict(net, tile, dev: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pred, _, _, seen = infer_tile(
        net, tile, dev, BLOCK_M, POINTS_PER_CROP, BLOCKS, batch=2, seed=seed
    )
    return pred, seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location=dev, weights_only=False)
    channels = int(blob.get("args", {}).get("channels") or 4)
    net = SurreyNet(N_CLASSES, width=128, extra_features=channels - 3).to(dev)
    net.load_state_dict(blob["model"])
    net.eval()

    jobs = [(f"sf_{a}", TARGET_PREP / a) for a in SF_AREAS]
    test = [
        t["name"]
        for t in json.loads(PREP_ARTIFACT.read_text(encoding="utf-8"))["tiles"]
        if t["split"] == "test"
    ]
    jobs += [(f"surrey_{n}", DALES_PREP / n) for n in sorted(test)[:SURREY_TILES]]

    coarse = coarse_lookup()
    rows = {}
    t0 = time.time()
    for name, path in jobs:
        old_a, seen_old = predict(net, LegacyTile(path), dev, 0)
        old_b, _ = predict(net, LegacyTile(path), dev, 1)
        # Inference seeds the deal and the network's draws together, so seed 1 is a different deal too.
        new_a, seen_new = predict(net, CachedTile(path), dev, 0)
        new_b, _ = predict(net, CachedTile(path), dev, 1)
        if not np.array_equal(seen_old, seen_new):
            raise SystemExit(f"{name}: the two iterators visited different points")
        m = seen_new
        row = {"points": int(m.sum())}
        for level, f in (("fine", lambda a: a), ("coarse", lambda a: coarse[a])):
            row[level] = {
                "old_vs_new": same(f(old_a[m]), f(new_a[m])),
                "new_vs_new_other_seed": same(f(new_a[m]), f(new_b[m])),
                "old_vs_old_other_seed": same(f(old_a[m]), f(old_b[m])),
            }
        if name.startswith("surrey_"):
            label = np.asarray(CachedTile(path).label)
            lm = m & (label >= 0)
            row["accuracy_old"] = round(float((old_a[lm] == label[lm]).mean()), 4)
            row["accuracy_new"] = round(float((new_a[lm] == label[lm]).mean()), 4)
        rows[name] = row
        print(f"  {name:22s} {row}", flush=True)

    doc = {
        "checkpoint": args.ckpt.relative_to(ROOT).as_posix(),
        "blocks_per_tile": BLOCKS,
        "points_per_tensor": POINTS_PER_CROP,
        "block_m": BLOCK_M,
        "device": dev,
        "what": "share of points with the same predicted class, over the fifteen classes and over the "
        "six coarse groups; base rates are the same iterator run again with a different seed",
        "tiles": rows,
        "seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
