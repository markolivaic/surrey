"""How many voting passes every model is predicted with, chosen on held-out Surrey before any San
Francisco figure exists.

The rule, written in docs/phase4_criteria.md before this ran: N is one of 1, 2, 4, 8, 16, the smallest
at which the coarse classes of the N-pass vote and the 2N-pass vote agree on at least 0.99 of the
points of the eleven held-out Surrey tiles, pooled; 16 if none of 1, 2, 4 or 8 does. Pass v is seeded
with v (surrey.inference.vote_probabilities), so the N-pass vote is the first N passes of the 2N-pass
one. One model decides it, M0 seed A, and every model is then predicted with the same N.

Writes data/artifacts/votes.json: the agreement at each N, pooled and per tile, the N chosen, and the
seconds each pass took on the card that ran it.

Run:
    python scripts/pipeline/choose_votes.py --ckpt data/checkpoints/m0_seedA.pt
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
from surrey.data import BLOCK_M, POINTS_PER_CROP  # noqa: E402
from surrey.inference import load_checkpoint, vote_probabilities  # noqa: E402
from surrey.metrics import coarse_lookup  # noqa: E402

DALES_PREP = ROOT / "data" / "prep" / "dales"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
OUT = ROOT / "data" / "artifacts" / "votes.json"
CANDIDATES = (1, 2, 4, 8, 16)
AGREEMENT = 0.99


def choose(agreement: dict[int, float]) -> int:
    """The smallest N whose vote agrees with the 2N vote on at least AGREEMENT of the points."""
    for n in CANDIDATES[:-1]:
        if agreement[n] >= AGREEMENT:
            return n
    return CANDIDATES[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "data" / "checkpoints" / "m0_seedA.pt")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net, _, _ = load_checkpoint(args.ckpt, dev)
    coarse = coarse_lookup()
    names = sorted(
        t["name"]
        for t in json.loads(PREP_ARTIFACT.read_text(encoding="utf-8"))["tiles"]
        if t["split"] == "test"
    )
    agree = dict.fromkeys(CANDIDATES[:-1], 0)
    points = 0
    per_tile, seconds = {}, []
    for name in names:
        tile = CachedTile(DALES_PREP / name, "surrey")
        total = None
        calls: dict[int, np.ndarray] = {}
        seen = None
        for v in range(CANDIDATES[-1]):
            t0 = time.time()
            probs, seen = vote_probabilities(
                net, tile, dev, BLOCK_M, POINTS_PER_CROP, batch=args.batch, seed=v, votes=1
            )
            seconds.append(time.time() - t0)
            total = probs if total is None else total + probs
            if v + 1 in CANDIDATES:
                calls[v + 1] = coarse[total.argmax(axis=1)]
        m = seen
        row = {}
        for n in CANDIDATES[:-1]:
            same = int((calls[n][m] == calls[2 * n][m]).sum())
            agree[n] += same
            row[str(n)] = round(same / int(m.sum()), 5)
        points += int(m.sum())
        per_tile[name] = row
        print(f"  {name}: {row}", flush=True)
    pooled = {n: agree[n] / points for n in agree}
    chosen = choose(pooled)
    doc = {
        "rule": "smallest N in 1, 2, 4, 8, 16 whose coarse classes agree with the 2N vote on at "
        f"least {AGREEMENT} of the held-out Surrey points, pooled; 16 if none does "
        "(docs/phase4_criteria.md)",
        "checkpoint": args.ckpt.resolve().relative_to(ROOT).as_posix()
        if args.ckpt.resolve().is_relative_to(ROOT)
        else args.ckpt.name,
        "tiles": names,
        "points": points,
        "agreement_n_with_2n": {str(n): round(a, 5) for n, a in pooled.items()},
        "per_tile": per_tile,
        "votes": chosen,
        "device": dev,
        "device_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "seconds_per_pass_per_tile_median": round(float(np.median(seconds)), 1),
        "seconds_all_passes": round(float(np.sum(seconds)), 1),
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"votes {chosen}; wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
