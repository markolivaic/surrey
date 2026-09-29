"""Run the model once, seeded, and keep every point's prediction on disk.

Three analyses need the same per-point predictions: the easy-error AUROC control,
the vegetation score against NAIP, and anything the viewer later draws. Each script
used to re-run inference itself, and before inference was seeded that meant each
one scored a slightly different model. So inference runs here once, the output is
cached under data/predictions, and every consumer reads the cache and checks that it
came from the checkpoint and seed it expects.

Not committed: the cache is regenerated from the checkpoint by this script.

The channel count is read from the checkpoint, so the three channel M0 is never
handed a fourth column. --out-dir names the variant's own directory.

Run:
    python scripts/pipeline/predict_cache.py
    python scripts/pipeline/predict_cache.py --ckpt data/checkpoints/m0_seedA.pt --out-dir data/predictions/m0_seedA
"""

from __future__ import annotations

import argparse
import hashlib
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
from surrey.inference import DEFAULT_SEED, infer_tile, load_checkpoint  # noqa: E402

DALES_PREP = ROOT / "data" / "prep" / "dales"
TARGET_PREP = ROOT / "data" / "prep" / "target"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
CKPT = ROOT / "data" / "checkpoints" / "m0_seedA.pt"
OUT = ROOT / "data" / "predictions"


def checkpoint_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--block-m", type=float, default=BLOCK_M)
    ap.add_argument("--points", type=int, default=POINTS_PER_CROP)
    ap.add_argument(
        "--width", type=int, default=None, help="default: the width the checkpoint records"
    )
    ap.add_argument(
        "--votes",
        type=int,
        default=1,
        help="passes averaged per point, chosen by scripts/pipeline/choose_votes.py",
    )
    ap.add_argument(
        "--ret-value",
        type=float,
        default=None,
        help="replace the return ratio with this constant (the four channel checkpoint's ret0)",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument(
        "--batch",
        type=int,
        default=4,
        help="blocks per forward pass; 1 fits beside a training run on a 4 GB card",
    )
    args = ap.parse_args()
    out_dir = args.out_dir

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # The project's own training checkpoints, which carry more than weights.
    net, blob, channels = load_checkpoint(args.ckpt, dev, args.width, allow_pickle=True)
    n_out = int(blob["model"]["head.4.bias"].shape[0])
    digest = checkpoint_digest(args.ckpt)
    print(
        f"device {dev}, checkpoint {digest} step {blob['step']}, {channels} channels, seed {args.seed}"
    )

    jobs = []
    for t in json.loads(PREP_ARTIFACT.read_text())["tiles"]:
        if t["split"] == "test":
            jobs.append(("surrey", t["name"], DALES_PREP / t["name"]))
    for p in sorted(TARGET_PREP.iterdir()):
        if p.is_dir() and (p / "meta.json").exists():
            jobs.append(("sf", p.name, p))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_sha256_16": digest,
        "checkpoint_step": int(blob["step"]),
        "checkpoint": str(args.ckpt.relative_to(ROOT)).replace("\\", "/")
        if args.ckpt.is_relative_to(ROOT)
        else str(args.ckpt),
        "channels": channels,
        "classes": n_out,
        # Entropy is divided by the log of this; predictions made before the field existed
        # divided by log 15 whatever the model's class count.
        "entropy_normalised_by_log_of": n_out,
        "recipe": blob.get("args", {}).get("recipe", "m0"),
        "batch": args.batch,
        "seed": args.seed,
        "votes": args.votes,
        "ret_value": args.ret_value,
        "block_m": args.block_m,
        "points": args.points,
        "device": dev,
        "tiles": {},
    }
    t0 = time.time()
    for city, name, path in sorted(jobs):
        target = out_dir / f"{city}_{name}.npz"
        tile = CachedTile(path, city)
        pred, conf, ent, seen = infer_tile(
            net,
            tile,
            dev,
            args.block_m,
            args.points,
            None,
            batch=args.batch,
            seed=args.seed,
            ret_value=args.ret_value,
            votes=args.votes,
        )
        if not seen.all():
            raise SystemExit(f"{name}: {int((~seen).sum())} points were never predicted")
        np.savez(target, pred=pred, conf=conf.astype(np.float16), ent=ent.astype(np.float16))
        manifest["tiles"][f"{city}_{name}"] = {"points": tile.n, "file": target.name}
        print(f"  {city:6s} {name:12s} {tile.n:>11,} points", flush=True)

    manifest["seconds"] = round(time.time() - t0, 1)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
