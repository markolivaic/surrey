"""Write a training checkpoint's weights as a file that loads without unpickling.

A training checkpoint carries the optimiser, the random states and the run's arguments, and reading it
needs unpickling, which can run code. The file written here holds the network's tensors and a few plain
values (channels, width, recipe, seed, step, the source checkpoint's SHA-256), so
`surrey.inference.load_checkpoint` reads it with torch.load(weights_only=True). This is the form weights
are published in.

Writes data/weights/<name>.pt for each checkpoint given.

Run:
    python scripts/pipeline/export_weights.py data/checkpoints/m0_seedA.pt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.inference import load_checkpoint  # noqa: E402

OUT = ROOT / "data" / "weights"
PLAIN = ("channels", "width", "recipe", "seed", "steps")


def export(ckpt: Path, out_dir: Path = OUT) -> Path:
    net, blob, channels = load_checkpoint(ckpt, "cpu", allow_pickle=True)
    args = blob.get("args") or {}
    plain = {k: args[k] for k in PLAIN if isinstance(args.get(k), (int, float, str))}
    plain["channels"] = channels
    plain["width"] = int(args.get("width") or 128)
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / ckpt.name
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "args": plain,
            "step": int(blob.get("step", 0)),
            "source_checkpoint_sha256_16": digest,
        },
        target,
    )
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    for ckpt in args.checkpoints:
        target = export(ckpt, args.out_dir)
        load_checkpoint(target, "cpu")  # proves it reads without unpickling
        print(f"wrote {target}, {target.stat().st_size / 1e6:.1f} MB, reads with weights_only=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
