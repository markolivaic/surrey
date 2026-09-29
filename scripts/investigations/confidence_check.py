"""How sure is the model when it is wrong? Measured, not inferred from class shares.

The claim that the model is most confident exactly where it has no concept was read
off the predicted class distribution. A share is not a confidence, and the rule in
this project is that a number nobody measured does not get asserted. So this reads
the probabilities themselves.

No training. The checkpoint is the one from the finished run and the predictions are
recomputed only because the earlier pass stored the argmax and threw the rest away.

Reported for each group: the maximum softmax probability, and the entropy of the
whole distribution normalised so that 0 is certain and 1 is a uniform guess over
fifteen classes. Max probability alone can hide a two-way split.

The groups, and why the Surrey ones are not optional. A max probability of 0.9 over
the bay means nothing without knowing what 0.9 looks like when the model is right at
home, and what it looks like when it is wrong at home. Those two are the scale
everything else is read against.

    surrey_correct      held-out Surrey, prediction matches the label
    surrey_wrong        held-out Surrey, prediction does not match
    sf_ground           San Francisco, delivery calls it ground
    sf_water            San Francisco, delivery calls it water, model has no such class
    sf_bridge_deck      San Francisco, delivery calls it bridge deck, no such class
    sf_unassigned       San Francisco, delivery says nothing

Two outcomes and both are useful. High confidence over water confirms the headline
by measurement. Low confidence over water means the information was there and argmax
discarded it, which earns the viewer a third colour that lights up over the bay.

Run:
    python scripts/investigations/confidence_check.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import BLOCK_M, N_CLASSES, POINTS_PER_CROP  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402

DALES_PREP = ROOT / "data" / "prep" / "dales"
TARGET_PREP = ROOT / "data" / "prep" / "target"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
CKPT = ROOT / "data" / "checkpoints" / "dayton.pt"
DEFAULT_OUT = ROOT / "data" / "artifacts" / "confidence.json"

ASPRS = {2: "sf_ground", 9: "sf_water", 17: "sf_bridge_deck", 1: "sf_unassigned"}
ASPRS_NOISE = (7, 18)
LOG_N = float(np.log(N_CLASSES))


@torch.no_grad()
def probabilities(
    net,
    tile: CachedTile,
    dev: str,
    block_m: float,
    n_points: int,
    max_blocks: int | None,
    batch: int = 4,
):
    """Per point: argmax class, its probability, and the normalised entropy."""
    pred = np.full(tile.n, -1, dtype=np.int8)
    conf = np.zeros(tile.n, dtype=np.float32)
    ent = np.zeros(tile.n, dtype=np.float32)
    seen = np.zeros(tile.n, dtype=bool)
    amp = dev == "cuda"
    pending: list[tuple[np.ndarray, np.ndarray, int]] = []

    def flush() -> None:
        if not pending:
            return
        x = torch.from_numpy(np.stack([p[0] for p in pending])).to(dev, non_blocking=True)
        with torch.amp.autocast(dev, dtype=torch.float16, enabled=amp):
            logits = net(x)
        # float32 for the softmax. In half precision the tail of a 15-way
        # distribution underflows and the entropy comes back flattered.
        p = torch.softmax(logits.float(), dim=-1)
        top = p.max(dim=-1)
        e = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1) / LOG_N
        cls = top.indices.cpu().numpy().astype(np.int8)
        prob = top.values.cpu().numpy().astype(np.float32)
        ent_b = e.cpu().numpy().astype(np.float32)
        for row, (_, chunk, valid) in enumerate(pending):
            idx = chunk[:valid]
            pred[idx] = cls[row, :valid]
            conf[idx] = prob[row, :valid]
            ent[idx] = ent_b[row, :valid]
            seen[idx] = True
        pending.clear()

    for feat, _, chunk, valid in tile.iter_full(block_m, n_points, max_blocks):
        pending.append((feat, chunk, valid))
        if len(pending) == batch:
            flush()
    flush()
    return pred, conf, ent, seen


def summarise(name: str, conf: np.ndarray, ent: np.ndarray) -> dict:
    if conf.size == 0:
        return {"group": name, "points": 0}
    return {
        "group": name,
        "points": int(conf.size),
        "max_prob_median": round(float(np.median(conf)), 4),
        "max_prob_p10": round(float(np.percentile(conf, 10)), 4),
        "max_prob_p90": round(float(np.percentile(conf, 90)), 4),
        "max_prob_over_0_9": round(float((conf > 0.9).mean()), 4),
        "max_prob_under_0_5": round(float((conf < 0.5).mean()), 4),
        "entropy_median": round(float(np.median(ent)), 4),
        "entropy_p90": round(float(np.percentile(ent, 90)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--block-m", type=float, default=BLOCK_M)
    ap.add_argument("--points", type=int, default=POINTS_PER_CROP)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument(
        "--surrey-blocks", type=int, default=40, help="blocks per held-out tile, spread across it"
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"{args.ckpt} missing, run scripts/pipeline/train.py first")
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net = SurreyNet(N_CLASSES, width=args.width).to(dev)
    net.load_state_dict(blob["model"])
    net.eval()
    print(f"device {dev}, checkpoint step {blob['step']}")

    t0 = time.time()
    pools: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    def add(group: str, c: np.ndarray, e: np.ndarray) -> None:
        pools.setdefault(group, []).append((c, e))

    test_names = [
        t["name"]
        for t in json.loads(PREP_ARTIFACT.read_text(encoding="utf-8"))["tiles"]
        if t["split"] == "test"
    ]
    print(f"Surrey, {len(test_names)} held-out tiles, {args.surrey_blocks} blocks each")
    for n in sorted(test_names):
        tile = CachedTile(DALES_PREP / n, "test")
        pred, conf, ent, seen = probabilities(
            net, tile, dev, args.block_m, args.points, args.surrey_blocks
        )
        label = np.asarray(tile.label[:], dtype=np.int16)
        ok = seen & (label >= 0)
        right = ok & (pred == label)
        wrong = ok & (pred != label)
        add("surrey_correct", conf[right], ent[right])
        add("surrey_wrong", conf[wrong], ent[wrong])
        print(f"  {n}", flush=True)

    per_area = []
    print("San Francisco, every area, every point")
    for p in sorted(TARGET_PREP.iterdir()):
        if not (p.is_dir() and (p / "meta.json").exists()):
            continue
        tile = CachedTile(p, "sf")
        pred, conf, ent, seen = probabilities(net, tile, dev, args.block_m, args.points, None)
        asprs = np.asarray(tile.asprs[:], dtype=np.int16)
        clean = seen & ~np.isin(asprs, ASPRS_NOISE)
        row = {"aoi": tile.name, "groups": []}
        for code, group in ASPRS.items():
            m = clean & (asprs == code)
            if m.sum() >= 1000:
                add(group, conf[m], ent[m])
                row["groups"].append(summarise(group, conf[m], ent[m]))
        per_area.append(row)
        print(f"  {tile.name}", flush=True)

    pooled = []
    order = [
        "surrey_correct",
        "surrey_wrong",
        "sf_ground",
        "sf_unassigned",
        "sf_water",
        "sf_bridge_deck",
    ]
    for group in order:
        if group not in pools:
            continue
        c = np.concatenate([a for a, _ in pools[group]])
        e = np.concatenate([b for _, b in pools[group]])
        pooled.append(summarise(group, c, e))

    by = {r["group"]: r for r in pooled}
    water = by.get("sf_water")
    anchor = by.get("surrey_correct")
    verdict = "water was not measured"
    if water and anchor:
        verdict = (
            "confident over water: its median max probability is not below the "
            "model's median when it is correct at home, so argmax is not discarding "
            "a doubt the network had"
            if water["max_prob_median"] >= anchor["max_prob_median"] - 0.05
            else "less confident over water: the network carried doubt that argmax threw "
            "away, which earns the viewer a third colour for 'no answer'"
        )

    result = {
        "question": "how sure is the model where it has no concept",
        "no_training": "the checkpoint is unchanged, only inference was rerun",
        "checkpoint_step": int(blob["step"]),
        "device": dev,
        "device_name": (
            torch.cuda.get_device_name(0)
            if dev == "cuda"
            else platform.processor() or platform.machine()
        ),
        "torch": torch.__version__,
        "entropy_note": (
            "normalised by log of the class count, so 0 is certain and "
            "1 is a uniform guess over all fifteen classes"
        ),
        "softmax_precision": "float32, because a half precision 15-way tail underflows",
        "surrey_blocks_per_tile": args.surrey_blocks,
        "pooled": pooled,
        "verdict": verdict,
        "per_area": per_area,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(
        f"\n{'group':18s}{'points':>13}{'max prob med':>14}{'p10':>8}"
        f"{'>0.9':>8}{'<0.5':>8}{'entropy med':>13}"
    )
    for r in pooled:
        print(
            f"{r['group']:18s}{r['points']:>13,}{r['max_prob_median']:>14.4f}"
            f"{r['max_prob_p10']:>8.3f}{r['max_prob_over_0_9']:>8.3f}"
            f"{r['max_prob_under_0_5']:>8.3f}{r['entropy_median']:>13.4f}"
        )
    print(f"\n{verdict}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
