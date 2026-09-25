"""Does confidence still tell the model when it is wrong, once it leaves home?

Not "is the model less confident in San Francisco". That was the soft reading. The
measured one is that confidence stops discriminating: at home, being wrong shows up
as lower confidence, and in the new city it does not. That is a claim about a
detector, so it is scored as one, with AUROC, and the task is built the same way on
both sides so the two numbers are comparable.

The task: take every point the model calls ground. Split it into the points where
that call is right and the points where it is wrong. Ask whether the maximum softmax
probability ranks the right ones above the wrong ones.

    Surrey, held-out tiles, full pass
        right   the DALES label is ground
        wrong   the DALES label is anything else
    San Francisco, every area
        right   the delivery calls it ground
        wrong   the delivery calls it water

One asymmetry, stated rather than hidden. At home the wrong side is every class the
model mistook for ground. In San Francisco the only certain wrong answer is water,
because unassigned might be ground and the model rarely calls a bridge deck ground.
So the San Francisco negative set is narrower. It is also the cleanest negative
available anywhere in the target, which is why it was chosen.

Pooling. The San Francisco AUROC pools ground from seven areas against water from
two, which mixes populations, the same mistake as a pooled null. So it is reported
both pooled and for Embarcadero alone, the one area that contains both.

Inference is seeded, see backend/surrey/inference.py, and the full held-out pass
recomputes accuracy so it can be checked against the training artefact.

Run:
    python scripts/investigations/auroc_check.py
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
from surrey.data import BLOCK_M, CLASS_NAMES, N_CLASSES, POINTS_PER_CROP  # noqa: E402
from surrey.inference import DEFAULT_SEED, infer_tile  # noqa: E402
from surrey.metrics import auroc_from_histograms, score_histogram  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402

DALES_PREP = ROOT / "data" / "prep" / "dales"
TARGET_PREP = ROOT / "data" / "prep" / "target"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
CKPT = ROOT / "data" / "checkpoints" / "dayton.pt"
DEFAULT_OUT = ROOT / "data" / "artifacts" / "auroc.json"

GROUND = CLASS_NAMES.index("ground")
ASPRS_GROUND, ASPRS_WATER = 2, 9
ASPRS_NOISE = (7, 18)
FIGURE_BINS = 200


class Group:
    """Scores for one side of the task, kept as a histogram so tiles pool cheaply."""

    def __init__(self) -> None:
        self.hist = np.zeros(20000, dtype=np.int64)
        self.under_half = 0
        self.over_09 = 0

    def add(self, scores: np.ndarray) -> None:
        if scores.size == 0:
            return
        self.hist += score_histogram(scores)
        self.under_half += int((scores < 0.5).sum())
        self.over_09 += int((scores > 0.9).sum())

    @property
    def n(self) -> int:
        return int(self.hist.sum())

    def quantile(self, q: float) -> float | None:
        if self.n == 0:
            return None
        c = np.cumsum(self.hist)
        b = int(np.searchsorted(c, q * self.n))
        return round((b + 0.5) / self.hist.size, 4)

    def summary(self) -> dict:
        n = self.n
        return {
            "points": n,
            "max_prob_median": self.quantile(0.5),
            "max_prob_p10": self.quantile(0.1),
            "points_under_0_5": self.under_half,
            "share_under_0_5": round(self.under_half / n, 6) if n else None,
            "points_over_0_9": self.over_09,
            "share_over_0_9": round(self.over_09 / n, 6) if n else None,
        }

    def figure(self, bins: int = FIGURE_BINS) -> list[int]:
        return self.hist.reshape(bins, -1).sum(axis=1).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--block-m", type=float, default=BLOCK_M)
    ap.add_argument("--points", type=int, default=POINTS_PER_CROP)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net = SurreyNet(N_CLASSES, width=args.width).to(dev)
    net.load_state_dict(blob["model"])
    net.eval()
    print(f"device {dev}, checkpoint step {blob['step']}, seed {args.seed}")
    t0 = time.time()

    # ---------------- Surrey, every point of every held-out tile ----------------
    test_names = sorted(
        t["name"] for t in json.loads(PREP_ARTIFACT.read_text())["tiles"] if t["split"] == "test"
    )
    s_right, s_wrong = Group(), Group()
    correct_total = labelled_total = 0
    per_tile = []
    for n in test_names:
        tile = CachedTile(DALES_PREP / n, "test")
        pred, conf, _, seen = infer_tile(
            net, tile, dev, args.block_m, args.points, None, seed=args.seed
        )
        label = np.asarray(tile.label[:], dtype=np.int16)
        ok = seen & (label >= 0)
        correct_total += int((ok & (pred == label)).sum())
        labelled_total += int(ok.sum())

        called = ok & (pred == GROUND)
        right = conf[called & (label == GROUND)]
        wrong = conf[called & (label != GROUND)]
        t_right, t_wrong = Group(), Group()
        t_right.add(right)
        t_wrong.add(wrong)
        s_right.add(right)
        s_wrong.add(wrong)
        per_tile.append(
            {
                "tile": n,
                "called_ground": int(called.sum()),
                "right": t_right.n,
                "wrong": t_wrong.n,
                "auroc": auroc_from_histograms(t_right.hist, t_wrong.hist),
            }
        )
        print(f"  surrey {n}  auroc {per_tile[-1]['auroc']}", flush=True)

    # ---------------- San Francisco, every point of every area ----------------
    f_right, f_wrong = Group(), Group()
    per_area = []
    for p in sorted(TARGET_PREP.iterdir()):
        if not (p.is_dir() and (p / "meta.json").exists()):
            continue
        tile = CachedTile(p, "sf")
        pred, conf, _, seen = infer_tile(
            net, tile, dev, args.block_m, args.points, None, seed=args.seed
        )
        asprs = np.asarray(tile.asprs[:], dtype=np.int16)
        called = seen & ~np.isin(asprs, ASPRS_NOISE) & (pred == GROUND)
        right = conf[called & (asprs == ASPRS_GROUND)]
        wrong = conf[called & (asprs == ASPRS_WATER)]
        a_right, a_wrong = Group(), Group()
        a_right.add(right)
        a_wrong.add(wrong)
        f_right.add(right)
        f_wrong.add(wrong)
        codes, counts = np.unique(asprs[called], return_counts=True)
        per_area.append(
            {
                "aoi": tile.name,
                "called_ground": int(called.sum()),
                "called_ground_by_delivery_class": {
                    str(int(c)): int(k) for c, k in zip(codes, counts, strict=True)
                },
                "right_vendor_ground": a_right.n,
                "wrong_water": a_wrong.n,
                "auroc": auroc_from_histograms(a_right.hist, a_wrong.hist),
            }
        )
        print(
            f"  sf {tile.name}  ground {a_right.n:,}  water {a_wrong.n:,}  "
            f"auroc {per_area[-1]['auroc']}",
            flush=True,
        )

    in_domain = auroc_from_histograms(s_right.hist, s_wrong.hist)
    out_pooled = auroc_from_histograms(f_right.hist, f_wrong.hist)
    emb = next((a for a in per_area if a["aoi"] == "embarcadero"), None)

    sr, sw, fr, fw = s_right.summary(), s_wrong.summary(), f_right.summary(), f_wrong.summary()
    gap_home = round(sr["max_prob_median"] - sw["max_prob_median"], 4)
    gap_away = round(fr["max_prob_median"] - fw["max_prob_median"], 4)

    result = {
        "question": "does confidence still flag the model's own errors in a new city",
        "task": "among points the model calls ground, rank right calls above wrong calls",
        "score": "maximum softmax probability, float32",
        "checkpoint_step": int(blob["step"]),
        "seed": args.seed,
        "device": dev,
        "device_name": (
            torch.cuda.get_device_name(0)
            if dev == "cuda"
            else platform.processor() or platform.machine()
        ),
        "torch": torch.__version__,
        "asymmetry": (
            "at home the wrong side is every class mistaken for ground; in San "
            "Francisco it is water only, the one certain wrong answer"
        ),
        "surrey_full_pass": {
            "population": "held-out DALES-2 tiles, every voxel-thinned labelled point",
            "points_labelled": labelled_total,
            "accuracy": round(correct_total / labelled_total, 4),
        },
        "in_domain": {
            "auroc": None if in_domain is None else round(in_domain, 4),
            "right": sr,
            "wrong": sw,
            "median_gap_right_minus_wrong": gap_home,
            "per_tile": per_tile,
        },
        "out_of_domain": {
            "auroc_pooled": None if out_pooled is None else round(out_pooled, 4),
            "auroc_embarcadero_only": None
            if emb is None or emb["auroc"] is None
            else round(emb["auroc"], 4),
            "pooling_note": (
                "pooled mixes ground from seven areas with water from two; "
                "Embarcadero is the only area holding both"
            ),
            "right": fr,
            "wrong": fw,
            "median_gap_right_minus_wrong": gap_away,
            "per_area": per_area,
        },
        "figure_histograms": {
            "bins": FIGURE_BINS,
            "surrey_right": s_right.figure(),
            "surrey_wrong": s_wrong.figure(),
            "sf_vendor_ground": f_right.figure(),
            "sf_water": f_wrong.figure(),
        },
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\nfull held-out pass accuracy {result['surrey_full_pass']['accuracy']}")
    print(f"AUROC in domain            {result['in_domain']['auroc']}")
    print(f"AUROC San Francisco pooled {result['out_of_domain']['auroc_pooled']}")
    print(f"AUROC Embarcadero alone    {result['out_of_domain']['auroc_embarcadero_only']}")
    print(f"median gap right minus wrong: home {gap_home}  away {gap_away}")
    print(
        f"under 0.5: home wrong {sw['points_under_0_5']:,} of {sw['points']:,}, "
        f"away water {fw['points_under_0_5']:,} of {fw['points']:,}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
