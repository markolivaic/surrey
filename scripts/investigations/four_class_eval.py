"""In-domain scores on the four classes a national set and San Francisco can share.

The fifteen DALES-2 classes and the six coarse groups are both finer than what any
target in this project labels. A multi-source model will be scored on ground,
vegetation, building and everything else, so the in-domain figure it is compared
against has to be on the same four. Regrouped from the held-out confusion matrix in
data/artifacts/train_dales.json, with the do-nothing baseline on the same points.

Run:
    python scripts/investigations/four_class_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.data import CLASS_NAMES  # noqa: E402

ARTIFACT = ROOT / "data" / "artifacts" / "train_dales.json"
OUT = ROOT / "data" / "artifacts" / "four_class_eval.json"
FOUR = {
    "ground": ["ground"],
    "vegetation": ["vegetation", "tree"],
    "building": ["habitat", "complex", "annex"],
    "other": [
        "car",
        "pickup",
        "van-truck",
        "heavy-duty",
        "utility-pole",
        "light-pole",
        "traffic-pole",
        "powerline",
        "fence",
    ],
}


def regroup(m: np.ndarray) -> np.ndarray:
    names = list(FOUR)
    index = {CLASS_NAMES.index(c): names.index(g) for g, cs in FOUR.items() for c in cs}
    if sorted(index) != list(range(len(CLASS_NAMES))):
        raise SystemExit("the four groups do not cover every class exactly once")
    out = np.zeros((len(names), len(names)), np.int64)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            out[index[i], index[j]] += m[i, j]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    final = json.loads(ARTIFACT.read_text(encoding="utf-8"))["final"]
    m = np.asarray(final["confusion"], np.int64)
    c = regroup(m)
    tp = np.diag(c)
    union = c.sum(0) + c.sum(1) - tp
    names = list(FOUR)
    support = c.sum(1)
    result = {
        "population": "held-out Surrey tiles, voxel-thinned, the points of the 0.8948 accuracy",
        "rows_are": "reference class; columns are prediction",
        "groups": FOUR,
        "points": int(c.sum()),
        "accuracy": round(float(tp.sum() / c.sum()), 4),
        "do_nothing_accuracy": round(float(support.max() / c.sum()), 4),
        "iou": {n: round(float(tp[i] / union[i]), 4) for i, n in enumerate(names)},
        "mean_iou": round(float(np.mean(tp / union)), 4),
        "support_share": {n: round(float(support[i] / c.sum()), 4) for i, n in enumerate(names)},
        "confusion": c.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps({k: result[k] for k in ("accuracy", "do_nothing_accuracy", "iou", "mean_iou")})
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
