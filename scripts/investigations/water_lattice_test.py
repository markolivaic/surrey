"""Does M1 read water by the 25 cm grid the Swiss water points sit on?

scripts/investigations/water_lattice.py measured that the points swisstopo labels water sit on a regular 25 cm grid,
in training and in the held-out Zug tile, and that San Francisco's water points and the Swiss ground
points do not. That is what the points are. This asks whether it is what M1 reads, with the one change
that separates the two: move the Zug water points off the grid and nothing else.

Conditions, on the four held-out Zug sub-tiles, for M1 seeds A, B and C:

  as_prepared      the tile as scored before
  water_jittered   every point labelled water moved in x and y by a uniform whole number of
                   centimetres in [-12, 12], half the grid step, kept inside its own 10 m cell so the
                   blocks gather the same points; z and every other point untouched
  ground_jittered  the control: the same move for the points labelled ground instead, which do not sit
                   on a grid, so it shows what moving points by that much does on its own

The reading, written 2026-09-23 before the first run:

  water IoU on Zug falls well below as_prepared under water_jittered, and does not fall that way under
      ground_jittered: M1's water depends on the grid the Swiss water points sit on, which San
      Francisco's water does not have. That is a reason for the San Francisco result that the spacing
      measurements did not give, and the text says so as measured, on these tiles.
  water IoU barely moves under water_jittered: the grid is not what M1 reads, and the rest of the San
      Francisco result stays unexplained.

The lattice share of the moved points is measured too, so the change is shown to have taken the grid
away rather than assumed to.

Every score carries the null it is read against, calling every point water: the labels and the number
of points are the same in all three conditions, so the null is too.

Run:
    python scripts/investigations/water_lattice_test.py
    python scripts/investigations/water_lattice_test.py --nulls-only     put the nulls into an existing artefact
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from score_swiss import prevalence_null  # noqa: E402
from surrey.cache import CachedTile  # noqa: E402
from surrey.data import BLOCK_M, POINTS_PER_CROP  # noqa: E402
from surrey.inference import DEFAULT_SEED, infer_tile  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402
from surrey.sources import M1_CLASS_NAMES, WATER  # noqa: E402
from water_lattice import reading  # noqa: E402

ZUG = ROOT / "data" / "prep" / "swiss_holdout"
WORK = ROOT / "data" / "cache" / "zug_lattice"
OUT = ROOT / "data" / "artifacts" / "water_lattice_test.json"
GROUND = M1_CLASS_NAMES.index("ground")
JITTER_CM = 12
VARIANTS = ("m1_seedA", "m1_seedB", "m1_seedC")
READING = (
    "written 2026-09-23 before the first run: if water IoU on Zug falls well below as_prepared when"
    " the water points are moved off their 25 cm grid, and does not fall that way when ground points"
    " are moved the same distance, M1's water depends on the grid; if it barely moves, the grid is not"
    " what M1 reads and the rest of the San Francisco result stays unexplained"
)


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    called, is_w = pred == WATER, truth == WATER
    tp = int((called & is_w).sum())
    c, s = int(called.sum()), int(is_w.sum())
    u = c + s - tp
    return {
        "support": s,
        "called": c,
        "true_positive": tp,
        "precision": round(tp / c, 4) if c else None,
        "recall": round(tp / s, 4) if s else None,
        "iou": round(tp / u, 4) if u else None,
    }


def jittered_copy(src: Path, dst: Path, cls: int, rng: np.random.Generator) -> dict:
    """The tile with one class's points moved, each kept inside its own cell."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    cell_cm = int(round(meta["cell_m"] * 100))
    xy = np.load(src / "xy_cm.npy").astype(np.int64)
    label = np.load(src / "label.npy")
    sel = np.nonzero(label == cls)[0]
    before = reading(xy[sel].astype(np.float64))
    move = rng.integers(-JITTER_CM, JITTER_CM + 1, size=(sel.size, 2))
    lo = (xy[sel] // cell_cm) * cell_cm
    xy[sel] = np.clip(xy[sel] + move, lo, lo + cell_cm - 1)
    np.save(dst / "xy_cm.npy", xy.astype(np.load(src / "xy_cm.npy", mmap_mode="r").dtype))
    after = reading(xy[sel].astype(np.float64))
    return {
        "moved": int(sel.size),
        "lattice_share_before": before.get("share_on_the_lattice"),
        "lattice_share_after": after.get("share_on_the_lattice"),
        "chance_share": after.get("chance_share"),
    }


def zug_points(names: list[str]) -> int:
    return sum(
        int(json.loads((ZUG / n / "meta.json").read_text(encoding="utf-8"))["points"])
        for n in names
    )


def add_nulls(result: dict) -> dict:
    points = zug_points(result["sub_tiles"])
    result["points"] = points
    for row in result["variants"].values():
        for score in row.values():
            score.update(prevalence_null(score["support"], points))
    return result


def main() -> int:
    if "--nulls-only" in sys.argv:
        OUT.write_text(
            json.dumps(add_nulls(json.loads(OUT.read_text(encoding="utf-8"))), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"nulls written into {OUT}")
        return 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = sorted(p.name for p in ZUG.iterdir() if p.is_dir())
    conditions = {
        "as_prepared": (None, None),
        "water_jittered": (WATER, 1),
        "ground_jittered": (GROUND, 2),
    }
    dirs, moved = {}, {}
    for cond, (cls, seed) in conditions.items():
        if cls is None:
            dirs[cond] = ZUG
            continue
        rng = np.random.default_rng(seed)
        dirs[cond] = WORK / cond
        moved[cond] = {n: jittered_copy(ZUG / n, WORK / cond / n, cls, rng) for n in names}
        print(cond, moved[cond], flush=True)

    result = {
        "reading_written_before_running": READING,
        "jitter_cm": JITTER_CM,
        "device": dev,
        "sub_tiles": names,
        "moved": moved,
        "variants": {},
    }
    for variant in VARIANTS:
        blob = torch.load(
            ROOT / "data" / "checkpoints" / f"{variant}.pt", map_location=dev, weights_only=False
        )
        channels = int(blob.get("args", {}).get("channels", 3))
        n_out = int(blob["model"]["head.4.bias"].shape[0])
        net = SurreyNet(n_out, width=128, extra_features=channels - 3).to(dev)
        net.load_state_dict(blob["model"])
        net.eval()
        row = {}
        for cond in conditions:
            preds, truths = [], []
            t0 = time.time()
            for n in names:
                tile = CachedTile(dirs[cond] / n, "train")
                pred, _c, _e, seen = infer_tile(
                    net,
                    tile,
                    dev,
                    BLOCK_M,
                    POINTS_PER_CROP,
                    None,
                    batch=4,
                    seed=DEFAULT_SEED,
                    channels=channels,
                )
                if not seen.all():
                    raise SystemExit(
                        f"{cond} {n}: {int((~seen).sum())} points were never predicted"
                    )
                preds.append(pred.astype(np.int16))
                truths.append(np.asarray(tile.label[:], np.int16))
            row[cond] = scores(np.concatenate(preds), np.concatenate(truths)) | {
                "seconds": round(time.time() - t0, 1)
            }
            print(variant, cond, row[cond], flush=True)
        result["variants"][variant] = row
    OUT.write_text(json.dumps(add_nulls(result), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
