"""Does M1 know water at all? The same model, scored on the Swiss sub-tiles it learned water from.

Nothing on San Francisco can answer this. "M1 does not recognise water in San Francisco" and "M1 never
learned water" produce the same San Francisco figures and mean opposite things, so water is scored here
on the data the class came from.

In-sample, and labelled as such. Every one of the sixteen swissSURFACE3D sub-tiles is a training tile
(scripts/pipeline/train.py takes the whole list in data/artifacts/prep_swiss.json), so there is no held-out Swiss
part and this is not evidence of generalisation. It is an upper bound on what the model can do with
water: a class that scores badly on its own training data was not learned.

The reading, written before the first run (2026-09-20):

  high water IoU here  the class was learned, and the collapse on San Francisco is a real transfer
                       finding about a class the model holds
  low water IoU here   the class was never learned, which is a training fault and not a transfer
                       finding, and the fault is then what gets chased

M0prime is scored the same way as a control on the scoring code. It has no output for water or bridge
deck at all, so it must call water on no point; anything else means this script, not the model, is wrong.

Also reported, because they bear on the same question: how many water and bridge deck points the
training saw and their share, the loss weight each class carried and how it was computed, and a check
by code that the class ids in training, in the cached Swiss labels and in scoring are the same numbers.

Run:
    python scripts/pipeline/score_swiss.py --variants m1_seedA m0prime_seedA
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
from surrey.data import BLOCK_M, CLASS_NAMES, POINTS_PER_CROP  # noqa: E402
from surrey.inference import DEFAULT_SEED, infer_tile  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402
from surrey.sources import (  # noqa: E402
    BRIDGE_DECK,
    BUILDING_SUPER,
    IGNORE,
    M1_CLASS_NAMES,
    SWISS_2024,
    WATER,
)

SWISS_PREP = ROOT / "data" / "prep" / "swiss"
SWISS_ARTIFACT = ROOT / "data" / "artifacts" / "prep_swiss.json"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
ARTIFACTS = ROOT / "data" / "artifacts"
OUT = ARTIFACTS / "swiss_scores.json"
READING = (
    "written 2026-09-20 before the first run: high water IoU here means the class was learned and "
    "the San Francisco collapse is a transfer finding; low water IoU here means it was never "
    "learned, which is a training fault to chase instead"
)


def scores(pred: np.ndarray, truth: np.ndarray, cls: int) -> dict:
    """Precision, recall and IoU for one class over the points that carry a label."""
    called, is_class = pred == cls, truth == cls
    tp = int((called & is_class).sum())
    called_n, support = int(called.sum()), int(is_class.sum())
    union = called_n + support - tp
    return {
        "support": support,
        "called": called_n,
        "true_positive": tp,
        "precision": round(tp / called_n, 4) if called_n else None,
        "recall": round(tp / support, 4) if support else None,
        "iou": round(tp / union, 4) if union else None,
    }


def prevalence_null(support: int, points: int) -> dict:
    """The null every class score here is read against: call every point this class.

    On a sub-tile that is mostly lake, calling everything water already scores most of the available
    precision, so a bare 0.959 would say nothing without this beside it.
    """
    share = support / points if points else None
    return {
        "null": "call every point this class",
        "null_precision": round(share, 4) if share is not None else None,
        "null_recall": 1.0 if support else None,
        "null_iou": round(share, 4) if share is not None else None,
    }


def add_nulls(result: dict) -> dict:
    """Put the prevalence null beside every score already measured, from the counts in the artefact."""
    for row in result.get("variants", {}).values():
        points = 0
        for tile in row.get("per_sub_tile", {}).values():
            points += tile["points"]
            for cls in ("water", "bridge_deck"):
                tile[cls].update(prevalence_null(tile[cls]["support"], tile["points"]))
        for cls in ("water", "bridge_deck"):
            row["pooled"][cls].update(prevalence_null(row["pooled"][cls]["support"], points))
        row["points"] = points
    return result


def mapping_check() -> dict:
    """The same class id in the taxonomy, in the cached Swiss labels, and in each training artefact."""
    ids = {"water": WATER, "bridge_deck": BRIDGE_DECK}
    expected = {IGNORE, BUILDING_SUPER} | {v[1] for v in SWISS_2024.values()}
    # The prepared Swiss tiles are downloaded, not committed: in a fresh clone the ids are checked
    # against the taxonomy and the committed training artefacts, and the cached labels where they exist.
    seen: set[int] | None = None
    if SWISS_PREP.exists():
        seen = set()
        for name in sorted(p.name for p in SWISS_PREP.iterdir() if p.is_dir()):
            seen |= {
                int(v)
                for v in np.unique(np.asarray(CachedTile(SWISS_PREP / name, "train").label[:]))
            }
    artefacts = {}
    for art in sorted(ARTIFACTS.glob("train_m1_seed*.json")):
        blob = json.loads(art.read_text(encoding="utf-8"))
        artefacts[art.stem] = {
            "new_class_ids": blob.get("new_class_ids"),
            "class_weight_count": len(blob.get("class_weights", [])),
        }
    manifests = {}
    for m in sorted((ROOT / "data" / "predictions").glob("m1_seed*/manifest.json")):
        manifests[m.parent.name] = json.loads(m.read_text(encoding="utf-8"))["classes"]
    labels_agree = seen is None or (seen <= expected and WATER in seen and BRIDGE_DECK in seen)
    agree = (
        bool(artefacts)
        and all(a["new_class_ids"] == ids for a in artefacts.values())
        and all(a["class_weight_count"] == len(M1_CLASS_NAMES) for a in artefacts.values())
        and all(c == len(M1_CLASS_NAMES) for c in manifests.values())
        and labels_agree
    )
    return {
        "taxonomy": dict(enumerate(M1_CLASS_NAMES)),
        "water_id": WATER,
        "bridge_deck_id": BRIDGE_DECK,
        "ids_in_cached_swiss_labels": sorted(seen) if seen is not None else None,
        "cached_labels_checked": seen is not None,
        "ids_the_table_allows": sorted(expected),
        "training_artefacts": artefacts,
        "prediction_manifest_classes": manifests,
        "all_agree": bool(agree),
    }


def training_share() -> dict:
    """How many water and bridge deck points the training saw, and their share of all training points."""
    swiss = json.loads(SWISS_ARTIFACT.read_text(encoding="utf-8"))
    dales = sum(
        t["points_kept"]
        for t in json.loads(PREP_ARTIFACT.read_text(encoding="utf-8"))["tiles"]
        if t["split"] == "train"
    )
    total = dales + swiss["points_kept_total"]
    out = {
        "dales_points": dales,
        "swiss_points": swiss["points_kept_total"],
        "training_points": total,
        "classes": {},
    }
    for name in ("water", "bridge_deck"):
        n = sum(t["points_kept"] * t["label_mix"].get(name, 0.0) for t in swiss["tiles"])
        out["classes"][name] = {
            "points": round(n),
            "share_of_training_points": round(n / total, 6),
            "sub_tiles_with_any": sum(
                1 for t in swiss["tiles"] if t["label_mix"].get(name, 0.0) > 0
            ),
        }
    weights = {}
    for art in sorted(ARTIFACTS.glob("train_m1_seed*.json")):
        w = json.loads(art.read_text(encoding="utf-8"))["class_weights"]
        weights[art.stem] = {
            "water": w[WATER],
            "bridge_deck": w[BRIDGE_DECK],
            "ground": w[CLASS_NAMES.index("ground")],
            "mean_of_the_fifteen": round(float(np.mean(w[: len(CLASS_NAMES)])), 4),
        }
    out["loss_weights"] = weights
    out["loss_weight_formula"] = (
        "inverse frequency to the power 0.5: the fifteen DALES weights from the "
        "fractions of the 29 DALES training tiles, rescaled to mean one, and water and "
        "bridge deck from their share of all M1 training points by the same "
        "formula (m1_class_weights in scripts/pipeline/train.py)"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["m1_seedA", "m0prime_seedA"])
    # The held-out tile sits in its own prepared directory with its own artefact, and is scored by the
    # same code as the training tiles so the two figures can be read side by side.
    ap.add_argument("--prep", type=Path, default=SWISS_PREP)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--held-out", action="store_true", help="say in the artefact that this tile never trained"
    )
    ap.add_argument(
        "--nulls-only",
        action="store_true",
        help="put the prevalence null beside scores already measured, running no model",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    if args.nulls_only:
        blob = add_nulls(json.loads(OUT.read_text(encoding="utf-8")))
        OUT.write_text(json.dumps(blob, indent=2) + chr(10), encoding="utf-8", newline="\n")
        print(f"nulls written into {OUT}")
        return 0
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = sorted(q.name for q in args.prep.iterdir() if q.is_dir())
    result = {
        "reading_written_before_running": READING,
        "in_sample": not args.held_out,
        "in_sample_note": (
            "every Swiss sub-tile is a training tile, so this measures whether the "
            "class was learned at all, never generalisation"
        )
        if not args.held_out
        else (
            "a tile from a city none of the training tiles came from, chosen by the same rule after M1 "
            "was trained and scored, and never trained on: this one does measure generalisation"
        ),
        "prep": args.prep.name,
        "sub_tiles": len(names),
        "device": dev,
        "seed": args.seed,
        "class_id_check": mapping_check(),
        "training_share": training_share(),
        "variants": {},
    }
    print("class ids agree:", result["class_id_check"]["all_agree"], flush=True)

    for variant in args.variants:
        ckpt = ROOT / "data" / "checkpoints" / f"{variant}.pt"
        blob = torch.load(ckpt, map_location=dev, weights_only=False)
        channels = int(blob.get("args", {}).get("channels", 3))
        n_out = int(blob["model"]["head.4.bias"].shape[0])
        net = SurreyNet(n_out, width=args.width, extra_features=channels - 3).to(dev)
        net.load_state_dict(blob["model"])
        net.eval()
        row = {
            "checkpoint_step": int(blob["step"]),
            "outputs": n_out,
            "has_water_output": n_out > WATER,
            "has_bridge_deck_output": n_out > BRIDGE_DECK,
            "per_sub_tile": {},
            "pooled": {},
        }
        pooled_pred, pooled_truth = [], []
        t0 = time.time()
        for name in names:
            tile = CachedTile(args.prep / name, "train")
            pred, _conf, _ent, seen = infer_tile(
                net,
                tile,
                dev,
                BLOCK_M,
                POINTS_PER_CROP,
                None,
                batch=args.batch,
                seed=args.seed,
                channels=channels,
            )
            if not seen.all():
                raise SystemExit(f"{name}: {int((~seen).sum())} points were never predicted")
            truth = np.asarray(tile.label[:], np.int16)
            row["per_sub_tile"][name] = {
                "points": tile.n,
                "water": scores(pred, truth, WATER),
                "bridge_deck": scores(pred, truth, BRIDGE_DECK),
            }
            pooled_pred.append(pred.astype(np.int16))
            pooled_truth.append(truth)
            print(
                f"  {variant} {name} water {row['per_sub_tile'][name]['water']['iou']} "
                f"deck {row['per_sub_tile'][name]['bridge_deck']['iou']}",
                flush=True,
            )
        p, t = np.concatenate(pooled_pred), np.concatenate(pooled_truth)
        row["pooled"] = {
            "water": scores(p, t, WATER),
            "bridge_deck": scores(p, t, BRIDGE_DECK),
            "note": "pooled over all sixteen sub-tiles, every one of them a training tile",
        }
        row["seconds"] = round(time.time() - t0, 1)
        result["variants"][variant] = row
        print(f"{variant} pooled water {row['pooled']['water']}", flush=True)

    add_nulls(result)
    control = result["variants"].get("m0prime_seedA", {}).get("pooled", {}).get("water", {})
    result["control_reading"] = {
        "m0prime_water_called": control.get("called"),
        "m0prime_water_iou": control.get("iou"),
        "passes": control.get("called") == 0,
        "why": (
            "M0prime has no water output, so it must call water on no point; anything else is a bug in "
            "this script rather than a finding about the model"
        ),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
