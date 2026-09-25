"""Train the segmentation model on Surrey and score it on held-out Surrey tiles.

Device is chosen at runtime and recorded in every artefact this writes, because
requirements.txt pins a CPU wheel so that CI can run the suite on a GitHub runner
with no GPU. A number produced on the laptop's RTX 3050 and a number produced on
a runner's CPU are not interchangeable, so each one says which it is.

Resumable, exactly. A run that dies at step 4,000 restarts from its last checkpoint with the
same schedule and the same batches, because the checkpoint carries the step, the optimiser and
scaler, and every random state the loop touches (surrey.resume). Checkpoints written before that
change hold no random state; resuming one is recorded as not exact.

Per-class IoU only. Four of the fifteen classes sit under half a percent of the
points and will hover near zero however long this trains; a mean over fifteen
classes moves when those four twitch and says nothing about the question.

Recipes (surrey.sources.RECIPES), all three channels, same steps and voxel:

    m0       DALES-2 alone
    m0prime  DALES-2 and the Swiss sub-tiles, Swiss water and bridge deck ignored
    m1       the same data, water and bridge deck as classes

Seeds are 0, 1, 2 for A, B, C. Seed A is the published model of each recipe, decided
before any result.

Run:
    python scripts/pipeline/train.py --recipe m0prime --seed 0 --ckpt data/checkpoints/m0prime_seedA.pt --out data/artifacts/train_m0prime_seedA.json
    python scripts/pipeline/train.py --channels 3 --seed 0 --ckpt data/checkpoints/m0_seedA.pt --out data/artifacts/train_m0_seedA.json
    python scripts/pipeline/train.py --channels 3 --seed 0 --ckpt data/checkpoints/m0_seedA.pt --out data/artifacts/train_m0_seedA.json --resume

The checkpoint at data/checkpoints/dayton.pt was trained with --channels 4 before that
option existed; the fourth channel is 0 on every DALES-2 point.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import CachedTile  # noqa: E402
from surrey.data import BLOCK_M, CLASS_NAMES, N_CLASSES, POINTS_PER_CROP  # noqa: E402
from surrey.metrics import Confusion  # noqa: E402
from surrey.model import SurreyNet, parameter_count  # noqa: E402
from surrey.resume import capture, interruption_record, restore, resume_event  # noqa: E402
from surrey.sources import (  # noqa: E402
    BRIDGE_DECK,
    RECIPES,
    WATER,
    class_names_for,
    recipe_ignore_mask,
    superclass_nll,
)

PREP = ROOT / "data" / "prep" / "dales"
SWISS_PREP = ROOT / "data" / "prep" / "swiss"
SWISS_ARTIFACT = ROOT / "data" / "artifacts" / "prep_swiss.json"
WATER_IGNORE = ROOT / "data" / "prep" / "dales_water_ignore"
PREP_ARTIFACT = ROOT / "data" / "artifacts" / "prep_dales.json"
CHECKPOINTS = ROOT / "data" / "checkpoints"
ARTIFACTS = ROOT / "data" / "artifacts"
HOLD_NOTE_SECONDS = 300
# The longest planned pause is the prototype measurement, three repeats of a few minutes each, well
# under half an hour. A hold past 45 minutes is almost certainly a hold left behind.
HOLD_WARN_SECONDS = 45 * 60


def device_record(dev: str) -> dict:
    if dev == "cuda":
        p = torch.cuda.get_device_properties(0)
        name, mem = p.name, round(p.total_memory / 1e9, 2)
    else:
        name, mem = platform.processor() or platform.machine(), None
    return {
        "device": dev,
        "device_name": name,
        "device_total_gb": mem,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def class_weights(stats_path: Path, power: float = 0.5) -> torch.Tensor:
    """Inverse frequency to a fractional power, rescaled to mean one.

    Rescaling matters: it keeps the loss in the same units as an unweighted run,
    so the two are comparable. The power is a blunt instrument against a 3,000 to
    1 imbalance between ground and traffic poles, chosen over ignoring the rare
    classes (which stops the model learning them at all) and over full inverse
    frequency (which makes a handful of points dominate every batch).

    The fractions are those of the 29 training tiles only. The eleven held-out tiles are
    what the model is scored on, and nothing it is trained with may be read from them.
    """
    freq = training_fractions(stats_path)
    w = (1.0 / freq) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def training_fractions(stats_path: Path) -> np.ndarray:
    """Class fractions of the DALES-2 training split, in CLASS_NAMES order. A missing file stops the
    run: training on equal weights by accident looks like a result, not like an error."""
    if not stats_path.exists():
        raise SystemExit(f"{stats_path} is missing; run scripts/pipeline/dales_stats.py")
    split = json.loads(stats_path.read_text())["per_split"]["train"]
    if split["tiles"] != 29:
        raise SystemExit(f"{stats_path} counts {split['tiles']} training tiles, expected 29")
    return np.array([max(split["fractions"][n], 1e-9) for n in CLASS_NAMES])


def m1_class_weights(stats_path: Path, swiss_path: Path, power: float = 0.5) -> torch.Tensor:
    """M0's fifteen weights unchanged, and water and bridge deck by the same formula.

    The DALES weights come from the same training fractions M0 uses, so the one thing M1
    changes about the fifteen classes is the data. The two new classes take their
    fraction over all M1 training points. The loss divides by the summed weight, so the
    scale of the vector does not matter, only the ratios.
    """
    w15 = (1.0 / training_fractions(stats_path)) ** power
    swiss = json.loads(swiss_path.read_text())
    dales_points = sum(
        t["points_kept"]
        for t in json.loads(PREP_ARTIFACT.read_text())["tiles"]
        if t["split"] == "train"
    )
    total = dales_points + swiss["points_kept_total"]
    extra = []
    for name in ("water", "bridge_deck"):
        n = sum(t["points_kept"] * t["label_mix"].get(name, 0.0) for t in swiss["tiles"])
        extra.append((1.0 / max(n / total, 1e-9)) ** power)
    w = np.concatenate([w15, extra])
    return torch.tensor(w / w[:15].mean(), dtype=torch.float32)


def split_names(artifact: Path) -> tuple[list[str], list[str]]:
    spec = json.loads(artifact.read_text())["tiles"]
    train = [t["name"] for t in spec if t["split"] == "train"]
    test = [t["name"] for t in spec if t["split"] == "test"]
    return sorted(train), sorted(test)


@torch.no_grad()
def evaluate(
    net,
    tiles: list[CachedTile],
    dev: str,
    block_m: float,
    n_points: int,
    max_blocks: int | None = None,
    batch: int = 4,
    channels: int = 4,
    names: list[str] | None = None,
) -> Confusion:
    """Score every point of every tile, in batches.

    Batched because the full pass covers roughly 110 million points and gets run
    more than once. One block at a time left the card idle between kernels.

    `max_blocks` is for the periodic look during training and is spread across
    each tile. The final number in the artefact is always a full pass.
    """
    net.eval()
    conf = Confusion(names)
    amp = dev == "cuda"

    def flush(pending: list[tuple[np.ndarray, np.ndarray, int]]) -> None:
        if not pending:
            return
        x = torch.from_numpy(np.stack([p[0] for p in pending])).to(dev, non_blocking=True)
        with torch.amp.autocast(dev, dtype=torch.float16, enabled=amp):
            pred = net(x).argmax(dim=-1).cpu().numpy()
        for row, (_, label, valid) in enumerate(pending):
            conf.add(label[:valid], pred[row, :valid])

    for tile in tiles:
        pending: list[tuple[np.ndarray, np.ndarray, int]] = []
        # max_blocks goes down into the tile so it spreads the subset across the
        # whole thing. Breaking out of the generator here would sample the corner
        # the blocks happen to start from.
        for feat, label, _, valid in tile.iter_full(
            block_m, n_points, max_blocks, channels=channels
        ):
            pending.append((feat, label, valid))
            if len(pending) == batch:
                flush(pending)
                pending = []
        flush(pending)
    net.train()
    return conf


def repo_relative(path: Path) -> str:
    """A path as the repository sees it, with forward slashes; outside the repository, as given."""
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--points", type=int, default=POINTS_PER_CROP)
    ap.add_argument("--block-m", type=float, default=BLOCK_M)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--recipe",
        default="m0",
        choices=RECIPES,
        help="m0 DALES alone; m0prime adds the Swiss sub-tiles with their water and "
        "bridge deck ignored; m1 adds them as classes. DALES water is ignored "
        "whenever Swiss data is in",
    )
    ap.add_argument(
        "--channels",
        type=int,
        default=3,
        choices=(3, 4),
        help="3 is x, y, z; 4 adds the return ratio, which is 0 on every "
        "DALES-2 point and was the input that broke the transfer",
    )
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument(
        "--ckpt-every",
        type=int,
        default=250,
        help="save without evaluating, so a killed run leaves something",
    )
    ap.add_argument(
        "--final-eval-blocks",
        type=int,
        default=0,
        help="blocks per test tile in the final evaluation, 0 for every block; only "
        "a smoke run sets this",
    )
    ap.add_argument(
        "--eval-blocks", type=int, default=6, help="blocks per test tile during training, 0 for all"
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="default data/checkpoints/<recipe>_seed<A, B or C>.pt",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default data/artifacts/train_<recipe>_seed<A, B or C>.json",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--prep",
        type=Path,
        default=PREP,
        help="prepared training tiles; tests point it at synthetic ones",
    )
    ap.add_argument("--split-artifact", type=Path, default=PREP_ARTIFACT)
    ap.add_argument(
        "--stop-after-step",
        type=int,
        default=0,
        help="save a checkpoint after this step and exit, as a crash would; for the resume test",
    )
    # Water was learned at about 11.55 points per square metre, what a crop carries on the Swiss tiles,
    # and is asked to work at 20.48, what a crop carries on San Francisco and on Surrey. This trains
    # with the crop's density drawn at random across that gap and below it, so the class cannot be
    # learned at one spacing only. The range is in distinct points per crop; 12800 to 32768 over a 40 m
    # block is 8.0 to 20.48 points per square metre.
    ap.add_argument(
        "--crop-distinct-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="draw each crop's distinct point count uniformly in this range",
    )
    ap.add_argument(
        "--batch-log",
        type=Path,
        default=None,
        help="append one SHA-256 of each step's input batch, to compare batch order across runs",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--ignore-hold",
        action="store_true",
        help="for a short check run inside a pause by the process that holds it",
    )
    args = ap.parse_args()
    variant = f"{args.recipe}_seed{'ABC'[args.seed] if 0 <= args.seed < 3 else args.seed}"
    args.ckpt = args.ckpt or CHECKPOINTS / f"{variant}.pt"
    args.out = args.out or ARTIFACTS / f"train_{variant}.json"

    # A pause between trainings for measurements that need the card to themselves: while
    # data/logs/HOLD exists, nothing touches the GPU. The wait is recorded in the artefact.
    hold = ROOT / "data" / "logs" / "HOLD"
    held_since = time.time()
    # The hold keeps the card free; a run pinned to the CPU (the resume test) never waits on it, and
    # neither does a check run inside the pause by the script holding it.
    if args.device == "cpu" or args.ignore_hold:
        hold = ROOT / "data" / "logs" / "HOLD_does_not_apply_to_this_run"
    if hold.exists():
        print(f"held by {hold}, waiting", flush=True)
    last_note = held_since
    while hold.exists():
        time.sleep(20)
        # Say so every few minutes, so a hold left behind is visible in the log and not a silence.
        if time.time() - last_note >= HOLD_NOTE_SECONDS:
            minutes = round((time.time() - held_since) / 60)
            print(f"still held by {hold}, {minutes} min so far", flush=True)
            last_note = time.time()
            if time.time() - held_since >= HOLD_WARN_SECONDS:
                # Training does not start by itself; the chain log says plainly that it is standing still.
                with (ROOT / "data" / "logs" / "gpu_chain.log").open(
                    "a", encoding="utf-8", newline="\n"
                ) as fh:
                    fh.write(
                        f"WARNING {time.strftime('%Y-%m-%d %H:%M:%S')}: {args.ckpt.stem} held by HOLD for "
                        f"{minutes} min and not training\n"
                    )
    held_seconds = round(time.time() - held_since, 1)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rec = device_record(dev)
    print(f"device {rec['device']}  {rec['device_name']}  torch {rec['torch']}")

    train_names, test_names = split_names(args.split_artifact)
    train_tiles = [CachedTile(args.prep / n, "train") for n in train_names]
    test_tiles = [CachedTile(args.prep / n, "test") for n in test_names]
    names = class_names_for(args.recipe)
    swiss_names: list[str] = []
    ignored_water = 0
    swiss_extra_ignored = 0
    if args.recipe != "m0":
        for t in train_tiles:
            mask_path = WATER_IGNORE / f"{t.name}.npy"
            if mask_path.exists():
                t.ignore_mask = np.load(mask_path)
                ignored_water += int(t.ignore_mask.sum())
        swiss_names = [r["name"] for r in json.loads(SWISS_ARTIFACT.read_text())["tiles"]]
        # Every tile is 500 m square in both sources, so drawing a tile uniformly draws
        # area uniformly, as M0 did over DALES alone.
        swiss_tiles = [CachedTile(SWISS_PREP / n, "train") for n in swiss_names]
        for t in swiss_tiles:
            mask = recipe_ignore_mask(args.recipe, np.asarray(t.label[:]))
            if mask is not None:
                t.ignore_mask = mask
                swiss_extra_ignored += int(mask.sum())
        train_tiles += swiss_tiles
        print(
            f"{args.recipe}: {len(swiss_names)} swiss sub-tiles, {ignored_water:,} DALES water points "
            f"ignored, {swiss_extra_ignored:,} swiss water and bridge deck points ignored"
        )
    print(
        f"tiles  {len(train_tiles)} train, {len(test_tiles)} test, "
        f"{sum(t.n for t in train_tiles):,} training points"
    )

    torch.manual_seed(args.seed)
    n_out = len(names)
    net = SurreyNet(n_out, width=args.width, extra_features=args.channels - 3).to(dev)
    stats = ROOT / "data" / "artifacts" / "dales_stats.json"
    weights = (
        m1_class_weights(stats, SWISS_ARTIFACT) if args.recipe == "m1" else class_weights(stats)
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")

    start = 0
    history: list[dict] = []
    interruptions: list[dict] = []
    repeated_crops = 0
    rng = np.random.default_rng(args.seed)
    running = 0.0
    if args.resume and args.ckpt.exists():
        blob = torch.load(args.ckpt, map_location=dev, weights_only=False)
        net.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        scaler.load_state_dict(blob["scaler"])
        start = int(blob["step"])
        history = blob.get("history", [])
        interruptions = list(blob.get("interruptions", []))
        repeated_crops = int(blob.get("repeated_crops", 0))
        running = float(blob.get("running", 0.0))
        if "rng" in blob:
            restore(blob["rng"], rng)
            interruptions.append(resume_event(start, True, "every random state restored"))
        else:
            # A checkpoint from before random states were saved: the old behaviour, recorded as such.
            rng = np.random.default_rng(args.seed + start)
            interruptions.append(
                resume_event(
                    start,
                    False,
                    "no random state in the checkpoint; sampler reseeded "
                    "with seed plus step, torch generators fresh",
                )
            )
        print(f"resumed from {args.ckpt} at step {start}, exact: {interruptions[-1]['exact']}")

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    amp = dev == "cuda"

    def save(step_done: int) -> None:
        torch.save(
            {
                "model": net.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step_done,
                "history": history,
                "rng": capture(rng),
                "repeated_crops": repeated_crops,
                "interruptions": interruptions,
                "running": running,
                "args": vars(args) | {"ckpt": str(args.ckpt), "out": str(args.out)},
            },
            args.ckpt,
        )

    t_start = time.time()
    t_window = t_start

    for step in range(start, args.steps):
        feats, labels = [], []
        for _ in range(args.batch):
            tile = train_tiles[rng.integers(len(train_tiles))]
            distinct = (
                int(rng.integers(args.crop_distinct_range[0], args.crop_distinct_range[1] + 1))
                if args.crop_distinct_range
                else None
            )
            feat, label, rep = tile.random_crop(
                rng, args.block_m, args.points, args.channels, distinct
            )
            repeated_crops += int(rep)
            feats.append(feat)
            labels.append(label)
        batch = np.stack(feats)
        if args.batch_log is not None:
            import hashlib

            with args.batch_log.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(f"{step + 1} {hashlib.sha256(batch.tobytes()).hexdigest()}\n")
        x = torch.from_numpy(batch).to(dev, non_blocking=True)
        y = torch.from_numpy(np.stack(labels)).to(dev, non_blocking=True)

        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / max(args.steps, 1)))
        for gp in opt.param_groups:
            gp["lr"] = lr

        with torch.amp.autocast(dev, dtype=torch.float16, enabled=amp):
            logits = net(x)
            if args.recipe != "m0":
                loss = superclass_nll(
                    F.log_softmax(logits.reshape(-1, n_out).float(), dim=-1), y.reshape(-1), weights
                )
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, N_CLASSES), y.reshape(-1), weight=weights, ignore_index=-1
                )
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        running = loss.item() if step == 0 else 0.98 * running + 0.02 * loss.item()
        if (step + 1) % 50 == 0:
            done = step + 1 - start
            now = time.time()
            rate = done * args.batch * args.points / (now - t_start)
            eta = (args.steps - step - 1) / max(done / (now - t_start), 1e-9)
            # Steps per second over the last 50 steps alone. The run average above hides a
            # slowdown; this one shows something sharing the card within a minute.
            window = 50 / max(now - t_window, 1e-9)
            t_window = now
            print(
                f"  step {step + 1:>6}/{args.steps}  loss {running:.4f}  "
                f"lr {lr:.2e}  {rate / 1000:.0f}k pts/s  {window:.3f} steps/s  eta {eta / 60:.0f} min",
                flush=True,
            )

        is_eval = (step + 1) % args.eval_every == 0 or step + 1 == args.steps
        if is_eval:
            conf = evaluate(
                net,
                test_tiles,
                dev,
                args.block_m,
                args.points,
                args.eval_blocks or None,
                channels=args.channels,
                names=names,
            )
            iou = conf.iou()
            history.append(
                {
                    "step": step + 1,
                    "loss": round(running, 4),
                    "accuracy": round(conf.accuracy(), 4),
                    "iou": iou,
                }
            )
            shown = {
                k: v
                for k, v in iou.items()
                if k in ("ground", "tree", "vegetation", "habitat", "complex", "car")
            }
            print(
                f"  eval  step {step + 1}  acc {conf.accuracy():.4f}  "
                + "  ".join(f"{k}={v}" for k, v in shown.items()),
                flush=True,
            )

        # Save far more often than we evaluate. An earlier run was killed at step
        # 500 with the checkpoint interval at 1,500, so it left nothing behind and
        # a comparison that was worth making became unavailable. Saving is cheap;
        # the model is 726k parameters. The save comes after any evaluation at this
        # step, because evaluation draws random centroids too: a checkpoint taken
        # before it would resume with the generator one evaluation behind.
        if (step + 1) % args.ckpt_every == 0 or is_eval or step + 1 == args.stop_after_step:
            save(step + 1)
        if step + 1 == args.stop_after_step:
            print(f"stopped after step {step + 1}, as asked", flush=True)
            return 0

    print("\nfull evaluation on every held-out tile")
    conf = evaluate(
        net,
        test_tiles,
        dev,
        args.block_m,
        args.points,
        args.final_eval_blocks or None,
        channels=args.channels,
        names=names,
    )
    iou = conf.iou()
    support = conf.support()

    result = {
        **rec,
        "parameters": parameter_count(net),
        # Paths are recorded from the repository root: an absolute path would put this machine's user
        # name and folder layout into a committed artefact.
        "config": {
            k: (repo_relative(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "train_tiles": train_names,
        "test_tiles": test_names,
        "train_points": sum(t.n for t in train_tiles),
        "recipe": args.recipe,
        "taxonomy": "m1: dales15 + water + bridge_deck" if args.recipe == "m1" else "dales15",
        "swiss_tiles": swiss_names,
        "dales_water_points_ignored": ignored_water,
        "swiss_water_and_bridge_deck_points_ignored": swiss_extra_ignored,
        "class_weights": [round(float(v), 4) for v in weights.cpu()],
        "new_class_ids": {"water": WATER, "bridge_deck": BRIDGE_DECK}
        if args.recipe == "m1"
        else None,
        "held_before_start_seconds": held_seconds,
        "crop_distinct_range": args.crop_distinct_range,
        "crop_density_range_points_per_m2": (
            [
                round(args.crop_distinct_range[0] / (args.block_m**2), 2),
                round(args.crop_distinct_range[1] / (args.block_m**2), 2),
            ]
            if args.crop_distinct_range
            else None
        ),
        "interruptions": interruption_record(interruptions),
        "steps_trained": args.steps,
        "points_seen": args.steps * args.batch * args.points,
        "crops_needing_repeats": repeated_crops,
        "class_weight_power": 0.5,
        "input_channels": ["x", "y", "z_above_block_floor"]
        + (["return_ratio"] if args.channels == 4 else []),
        "final": {
            "accuracy": round(conf.accuracy(), 4),
            "iou_per_class": iou,
            "support_per_class": support,
            "confusion": conf.m.tolist(),
        },
        "history": history,
        "wall_seconds": round(time.time() - t_start, 1),
        "note": (
            "iou is per class and deliberately not averaged. four classes hold "
            "under 0.5 percent of points and a fifteen class mean moves with them."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"\n{'class':<14}{'IoU':>8}{'support':>14}")
    for name in names:
        v = iou[name]
        print(f"{name:<14}{'n/a' if v is None else f'{v:.4f}':>8}{support[name]:>14,}")
    print(f"\noverall point accuracy {conf.accuracy():.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
