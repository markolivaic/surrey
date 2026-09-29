"""Per-point inference with probabilities, seeded so a rerun gives the same answer.

The network draws its set abstraction centroids at random (see model.random_sample).
Nothing seeded that draw, so every script that scored the same checkpoint got a
slightly different prediction, and the same quantity came back with different
values depending on which script produced it. A registry of quantities cannot hold
anything while that is true, so inference seeds the generator per tile, from a seed
that is recorded in whatever artefact the call ends up in.

Even seeded, one pass is one draw: the same checkpoint on the same points gives a different class to a
share of them from one seed to the next. Voting averages the softmax over several seeded passes, the
test-time voting point cloud segmentation models use, and the number of passes is chosen by the rule
in scripts/pipeline/choose_votes.py.
"""

from __future__ import annotations

import pathlib
import pickle
import types

import numpy as np
import torch

from .cache import CachedTile
from .data import N_CLASSES

DEFAULT_SEED = 0


class _PathTolerantUnpickler(pickle.Unpickler):
    """Checkpoints trained on Windows store their run's arguments with pathlib.WindowsPath, which
    cannot be made on Linux. Those arguments are only a record, so a stored path of either kind is read
    as a pure path; the weights are untouched."""

    def find_class(self, module, name):
        if module == "pathlib" and name in ("WindowsPath", "PosixPath"):
            return pathlib.PurePath
        return super().find_class(module, name)


_PICKLE = types.ModuleType("surrey_checkpoint_pickle")
_PICKLE.Unpickler = _PathTolerantUnpickler  # type: ignore[attr-defined]
_PICKLE.load = pickle.load  # type: ignore[attr-defined]
_PICKLE.loads = pickle.loads  # type: ignore[attr-defined]


def load_checkpoint(path, dev: str, width: int | None = None, allow_pickle: bool = False):
    """A checkpoint as the network it was trained as: its head's size, so M1's seventeen outputs are
    never cut to fifteen, its channel count and its width. Returns the network in eval mode, the
    stored blob and the channel count. A checkpoint that predates the channel option read four.

    By default only weights and plain values are read (torch.load with weights_only=True), which is
    what a file from elsewhere should get: the weights scripts/pipeline/export_weights.py writes. A
    training checkpoint also carries the run's random states and arguments, which only unpickling
    reads, and unpickling can run code; the project's own scripts pass allow_pickle=True for the
    checkpoints they trained themselves."""
    from .model import SurreyNet

    if allow_pickle:
        blob = torch.load(path, map_location=dev, weights_only=False, pickle_module=_PICKLE)
    else:
        try:
            blob = torch.load(path, map_location=dev, weights_only=True)
        except pickle.UnpicklingError as e:
            raise SystemExit(
                f"{path} carries more than weights. Load it with allow_pickle=True only if it was "
                "trained here, or read the weights file scripts/pipeline/export_weights.py writes."
            ) from e
    args = blob.get("args") or {}
    channels = int(args.get("channels") or 4)
    width = int(width or args.get("width") or 128)
    n_out = int(blob["model"]["head.4.bias"].shape[0])
    net = SurreyNet(n_out, width=width, extra_features=channels - 3).to(dev)
    net.load_state_dict(blob["model"])
    net.eval()
    return net, blob, channels


@torch.no_grad()
def vote_probabilities(
    net,
    tile: CachedTile,
    dev: str,
    block_m: float,
    n_points: int,
    max_blocks: int | None = None,
    batch: int = 4,
    seed: int = DEFAULT_SEED,
    votes: int = 1,
    z_override=None,
    ret_value: float | None = None,
    channels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The softmax per point averaged over `votes` passes, and which points were predicted.

    Pass v seeds both of the draws inference makes with seed + v: the network's set abstraction
    centroids and the dealing of each block into partitions (CachedTile.iter_full). One pass is what
    every script did before voting; more passes average out the answer that changes with the draw.

    Softmax runs in float32. In half precision the tail of a fifteen way distribution underflows
    and the entropy comes back flattered.

    `z_override` and `ret_value` pass through to CachedTile.iter_full, for diagnostics. `channels`
    defaults to what the network reads, so a three channel checkpoint is never handed a fourth column.
    """
    if votes < 1:
        raise ValueError(f"votes must be at least 1, got {votes}")
    if channels is None:
        channels = 3 + int(getattr(net, "extra", 1))
    total: np.ndarray | None = None
    seen = np.zeros(tile.n, dtype=bool)
    amp = dev == "cuda"
    pending: list[tuple[np.ndarray, np.ndarray, int]] = []

    def flush() -> None:
        nonlocal total
        if not pending:
            return
        x = torch.from_numpy(np.stack([p[0] for p in pending])).to(dev, non_blocking=True)
        with torch.amp.autocast(dev, dtype=torch.float16, enabled=amp):
            logits = net(x)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        if total is None:
            total = np.zeros((tile.n, probs.shape[-1]), dtype=np.float32)
        for row, (_, chunk, valid) in enumerate(pending):
            idx = chunk[:valid]
            total[idx] += probs[row, :valid]
            seen[idx] = True
        pending.clear()

    for v in range(votes):
        torch.manual_seed(seed + v)
        for feat, _, chunk, valid in tile.iter_full(
            block_m, n_points, max_blocks, z_override, ret_value, channels, seed=seed + v
        ):
            pending.append((feat, chunk, valid))
            if len(pending) == batch:
                flush()
        flush()
    if total is None:
        return np.zeros((tile.n, N_CLASSES), dtype=np.float32), seen
    return total / np.float32(votes), seen


def infer_tile(
    net,
    tile: CachedTile,
    dev: str,
    block_m: float,
    n_points: int,
    max_blocks: int | None = None,
    batch: int = 4,
    seed: int = DEFAULT_SEED,
    z_override=None,
    ret_value: float | None = None,
    channels: int | None = None,
    votes: int = 1,
):
    """Per point: argmax class, its probability, normalised entropy, and coverage, from the mean
    softmax over `votes` passes (vote_probabilities)."""
    probs, seen = vote_probabilities(
        net,
        tile,
        dev,
        block_m,
        n_points,
        max_blocks,
        batch,
        seed,
        votes,
        z_override,
        ret_value,
        channels,
    )
    pred = np.where(seen, probs.argmax(axis=1), -1).astype(np.int8)
    conf = np.where(seen, probs.max(axis=1), 0.0).astype(np.float32)
    # Normalised by the model's own class count, so 1 is a uniform guess over its outputs.
    ent = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1) / np.float32(
        np.log(probs.shape[1])
    )
    ent = np.where(seen, ent, 0.0).astype(np.float32)
    return pred, conf, ent, seen
