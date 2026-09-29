"""A run stopped at step k and resumed draws the same batches and ends with the same weights as a
run that never stopped. On synthetic tiles and the CPU, so it runs anywhere, CI included.

The GPU is not bit for bit deterministic (non-deterministic kernels), so on the card only the batch
order is expected to match; scripts/investigations/check_resume_gpu.py records that separately. On the CPU both
must match exactly, with one thread: two unbroken CPU runs with the default thread count already
differed in the third decimal of some weights after four steps (seen 2026-09-15), and
with OMP_NUM_THREADS=1 two unbroken runs matched bit for bit. The thread count is pinned for that
reason, and the self-test below checks that two unbroken runs agree, so a weight match means
something. A checkpoint from before random states were saved must resume with a
different batch order and say it was not exact.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import write_tile  # noqa: E402

TRAIN = ROOT / "scripts" / "pipeline" / "train.py"
COMMON = [
    "--device",
    "cpu",
    "--recipe",
    "m0",
    "--channels",
    "3",
    "--steps",
    "4",
    "--batch",
    "2",
    "--points",
    "4096",
    "--width",
    "16",
    "--eval-every",
    "1000",
    "--eval-blocks",
    "1",
    "--final-eval-blocks",
    "1",
    "--ckpt-every",
    "1000",
    "--seed",
    "3",
]


def synthetic_tiles(root: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for i, split in enumerate(("train", "train", "test")):
        n = 30000
        xy = rng.uniform(0, 60, (n, 2))
        z = rng.uniform(0, 5, n)
        arrays = {
            "xy_cm": (xy * 100).astype(np.uint16),
            "z_cm": (z * 100).astype(np.uint32),
            "hag_cm": (z * 100).astype(np.int16),
            "ret_q": np.zeros(n, np.uint8),
            "label": rng.integers(0, 15, n).astype(np.int8),
            "asprs": np.ones(n, np.uint8),
        }
        name = f"synthetic_{i}"
        write_tile(
            root / name,
            arrays,
            {"name": name, "split": split, "origin": [0.0, 0.0, 0.0], "epsg": 32610},
        )
        rows.append({"name": name, "split": split})
    art = root / "split.json"
    art.write_text(json.dumps({"tiles": rows}))
    return art


def train(tmp: Path, art: Path, name: str, *extra: str) -> None:
    cmd = [
        sys.executable,
        str(TRAIN),
        *COMMON,
        "--prep",
        str(tmp),
        "--split-artifact",
        str(art),
        "--ckpt",
        str(tmp / f"{name}.pt"),
        "--out",
        str(tmp / f"{name}.json"),
        "--batch-log",
        str(tmp / f"{name}.batches"),
        *extra,
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]


def weights(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("resume")
    art = synthetic_tiles(tmp)
    train(tmp, art, "unbroken")
    train(tmp, art, "unbroken_again")
    train(tmp, art, "broken", "--stop-after-step", "2")
    train(tmp, art, "broken", "--resume")
    # The legacy case: the same stop, then the checkpoint's random states removed.
    train(tmp, art, "legacy", "--stop-after-step", "2")
    blob = torch.load(tmp / "legacy.pt", map_location="cpu", weights_only=False)
    del blob["rng"]
    torch.save(blob, tmp / "legacy.pt")
    train(tmp, art, "legacy", "--resume")
    return tmp


def batches(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").split()


def test_a_resumed_run_draws_the_same_batches(runs):
    assert batches(runs / "broken.batches") == batches(runs / "unbroken.batches")


def test_a_resumed_run_ends_with_the_same_weights_on_the_cpu(runs):
    a, b = weights(runs / "unbroken.pt"), weights(runs / "broken.pt")
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_self_test_two_unbroken_runs_agree_bit_for_bit(runs):
    a, b = weights(runs / "unbroken.pt"), weights(runs / "unbroken_again.pt")
    assert all(torch.equal(a[k], b[k]) for k in a), (
        "the run is not deterministic; a weight match would be luck"
    )


def test_the_artefact_records_one_exact_interruption(runs):
    rec = json.loads((runs / "broken.json").read_text(encoding="utf-8"))["interruptions"]
    assert rec["count"] == 1 and rec["all_exact"] is True
    assert rec["events"][0]["resumed_from_checkpoint_step"] == 2
    assert (
        json.loads((runs / "unbroken.json").read_text(encoding="utf-8"))["interruptions"]["count"]
        == 0
    )


def test_self_test_a_checkpoint_without_random_state_resumes_differently_and_says_so(runs):
    legacy, unbroken = batches(runs / "legacy.batches"), batches(runs / "unbroken.batches")
    assert legacy[:4] == unbroken[:4], "the first two steps ran before the stop"
    assert legacy[4:] != unbroken[4:], (
        "a reseeded sampler drew the same batches; the test proves nothing"
    )
    rec = json.loads((runs / "legacy.json").read_text(encoding="utf-8"))["interruptions"]
    assert rec["count"] == 1 and rec["all_exact"] is False
