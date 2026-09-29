"""A published weights file loads without unpickling, and a training checkpoint does not load that way.

scripts/pipeline/export_weights.py writes the tensors and plain values only; surrey.inference.load_checkpoint
reads that with torch.load(weights_only=True) by default. A training checkpoint, with its random states and
a path in its arguments, is refused unless the caller says it trained it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.inference import load_checkpoint  # noqa: E402
from surrey.model import SurreyNet  # noqa: E402


def exporter():
    spec = importlib.util.spec_from_file_location(
        "export_weights", ROOT / "scripts" / "pipeline" / "export_weights.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def training_checkpoint(tmp_path: Path) -> Path:
    net = SurreyNet(15, width=32, extra_features=0)
    path = tmp_path / "m0_seedA.pt"
    torch.save(
        {
            "model": net.state_dict(),
            "step": 7,
            "rng": {"numpy": np.random.default_rng(0).bit_generator.state},
            "args": {
                "channels": 3,
                "width": 32,
                "recipe": "m0",
                "seed": 0,
                "prep": PurePosixPath("data/prep/dales"),
            },
        },
        path,
    )
    return path


def test_a_training_checkpoint_is_refused_without_allow_pickle(training_checkpoint):
    with pytest.raises(SystemExit):
        load_checkpoint(training_checkpoint, "cpu")
    net, _blob, channels = load_checkpoint(training_checkpoint, "cpu", allow_pickle=True)
    assert channels == 3


def test_the_exported_weights_load_safely_and_predict_the_same(training_checkpoint, tmp_path):
    target = exporter().export(training_checkpoint, tmp_path / "weights")
    safe, blob, channels = load_checkpoint(target, "cpu")
    full, _b, _c = load_checkpoint(training_checkpoint, "cpu", allow_pickle=True)
    assert channels == 3 and blob["step"] == 7 and blob["args"]["recipe"] == "m0"
    x = torch.rand(1, 2048, 3) * 20
    torch.manual_seed(0)
    a = safe(x)
    torch.manual_seed(0)
    b = full(x)
    assert torch.equal(a, b)
