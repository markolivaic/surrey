"""No prepared tile holds a point with a LAS flag the preparation drops (surrey.flags).

Swiss water was almost all synthetic, and a model learned water from those points before anything
read the flag. The prepared tiles are built locally and not committed, so this runs where they are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.cache import write_tile  # noqa: E402
from surrey.flags import DROP, SYNTHETIC, WITHHELD  # noqa: E402

PREP = ROOT / "data" / "prep"


def dropped_in(tile_dir: Path) -> int | None:
    """Points with a dropped flag in one prepared tile, or None if it carries no flags at all."""
    f = tile_dir / "flags.npy"
    if not f.exists():
        return None
    return int(((np.load(f) & DROP) != 0).sum())


def test_no_prepared_tile_holds_a_dropped_flag():
    tiles = (
        sorted(p for p in PREP.glob("*/*") if (p / "meta.json").exists()) if PREP.exists() else []
    )
    if not tiles:
        pytest.skip("no prepared data in this checkout; the pipeline builds it locally")
    unflagged = [p.relative_to(PREP).as_posix() for p in tiles if dropped_in(p) is None]
    assert not unflagged, "prepared before the flag filter, prepare again:\n" + "\n".join(unflagged)
    bad = {p.relative_to(PREP).as_posix(): dropped_in(p) for p in tiles if dropped_in(p)}
    assert not bad, f"prepared tiles holding points with a dropped flag: {bad}"


def test_self_test_a_tile_with_a_synthetic_or_withheld_point_is_caught(tmp_path):
    n = 50
    arrays = {
        "xy_cm": np.zeros((n, 2)),
        "z_cm": np.zeros(n),
        "hag_cm": np.zeros(n),
        "ret_q": np.zeros(n),
        "label": np.zeros(n),
        "asprs": np.full(n, 2),
        "flags": np.zeros(n, np.uint8),
    }
    write_tile(tmp_path / "clean", arrays, {"origin": [0.0, 0.0, 0.0]})
    arrays["flags"][[3, 7]] = [SYNTHETIC, WITHHELD]
    write_tile(tmp_path / "dirty", arrays, {"origin": [0.0, 0.0, 0.0]})
    assert dropped_in(tmp_path / "clean") == 0
    assert dropped_in(tmp_path / "dirty") == 2
