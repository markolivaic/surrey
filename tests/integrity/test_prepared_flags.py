"""No prepared tile holds a point with a LAS flag the preparation drops (surrey.flags).

Swiss water was almost all synthetic, and a model learned water from those points before anything
read the flag. The prepared tiles are built locally and not committed, so this runs where they are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

from surrey.flags import DROP  # noqa: E402

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
