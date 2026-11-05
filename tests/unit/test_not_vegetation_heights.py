"""Where the NAIP not-vegetation cores stand: their size and height, counted with no model involved.

Fed small inputs whose answer is known, including an area with no core at all and a mask that does
not match its tile.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


notveg = load("archive/investigations/not_vegetation_heights.py", "not_vegetation_heights")


def test_the_not_vegetation_cores_are_counted_with_their_height(tmp_path, monkeypatch):
    prep, ref = tmp_path / "prep", tmp_path / "reference"
    ref.mkdir()
    # A park where 2 of 5 points are a core, 3 m and 21 m up; a city block with no core at all.
    for name, core, hag in (
        ("park", [0, 1, 0, 1, 0], [0.1, 3.0, 0.2, 21.0, 15.0]),
        ("block", [0, 0, 0], [5.0, 6.0, 0.0]),
    ):
        (prep / name).mkdir(parents=True)
        np.save(prep / name / "hag_cm.npy", (np.array(hag) * 100).astype(np.int16))
        np.save(prep / name / "asprs.npy", np.ones(len(hag), dtype=np.uint8))
        np.savez(ref / f"naip_{name}.npz", not_vegetation=np.array(core, dtype=bool))
    out = tmp_path / "out.json"
    monkeypatch.setattr(notveg, "PREP", prep)
    monkeypatch.setattr(notveg, "REFERENCE", ref)
    monkeypatch.setattr(notveg, "OUT", out)
    assert notveg.main() == 0
    areas = json.loads(out.read_text(encoding="utf-8"))["areas"]
    park = areas["park"]
    assert (park["points"], park["share_of_area"], park["share_above_2m"]) == (2, 0.4, 1.0)
    assert park["height_above_floor_m"]["median"] == 12.0
    assert park["delivery_asprs_share"] == {"1": 1.0}
    assert areas["block"] == {"points": 0, "area_points": 3, "share_of_area": 0.0}


def test_a_mask_that_does_not_match_its_tile_stops_the_count(tmp_path, monkeypatch):
    prep, ref = tmp_path / "prep", tmp_path / "reference"
    (prep / "park").mkdir(parents=True)
    ref.mkdir()
    np.save(prep / "park" / "hag_cm.npy", np.zeros(4, dtype=np.int16))
    np.save(prep / "park" / "asprs.npy", np.ones(4, dtype=np.uint8))
    np.savez(ref / "naip_park.npz", not_vegetation=np.ones(3, dtype=bool))
    monkeypatch.setattr(notveg, "PREP", prep)
    monkeypatch.setattr(notveg, "REFERENCE", ref)
    monkeypatch.setattr(notveg, "OUT", tmp_path / "out.json")
    with pytest.raises(SystemExit):
        notveg.main()
