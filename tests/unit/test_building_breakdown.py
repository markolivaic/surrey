"""The building breakdown's rules, the README's reading of it, and the not-vegetation count.

Each produces figures the README prints: the direction of a change between the first round's seeds and
the retrained ones, the reading of precision against coverage written before the comparison ran, and
the size and height of the NAIP not-vegetation cores. Each is fed small inputs whose answer is known,
including ones where the answer is that nothing moved.
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


breakdown = load("scripts/investigations/building_breakdown.py", "building_breakdown")
readme = load("scripts/pipeline/render_readme.py", "render_readme")
notveg = load("scripts/investigations/not_vegetation_heights.py", "not_vegetation_heights")


def test_a_figure_moves_only_when_every_new_seed_lies_beyond_every_old_one():
    assert breakdown.direction([0.9, 0.95, 0.92], [0.8, 0.85, 0.88]) == "rose"
    assert breakdown.direction([0.1, 0.12, 0.15], [0.2, 0.3, 0.25]) == "fell"
    # One new seed inside the old range: nothing moved, however far the medians sit apart.
    assert breakdown.direction([0.5, 0.9, 0.95], [0.4, 0.6, 0.45]) == "did not move"
    assert breakdown.direction([0.5, None, 0.6], [0.4, 0.3, 0.2]) == "not measured"


def test_the_summary_is_median_min_max_over_the_seeds_that_have_a_value():
    s = breakdown.summary([0.3, None, 0.1])
    assert (s["median"], s["min"], s["max"]) == (0.2, 0.1, 0.3)
    assert breakdown.summary([None, None]) is None


def cells(area: str, fig: str, old: tuple[float, float], new: tuple[float, float]) -> dict:
    return {
        f"buildings.{area}.{fig}.old.min": old[0],
        f"buildings.{area}.{fig}.old.max": old[1],
        f"buildings.{area}.{fig}.new.min": new[0],
        f"buildings.{area}.{fig}.new.max": new[1],
    }


@pytest.mark.parametrize(
    ("precision", "coverage", "reading"),
    [
        (((0.8, 0.9), (0.6, 0.7)), ((0.5, 0.6), (0.8, 0.9)), "a trade"),
        (((0.8, 0.9), (0.6, 0.7)), ((0.8, 0.9), (0.5, 0.6)), "worse at buildings here"),
        (
            ((0.8, 0.9), (0.85, 0.95)),
            ((0.5, 0.6), (0.55, 0.65)),
            "precision did not move, coverage did not move",
        ),
        (((0.1, 0.2), (0.3, 0.4)), ((0.5, 0.6), (0.8, 0.9)), "precision rose, coverage rose"),
    ],
)
def test_the_readme_reads_precision_against_coverage_as_written_beforehand(
    precision, coverage, reading
):
    vals = cells("x", "precision", *precision) | cells("x", "coverage_median", *coverage)
    assert readme.building_reading(vals, "x").startswith(reading)


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
