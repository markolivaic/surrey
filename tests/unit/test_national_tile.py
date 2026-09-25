"""The contamination proxy on synthetic tiles whose answer is known."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path[:0] = [str(ROOT / "scripts" / "pipeline"), str(ROOT / "scripts" / "investigations")]

from national_tile_check import TILES, contamination  # noqa: E402


def street_with_roofs(n_unclassified_roof: int, n_classified_roof: int, n_car: int):
    """Flat ground at z 100 on a 50 m square, roofs 12 m up, cars 1.5 m up."""
    rng = np.random.default_rng(0)
    g = 5000
    xs = [rng.uniform(0, 50, g)]
    ys = [rng.uniform(0, 50, g)]
    zs = [np.full(g, 100.0)]
    cls = [np.full(g, 2)]
    for count, height, code in (
        (n_unclassified_roof, 12.0, 1),
        (n_classified_roof, 12.0, 6),
        (n_car, 1.5, 1),
    ):
        xs.append(rng.uniform(0, 50, count))
        ys.append(rng.uniform(0, 50, count))
        zs.append(np.full(count, 100.0 + height))
        cls.append(np.full(count, code))
    x, y, z, c = (np.concatenate(v) for v in (xs, ys, zs, cls))
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), c.astype(np.uint8)


def test_the_share_counts_only_points_above_two_metres():
    x, y, z, c = street_with_roofs(n_unclassified_roof=300, n_classified_roof=700, n_car=900)
    r = contamination(x, y, z, c, 2, [1], [], np.ones(c.size, bool))
    # Cars at 1.5 m are unclassified but below the gate, so they do not count.
    assert r["points_over_2m"] == 1000
    assert abs(r["unclassified_share_of_points_over_2m"] - 0.3) < 1e-9
    assert r["unclassified_over_2m_height_m_p10_p50_p90"] == [12.0, 12.0, 12.0]


def test_codes_outside_the_keep_mask_are_not_counted():
    x, y, z, c = street_with_roofs(n_unclassified_roof=500, n_classified_roof=500, n_car=0)
    keep = np.ones(c.size, bool)
    keep[c == 1] = False
    r = contamination(x, y, z, c, 2, [1], [], keep)
    assert r["unclassified_share_of_points_over_2m"] == 0.0


def test_a_delivery_with_no_vegetation_class_reports_no_vegetation_share():
    x, y, z, c = street_with_roofs(10, 10, 0)
    r = contamination(x, y, z, c, 2, [1], [], np.ones(c.size, bool))
    assert r["vegetation_share_of_points_over_2m"] is None


def test_every_tile_names_its_class_table_source_and_the_gate_classes():
    for name, spec in TILES.items():
        assert spec["class_table_source"], name
        assert spec["url"].startswith("https://"), name
        for code in spec["vegetation"]:
            assert code in spec["classes"], f"{name}: vegetation code {code} has no sourced name"
