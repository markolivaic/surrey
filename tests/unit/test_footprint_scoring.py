"""The footprint score, including the cases where it must refuse to flatter.

The null is the point of these tests. A footprint interior holds ground, cars and
overhanging branches as well as roofs, so a model that calls everything a building
scores well on precision by itself. Every case here checks the measured number
against the null for the same population.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.footprints import (  # noqa: E402
    clip_to_box,
    coverage_with_null,
    footprint_coverage,
    footprints_hit,
    inside_mask,
    score_building,
)

shapely = pytest.importorskip("shapely")


def two_squares():
    """Two 10 m squares, at x 0-10 and x 20-30, both y 0-10."""
    return [shapely.box(0, 0, 10, 10), shapely.box(20, 0, 30, 10)]


def grid_points(x0, x1, y0, y1, step=1.0):
    gx, gy = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return gx.ravel().astype(float), gy.ravel().astype(float)


def test_inside_mask_matches_the_polygons():
    polys = two_squares()
    x = np.array([5.0, 15.0, 25.0, 35.0])
    y = np.array([5.0, 5.0, 5.0, 5.0])
    assert inside_mask(polys, x, y).tolist() == [True, False, True, False]


def test_inside_mask_with_no_polygons_is_all_false():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    assert inside_mask([], x, y).tolist() == [False, False]


def test_clip_to_box_drops_footprints_outside_the_points():
    polys = two_squares()
    x, y = grid_points(0, 11, 0, 11)
    clipped, area = clip_to_box(polys, x, y)
    assert len(clipped) == 1
    assert area == pytest.approx(100.0, rel=0.05)


def test_footprints_hit_only_counts_selected_points():
    polys = two_squares()
    x = np.array([5.0, 25.0])
    y = np.array([5.0, 5.0])
    only_first = np.array([True, False])
    assert footprints_hit(polys, x, y, only_first).tolist() == [True, False]
    assert footprints_hit(polys, x, y, np.ones(2, bool)).tolist() == [True, True]


def test_a_perfect_prediction_beats_its_null():
    # A model that picks exactly the points inside the footprint takes all of the
    # room above the null. The null is asserted against the measured inside
    # fraction rather than a written constant: clip_to_box shrinks the polygon to
    # the point bounding box and boundary points are excluded, so a hand written
    # 0.5 here tests the fixture instead of the code.
    polys = [shapely.box(0, 0, 10, 10)]
    x, y = grid_points(0, 20, 0, 10)
    clipped, _ = clip_to_box(polys, x, y)
    inside = inside_mask(clipped, x, y)
    assert 0.2 < inside.mean() < 0.8, "fixture should be genuinely mixed"

    r = score_building(clipped, x, y, predicted_building=inside)
    assert r["null_precision"] == pytest.approx(float(inside.mean()), abs=1e-4)
    assert r["precision"] == pytest.approx(1.0)
    assert r["precision_lift_over_null"] == pytest.approx(1.0, abs=0.05)
    assert r["recall"] == pytest.approx(1.0)


def test_predicting_everything_scores_exactly_the_null_and_no_lift():
    # This is the case the metric must not reward. Calling every point a building
    # has to come back with zero lift, not with a precision that reads as good.
    polys = [shapely.box(0, 0, 10, 10)]
    x, y = grid_points(0, 20, 0, 10)
    clipped, _ = clip_to_box(polys, x, y)

    r = score_building(clipped, x, y, predicted_building=np.ones(x.size, bool))
    assert r["precision"] == pytest.approx(r["null_precision"])
    assert r["precision_lift_over_null"] == pytest.approx(0.0, abs=1e-6)
    assert r["recall"] == pytest.approx(r["null_recall"])


def test_a_worse_than_null_prediction_reports_negative_lift():
    # Picking only points outside the footprint must go negative, not clamp to
    # zero. A guard that cannot report worse than nothing hides a broken model.
    polys = [shapely.box(0, 0, 10, 10)]
    x, y = grid_points(0, 20, 0, 10)
    clipped, _ = clip_to_box(polys, x, y)
    outside = ~inside_mask(clipped, x, y)

    r = score_building(clipped, x, y, predicted_building=outside)
    assert r["precision"] == pytest.approx(0.0)
    assert r["precision_lift_over_null"] < 0.0


def test_predicting_nothing_gives_no_precision_rather_than_a_perfect_one():
    # An empty prediction has an undefined precision. Returning 1.0 here, or 0.0,
    # would let a model that predicts no buildings at all look like a result.
    polys = [shapely.box(0, 0, 10, 10)]
    x, y = grid_points(0, 20, 0, 10)
    clipped, _ = clip_to_box(polys, x, y)

    r = score_building(clipped, x, y, predicted_building=np.zeros(x.size, bool))
    assert r["precision"] is None
    assert r["recall"] == pytest.approx(0.0)
    assert r["predicted_building_points"] == 0


def test_the_null_follows_the_eligible_population():
    # Restricting to points over 2 m raises the null, because ground leaves the
    # denominator. If the null did not move with the population the pair would
    # not be comparable.
    polys = [shapely.box(0, 0, 10, 10)]
    x, y = grid_points(0, 20, 0, 10)
    clipped, _ = clip_to_box(polys, x, y)
    inside = inside_mask(clipped, x, y)

    # Everything inside the footprint is "tall", plus a quarter of the outside.
    eligible = inside.copy()
    outside_idx = np.flatnonzero(~inside)
    eligible[outside_idx[: outside_idx.size // 4]] = True

    wide = score_building(clipped, x, y, np.ones(x.size, bool))
    narrow = score_building(clipped, x, y, np.ones(x.size, bool), eligible=eligible)
    assert narrow["null_precision"] > wide["null_precision"]
    assert narrow["eligible_points"] < wide["eligible_points"]


# --------------------------------------------------------------------------
# Area coverage. The reason this exists: one point inside a polygon should not
# count as having found the building.
# --------------------------------------------------------------------------


def test_one_point_inside_a_footprint_is_not_coverage():
    # This is the whole case. A single point wins the touch test outright and has
    # to score almost nothing on area.
    poly = [shapely.box(0, 0, 20, 20)]
    x = np.array([10.0])
    y = np.array([10.0])
    sel = np.array([True])

    assert footprints_hit(poly, x, y, sel).tolist() == [True]
    cov = footprint_coverage(poly, x, y, sel, cell_m=1.0)
    assert cov["footprints_scored"] == 1
    assert cov["coverage_median"] < 0.01


def test_a_fully_covered_footprint_scores_one():
    poly = [shapely.box(0, 0, 10, 10)]
    x, y = grid_points(0, 10, 0, 10, step=0.5)
    sel = np.ones(x.size, bool)
    cov = footprint_coverage(poly, x, y, sel, cell_m=1.0)
    assert cov["coverage_median"] == pytest.approx(1.0)


def test_half_a_footprint_covered_scores_about_half():
    poly = [shapely.box(0, 0, 20, 10)]
    x, y = grid_points(0, 20, 0, 10, step=0.5)
    sel = x < 10.0
    cov = footprint_coverage(poly, x, y, sel, cell_m=1.0)
    assert cov["coverage_median"] == pytest.approx(0.5, abs=0.06)


def test_no_selected_points_gives_zero_coverage_not_none():
    poly = [shapely.box(0, 0, 10, 10)]
    x, y = grid_points(0, 10, 0, 10, step=0.5)
    cov = footprint_coverage(poly, x, y, np.zeros(x.size, bool), cell_m=1.0)
    assert cov["coverage_median"] == pytest.approx(0.0)
    assert cov["footprints_scored"] == 1


def test_coverage_has_a_high_null_and_that_is_reported_not_hidden():
    # Lidar covers the whole footprint whatever the labels say, so selecting every
    # point saturates coverage. The metric is therefore one sided: it catches
    # under-covering only. This pins that behaviour so nobody reads a high
    # coverage as a strong result.
    poly = [shapely.box(0, 0, 20, 20)]
    x, y = grid_points(0, 20, 0, 20, step=0.5)
    null = footprint_coverage(poly, x, y, np.ones(x.size, bool), cell_m=1.0)
    assert null["coverage_median"] > 0.95


def test_footprints_too_small_to_score_are_counted_separately():
    # A 1.5 m square holds fewer cells than min_cells, so its coverage would be a
    # ratio of small integers. It must be excluded and declared, not averaged in.
    polys = [shapely.box(0, 0, 1.5, 1.5), shapely.box(5, 5, 15, 15)]
    x, y = grid_points(0, 20, 0, 20, step=0.5)
    cov = footprint_coverage(polys, x, y, np.ones(x.size, bool), cell_m=1.0, min_cells=4)
    assert cov["footprints"] == 2
    assert cov["footprints_scored"] == 1
    assert cov["footprints_too_small"] == 1


def test_coverage_spread_is_reported_so_a_median_cannot_hide_a_split():
    # Four footprints, two fully covered and two untouched. The median reads 0.5
    # and looks like uniform mediocrity, so the percentiles have to show that
    # nothing is actually near 0.5.
    polys = [
        shapely.box(0, 0, 10, 10),
        shapely.box(15, 0, 25, 10),
        shapely.box(30, 0, 40, 10),
        shapely.box(45, 0, 55, 10),
    ]
    x, y = grid_points(0, 55, 0, 10, step=0.5)
    sel = x < 25.0
    cov = footprint_coverage(polys, x, y, sel, cell_m=1.0)
    assert cov["footprints_scored"] == 4
    assert cov["coverage_median"] == pytest.approx(0.5, abs=0.1)
    assert cov["coverage_p10"] < 0.15
    assert cov["coverage_p90"] > 0.85
    assert not any(0.2 < v < 0.8 for v in cov["per_footprint"])


def test_coverage_with_no_footprints_returns_empty_rather_than_failing():
    x, y = grid_points(0, 10, 0, 10)
    cov = footprint_coverage([], x, y, np.ones(x.size, bool))
    assert cov["footprints"] == 0
    assert cov["coverage_median"] is None


def test_coverage_with_null_never_returns_one_half_without_the_other():
    # The pairing is the point. Any caller gets both keys or the call fails.
    polys = [shapely.box(0, 0, 20, 20)]
    x, y = grid_points(0, 20, 0, 20, step=0.5)
    eligible = np.ones(x.size, bool)
    r = coverage_with_null(polys, x, y, x < 10.0, eligible, cell_m=1.0)
    for key in ("coverage_median", "null_coverage_median", "shortfall_against_null"):
        assert key in r and r[key] is not None, key


def test_a_perfect_coverage_has_no_shortfall_against_its_null():
    polys = [shapely.box(0, 0, 20, 20)]
    x, y = grid_points(0, 20, 0, 20, step=0.5)
    eligible = np.ones(x.size, bool)
    r = coverage_with_null(polys, x, y, eligible, eligible, cell_m=1.0)
    assert r["coverage_median"] == pytest.approx(r["null_coverage_median"])
    assert r["shortfall_against_null"] == pytest.approx(0.0, abs=1e-6)


def test_covering_half_the_reachable_area_shows_up_as_a_shortfall():
    polys = [shapely.box(0, 0, 20, 20)]
    x, y = grid_points(0, 20, 0, 20, step=0.5)
    eligible = np.ones(x.size, bool)
    r = coverage_with_null(polys, x, y, x < 10.0, eligible, cell_m=1.0)
    assert r["shortfall_against_null"] == pytest.approx(0.5, abs=0.08)


def test_the_null_follows_the_eligible_set_in_coverage_too():
    # Restricting eligibility lowers what is reachable, so the null must drop with
    # it. A fixed null would make the shortfall meaningless.
    polys = [shapely.box(0, 0, 20, 20)]
    x, y = grid_points(0, 20, 0, 20, step=0.5)
    wide = coverage_with_null(polys, x, y, np.ones(x.size, bool), np.ones(x.size, bool), cell_m=1.0)
    narrow = coverage_with_null(polys, x, y, np.ones(x.size, bool), x < 10.0, cell_m=1.0)
    assert narrow["null_coverage_median"] < wide["null_coverage_median"]
